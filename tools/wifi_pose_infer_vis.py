import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.utils import compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root

from opera.datasets import build_dataloader, build_dataset
from opera.models import build_model


SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 5), (3, 0), (4, 2), (5, 7), (6, 3),
    (7, 3), (8, 4), (9, 5), (10, 6), (11, 7), (12, 9), (13, 11)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run WiFiPose inference and save per-sample visualizations.'
    )
    parser.add_argument('--config', default='configs/wifi/petr_wifi_remote.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--mode', default='test', choices=['train', 'test'])
    parser.add_argument('--list-file', default=None)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--sample-name', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-samples', type=int, default=1)
    parser.add_argument('--sample-indices', type=int, nargs='+', default=None)
    parser.add_argument('--save-raw-pred', action='store_true')
    parser.add_argument(
        '--vis-mode',
        default='raw',
        choices=['raw', 'matched', 'both'],
        help='raw: 直接看模型原始预测；matched: 看和GT配对后的预测；both: 两种都保存',
    )
    parser.add_argument(
        '--summary-sort-key',
        default='sample_index',
        choices=['sample_index', 'sample_name', 'mpjpe', 'matched_persons', 'num_pred_persons'],
    )
    parser.add_argument('--summary-sort-desc', action='store_true')
    return parser.parse_args()


def load_model(cfg, checkpoint_path, device):
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = cfg
    model = MMDataParallel(model, device_ids=[0] if device.startswith('cuda') else None)
    model.to(device)
    model.eval()
    return model


def build_test_dataset(cfg, dataset_root, mode, list_file):
    data_cfg = cfg.data.test.copy()
    data_cfg.dataset_root = dataset_root
    data_cfg.mode = mode
    data_cfg.test_mode = True
    if list_file:
        data_cfg.list_file = list_file
    return build_dataset(data_cfg)


def greedy_match(gt_keypoints, pred_keypoints, threshold=50.0):
    gt = torch.as_tensor(gt_keypoints, dtype=torch.float32)
    pred = torch.as_tensor(pred_keypoints, dtype=torch.float32)
    if pred.numel() == 0:
        return torch.full_like(gt, float('nan')), []

    n = gt.shape[0]
    m = pred.shape[0]
    distances = torch.full((n, m), float('inf'))
    for i in range(n):
        for j in range(m):
            distances[i, j] = torch.norm(gt[i] - pred[j], p=2, dim=-1).mean()

    matched_pred = torch.full_like(gt, float('nan'))
    pairs = []
    occupied = torch.zeros(m, dtype=torch.bool)
    while True:
        min_value = distances.min()
        if not torch.isfinite(min_value) or min_value >= threshold:
            break
        i, j = torch.where(distances == min_value)
        i = int(i[0].item())
        j = int(j[0].item())
        distances[i, :] = float('inf')
        distances[:, j] = float('inf')
        if occupied[j]:
            continue
        occupied[j] = True
        matched_pred[i] = pred[j]
        pairs.append((i, j, float(min_value.item())))
    return matched_pred, pairs


def compute_metrics(gt_keypoints, pred_keypoints):
    gt = torch.as_tensor(gt_keypoints, dtype=torch.float32)
    pred = torch.as_tensor(pred_keypoints, dtype=torch.float32)
    valid = torch.isfinite(pred).all(dim=-1).all(dim=-1)
    if not valid.any():
        return {
            'matched_persons': 0,
            'mpjpe': None,
            'mpjpeh': None,
            'mpjpev': None,
            'mpjped': None,
        }

    gt = gt[valid]
    pred = pred[valid]
    diff = gt - pred
    return {
        'matched_persons': int(valid.sum().item()),
        'mpjpe': float(torch.sqrt(torch.pow(diff, 2).sum(-1)).mean().item() * 1000),
        'mpjpeh': float(torch.abs(diff[:, :, 0]).mean().item() * 1000),
        'mpjpev': float(torch.abs(diff[:, :, 1]).mean().item() * 1000),
        'mpjped': float(torch.abs(diff[:, :, 2]).mean().item() * 1000),
    }


def plot_people(ax, keypoints, title, color):
    pts = np.asarray(keypoints, dtype=np.float32)
    valid_people = 0
    for person in pts:
        if not np.isfinite(person).all():
            continue
        valid_people += 1
        ax.scatter(person[:, 0], person[:, 1], person[:, 2], c=color, s=18)
        for start, end in SKELETON_EDGES:
            ax.plot(
                [person[start, 0], person[end, 0]],
                [person[start, 1], person[end, 1]],
                [person[start, 2], person[end, 2]],
                color=color,
                linewidth=1.5,
            )
    ax.set_title(f'{title} ({valid_people})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=18, azim=-62)


def save_visualization(output_path, gt_keypoints, pred_keypoints, sample_name, metrics, scores, pred_title='Pred'):
    fig = plt.figure(figsize=(12, 6))
    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pred = fig.add_subplot(1, 2, 2, projection='3d')
    plot_people(ax_gt, gt_keypoints, 'GT', '#1f77b4')
    plot_people(ax_pred, pred_keypoints, pred_title, '#d62728')
    fig.suptitle(
        f'{sample_name} | mpjpe={metrics["mpjpe"]} | scores={scores}',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def sanitize_name(sample_name):
    return sample_name.replace('/', '_')


def build_sample_indices(args, dataset):
    if args.sample_name is not None:
        if args.sample_name not in dataset.filename_list:
            raise ValueError(f'sample not found: {args.sample_name}')
        return [dataset.filename_list.index(args.sample_name)]
    if args.sample_indices:
        indices = args.sample_indices
    else:
        start = args.sample_index
        end = min(start + max(args.max_samples, 1), len(dataset))
        indices = list(range(start, end))
    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f'sample_index out of range: {idx}')
    return indices


def extract_predictions(outputs, score_thr, top_k):
    bbox_results, keypoint_results = outputs[0]
    pred_bboxes = bbox_results[0] if bbox_results else np.zeros((0, 5), dtype=np.float32)
    pred_keypoints = keypoint_results[0] if keypoint_results else np.zeros((0, 14, 3), dtype=np.float32)

    if pred_bboxes.size > 0 and pred_bboxes.shape[1] >= 5:
        keep = pred_bboxes[:, 4] >= score_thr
        pred_bboxes = pred_bboxes[keep]
        pred_keypoints = pred_keypoints[keep]
        if top_k > 0 and pred_bboxes.shape[0] > top_k:
            order = np.argsort(-pred_bboxes[:, 4])[:top_k]
            pred_bboxes = pred_bboxes[order]
            pred_keypoints = pred_keypoints[order]
    return pred_bboxes, pred_keypoints


def save_sample_outputs(
    base_output_dir,
    sample_name,
    gt_keypoints,
    pred_keypoints,
    matched_pred,
    result_summary,
    save_raw_pred,
):
    safe_name = sanitize_name(sample_name)
    sample_dir = base_output_dir / safe_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.save(sample_dir / f'{safe_name}_gt.npy', gt_keypoints)
    np.save(sample_dir / f'{safe_name}_pred_matched.npy', matched_pred.cpu().numpy())
    if save_raw_pred:
        np.save(sample_dir / f'{safe_name}_pred.npy', pred_keypoints)

    if result_summary['vis_mode'] in ('raw', 'both'):
        save_visualization(
            output_path=sample_dir / f'{safe_name}_vis_raw.png',
            gt_keypoints=gt_keypoints,
            pred_keypoints=pred_keypoints,
            sample_name=sample_name,
            metrics=result_summary['metrics_mm'],
            scores=result_summary['prediction_scores'],
            pred_title='Pred Raw',
        )

    if result_summary['vis_mode'] in ('matched', 'both'):
        save_visualization(
            output_path=sample_dir / f'{safe_name}_vis_matched.png',
            gt_keypoints=gt_keypoints,
            pred_keypoints=matched_pred.cpu().numpy(),
            sample_name=sample_name,
            metrics=result_summary['metrics_mm'],
            scores=result_summary['prediction_scores'],
            pred_title='Pred Matched',
        )

    with open(sample_dir / f'{safe_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)

    return sample_dir


def summarize_results(results):
    matched = [item['metrics_mm']['matched_persons'] for item in results]
    mpjpe_values = [
        item['metrics_mm']['mpjpe']
        for item in results
        if item['metrics_mm']['mpjpe'] is not None
    ]
    return {
        'num_samples': len(results),
        'num_samples_with_match': sum(1 for value in matched if value > 0),
        'mean_mpjpe': float(np.mean(mpjpe_values)) if mpjpe_values else None,
        'median_mpjpe': float(np.median(mpjpe_values)) if mpjpe_values else None,
        'min_mpjpe': float(np.min(mpjpe_values)) if mpjpe_values else None,
        'max_mpjpe': float(np.max(mpjpe_values)) if mpjpe_values else None,
    }


def sort_results(results, sort_key, descending):
    def key_fn(item):
        if sort_key == 'mpjpe':
            value = item['metrics_mm']['mpjpe']
            return float('inf') if value is None else value
        if sort_key == 'matched_persons':
            return item['metrics_mm']['matched_persons']
        return item[sort_key]

    return sorted(results, key=key_fn, reverse=descending)


def save_batch_summary(output_dir, args, sample_results):
    ordered_results = sort_results(sample_results, args.summary_sort_key, args.summary_sort_desc)
    summary = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'dataset_root': os.path.abspath(args.dataset_root),
        'mode': args.mode,
        'list_file': os.path.abspath(args.list_file) if args.list_file else None,
        'device': args.device,
        'score_threshold': args.score_thr,
        'top_k': args.top_k,
        'max_samples': args.max_samples,
        'requested_sample_index': args.sample_index,
        'requested_sample_name': args.sample_name,
        'requested_sample_indices': args.sample_indices,
        'summary_sort_key': args.summary_sort_key,
        'summary_sort_desc': args.summary_sort_desc,
        'aggregate': summarize_results(sample_results),
        'samples': ordered_results,
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.device = get_device()

    dataset = build_test_dataset(
        cfg=cfg,
        dataset_root=args.dataset_root,
        mode=args.mode,
        list_file=args.list_file,
    )
    target_indices = build_sample_indices(args, dataset)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    pending = set(target_indices)
    selected_batches = {}
    for idx, data in enumerate(data_loader):
        if idx in pending:
            selected_batches[idx] = data
            pending.remove(idx)
        if not pending:
            break
    if pending:
        raise IndexError(f'sample_index out of range: {sorted(pending)}')

    model = load_model(cfg, args.checkpoint, args.device)

    sample_results = []
    for sample_index in target_indices:
        sample_name = dataset.filename_list[sample_index]
        gt_keypoints = dataset._load_keypoints(sample_name).cpu().numpy()

        with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, **selected_batches[sample_index])
        pred_bboxes, pred_keypoints = extract_predictions(outputs, args.score_thr, args.top_k)
        matched_pred, pairs = greedy_match(gt_keypoints, pred_keypoints)
        metrics = compute_metrics(gt_keypoints, matched_pred)
        scores = pred_bboxes[:, 4].tolist() if pred_bboxes.size > 0 else []

        safe_name = sanitize_name(sample_name)
        result_summary = {
            'sample_name': sample_name,
            'sample_index': sample_index,
            'checkpoint': os.path.abspath(args.checkpoint),
            'dataset_root': os.path.abspath(args.dataset_root),
            'list_file': os.path.abspath(args.list_file) if args.list_file else None,
            'score_threshold': args.score_thr,
            'top_k': args.top_k,
            'vis_mode': args.vis_mode,
            'num_gt_persons': int(gt_keypoints.shape[0]),
            'num_pred_persons': int(pred_keypoints.shape[0]),
            'prediction_scores': scores,
            'matches': [
                {'gt_index': gt_idx, 'pred_index': pred_idx, 'mean_l2': dist}
                for gt_idx, pred_idx, dist in pairs
            ],
            'metrics_mm': metrics,
            'artifacts': {
                'sample_dir': safe_name,
                'gt_npy': f'{safe_name}_gt.npy',
                'pred_npy': f'{safe_name}_pred.npy' if args.save_raw_pred else None,
                'pred_matched_npy': f'{safe_name}_pred_matched.npy',
                'vis_raw_png': f'{safe_name}_vis_raw.png' if args.vis_mode in ('raw', 'both') else None,
                'vis_matched_png': f'{safe_name}_vis_matched.png' if args.vis_mode in ('matched', 'both') else None,
            },
        }
        save_sample_outputs(
            base_output_dir=output_dir,
            sample_name=sample_name,
            gt_keypoints=gt_keypoints,
            pred_keypoints=pred_keypoints,
            matched_pred=matched_pred,
            result_summary=result_summary,
            save_raw_pred=args.save_raw_pred,
        )
        sample_results.append(result_summary)
        print(json.dumps(result_summary, ensure_ascii=False))

    summary = save_batch_summary(output_dir, args, sample_results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
