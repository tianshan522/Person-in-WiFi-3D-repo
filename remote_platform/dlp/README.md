# Person-in-WiFi-3D 远程训练复用流程

这套文件从旧的 `SAM` 项目里复用了三层能力：

- 本地 NFS 云盘挂载后的增量同步
- DLP 作业提交和状态轮询
- 远端容器内的 smoke test / train 统一入口

所有新增文件都集中在 `remote_platform/dlp/`，不改现有 `tools/` 和 `configs/` 主流程。

当前请优先以 `docs/远端训练说明.md` 作为实际操作说明，本 README 更偏向说明 `remote_platform/dlp/` 目录本身的职责。

## 1. 本地挂载云盘

```bash
mkdir -p ~/Documents/<你的云盘目录名>
sudo mount -t nfs -o nolock,resvport,rw 172.30.2.93:/cloud/tmp/<你的云盘目录名> ~/Documents/<你的云盘目录名>
```

后文假设挂载目录是：

```bash
~/Documents/<你的云盘目录名>
```

## 2. 同步仓库到云盘

```bash
PIWIFI_CLOUD_MOUNT_DIR=~/Documents/<你的云盘目录名> \
PIWIFI_CLOUD_REPO_DIR=~/Documents/<你的云盘目录名>/Person-in-WiFi-3D-repo \
zsh remote_platform/dlp/sync_to_cloud_nfs.sh
```

脚本会同步这些必要目录：

- `configs`
- `docs`
- `opera`
- `requirements`
- `third_party/mmcv`
- `third_party/mmdet`
- `tools`
- `remote_platform`
- `setup.py`
- `setup.cfg`
- `requirements.txt`
- `README.md`

如果本地存在 `data/wifipose`，也会一起同步。

## 3. 远端路径约定

Mac 上的挂载路径不能直接拿给 DLP，用容器内可见的 NFS 路径，例如：

```bash
export DLP_REMOTE_NFS_PATH=/cloud/tmp/yc
export DLP_REMOTE_REPO_DIR=/cloud/tmp/yc/Person-in-WiFi-3D-repo
export PIWIFI_DATASET_ROOT=/cloud/tmp/yc/Person-in-WiFi-3D-repo/data/wifipose
```

如果远端已经准备好 Python，也可以显式指定：

```bash
export PIWIFI_REMOTE_PYTHON=/cloud/tmp/yc/miniconda3/envs/piwifi/bin/python
```

## 4. 先跑 smoke test

先确认四件事：

- DLP 能挂到 NFS
- 远端 Python 可用
- `mmcv/mmdet/pywt/h5py/scipy/torch` 能导入
- WiFiPose 数据目录结构完整

```bash
DLP_REMOTE_NFS_PATH=/cloud/tmp/yc \
DLP_REMOTE_REPO_DIR=/cloud/tmp/yc/Person-in-WiFi-3D-repo \
PIWIFI_DATASET_ROOT=/cloud/tmp/yc/Person-in-WiFi-3D-repo/data/wifipose \
PIWIFI_REMOTE_PYTHON=/cloud/tmp/yc/miniconda3/envs/piwifi/bin/python \
python remote_platform/dlp/dlpflow_remote_smoke_test.py
```

通过标志：

- 日志里有 `remote_dataset_ok`
- 日志里有 `config_load_ok`
- 最后打印 `remote_smoke_ok`

## 5. 启动正式训练

单卡：

```bash
DLP_REMOTE_NFS_PATH=/cloud/tmp/yc \
DLP_REMOTE_REPO_DIR=/cloud/tmp/yc/Person-in-WiFi-3D-repo \
PIWIFI_DATASET_ROOT=/cloud/tmp/yc/Person-in-WiFi-3D-repo/data/wifipose \
PIWIFI_REMOTE_PYTHON=/cloud/tmp/yc/miniconda3/envs/piwifi/bin/python \
PIWIFI_CONFIG_PATH=configs/wifi/petr_wifi_remote.py \
PIWIFI_WORK_DIR=/cloud/tmp/yc/Person-in-WiFi-3D-repo/work_dirs/remote_train_full_resume20epochs_20260406 \
python remote_platform/dlp/dlpflow_train.py
```

多卡：

```bash
DLP_REMOTE_NFS_PATH=/cloud/tmp/yc \
DLP_REMOTE_REPO_DIR=/cloud/tmp/yc/Person-in-WiFi-3D-repo \
PIWIFI_DATASET_ROOT=/cloud/tmp/yc/Person-in-WiFi-3D-repo/data/wifipose \
PIWIFI_REMOTE_PYTHON=/cloud/tmp/yc/miniconda3/envs/piwifi/bin/python \
DLP_GPU_NUM=2 \
PIWIFI_GPUS=2 \
PIWIFI_USE_DIST=1 \
PIWIFI_CONFIG_PATH=configs/wifi/petr_wifi_remote.py \
PIWIFI_WORK_DIR=/cloud/tmp/yc/Person-in-WiFi-3D-repo/work_dirs/petr_wifi_remote_2gpu \
python remote_platform/dlp/dlpflow_train.py
```

## 6. 常用覆盖项

训练脚本不会改原始配置文件，而是通过 `--cfg-options` 覆盖数据和输出路径。常用环境变量：

- `PIWIFI_CONFIG_PATH`
- `PIWIFI_DATASET_ROOT`
- `PIWIFI_WORK_DIR`
- `PIWIFI_GPUS`
- `PIWIFI_USE_DIST`
- `PIWIFI_SAMPLES_PER_GPU`
- `PIWIFI_WORKERS_PER_GPU`
- `PIWIFI_RESUME_FROM`
- `PIWIFI_CFG_OPTIONS`

例如额外覆盖学习率和 batch：

```bash
PIWIFI_CFG_OPTIONS='optimizer.lr=1e-5 data.samples_per_gpu=16'
```

恢复训练：

```bash
PIWIFI_RESUME_FROM=/cloud/tmp/yc/Person-in-WiFi-3D-repo/work_dirs/petr_wifi_remote/latest.pth
```

## 7. 远端安装依赖

如果远端环境还没准备好，可以在作业里临时安装：

```bash
PIWIFI_REMOTE_INSTALL_DEPS=1
```

这会执行：

- `pip install -r remote_platform/dlp/requirements-remote-base.txt`
- `pip install -e third_party/mmcv`
- `pip install -e third_party/mmdet`
- `pip install -e .`

更稳妥的做法仍然是提前在你们的远端镜像或 conda 环境里装好 `torch` 和 CUDA 匹配版本，再用这套脚本负责挂载、同步和提交流程。

## 8. 当前这套目录中最常用的文件

- `dlpflow_train.py`
  - DLP 训练任务提交入口
- `run_remote_train.sh`
  - 远端训练主脚本
- `validate_remote_dataset.py`
  - 训练前数据校验
- `sync_to_cloud_nfs.sh`
  - macOS 本地仓库到挂载目录的同步
- `copy_best_checkpoint.sh`
  - 在远端 NFS 内复制 best checkpoint 的辅助脚本
