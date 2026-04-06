#!/usr/bin/env bash
set -euo pipefail

export http_proxy="http://p.truesightai.com:1087"
export https_proxy="http://p.truesightai.com:1087"

PROJECT_DIR="${PIWIFI_REMOTE_REPO_DIR:-/cloud/tmp/yc/Person-in-WiFi-3D-repo}"
PYTHON_BIN="${PIWIFI_REMOTE_PYTHON:-python3}"
CONFIG_PATH="${PIWIFI_CONFIG_PATH:-configs/wifi/petr_wifi.py}"
DATASET_ROOT="${PIWIFI_DATASET_ROOT:-$PROJECT_DIR/data/wifipose}"
WORK_DIR="${PIWIFI_WORK_DIR:-$PROJECT_DIR/work_dirs/remote_train}"
GPUS="${PIWIFI_GPUS:-1}"
PORT="${PIWIFI_PORT:-29500}"
INSTALL_DEPS="${PIWIFI_REMOTE_INSTALL_DEPS:-0}"
USE_DIST="${PIWIFI_USE_DIST:-0}"

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party/mmcv:$PROJECT_DIR/third_party/mmdet:${PYTHONPATH:-}"
export MMCV_WITH_OPS="${MMCV_WITH_OPS:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_DIR/.cache/torch_extensions}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"

mkdir -p "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$WORK_DIR"

echo "[train] project_dir: $PROJECT_DIR"
echo "[train] python_bin: $PYTHON_BIN"
echo "[train] config_path: $CONFIG_PATH"
echo "[train] dataset_root: $DATASET_ROOT"
echo "[train] work_dir: $WORK_DIR"
echo "[train] gpus: $GPUS"

echo "[train] step: remove macOS metadata files"
find "$DATASET_ROOT" -type f -name '._*' -delete || true

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[train] step: install remote base requirements"
  "$PYTHON_BIN" -m pip install --no-cache-dir -r remote_platform/dlp/requirements-remote-base.txt
  echo "[train] step: install bundled mmcv"
  MMCV_WITH_OPS=1 "$PYTHON_BIN" -m pip install --no-cache-dir -e third_party/mmcv
  echo "[train] step: install bundled mmdet"
  "$PYTHON_BIN" -m pip install --no-cache-dir -e third_party/mmdet
  echo "[train] step: install opera"
  "$PYTHON_BIN" -m pip install --no-cache-dir -e .
else
  echo "[train] step: skip dependency installation"
fi

echo "[train] step: validate dataset"
PIWIFI_DATASET_ROOT="$DATASET_ROOT" "$PYTHON_BIN" remote_platform/dlp/validate_remote_dataset.py

TRAIN_ARGS=(
  "$CONFIG_PATH"
  "--work-dir" "$WORK_DIR"
  "--cfg-options"
  "data.train.dataset_root=$DATASET_ROOT/train_data"
  "data.val.dataset_root=$DATASET_ROOT/test_data"
  "data.test.dataset_root=$DATASET_ROOT/test_data"
  "data.samples_per_gpu=${PIWIFI_SAMPLES_PER_GPU:-32}"
  "data.workers_per_gpu=${PIWIFI_WORKERS_PER_GPU:-2}"
  "work_dir=$WORK_DIR"
)

if [[ -n "${PIWIFI_RESUME_FROM:-}" ]]; then
  TRAIN_ARGS+=("--resume-from" "$PIWIFI_RESUME_FROM")
fi

if [[ -n "${PIWIFI_CFG_OPTIONS:-}" ]]; then
  # Split a plain space-delimited cfg-options string in bash.
  read -r -a EXTRA_CFG_OPTIONS <<< "${PIWIFI_CFG_OPTIONS}"
  TRAIN_ARGS+=("${EXTRA_CFG_OPTIONS[@]}")
fi

if [[ "$USE_DIST" == "1" && "$GPUS" != "1" ]]; then
  echo "[train] step: distributed training"
  DIST_ARGS=(
    "$CONFIG_PATH"
    "$GPUS"
    "--work-dir" "$WORK_DIR"
    "--cfg-options"
    "data.train.dataset_root=$DATASET_ROOT/train_data"
    "data.val.dataset_root=$DATASET_ROOT/test_data"
    "data.test.dataset_root=$DATASET_ROOT/test_data"
    "data.samples_per_gpu=${PIWIFI_SAMPLES_PER_GPU:-32}"
    "data.workers_per_gpu=${PIWIFI_WORKERS_PER_GPU:-2}"
    "work_dir=$WORK_DIR"
  )
  if [[ -n "${PIWIFI_RESUME_FROM:-}" ]]; then
    DIST_ARGS+=("--resume-from" "$PIWIFI_RESUME_FROM")
  fi
  if [[ -n "${PIWIFI_CFG_OPTIONS:-}" ]]; then
    DIST_ARGS+=("${EXTRA_CFG_OPTIONS[@]}")
  fi
  exec env PORT="$PORT" bash tools/dist_train.sh "${DIST_ARGS[@]}"
else
  echo "[train] step: single-process training"
  exec "$PYTHON_BIN" tools/train.py "${TRAIN_ARGS[@]}"
fi
