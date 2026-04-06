#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PIWIFI_REMOTE_REPO_DIR:-/cloud/tmp/yc/Person-in-WiFi-3D-repo}"
PYTHON_BIN="${PIWIFI_REMOTE_PYTHON:-python3}"
DATASET_ROOT="${PIWIFI_DATASET_ROOT:-$PROJECT_DIR/data/wifipose}"

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party/mmcv:$PROJECT_DIR/third_party/mmdet:${PYTHONPATH:-}"

echo "[smoke] pwd: $(pwd)"
echo "[smoke] python: $PYTHON_BIN"
"$PYTHON_BIN" -V
echo "[smoke] repo:"
ls -la
echo "[smoke] dataset_root: $DATASET_ROOT"
"$PYTHON_BIN" remote_platform/dlp/validate_remote_dataset.py

echo "[smoke] install minimal dependencies"
"$PYTHON_BIN" -m pip install --no-cache-dir \
  -r remote_platform/dlp/requirements-remote-base.txt \
  -r requirements/runtime.txt

echo "[smoke] import checks"
"$PYTHON_BIN" - <<'PY'
import mmcv  # noqa: F401
import mmdet  # noqa: F401
import pywt  # noqa: F401
import h5py  # noqa: F401
import scipy  # noqa: F401
import torch  # noqa: F401
from mmcv import Config
cfg = Config.fromfile("configs/wifi/petr_wifi.py")
print("config_load_ok")
print(cfg.data.train.type)
PY

echo "remote_smoke_ok"
