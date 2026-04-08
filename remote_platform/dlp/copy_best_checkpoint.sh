#!/usr/bin/env bash
set -euo pipefail

SRC_PATH="${1:-/cloud/tmp/yc/Person-in-WiFi-3D-repo/work_dirs/remote_train_full_resume20epochs_20260406/best_mpjpe_epoch_10.pth}"
DST_PATH="${2:-/cloud/tmp/yc/best_mpjpe_epoch_10_20260408.pth}"

cp "$SRC_PATH" "$DST_PATH"
chmod 644 "$DST_PATH"
echo "[copy] $SRC_PATH -> $DST_PATH"
