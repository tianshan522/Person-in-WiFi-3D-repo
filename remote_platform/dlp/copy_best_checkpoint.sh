#!/usr/bin/env bash
set -euo pipefail

SRC_PATH="${1:-/cloud/tmp/yc/Person-in-WiFi-3D-repo/work_dirs/remote_validate_resume_epoch1_best_20260406_retry2/best_mpjpe_epoch_2.pth}"
DST_PATH="${2:-/cloud/tmp/yc/best_mpjpe_epoch_2_20260406.pth}"

cp "$SRC_PATH" "$DST_PATH"
chmod 644 "$DST_PATH"
echo "[copy] $SRC_PATH -> $DST_PATH"
