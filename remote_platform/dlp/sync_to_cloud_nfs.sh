#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PIWIFI_CLOUD_MOUNT_DIR="${PIWIFI_CLOUD_MOUNT_DIR:-/private/tmp/piwifi/cloud}"
export PIWIFI_CLOUD_REPO_DIR="${PIWIFI_CLOUD_REPO_DIR:-}"

if [[ -z "$PIWIFI_CLOUD_REPO_DIR" ]]; then
  PIWIFI_CLOUD_REPO_DIR="$PIWIFI_CLOUD_MOUNT_DIR/Person-in-WiFi-3D-repo"
fi

mkdir -p "$PIWIFI_CLOUD_REPO_DIR"

RSYNC_COMMON_ARGS=(
  -a
  --delete
  --exclude ".git"
  --exclude ".conda"
  --exclude "__pycache__"
  --exclude "*.pyc"
  --exclude ".DS_Store"
  --exclude "._*"
  --exclude ".ipynb_checkpoints"
  --exclude "work_dirs"
  --exclude "result"
  --exclude "*.pth"
  --exclude "*.pt"
  --exclude "*.ckpt"
)

SYNC_PATHS=(
  "configs"
  "docs"
  "opera"
  "requirements"
  "third_party/mmcv"
  "third_party/mmdet"
  "tools"
  "remote_platform"
  "setup.py"
  "setup.cfg"
  "requirements.txt"
  "README.md"
)

OPTIONAL_PATHS=(
  "data/wifipose"
)

echo "[info] sync target: $PIWIFI_CLOUD_REPO_DIR"

for rel_path in "${SYNC_PATHS[@]}"; do
  src_path="$REPO_DIR/$rel_path"
  dst_path="$PIWIFI_CLOUD_REPO_DIR/$rel_path"

  if [[ ! -e "$src_path" ]]; then
    echo "[skip] missing: $rel_path"
    continue
  fi

  mkdir -p "$(dirname "$dst_path")"
  echo "[sync] $rel_path"
  if [[ -d "$src_path" ]]; then
    mkdir -p "$dst_path"
    rsync "${RSYNC_COMMON_ARGS[@]}" "$src_path/" "$dst_path/"
  else
    rsync "${RSYNC_COMMON_ARGS[@]}" "$src_path" "$dst_path"
  fi
done

for rel_path in "${OPTIONAL_PATHS[@]}"; do
  src_path="$REPO_DIR/$rel_path"
  dst_path="$PIWIFI_CLOUD_REPO_DIR/$rel_path"

  if [[ ! -e "$src_path" ]]; then
    echo "[skip] optional missing: $rel_path"
    continue
  fi

  mkdir -p "$(dirname "$dst_path")"
  echo "[sync] $rel_path"
  mkdir -p "$dst_path"
  rsync "${RSYNC_COMMON_ARGS[@]}" "$src_path/" "$dst_path/"
done

echo "[done] cloud sync completed."
