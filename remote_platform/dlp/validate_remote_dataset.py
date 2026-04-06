#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def count_files(root: Path, suffix: str) -> int:
    return sum(
        1
        for path in root.rglob(f"*{suffix}")
        if path.is_file() and not path.name.startswith("._")
    )


def main() -> int:
    dataset_root = Path(os.environ.get("PIWIFI_DATASET_ROOT", "")).expanduser()
    if not dataset_root:
        print("[error] PIWIFI_DATASET_ROOT is required")
        return 1

    required = [
        dataset_root / "train_data" / "csi",
        dataset_root / "train_data" / "keypoint",
        dataset_root / "test_data" / "csi",
        dataset_root / "test_data" / "keypoint",
        dataset_root / "train_data" / "train_data_list.txt",
        dataset_root / "test_data" / "test_data_list.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("[error] missing dataset paths:")
        for path in missing:
            print(path)
        return 1

    train_csi = count_files(dataset_root / "train_data" / "csi", ".mat")
    train_kpt = count_files(dataset_root / "train_data" / "keypoint", ".npy")
    test_csi = count_files(dataset_root / "test_data" / "csi", ".mat")
    test_kpt = count_files(dataset_root / "test_data" / "keypoint", ".npy")

    print(f"[info] dataset_root: {dataset_root}")
    print(f"[info] train_csi: {train_csi}")
    print(f"[info] train_keypoint: {train_kpt}")
    print(f"[info] test_csi: {test_csi}")
    print(f"[info] test_keypoint: {test_kpt}")

    if min(train_csi, train_kpt, test_csi, test_kpt) == 0:
        print("[error] dataset is empty")
        return 1

    print("remote_dataset_ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
