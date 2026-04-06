#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from dlpctl.flow.flow import JobFlow
from dlpctl.flow.helper import generate_job_config


LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))


def main() -> None:
    remote_nfs_path = os.environ.get("DLP_REMOTE_NFS_PATH", "/cloud/tmp/yc").strip()
    project_dir = os.environ.get(
        "DLP_REMOTE_REPO_DIR", f"{remote_nfs_path}/Person-in-WiFi-3D-repo"
    ).strip()
    gpu_num = int(os.environ.get("DLP_GPU_NUM", "1"))
    gpu_type = os.environ.get("DLP_GPU_TYPE", "3090").strip()
    image = os.environ.get("DLP_IMAGE", "").strip()
    pre_process = os.environ.get("DLP_PRE_PROCESS", "").strip()

    envs = {
        "PIWIFI_REMOTE_REPO_DIR": project_dir,
        "PIWIFI_CONFIG_PATH": os.environ.get("PIWIFI_CONFIG_PATH", "configs/wifi/petr_wifi.py"),
        "PIWIFI_DATASET_ROOT": os.environ.get("PIWIFI_DATASET_ROOT", f"{project_dir}/data/wifipose"),
        "PIWIFI_WORK_DIR": os.environ.get("PIWIFI_WORK_DIR", f"{project_dir}/work_dirs/remote_train"),
        "PIWIFI_GPUS": os.environ.get("PIWIFI_GPUS", str(gpu_num)),
        "PIWIFI_USE_DIST": os.environ.get("PIWIFI_USE_DIST", "0"),
        "PIWIFI_SAMPLES_PER_GPU": os.environ.get("PIWIFI_SAMPLES_PER_GPU", "32"),
        "PIWIFI_WORKERS_PER_GPU": os.environ.get("PIWIFI_WORKERS_PER_GPU", "2"),
    }

    for key in [
        "PIWIFI_PORT",
        "PIWIFI_RESUME_FROM",
        "PIWIFI_CFG_OPTIONS",
        "PIWIFI_REMOTE_INSTALL_DEPS",
    ]:
        value = os.environ.get(key, "").strip()
        if value:
            envs[key] = value

    remote_python = os.environ.get("PIWIFI_REMOTE_PYTHON", "").strip()
    if remote_python:
        envs["PIWIFI_REMOTE_PYTHON"] = remote_python

    generate_kwargs = {
        "gpu_num": gpu_num,
        "entrypoint": "bash remote_platform/dlp/run_remote_train.sh",
        "gpu_type": gpu_type,
        "envs": envs,
        "custom_volumes": [{"path": remote_nfs_path, "mode": "rw"}],
        "working_dir": project_dir,
        "python_interpreter_path": "",
    }
    if pre_process:
        generate_kwargs["pre_process"] = pre_process
    if image:
        generate_kwargs["image"] = image

    job_config, err = generate_job_config(**generate_kwargs)
    if err:
        print(f"failed to generate job config: {err}")
        sys.exit(1)

    job_config.description = "Person-in-WiFi-3D remote training"
    job_config.print()
    JobFlow(job_config, follow=True).start()
    sys.exit(0)


if __name__ == "__main__":
    main()
