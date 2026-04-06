#!/usr/bin/env python3
import os
import sys
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from dlpctl.flow.flow import JobFlow
from dlpctl.flow.helper import generate_job_config


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
        "PIWIFI_DATASET_ROOT": os.environ.get("PIWIFI_DATASET_ROOT", f"{project_dir}/data/wifipose"),
    }

    remote_python = os.environ.get("PIWIFI_REMOTE_PYTHON", "python3").strip()
    envs["PIWIFI_REMOTE_PYTHON"] = remote_python

    generate_kwargs = {
        "gpu_num": gpu_num,
        "entrypoint": "bash remote_platform/dlp/run_remote_smoke_test.sh",
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

    job_config.description = "Person-in-WiFi-3D remote smoke test"
    job_config.print()
    job_flow = JobFlow(job_config, follow=True)
    job_flow.start()


if __name__ == "__main__":
    main()
