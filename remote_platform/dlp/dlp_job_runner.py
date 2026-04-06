#!/usr/bin/env python3
import os
import time
from typing import Literal

from dlpctl.api.jobs.api_create_job import create_job
from dlpctl.api.jobs.api_get_jobs import get_job_detail
from dlpctl.api.model.job import JobConfig


POLL_INTERVAL_SECONDS = 5
WAITING_STATUSES = {"Waiting", "Queued", "Scheduling"}
SUCCESS_STATUSES = {"Finished"}
FAILURE_STATUSES = {"Failed", "Stopped"}


def _frontend_url() -> str:
    return "http://dlp-test.truesightai.com" if os.getenv("ENV") in {"test", "dev"} else "http://dlp.truesightai.com"


def submit_and_wait(job_config: JobConfig, wait_mode: Literal["running", "terminal"]) -> int:
    print("[info] submitting DLP job...")
    job = create_job(job_config)
    print(f"[info] created: job-{job.id}")
    print(f"[info] url: {_frontend_url()}/jobs/info?id={job.id}")

    while True:
        detail = get_job_detail(job.id)
        status = detail.status
        print(f"[info] job-{job.id} status: [{status}]")

        if status in WAITING_STATUSES:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if status == "Running":
            if wait_mode == "running":
                print("[done] job is running.")
                print(f"[info] tasks: dlpctl get tasks -j job-{job.id}")
                print(f"[info] logs: dlpctl logs -f job-{job.id}-worker-0")
                return 0
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if status in SUCCESS_STATUSES:
            print("[done] job finished.")
            return 0

        if status in FAILURE_STATUSES:
            print(f"[error] job ended with status: {status}")
            print(f"[info] tasks: dlpctl get tasks -j job-{job.id}")
            print(f"[info] logs: dlpctl logs -f job-{job.id}-worker-0")
            return 1

        print(f"[warn] unknown status: {status}")
        time.sleep(POLL_INTERVAL_SECONDS)
