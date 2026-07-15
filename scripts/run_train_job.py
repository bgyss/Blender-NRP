#!/usr/bin/env python
"""Consume a V3 ``train_job.json`` in plain Python on a worker."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from _job_worker import run_worker

from blender_nrp.core.jobs import TrainJob
from blender_nrp.core.path_cache import load_arrays
from blender_nrp.core.reports import write_json_report
from blender_nrp.core.torch_proxy import torch_status


def execute(job: TrainJob, progress) -> dict[str, Path]:
    if job.kind != "train":
        raise ValueError(f"run_train_job cannot consume {job.kind!r}")
    ok, detail = torch_status()
    if not ok:
        raise RuntimeError(detail)
    progress(0.1, "Training", "loading cache")
    from blender_nrp.core.torch_proxy.train import train_proxy

    output_dir = Path(job.output_dir)
    model_path = output_dir / "model.pt"
    report = train_proxy(
        load_arrays(job.cache_path).arrays,
        model_path,
        iterations=job.iterations,
        device=job.torch_device,
    )
    report_path = output_dir / "train_report.json"
    write_json_report(report_path, report)
    return {"model": model_path, "train_report": report_path}


if __name__ == "__main__":
    raise SystemExit(run_worker("Run Blender-NRP train job", execute))
