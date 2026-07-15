#!/usr/bin/env python
"""Consume a V3 ``bake_job.json`` in Blender background mode or plain Python."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from _job_worker import run_worker

from blender_nrp.backends.cycles_capture import bake
from blender_nrp.backends.interface import BakeSettings
from blender_nrp.core.jobs import BakeJob


def execute(job: BakeJob, progress) -> dict[str, Path]:
    if job.kind != "bake":
        raise ValueError(f"run_bake_job cannot consume {job.kind!r}")
    progress(0.1, "Baking", "tracing paths")
    context = None
    try:
        import bpy

        context = bpy.context
    except ModuleNotFoundError:
        pass
    result = bake(
        context,
        BakeSettings(
            scene_id=job.scene_id,
            output_dir=Path(job.output_dir),
            width=job.width,
            height=job.height,
            segment_count=1,
            max_segment_distance=100.0,
            camera_id=job.camera_id,
            seed=job.seed,
            paths_per_pixel=job.paths_per_pixel,
            max_bounces=job.max_bounces,
            packed=job.packed,
            torch_device=job.torch_device,
            tracer_engine=job.tracer_engine,
            reference_check=context is not None,
        ),
    )
    progress(0.95, "Validating", "cache and metadata written")
    return {
        "path_cache": result.cache_path,
        "metadata": result.metadata_path,
        "bake_report": result.bake_report_path,
    }


if __name__ == "__main__":
    raise SystemExit(run_worker("Run Blender-NRP bake job", execute))
