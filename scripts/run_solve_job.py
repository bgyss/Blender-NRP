#!/usr/bin/env python
"""Consume a V3 ``solve_job.json`` in plain Python on a worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _job_worker import run_worker

from blender_nrp.core.jobs import SolveJob
from blender_nrp.core.lights import LightRig, light_from_dict
from blender_nrp.core.optimize_fallback import optimize_lights_fallback
from blender_nrp.core.path_cache import load_arrays
from blender_nrp.core.reports import write_json_report


def execute(job: SolveJob, progress) -> dict[str, Path]:
    if job.kind != "solve":
        raise ValueError(f"run_solve_job cannot consume {job.kind!r}")
    progress(0.1, "Matching Reference", "loading cache and rig")
    arrays = load_arrays(job.cache_path).arrays
    rig = LightRig.load(job.lights_path)
    target = np.load(job.target_path)
    report = optimize_lights_fallback(arrays, rig.lights, target, sweeps=max(1, job.steps // 75))
    output_dir = Path(job.output_dir)
    solved_path = output_dir / "solved_lights.json"
    solved_lights = tuple(light_from_dict(item) for item in report["optimized_lights"])
    LightRig(solved_lights, scene_id=rig.scene_id, camera_id=rig.camera_id).save(solved_path)
    report_path = output_dir / "solve_report.json"
    write_json_report(report_path, report)
    return {"solved_lights": solved_path, "solve_report": report_path}


if __name__ == "__main__":
    raise SystemExit(run_worker("Run Blender-NRP solve job", execute))
