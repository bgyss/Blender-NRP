#!/usr/bin/env python
"""Consume a V3 ``solve_job.json`` in plain Python on a worker."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

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
    locks = job.locks or tuple(() for _ in rig.lights)
    report = None
    if job.model_path:
        try:
            from blender_nrp.core.torch_proxy.model import TorchNRP
            from blender_nrp.core.torch_proxy.optimize import optimize_lights

            model = TorchNRP.load(job.model_path)
            if {light.light_type for light in rig.lights} == {model.light_type}:
                report = optimize_lights(
                    model,
                    arrays,
                    rig.lights,
                    target,
                    steps=job.steps,
                    device=job.torch_device,
                    locks=locks,
                )
        except Exception as exc:
            # Preserve a useful fallback report when an optional model/runtime
            # is unavailable on a remote worker.
            report = None
            model_error = str(exc)
        else:
            model_error = None
    else:
        model_error = None
    if report is None:
        report = optimize_lights_fallback(
            arrays, rig.lights, target, locks=locks, sweeps=max(1, job.steps // 75)
        )
        if model_error:
            report.setdefault("limitations", []).append(
                f"Gradient proxy solver unavailable; used coordinate descent: {model_error}"
            )
    output_dir = Path(job.output_dir)
    solved_path = output_dir / "solved_lights.json"
    solved_lights = tuple(light_from_dict(item) for item in report["optimized_lights"])
    LightRig(solved_lights, scene_id=rig.scene_id, camera_id=rig.camera_id).save(solved_path)
    report_path = output_dir / "solve_report.json"
    write_json_report(report_path, report)
    return {"solved_lights": solved_path, "solve_report": report_path}


if __name__ == "__main__":
    raise SystemExit(run_worker("Run Blender-NRP solve job", execute))
