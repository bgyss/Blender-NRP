#!/usr/bin/env python
"""Background-mode fixture relight entrypoint (V2: cycles_capture cache).

Reuses (or bakes) the fixture cache, writes a sphere+quad rig, renders the exact
cache-gather preview, and — when torch is importable — also runs the inverse light
solver against the known-light image and writes its solve_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from blender_nrp.backends.cycles_capture import bake
from blender_nrp.backends.interface import BakeSettings
from blender_nrp.core.gather import gather_hdr, write_relight_preview
from blender_nrp.core.lights import LightRig, QuadLight, SphereLight
from blender_nrp.core.path_cache import load_arrays

SCENE_ID = "fixture_room_001"


def main() -> int:
    output_dir = ROOT / "build" / "nrp" / SCENE_ID
    cache_path = output_dir / "path_cache.npz"
    if not cache_path.exists():
        result = bake(
            _context_or_none(),
            BakeSettings(
                scene_id=SCENE_ID,
                output_dir=ROOT / "build" / "nrp",
                width=64,
                height=64,
                segment_count=8,
                max_segment_distance=20.0,
                camera_id="Camera",
                seed=7,
                paths_per_pixel=16,
                max_bounces=4,
            ),
        )
        cache_path = result.cache_path
        output_dir = result.output_dir

    rig = LightRig(
        (
            SphereLight(
                position=(0.0, 2.0, 2.0),
                radius=0.35,
                color=(1.0, 0.86, 0.68),
                intensity=8.0,
            ),
            QuadLight(
                position=(-0.8, 1.0, 2.2),
                normal=(0.3, 0.2, -0.93),
                width=1.0,
                height=0.7,
                color=(0.7, 0.8, 1.0),
                intensity=4.0,
            ),
        ),
        scene_id=SCENE_ID,
        camera_id="Camera",
    )
    rig.save(output_dir / "solved_lights.json")
    write_relight_preview(cache_path, rig, output_dir / "relight_preview.png", exposure=12.0)

    arrays = load_arrays(cache_path).arrays
    solve_report: dict = {
        "ok": True,
        "solver": "fixture_known_light",
        "updated_fields": ["position", "radius", "color", "intensity"],
    }
    try:
        import torch  # noqa: F401

        from blender_nrp.core.optimize_fallback import optimize_lights_fallback

        true_light = rig.lights[0]
        target = gather_hdr(arrays, (true_light,))
        init = SphereLight(
            position=(1.0, 0.0, 1.0), radius=0.2, color=(1.0, 1.0, 1.0), intensity=1.0
        )
        solve_report = optimize_lights_fallback(arrays, (init,), target, sweeps=4)
        solve_report["target"] = "gather of the known fixture light"
    except ImportError:
        solve_report["limitations"] = ["torch unavailable; solver not exercised in fixture"]
    (output_dir / "solve_report.json").write_text(
        json.dumps(solve_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preview = np.asarray(gather_hdr(arrays, rig.lights))
    print(f"relit fixture: preview mean {preview.mean():.5f}")
    print(output_dir / "relight_preview.png")
    return 0


def _context_or_none() -> object | None:
    try:
        import bpy
    except ModuleNotFoundError:
        return None
    return bpy.context


if __name__ == "__main__":
    raise SystemExit(main())
