#!/usr/bin/env python
"""Background-mode fixture relight entrypoint."""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_nrp.backends.interface import BakeSettings
from blender_nrp.backends.stock_blender_hemi import bake
from blender_nrp.core.gather import write_relight_preview
from blender_nrp.core.lights import LightRig, SphereLight
from blender_nrp.core.proxy import train_basic_proxy


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
            ),
        )
        cache_path = result.cache_path
        output_dir = result.output_dir
    model_path = output_dir / "model.pt"
    if not model_path.exists():
        train_basic_proxy(cache_path, model_path, output_dir / "train_report.json")

    rig = LightRig(
        (
            SphereLight(
                position=(0.0, 2.0, 2.0),
                radius=0.35,
                color=(1.0, 0.86, 0.68),
                intensity=8.0,
            ),
        ),
        scene_id=SCENE_ID,
        camera_id="Camera",
    )
    rig.save(output_dir / "solved_lights.json")
    write_relight_preview(cache_path, rig, output_dir / "relight_preview.png", exposure=12.0)
    (output_dir / "solve_report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "solver": "fixture_known_light",
                "updated_fields": ["position", "radius", "color", "intensity"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
