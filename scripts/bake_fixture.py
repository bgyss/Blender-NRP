#!/usr/bin/env python
"""Background-mode fixture bake entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_nrp.backends.interface import BakeSettings
from blender_nrp.backends.stock_blender_hemi import bake
from blender_nrp.core.proxy import train_basic_proxy


def main() -> int:
    result = bake(
        globals().get("bpy_context") or _context_or_none(),
        BakeSettings(
            scene_id="fixture_room_001",
            output_dir=ROOT / "build" / "nrp",
            width=64,
            height=64,
            segment_count=8,
            max_segment_distance=20.0,
            camera_id="Camera",
            seed=7,
        ),
    )
    train_basic_proxy(
        result.cache_path,
        result.output_dir / "model.pt",
        result.output_dir / "train_report.json",
    )
    print(result.cache_path)
    return 0


def _context_or_none() -> object | None:
    try:
        import bpy
    except ModuleNotFoundError:
        return None
    return bpy.context


if __name__ == "__main__":
    raise SystemExit(main())
