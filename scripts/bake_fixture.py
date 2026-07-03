#!/usr/bin/env python
"""Background-mode fixture bake entrypoint (V2: cycles_capture + torch proxy).

Runs inside Blender (`blender --background ... --python scripts/bake_fixture.py`)
against the real scene, or as plain Python against the analytic test room. Trains a
real torch proxy when torch is importable; otherwise writes the V1 numpy-summary
fallback artifact, which labels itself as such in train_report.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_nrp.backends.cycles_capture import bake
from blender_nrp.backends.interface import BakeSettings


def _train(cache_path: Path, output_dir: Path) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        from blender_nrp.core.proxy import train_basic_proxy

        train_basic_proxy(cache_path, output_dir / "model.pt", output_dir / "train_report.json")
        print("torch unavailable: wrote numpy-summary fallback proxy (see train_report.json)")
        return
    from blender_nrp.core.path_cache import load_arrays
    from blender_nrp.core.reports import write_json_report
    from blender_nrp.core.torch_proxy.train import train_proxy

    report = train_proxy(
        load_arrays(cache_path).arrays,
        output_dir / "model.pt",
        iterations=400,
        batch_size=4096,
        pool_size=16,
        n_val_lights=4,
        device="cpu",
        checkpoint_every=0,
    )
    write_json_report(output_dir / "train_report.json", report)
    print(f"trained torch proxy: val PSNR {report['val_psnr_db_mean']:.1f} dB")


def main() -> int:
    result = bake(
        _context_or_none(),
        BakeSettings(
            scene_id="fixture_room_001",
            output_dir=ROOT / "build" / "nrp",
            width=64,
            height=64,
            segment_count=8,
            max_segment_distance=20.0,
            camera_id="Camera",
            seed=7,
            paths_per_pixel=16,
            max_bounces=4,
            reference_check=True,
        ),
    )
    _train(result.cache_path, result.output_dir)
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
