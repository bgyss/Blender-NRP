"""Lightweight proxy artifact helpers."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def train_basic_proxy(
    cache_path: str | Path,
    model_path: str | Path,
    report_path: str | Path,
) -> Path:
    """Write a tiny loadable fallback proxy artifact.

    The file is intentionally named ``model.pt`` for NRP workflow compatibility,
    but this fallback stores NumPy summaries rather than a Torch module.
    """
    with np.load(cache_path) as npz:
        albedo = npz["albedo"].astype(np.float32)
        normal = npz["normal"].astype(np.float32)
        depth = npz["depth"].astype(np.float32)
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez_compressed(
            handle,
            format=np.array(["blender_nrp_basic_proxy"]),
            albedo_mean=albedo.reshape(-1, 3).mean(axis=0),
            normal_mean=normal.reshape(-1, 3).mean(axis=0),
            depth_mean=np.array([float(depth.mean())], dtype=np.float32),
        )
    report = {
        "ok": True,
        "model_path": str(target),
        "format": "blender_nrp_basic_proxy",
        "training_backend": "numpy_summary",
        "limitations": [
            "This fallback artifact validates load/save workflow but is not a PyTorch neural proxy."
        ],
    }
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_basic_proxy(model_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(model_path) as npz:
        return {key: npz[key] for key in npz.files}
