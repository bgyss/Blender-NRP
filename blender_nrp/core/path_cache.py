"""Path-cache schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_KEYS = {
    "n_paths",
    "seg_pixel",
    "seg_origin",
    "seg_dir",
    "seg_tmax",
    "seg_throughput",
    "albedo",
    "normal",
    "depth",
    "position",
}


@dataclass(frozen=True)
class CacheValidationReport:
    width: int
    height: int
    segment_count: int
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_npz(path: str | Path) -> CacheValidationReport:
    with np.load(path) as npz:
        missing = sorted(REQUIRED_KEYS - set(npz.files))
        if missing:
            return CacheValidationReport(0, 0, 0, (f"missing arrays: {', '.join(missing)}",))
        arrays = {key: npz[key] for key in REQUIRED_KEYS}
    return validate_arrays(arrays)


def validate_arrays(arrays: dict[str, np.ndarray]) -> CacheValidationReport:
    errors: list[str] = []
    albedo = arrays["albedo"]
    if albedo.ndim != 3 or albedo.shape[2] != 3:
        return CacheValidationReport(0, 0, 0, ("albedo must have shape (H, W, 3)",))

    height, width, _channels = albedo.shape
    pixels = width * height
    segment_count = int(arrays["seg_pixel"].shape[0])
    expected = {
        "n_paths": (pixels,),
        "seg_pixel": (segment_count,),
        "seg_origin": (segment_count, 3),
        "seg_dir": (segment_count, 3),
        "seg_tmax": (segment_count,),
        "seg_throughput": (segment_count, 3),
        "albedo": (height, width, 3),
        "normal": (height, width, 3),
        "depth": (height, width),
        "position": (height, width, 3),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            errors.append(f"{key} has shape {arrays[key].shape}, expected {shape}")
        if not np.all(np.isfinite(arrays[key])) and key != "seg_tmax":
            errors.append(f"{key} contains non-finite values")

    if segment_count == 0:
        errors.append("cache contains no segments")
    else:
        seg_pixel = arrays["seg_pixel"]
        if int(seg_pixel.min()) < 0 or int(seg_pixel.max()) >= pixels:
            errors.append("seg_pixel indices are out of range")
        norms = np.linalg.norm(arrays["seg_dir"], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            errors.append("seg_dir rows must be unit length")
        if not np.all(arrays["seg_tmax"] > 0.0):
            errors.append("seg_tmax values must be positive")

    return CacheValidationReport(width, height, segment_count, tuple(errors))

