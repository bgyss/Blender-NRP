"""Path-cache schema helpers.

Two on-disk `.npz` layouts, matching the nrp reference (`nrp/path_cache.py`):

- default: every float array stored as float64 (or float32 from V1 bakes).
- packed (paper §4.2, ~4x smaller): geometry and G-buffer aux as fp16, per-segment
  throughput as shared-exponent rgb9e5 words under `seg_throughput_rgb9e5`, seg_pixel
  as int32, flagged by a `packed_layout` key.

`load_arrays` auto-detects the layout and always hands back float64 arrays, so
everything downstream (gather, training, validation) is layout-agnostic. fp16
directions are renormalized on load to restore unit length. Escape segments survive
packing: fp16 represents inf exactly, and finite t_max is clamped to the fp16 finite
range on write so it can never round *to* inf.

Schema versions follow nrp: v1 caches have no `schema_version` key; v2 adds it plus
an optional homogeneous-medium description (`medium_sigma_t`/`medium_albedo` scalars).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rgb9e5 import rgb9e5_decode, rgb9e5_encode

SCHEMA_VERSION = 2

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

_PACKED_KEYS = (REQUIRED_KEYS - {"seg_throughput"}) | {"seg_throughput_rgb9e5"}

_FP16_MAX = float(np.finfo(np.float16).max)
_FP16_TINY = float(np.finfo(np.float16).smallest_subnormal)


def _to_fp16(arr: np.ndarray) -> np.ndarray:
    """fp16 with finite values clamped into fp16's finite range (inf stays inf)."""
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    return np.where(finite, np.clip(a, -_FP16_MAX, _FP16_MAX), a).astype(np.float16)


@dataclass(frozen=True)
class CacheValidationReport:
    width: int
    height: int
    segment_count: int
    errors: tuple[str, ...] = ()
    schema_version: int | None = None
    packed: bool = False
    medium: dict | None = None
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class LoadedCache:
    """Layout-agnostic decoded cache: float64 arrays + surfaced metadata."""

    arrays: dict[str, np.ndarray]
    schema_version: int | None = None
    packed: bool = False
    medium: dict | None = field(default=None)

    @property
    def width(self) -> int:
        return int(self.arrays["albedo"].shape[1])

    @property
    def height(self) -> int:
        return int(self.arrays["albedo"].shape[0])


def load_arrays(path: str | Path) -> LoadedCache:
    """Load an .npz cache, decoding the packed layout if present.

    Returns float64 segment/aux arrays under the canonical REQUIRED_KEYS names
    regardless of the on-disk layout.
    """
    with np.load(path) as npz:
        files = set(npz.files)
        packed = "packed_layout" in files
        required = _PACKED_KEYS if packed else REQUIRED_KEYS
        missing = sorted(required - files)
        if missing:
            layout = "packed" if packed else "default"
            raise ValueError(f"{layout}-layout cache missing arrays: {', '.join(missing)}")
        raw = {key: npz[key] for key in required}
        schema_version = int(npz["schema_version"]) if "schema_version" in files else None
        medium = None
        if "medium_sigma_t" in files:
            medium = {
                "sigma_t": float(npz["medium_sigma_t"]),
                "albedo": float(npz["medium_albedo"]),
            }

    arrays: dict[str, np.ndarray] = {
        "n_paths": raw["n_paths"].astype(np.int64),
        "seg_pixel": raw["seg_pixel"].astype(np.int64),
        "seg_origin": raw["seg_origin"].astype(np.float64),
        "seg_tmax": raw["seg_tmax"].astype(np.float64),
        "albedo": raw["albedo"].astype(np.float64),
        "normal": raw["normal"].astype(np.float64),
        "depth": raw["depth"].astype(np.float64),
        "position": raw["position"].astype(np.float64),
    }
    seg_dir = raw["seg_dir"].astype(np.float64)
    if packed:
        norms = np.linalg.norm(seg_dir, axis=1, keepdims=True)
        seg_dir = np.divide(seg_dir, norms, out=seg_dir, where=norms > 0)
        arrays["seg_throughput"] = rgb9e5_decode(raw["seg_throughput_rgb9e5"])
    else:
        arrays["seg_throughput"] = raw["seg_throughput"].astype(np.float64)
    arrays["seg_dir"] = seg_dir
    return LoadedCache(arrays, schema_version=schema_version, packed=packed, medium=medium)


def save_arrays(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    width: int,
    height: int,
    packed: bool = False,
    medium: dict | None = None,
) -> None:
    """Write a cache .npz in the default or packed layout.

    Always writes `schema_version`, `width`, and `height` — required by the nrp
    reference loader (`nrp.path_cache.PathCache.load`).
    """
    extra: dict[str, float] = {}
    if medium is not None:
        extra["medium_sigma_t"] = float(medium["sigma_t"])
        extra["medium_albedo"] = float(medium["albedo"])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if packed:
        seg_tmax = np.asarray(arrays["seg_tmax"], dtype=np.float64)
        tmax16 = _to_fp16(seg_tmax)
        # Positive t_max that would round to fp16 zero is pinned to the smallest
        # subnormal so the positivity invariant survives the round trip.
        tmax16 = np.where((tmax16 == 0) & (seg_tmax > 0), np.float16(_FP16_TINY), tmax16)
        np.savez_compressed(
            target,
            schema_version=SCHEMA_VERSION,
            packed_layout=1,
            width=width,
            height=height,
            n_paths=np.asarray(arrays["n_paths"], dtype=np.int64),
            seg_pixel=np.asarray(arrays["seg_pixel"], dtype=np.int32),
            seg_origin=_to_fp16(arrays["seg_origin"]),
            seg_dir=_to_fp16(arrays["seg_dir"]),
            seg_tmax=tmax16,
            seg_throughput_rgb9e5=rgb9e5_encode(np.asarray(arrays["seg_throughput"])),
            albedo=_to_fp16(arrays["albedo"]),
            position=_to_fp16(arrays["position"]),
            depth=_to_fp16(arrays["depth"]),
            normal=_to_fp16(arrays["normal"]),
            **extra,
        )
        return
    np.savez_compressed(
        target,
        schema_version=SCHEMA_VERSION,
        width=width,
        height=height,
        **{key: arrays[key] for key in REQUIRED_KEYS},
        **extra,
    )


def validate_npz(path: str | Path) -> CacheValidationReport:
    try:
        loaded = load_arrays(path)
    except ValueError as exc:
        return CacheValidationReport(0, 0, 0, (str(exc),))
    report = validate_arrays(loaded.arrays)
    warnings = report.warnings
    if loaded.medium is not None:
        warnings = warnings + (
            "cache records a homogeneous medium "
            f"(sigma_t={loaded.medium['sigma_t']}, albedo={loaded.medium['albedo']}); "
            "Blender-side volume capture is out of scope, gather works unchanged",
        )
    return CacheValidationReport(
        report.width,
        report.height,
        report.segment_count,
        report.errors,
        schema_version=loaded.schema_version,
        packed=loaded.packed,
        medium=loaded.medium,
        warnings=warnings,
    )


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
