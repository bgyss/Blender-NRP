"""Cache gather relighting helpers (reference GATHERLIGHT semantics).

Matches the `nrp` reference implementation: a segment contributes its throughput to
its pixel iff the segment's parametric interval [0, t_max] overlaps the light
sphere's interior, and per-pixel sums are normalized by `n_paths`. Emission is
`color * intensity` per light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .images import write_png_rgb
from .lights import LightRig


def segment_hits_sphere(
    origins: np.ndarray,
    dirs: np.ndarray,
    t_max: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Vectorized segment-vs-sphere overlap test.

    Returns bool (S,): True iff [t0, t1] (the ray's interval inside the sphere)
    overlaps [0, t_max]. Counts segments starting inside the sphere and segments
    passing through it; t_max may be np.inf for escape segments.
    """
    oc = origins - np.asarray(center, dtype=origins.dtype)
    b = np.einsum("ij,ij->i", oc, dirs)
    c = np.einsum("ij,ij->i", oc, oc) - float(radius) ** 2
    disc = b * b - c
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    return (disc >= 0.0) & (t0 <= t_max) & (t1 >= 0.0)


def gather_relight(
    arrays: dict[str, np.ndarray],
    rig: LightRig,
    *,
    exposure: float = 1.0,
) -> np.ndarray:
    height, width, _ = arrays["albedo"].shape
    result = np.zeros((height * width, 3), dtype=np.float64)
    origins = arrays["seg_origin"].astype(np.float64)
    dirs = arrays["seg_dir"].astype(np.float64)
    t_max = arrays["seg_tmax"].astype(np.float64)
    throughput = arrays["seg_throughput"].astype(np.float64)
    seg_pixel = arrays["seg_pixel"].astype(np.int64)
    n_paths = arrays["n_paths"].astype(np.float64)

    for light in rig.lights if origins.shape[0] else ():
        hits = segment_hits_sphere(
            origins, dirs, t_max, np.asarray(light.position, dtype=np.float64), light.radius
        )
        if not np.any(hits):
            continue
        emission = np.asarray(light.color, dtype=np.float64) * float(light.intensity)
        np.add.at(result, seg_pixel[hits], throughput[hits] * emission[None, :])

    result /= np.maximum(n_paths, 1.0)[:, None]
    image = result.reshape((height, width, 3)) * float(exposure)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def write_relight_preview(
    cache_path: str | Path,
    rig: LightRig,
    output_path: str | Path,
    *,
    exposure: float = 1.0,
) -> Path:
    with np.load(cache_path) as npz:
        arrays = {key: npz[key] for key in npz.files}
    image = gather_relight(arrays, rig, exposure=exposure)
    target = Path(output_path)
    write_png_rgb(target, image)
    return target
