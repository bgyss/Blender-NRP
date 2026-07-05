"""Cache gather relighting helpers (reference GATHERLIGHT semantics).

Matches the `nrp` reference implementation: a segment contributes its throughput to
its pixel iff the segment's parametric interval [0, t_max] overlaps the light's
extent (sphere interior, or a rectangle crossing for quads), and per-pixel sums are
normalized by `n_paths`. Emission is `color * intensity` per light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .images import write_png_rgb
from .lights import AnyLight, LightRig, QuadLight, SphereLight
from .path_cache import load_arrays


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


def quad_tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal (u, v) spanning the quad's plane for a unit normal.

    Must match nrp's `quad_tangent_frame` exactly: (center, normal, width, height)
    fully determine the light only because both sides derive the same frame.
    """
    n = np.asarray(normal, dtype=np.float64)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def segment_hits_quad(
    origins: np.ndarray,
    dirs: np.ndarray,
    t_max: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """Vectorized segment-vs-rectangle test (mirrors nrp's `segment_hits_quad`).

    Returns bool (S,): True iff the segment crosses the quad's plane at t in
    [0, t_max] with the crossing point inside the (width x height) rectangle.
    Segments parallel to the plane never hit.
    """
    center = np.asarray(center, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    u, v = quad_tangent_frame(n)

    denom = dirs @ n
    parallel = np.abs(denom) < 1e-12
    safe = np.where(parallel, 1.0, denom)
    t = ((center - origins) @ n) / safe
    p = origins + t[:, None] * dirs
    local = p - center
    lu = local @ u
    lv = local @ v
    return (
        ~parallel
        & (t >= 0.0)
        & (t <= t_max)
        & (np.abs(lu) <= width / 2.0)
        & (np.abs(lv) <= height / 2.0)
    )


def segment_hits_light(
    origins: np.ndarray,
    dirs: np.ndarray,
    t_max: np.ndarray,
    light: AnyLight,
) -> np.ndarray:
    if isinstance(light, SphereLight):
        return segment_hits_sphere(
            origins, dirs, t_max, np.asarray(light.position, dtype=np.float64), light.radius
        )
    if isinstance(light, QuadLight):
        return segment_hits_quad(
            origins,
            dirs,
            t_max,
            np.asarray(light.position, dtype=np.float64),
            np.asarray(light.normal, dtype=np.float64),
            light.width,
            light.height,
        )
    raise TypeError(f"unsupported light type: {type(light).__name__}")


def gather_hdr(arrays: dict[str, np.ndarray], lights: tuple[AnyLight, ...]) -> np.ndarray:
    """(H, W, 3) float64 linear-HDR GATHERLIGHT image over a light list."""
    height, width, _ = arrays["albedo"].shape
    result = np.zeros((height * width, 3), dtype=np.float64)
    origins = arrays["seg_origin"].astype(np.float64)
    dirs = arrays["seg_dir"].astype(np.float64)
    t_max = arrays["seg_tmax"].astype(np.float64)
    throughput = arrays["seg_throughput"].astype(np.float64)
    seg_pixel = arrays["seg_pixel"].astype(np.int64)
    n_paths = arrays["n_paths"].astype(np.float64)

    for light in lights if origins.shape[0] else ():
        hits = segment_hits_light(origins, dirs, t_max, light)
        if not np.any(hits):
            continue
        emission = np.asarray(light.color, dtype=np.float64) * float(light.intensity)
        np.add.at(result, seg_pixel[hits], throughput[hits] * emission[None, :])

    result /= np.maximum(n_paths, 1.0)[:, None]
    return result.reshape((height, width, 3))


def gather_relight(
    arrays: dict[str, np.ndarray],
    rig: LightRig,
    *,
    exposure: float = 1.0,
) -> np.ndarray:
    """Display-ready (H, W, 3) float32 in [0, 1]: HDR gather scaled and clipped."""
    image = gather_hdr(arrays, rig.lights) * float(exposure)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def write_relight_preview(
    cache_path: str | Path,
    rig: LightRig,
    output_path: str | Path,
    *,
    exposure: float = 1.0,
) -> Path:
    image = gather_relight(load_arrays(cache_path).arrays, rig, exposure=exposure)
    target = Path(output_path)
    write_png_rgb(target, image)
    return target
