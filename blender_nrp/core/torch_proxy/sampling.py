"""Light-configuration sampling for training (nrp torch_backend/sampling.py port).

Positions are sampled on recorded path segments (paper §4.4): pick a segment
uniformly, then a point uniformly along it — implicitly importance-sampling
positions that contribute to the image. Escape segments (t_max = inf) sample over a
finite range set by the scene extent. Shape parameters are uniform within bounds;
quad normals are uniform on the unit sphere.
"""

from __future__ import annotations

import numpy as np

from ..lights import AnyLight, QuadLight, SphereLight


def scene_extent(arrays: dict[str, np.ndarray]) -> float:
    tmax = np.asarray(arrays["seg_tmax"])
    finite = tmax[np.isfinite(tmax)]
    return float(finite.max()) if finite.size else 1.0


def sample_position_on_segments(
    arrays: dict[str, np.ndarray], rng: np.random.Generator, n: int
) -> np.ndarray:
    """(n, 3) positions, each uniform along a uniformly chosen recorded segment."""
    count = int(np.asarray(arrays["seg_pixel"]).shape[0])
    if not count:
        raise ValueError("cannot segment-sample an empty path cache")
    idx = rng.integers(0, count, size=n)
    t_max = np.minimum(np.asarray(arrays["seg_tmax"])[idx], scene_extent(arrays))
    t = rng.random(n) * t_max
    return np.asarray(arrays["seg_origin"])[idx] + t[:, None] * np.asarray(arrays["seg_dir"])[idx]


def default_bounds(arrays: dict[str, np.ndarray]) -> dict:
    """Radius/size bounds scaled to the visible scene extent."""
    extent = scene_extent(arrays)
    return {
        "radius_min": 0.02 * extent,
        "radius_max": 0.25 * extent,
        "size_min": 0.05 * extent,
        "size_max": 0.5 * extent,
    }


def sample_light(
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    light_type: str,
    bounds: dict,
) -> AnyLight:
    """One random light configuration with unit emission (the network learns the
    pre-emission contribution; emission scales at inference time)."""
    center = tuple(float(v) for v in sample_position_on_segments(arrays, rng, 1)[0])
    if light_type == "sphere":
        radius = bounds["radius_min"] + rng.random() * (bounds["radius_max"] - bounds["radius_min"])
        return SphereLight(position=center, radius=float(radius), color=(1, 1, 1), intensity=1.0)
    if light_type == "quad":
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        width = bounds["size_min"] + rng.random() * (bounds["size_max"] - bounds["size_min"])
        height = bounds["size_min"] + rng.random() * (bounds["size_max"] - bounds["size_min"])
        return QuadLight(
            position=center,
            normal=tuple(float(v) for v in normal),
            width=float(width),
            height=float(height),
            color=(1, 1, 1),
            intensity=1.0,
        )
    raise ValueError(f"unknown light type {light_type!r}")


def light_param_vector(light: AnyLight) -> np.ndarray:
    if isinstance(light, SphereLight):
        return np.concatenate([np.asarray(light.position), [light.radius]])
    return np.concatenate(
        [np.asarray(light.position), np.asarray(light.normal), [light.width], [light.height]]
    )
