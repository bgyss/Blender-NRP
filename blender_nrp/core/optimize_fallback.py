"""No-torch inverse light optimization: coordinate descent over the numpy gather.

Slower and cruder than the torch proxy path (`torch_proxy/optimize.py`) but honest:
it evaluates the real reference gather at every probe, so what it reports is what
you get. To keep cost bounded, the search gathers over a random subset of segments
(an unbiased-in-expectation Monte Carlo thinning); the final report numbers are
recomputed with the full gather.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .gather import gather_hdr
from .lights import AnyLight, SphereLight


def _thin_arrays(arrays: dict[str, np.ndarray], max_segments: int, seed: int) -> dict:
    count = int(np.asarray(arrays["seg_pixel"]).shape[0])
    if count <= max_segments:
        return arrays
    rng = np.random.default_rng(seed)
    keep = rng.choice(count, size=max_segments, replace=False)
    keep.sort()
    fraction = max_segments / count
    thinned = dict(arrays)
    for key in ("seg_pixel", "seg_origin", "seg_dir", "seg_tmax", "seg_throughput"):
        thinned[key] = np.asarray(arrays[key])[keep]
    # Keep the estimator unbiased: n_paths scales down with the kept fraction.
    thinned["n_paths"] = np.maximum(
        (np.asarray(arrays["n_paths"], dtype=np.float64) * fraction), 1e-9
    )
    return thinned


def _get_params(light: AnyLight) -> list[tuple[str, int | None, float]]:
    """(field, component-index-or-None, value) rows for every scalar parameter."""
    rows = [("position", i, light.position[i]) for i in range(3)]
    rows += [("color", i, light.color[i]) for i in range(3)]
    rows.append(("intensity", None, light.intensity))
    if isinstance(light, SphereLight):
        rows.append(("radius", None, light.radius))
    else:
        rows.append(("width", None, light.width))
        rows.append(("height", None, light.height))
    return rows


def _set_param(light: AnyLight, field: str, index: int | None, value: float) -> AnyLight:
    if index is None:
        if field in ("radius", "width", "height") and value <= 0:
            value = 1e-4
        if field == "intensity":
            value = max(value, 0.0)
        return replace(light, **{field: value})
    vec = list(getattr(light, field))
    vec[index] = max(value, 0.0) if field == "color" else value
    return replace(light, **{field: tuple(vec)})


def optimize_lights_fallback(
    arrays: dict[str, np.ndarray],
    lights: tuple[AnyLight, ...],
    target: np.ndarray,
    *,
    sweeps: int = 4,
    initial_step: float = 0.25,
    max_segments: int = 200_000,
    seed: int = 0,
) -> dict:
    """Coordinate descent on every light parameter against `target` (H, W, 3)."""
    h, w, _ = arrays["albedo"].shape
    if target.shape != (h, w, 3):
        raise ValueError(f"target must be {(h, w, 3)}, got {target.shape}")
    search_arrays = _thin_arrays(arrays, max_segments, seed)
    target64 = np.asarray(target, dtype=np.float64)

    positions = arrays["position"].reshape(-1, 3)
    valid = np.asarray(arrays["n_paths"]) > 0
    pts = positions[valid] if np.any(valid) else positions
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0

    def loss(candidate: tuple[AnyLight, ...]) -> float:
        image = gather_hdr(search_arrays, candidate)
        return float(np.mean((image - target64) ** 2))

    current = tuple(lights)
    current_loss = loss(current)
    initial_loss = current_loss
    evaluations = 1

    # Visibility is a hard indicator: a light that hits no segments sits on a flat
    # loss plateau. Seed each light's position with a coarse global search over
    # points sampled on recorded segments (positions that provably intersect paths).
    rng = np.random.default_rng(seed + 1)
    seg_count = int(np.asarray(search_arrays["seg_pixel"]).shape[0])
    if seg_count:
        tmax = np.asarray(search_arrays["seg_tmax"])
        finite = tmax[np.isfinite(tmax)]
        span = float(finite.max()) if finite.size else 1.0
        idx = rng.integers(0, seg_count, size=16)
        t = rng.random(16) * np.minimum(tmax[idx], span)
        candidates = (
            np.asarray(search_arrays["seg_origin"])[idx]
            + t[:, None] * np.asarray(search_arrays["seg_dir"])[idx]
        )
        for light_index in range(len(current)):
            for point in candidates:
                candidate_light = replace(
                    current[light_index], position=tuple(float(v) for v in point)
                )
                candidate = (
                    current[:light_index] + (candidate_light,) + current[light_index + 1 :]
                )
                candidate_loss = loss(candidate)
                evaluations += 1
                if candidate_loss < current_loss:
                    current, current_loss = candidate, candidate_loss

    step = initial_step
    for _sweep in range(sweeps):
        for light_index in range(len(current)):
            for field, comp, value in _get_params(current[light_index]):
                scale = extent if field == "position" else max(abs(value), 0.05)
                for sign in (1.0, -1.0):
                    improved = False
                    # Line search: keep stepping while the loss keeps dropping.
                    while True:
                        base = _get_params(current[light_index])
                        base_value = next(
                            v for f, c, v in base if f == field and c == comp
                        )
                        candidate_light = _set_param(
                            current[light_index], field, comp, base_value + sign * step * scale
                        )
                        candidate = (
                            current[:light_index]
                            + (candidate_light,)
                            + current[light_index + 1 :]
                        )
                        candidate_loss = loss(candidate)
                        evaluations += 1
                        if candidate_loss < current_loss:
                            current, current_loss = candidate, candidate_loss
                            improved = True
                        else:
                            break
                    if improved:
                        break
        step *= 0.5

    gather_initial = gather_hdr(arrays, lights)
    gather_final = gather_hdr(arrays, current)

    def full_mse(image: np.ndarray) -> float:
        return float(np.mean((image - target64) ** 2))

    return {
        "ok": True,
        "solver": "numpy_coordinate_descent",
        "sweeps": sweeps,
        "light_count": len(lights),
        "gather_evaluations": evaluations,
        "search_segments": int(np.asarray(search_arrays["seg_pixel"]).shape[0]),
        "initial_lights": [light.to_dict() for light in lights],
        "optimized_lights": [light.to_dict() for light in current],
        "search_loss_first": initial_loss,
        "search_loss_last": current_loss,
        "gather_mse_vs_target_initial": full_mse(gather_initial),
        "gather_mse_vs_target_final": full_mse(gather_final),
        "limitations": [
            "Torch is unavailable: coordinate descent over the numpy gather, no "
            "gradients — expect coarser convergence than the proxy solver.",
            "Quad normals are held fixed (no rotation search).",
        ],
    }
