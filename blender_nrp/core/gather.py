"""Cache gather relighting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .images import write_png_rgb
from .lights import LightRig


def gather_relight(
    arrays: dict[str, np.ndarray],
    rig: LightRig,
    *,
    exposure: float = 1.0,
) -> np.ndarray:
    height, width, _ = arrays["albedo"].shape
    result = np.zeros((height * width, 3), dtype=np.float32)
    origins = arrays["seg_origin"].astype(np.float32)
    dirs = arrays["seg_dir"].astype(np.float32)
    throughput = arrays["seg_throughput"].astype(np.float32)
    seg_pixel = arrays["seg_pixel"].astype(np.int64)

    for light in rig.lights:
        light_pos = np.asarray(light.position, dtype=np.float32)
        to_light = light_pos[None, :] - origins
        dist2 = np.sum(to_light * to_light, axis=1)
        dist = np.sqrt(np.maximum(dist2, 1e-8))
        light_dir = to_light / dist[:, None]
        alignment = np.maximum(np.sum(dirs * light_dir, axis=1), 0.0)
        attenuation = (light.radius * light.radius) / np.maximum(dist2, 1e-4)
        color = np.asarray(light.color, dtype=np.float32) * float(light.intensity)
        contribution = throughput * alignment[:, None] * attenuation[:, None] * color[None, :]
        np.add.at(result, seg_pixel, contribution)

    image = result.reshape((height, width, 3)) * float(exposure)
    return np.clip(image, 0.0, 1.0)


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
