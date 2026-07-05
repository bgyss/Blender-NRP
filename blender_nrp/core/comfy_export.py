"""One-way export bundle for ComfyUI-NeuralRenderProxy (interop debts 1 + 2).

Decision, made once and documented here: the Blender-side cache keeps *raw*
per-segment throughput with gather-time division by `n_paths` (nrp-main convention,
recorded as `throughput_normalization: "n_paths"` in metadata.json). ComfyUI's
gather (`nrp/gather.py` in that repo) sums contributions without normalizing, so the
ComfyUI export path pre-divides each segment's throughput by its pixel's path count
at export time and labels the result `throughput_normalization: "pre_divided"`.

The exported bundle is also rotated into ComfyUI's default `right_handed_y_up`
frame — segment geometry and G-buffer vectors alike, so a rig exported with
`convert_rig` gathers identically against it (rotations preserve segment/light
intersections).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .coords import BLENDER_Z_UP, RIGHT_HANDED_Y_UP
from .metadata import NRPMetadata
from .path_cache import REQUIRED_KEYS

# blender_z_up -> right_handed_y_up as a row-vector rotation: (x, y, z) -> (x, z, -y).
_BLENDER_TO_Y_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _rotate_rows(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) @ _BLENDER_TO_Y_UP


def comfy_cache_arrays(
    arrays: dict[str, np.ndarray], *, convert_coords: bool = True
) -> dict[str, np.ndarray]:
    """ComfyUI-compatible copy: throughput pre-divided by per-pixel path counts,
    geometry optionally rotated from blender_z_up into right_handed_y_up."""
    n_paths = np.asarray(arrays["n_paths"], dtype=np.float64)
    seg_pixel = np.asarray(arrays["seg_pixel"], dtype=np.int64)
    denom = np.maximum(n_paths, 1.0)[seg_pixel][:, None]
    out = {key: np.asarray(arrays[key]) for key in REQUIRED_KEYS}
    out["seg_throughput"] = np.asarray(arrays["seg_throughput"], dtype=np.float64) / denom
    if convert_coords:
        h, w, _ = out["position"].shape
        out["seg_origin"] = _rotate_rows(out["seg_origin"])
        out["seg_dir"] = _rotate_rows(out["seg_dir"])
        out["position"] = _rotate_rows(out["position"].reshape(-1, 3)).reshape(h, w, 3)
        out["normal"] = _rotate_rows(out["normal"].reshape(-1, 3)).reshape(h, w, 3)
    return out


def export_comfy_bundle(
    arrays: dict[str, np.ndarray],
    metadata: NRPMetadata,
    cache_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    convert_coords: bool = True,
) -> Path:
    """Write the pre-divided (optionally y-up) cache npz + matching metadata.json."""
    converted = comfy_cache_arrays(arrays, convert_coords=convert_coords)
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **converted)
    if metadata_path is not None:
        payload = metadata.to_dict()
        payload["throughput_normalization"] = "pre_divided"
        if convert_coords:
            payload["coordinate_system"] = RIGHT_HANDED_Y_UP
            lo = [payload["bbox_min"][0], payload["bbox_min"][2], -payload["bbox_max"][1]]
            hi = [payload["bbox_max"][0], payload["bbox_max"][2], -payload["bbox_min"][1]]
            payload["bbox_min"], payload["bbox_max"] = lo, hi
        else:
            payload["coordinate_system"] = BLENDER_Z_UP
        NRPMetadata.from_dict(payload).save(metadata_path)
    return target
