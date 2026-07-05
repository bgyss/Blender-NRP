"""ComfyUI export bundle: pre-divided throughput + y-up rotation equivalence."""

from __future__ import annotations

import numpy as np

from blender_nrp.core.comfy_export import comfy_cache_arrays, export_comfy_bundle
from blender_nrp.core.coords import RIGHT_HANDED_Y_UP, convert_rig
from blender_nrp.core.gather import gather_hdr, segment_hits_sphere
from blender_nrp.core.lights import LightRig, SphereLight
from blender_nrp.core.metadata import NRPMetadata


def _demo_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    height, width = 2, 2
    pixels = height * width
    per_pixel = np.array([3, 1, 0, 2], dtype=np.int64)  # includes an undersampled pixel
    seg_pixel = np.repeat(np.arange(pixels, dtype=np.int64), per_pixel)
    segments = int(per_pixel.sum())
    dirs = rng.normal(size=(segments, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return {
        "n_paths": per_pixel,
        "seg_pixel": seg_pixel,
        "seg_origin": rng.normal(size=(segments, 3)),
        "seg_dir": dirs,
        "seg_tmax": rng.random(segments) * 5.0 + 0.5,
        "seg_throughput": rng.random((segments, 3)),
        "albedo": rng.random((height, width, 3)),
        "normal": np.tile([0.0, 0.0, 1.0], (height, width, 1)),
        "depth": rng.random((height, width)) + 0.1,
        "position": rng.normal(size=(height, width, 3)),
    }


def _comfy_style_gather(arrays: dict[str, np.ndarray], rig: LightRig) -> np.ndarray:
    """ComfyUI's gather semantics: sum contributions, no n_paths division."""
    height, width, _ = arrays["albedo"].shape
    result = np.zeros((height * width, 3), dtype=np.float64)
    for light in rig.lights:
        hits = segment_hits_sphere(
            arrays["seg_origin"],
            arrays["seg_dir"],
            arrays["seg_tmax"],
            np.asarray(light.position, dtype=np.float64),
            light.radius,
        )
        emission = np.asarray(light.color) * light.intensity
        np.add.at(result, arrays["seg_pixel"][hits], arrays["seg_throughput"][hits] * emission)
    return result.reshape(height, width, 3)


def test_pre_divided_comfy_gather_matches_reference_gather():
    arrays = _demo_arrays()
    rig = LightRig(
        (SphereLight(position=(0.0, 0.0, 0.0), radius=2.0, color=(1, 1, 1), intensity=1.5),)
    )
    reference = gather_hdr(arrays, rig.lights)
    comfy_arrays = comfy_cache_arrays(arrays, convert_coords=False)
    comfy = _comfy_style_gather(comfy_arrays, rig)
    np.testing.assert_allclose(comfy, reference, atol=1e-12)


def test_rotated_bundle_gathers_identically_with_rotated_rig():
    arrays = _demo_arrays()
    rig = LightRig(
        (SphereLight(position=(0.5, -0.5, 1.0), radius=1.5, color=(1, 1, 1), intensity=1.0),),
        coordinate_system="blender_z_up",
    )
    reference = gather_hdr(arrays, rig.lights)
    rotated = comfy_cache_arrays(arrays, convert_coords=True)
    rotated_rig = convert_rig(rig, RIGHT_HANDED_Y_UP)
    comfy = _comfy_style_gather(rotated, rotated_rig)
    np.testing.assert_allclose(comfy, reference, atol=1e-10)


def test_export_comfy_bundle_writes_cache_and_metadata(tmp_path):
    arrays = _demo_arrays()
    metadata = NRPMetadata(
        scene_id="s",
        camera_id="c",
        resolution=(2, 2),
        light_type="sphere",
        aux_features=["albedo", "normal", "depth"],
        bbox_min=[-1.0, -2.0, -3.0],
        bbox_max=[1.0, 2.0, 3.0],
        model_width=64,
        model_depth=4,
    )
    cache_path = tmp_path / "comfy_cache.npz"
    meta_path = tmp_path / "comfy_metadata.json"
    export_comfy_bundle(arrays, metadata, cache_path, meta_path)

    exported = NRPMetadata.load(meta_path)
    assert exported.throughput_normalization == "pre_divided"
    assert exported.coordinate_system == RIGHT_HANDED_Y_UP
    assert exported.bbox_min[1] <= exported.bbox_max[1]
    # Rotated bbox: y-up min/max come from z / -y of the blender bbox.
    assert exported.bbox_min == [-1.0, -3.0, -2.0]
    assert exported.bbox_max == [1.0, 3.0, 2.0]

    with np.load(cache_path) as npz:
        assert set(npz.files) >= {"seg_throughput", "seg_origin", "n_paths"}
