"""GATHERLIGHT-semantics tests: results must match the nrp reference conventions."""

from __future__ import annotations

import numpy as np

from blender_nrp.core.gather import gather_relight, segment_hits_sphere
from blender_nrp.core.lights import LightRig, SphereLight


def _single_pixel_arrays(n_paths: int, segments: list[dict]) -> dict[str, np.ndarray]:
    return {
        "n_paths": np.array([n_paths], dtype=np.int64),
        "seg_pixel": np.zeros(len(segments), dtype=np.int64),
        "seg_origin": np.array(
            [s["origin"] for s in segments], dtype=np.float64
        ).reshape(-1, 3),
        "seg_dir": np.array([s["dir"] for s in segments], dtype=np.float64).reshape(-1, 3),
        "seg_tmax": np.array([s["tmax"] for s in segments], dtype=np.float64),
        "seg_throughput": np.array(
            [s["throughput"] for s in segments], dtype=np.float64
        ).reshape(-1, 3),
        "albedo": np.zeros((1, 1, 3), dtype=np.float64),
        "normal": np.zeros((1, 1, 3), dtype=np.float64),
        "depth": np.zeros((1, 1), dtype=np.float64),
        "position": np.zeros((1, 1, 3), dtype=np.float64),
    }


def test_segment_hits_sphere_overlap_cases():
    origins = np.zeros((4, 3))
    dirs = np.tile([1.0, 0.0, 0.0], (4, 1))
    # through-sphere, stops-before, escape ray through, points-away
    centers_hit = segment_hits_sphere(
        origins,
        dirs,
        np.array([10.0, 1.0, np.inf, 10.0]),
        center=np.array([5.0, 0.0, 0.0]),
        radius=1.0,
    )
    assert centers_hit.tolist() == [True, False, True, True]
    behind = segment_hits_sphere(
        origins[:1], dirs[:1], np.array([10.0]), center=np.array([-5.0, 0.0, 0.0]), radius=1.0
    )
    assert not behind[0]


def test_gather_relight_accumulates_hits_and_normalizes_by_n_paths():
    segments = [
        # hits the light sphere at x=5
        {"origin": [0, 0, 0], "dir": [1, 0, 0], "tmax": 10.0, "throughput": [0.5, 0.5, 0.5]},
        # points away, must contribute nothing
        {"origin": [0, 0, 0], "dir": [-1, 0, 0], "tmax": 10.0, "throughput": [0.5, 0.5, 0.5]},
    ]
    arrays = _single_pixel_arrays(n_paths=2, segments=segments)
    rig = LightRig(
        (SphereLight(position=(5.0, 0.0, 0.0), radius=1.0, color=(1.0, 1.0, 1.0), intensity=2.0),)
    )
    image = gather_relight(arrays, rig)
    # one hit: throughput 0.5 * emission 2.0 / n_paths 2 = 0.5
    assert np.allclose(image[0, 0], 0.5)


def test_gather_relight_zero_paths_pixel_is_black():
    arrays = _single_pixel_arrays(n_paths=0, segments=[])
    rig = LightRig(
        (SphereLight(position=(0.0, 0.0, 0.0), radius=1.0, color=(1.0, 1.0, 1.0), intensity=1.0),)
    )
    image = gather_relight(arrays, rig)
    assert np.all(image == 0.0)
