"""Quad-light hit tests mirroring nrp's `segment_hits_quad` semantics."""

from __future__ import annotations

import numpy as np
import pytest

from blender_nrp.core.gather import (
    gather_hdr,
    quad_tangent_frame,
    segment_hits_quad,
)
from blender_nrp.core.lights import LightRig, QuadLight, SphereLight, light_from_dict


def _quad_kwargs(**overrides):
    base = {
        "position": (5.0, 0.0, 0.0),
        "normal": (1.0, 0.0, 0.0),
        "width": 2.0,
        "height": 2.0,
        "color": (1.0, 1.0, 1.0),
        "intensity": 1.0,
    }
    base.update(overrides)
    return base


def test_quad_tangent_frame_is_orthonormal():
    for normal in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.577, 0.577, 0.577]):
        n = np.asarray(normal) / np.linalg.norm(normal)
        u, v = quad_tangent_frame(n)
        assert abs(np.dot(u, v)) < 1e-12
        assert abs(np.dot(u, n)) < 1e-12
        assert abs(np.dot(v, n)) < 1e-12
        np.testing.assert_allclose([np.linalg.norm(u), np.linalg.norm(v)], 1.0)


def test_segment_hits_quad_cases():
    origins = np.zeros((5, 3))
    dirs = np.array(
        [
            [1.0, 0.0, 0.0],  # crosses inside the rectangle
            [1.0, 0.0, 0.0],  # stops before the plane
            [1.0, 0.0, 0.0],  # escape segment through the plane
            [0.0, 1.0, 0.0],  # parallel to the plane, never hits
            [-1.0, 0.0, 0.0],  # crossing behind the origin
        ]
    )
    t_max = np.array([10.0, 1.0, np.inf, 10.0, 10.0])
    hits = segment_hits_quad(
        origins, dirs, t_max, center=[5.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], width=2.0, height=2.0
    )
    assert hits.tolist() == [True, False, True, False, False]


def test_segment_misses_outside_rectangle():
    # Crossing point is 1.5 off-center in the plane; a 2x2 quad (half-extent 1) misses.
    origins = np.array([[0.0, 1.5, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    hits_small = segment_hits_quad(
        origins, dirs, np.array([10.0]), [5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0, 2.0
    )
    hits_big = segment_hits_quad(
        origins, dirs, np.array([10.0]), [5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 4.0, 4.0
    )
    assert not hits_small[0]
    assert hits_big[0]


def test_quad_light_json_round_trip():
    light = QuadLight(**_quad_kwargs(normal=(0.0, 3.0, 0.0)))
    assert light.normal == (0.0, 1.0, 0.0)  # normalized on construction
    data = light.to_dict()
    assert data["type"] == "quad"
    assert QuadLight.from_dict(data) == light


def test_light_from_dict_dispatch():
    sphere_spec = {"position": [0, 0, 0], "radius": 1.0, "color": [1, 1, 1], "intensity": 1.0}
    quad_spec = _quad_kwargs()
    quad_spec = {k: list(v) if isinstance(v, tuple) else v for k, v in quad_spec.items()}
    assert isinstance(light_from_dict(sphere_spec), SphereLight)  # untyped -> sphere
    assert isinstance(light_from_dict({"type": "quad", **quad_spec}), QuadLight)
    assert isinstance(light_from_dict(quad_spec), QuadLight)  # untyped with width -> quad
    with pytest.raises(ValueError):
        light_from_dict({"type": "disk", **sphere_spec})


def test_quad_light_validation():
    with pytest.raises(ValueError):
        QuadLight(**_quad_kwargs(width=0.0))
    with pytest.raises(ValueError):
        QuadLight(**_quad_kwargs(normal=(0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        QuadLight(**_quad_kwargs(intensity=-1.0))


def test_mixed_rig_gather_sums_both_light_types():
    arrays = {
        "n_paths": np.array([1], dtype=np.int64),
        "seg_pixel": np.zeros(1, dtype=np.int64),
        "seg_origin": np.zeros((1, 3)),
        "seg_dir": np.array([[1.0, 0.0, 0.0]]),
        "seg_tmax": np.array([10.0]),
        "seg_throughput": np.array([[1.0, 1.0, 1.0]]),
        "albedo": np.zeros((1, 1, 3)),
        "normal": np.zeros((1, 1, 3)),
        "depth": np.zeros((1, 1)),
        "position": np.zeros((1, 1, 3)),
    }
    rig = LightRig(
        (
            SphereLight(position=(5.0, 0.0, 0.0), radius=1.0, color=(1, 0, 0), intensity=1.0),
            QuadLight(**_quad_kwargs(position=(7.0, 0.0, 0.0), color=(0, 1, 0), intensity=2.0)),
        )
    )
    assert rig.light_types == ("quad", "sphere")
    image = gather_hdr(arrays, rig.lights)
    np.testing.assert_allclose(image[0, 0], [1.0, 2.0, 0.0])


def test_rig_json_round_trip_with_quads(tmp_path):
    rig = LightRig(
        (
            SphereLight(position=(1, 2, 3), radius=0.5, color=(1, 1, 1), intensity=1.0),
            QuadLight(**_quad_kwargs()),
        ),
        scene_id="s",
        camera_id="c",
    )
    path = tmp_path / "rig.json"
    rig.save(path)
    assert LightRig.load(path) == rig
