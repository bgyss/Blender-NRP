"""Coordinate conversion (blender_z_up <-> right_handed_y_up) round trips."""

from __future__ import annotations

import numpy as np
import pytest

from blender_nrp.core.coords import (
    BLENDER_Z_UP,
    RIGHT_HANDED_Y_UP,
    convert_rig,
    convert_vector,
)
from blender_nrp.core.lights import LightRig, QuadLight, SphereLight


def test_convert_vector_axes():
    # Blender's +Z (up) becomes +Y (up) in the y-up frame.
    assert convert_vector((0, 0, 1), BLENDER_Z_UP, RIGHT_HANDED_Y_UP) == (0.0, 1.0, 0.0)
    # Blender's +Y (forward/into screen) becomes -Z (away from viewer).
    assert convert_vector((0, 1, 0), BLENDER_Z_UP, RIGHT_HANDED_Y_UP) == (0.0, 0.0, -1.0)
    assert convert_vector((1, 0, 0), BLENDER_Z_UP, RIGHT_HANDED_Y_UP) == (1.0, 0.0, 0.0)


def test_convert_vector_round_trip_and_handedness():
    rng = np.random.default_rng(3)
    for v in rng.normal(size=(20, 3)):
        forward = convert_vector(tuple(v), BLENDER_Z_UP, RIGHT_HANDED_Y_UP)
        back = convert_vector(forward, RIGHT_HANDED_Y_UP, BLENDER_Z_UP)
        np.testing.assert_allclose(back, v)
        # Proper rotation: length preserved.
        np.testing.assert_allclose(np.linalg.norm(forward), np.linalg.norm(v))


def test_convert_vector_unknown_system():
    with pytest.raises(ValueError):
        convert_vector((1, 0, 0), "blender_z_up", "left_handed_z_down")


def test_convert_rig_converts_positions_and_normals_both_directions():
    rig = LightRig(
        (
            SphereLight(position=(1.0, 2.0, 3.0), radius=0.5, color=(1, 1, 1), intensity=1.0),
            QuadLight(
                position=(4.0, 5.0, 6.0),
                normal=(0.0, 1.0, 0.0),
                width=2.0,
                height=1.0,
                color=(1, 1, 1),
                intensity=1.0,
            ),
        ),
        scene_id="s",
        camera_id="c",
        coordinate_system=BLENDER_Z_UP,
    )
    converted = convert_rig(rig, RIGHT_HANDED_Y_UP)
    assert converted.coordinate_system == RIGHT_HANDED_Y_UP
    assert converted.scene_id == "s" and converted.camera_id == "c"
    assert converted.lights[0].position == (1.0, 3.0, -2.0)
    assert converted.lights[1].position == (4.0, 6.0, -5.0)
    assert converted.lights[1].normal == (0.0, 0.0, -1.0)
    # Scalar shape params untouched.
    assert converted.lights[0].radius == 0.5
    assert converted.lights[1].width == 2.0

    back = convert_rig(converted, BLENDER_Z_UP)
    assert back == rig


def test_convert_rig_noop_when_already_target():
    rig = LightRig(
        (SphereLight(position=(1, 2, 3), radius=0.5, color=(1, 1, 1), intensity=1.0),),
        coordinate_system=BLENDER_Z_UP,
    )
    assert convert_rig(rig, BLENDER_Z_UP) is rig
