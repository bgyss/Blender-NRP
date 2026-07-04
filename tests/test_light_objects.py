"""Object <-> light mapping tests (duck-typed, no bpy required)."""

from __future__ import annotations

from blender_nrp.core.light_objects import (
    apply_light_to_object,
    collect_rig_lights,
    light_from_object,
)
from blender_nrp.core.lights import QuadLight, SphereLight


class FakeObject(dict):
    def __init__(self, location=(0.0, 0.0, 0.0), z_axis=(0.0, 0.0, 1.0), **props):
        super().__init__(props)
        self.location = list(location)
        # Column-major-ish world matrix with the given local +Z axis in column 2.
        self.matrix_world = [
            [1.0, 0.0, z_axis[0], location[0]],
            [0.0, 1.0, z_axis[1], location[1]],
            [0.0, 0.0, z_axis[2], location[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]


def test_light_from_object_sphere_and_quad():
    sphere_obj = FakeObject(
        location=(1, 2, 3),
        nrp_light_type="sphere",
        nrp_radius=0.5,
        nrp_color=(1.0, 0.5, 0.25),
        nrp_intensity=2.0,
    )
    light = light_from_object(sphere_obj)
    assert isinstance(light, SphereLight)
    assert light.position == (1.0, 2.0, 3.0)
    assert light.radius == 0.5

    quad_obj = FakeObject(
        location=(0, 0, 2),
        z_axis=(0.0, 1.0, 0.0),
        nrp_light_type="quad",
        nrp_width=2.0,
        nrp_height=1.0,
    )
    quad = light_from_object(quad_obj)
    assert isinstance(quad, QuadLight)
    assert quad.normal == (0.0, 1.0, 0.0)
    assert quad.width == 2.0

    assert light_from_object(FakeObject()) is None
    assert len(collect_rig_lights([sphere_obj, FakeObject(), quad_obj])) == 2


def test_apply_light_to_object_writes_solved_params():
    obj = FakeObject(nrp_light_type="sphere", nrp_radius=0.1)
    solved = SphereLight(position=(4, 5, 6), radius=0.9, color=(0.2, 0.4, 0.6), intensity=7.0)
    apply_light_to_object(obj, solved)
    assert tuple(obj.location) == (4.0, 5.0, 6.0)
    assert obj["nrp_radius"] == 0.9
    assert obj["nrp_intensity"] == 7.0
    assert obj["nrp_color"] == [0.2, 0.4, 0.6]

    quad_obj = FakeObject(nrp_light_type="quad")
    solved_quad = QuadLight(
        position=(1, 1, 1), normal=(0, 0, 1), width=3.0, height=2.0,
        color=(1, 1, 1), intensity=1.5,
    )
    apply_light_to_object(quad_obj, solved_quad)
    assert quad_obj["nrp_width"] == 3.0
    assert quad_obj["nrp_height"] == 2.0


def test_intensity_change_round_trips_through_object():
    """Mutating nrp_intensity on the object is reflected when re-reading the light."""
    obj = FakeObject(
        location=(1, 2, 3),
        nrp_light_type="sphere",
        nrp_radius=0.25,
        nrp_color=(1.0, 1.0, 1.0),
        nrp_intensity=1.0,
    )
    light = light_from_object(obj)
    assert light.intensity == 1.0

    obj["nrp_intensity"] = 5.5
    light2 = light_from_object(obj)
    assert light2.intensity == 5.5

    obj["nrp_intensity"] = 0.0
    light3 = light_from_object(obj)
    assert light3.intensity == 0.0


def test_color_change_round_trips_through_object():
    """Mutating nrp_color on the object is reflected when re-reading the light."""
    obj = FakeObject(
        location=(0, 0, 0),
        nrp_light_type="quad",
        nrp_width=1.0,
        nrp_height=1.0,
        nrp_color=(1.0, 1.0, 1.0),
        nrp_intensity=1.0,
    )
    light = light_from_object(obj)
    assert light.color == (1.0, 1.0, 1.0)

    obj["nrp_color"] = [0.5, 0.3, 0.1]
    light2 = light_from_object(obj)
    assert light2.color == (0.5, 0.3, 0.1)


def test_radius_and_size_change_round_trips():
    """Mutating nrp_radius / nrp_width / nrp_height is reflected on re-read."""
    sphere = FakeObject(
        nrp_light_type="sphere", nrp_radius=0.25,
        nrp_color=(1, 1, 1), nrp_intensity=1.0,
    )
    sphere["nrp_radius"] = 0.75
    assert light_from_object(sphere).radius == 0.75

    quad = FakeObject(
        nrp_light_type="quad", nrp_width=1.0, nrp_height=1.0,
        nrp_color=(1, 1, 1), nrp_intensity=1.0,
    )
    quad["nrp_width"] = 2.5
    quad["nrp_height"] = 3.0
    q = light_from_object(quad)
    assert q.width == 2.5
    assert q.height == 3.0
