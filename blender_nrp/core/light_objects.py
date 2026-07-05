"""Mapping between NRP lights and Blender objects' custom properties.

Kept bpy-independent (duck-typed `obj` with `.location`, `.matrix_world`, and dict
access to custom properties) so the conversion logic is testable outside Blender.

Object convention:
- `nrp_light_type` = "sphere" | "quad" marks an object as an NRP light.
- Spheres: `nrp_radius`, `nrp_color`, `nrp_intensity`; position = object location.
- Quads: `nrp_width`, `nrp_height`, `nrp_color`, `nrp_intensity`; position = object
  location; normal = the object's local +Z axis in world space (third column of the
  world matrix), so rotating the Blender object aims the emitter.
"""

from __future__ import annotations

from typing import Any

from .lights import AnyLight, QuadLight, SphereLight


def _object_z_axis(obj: Any) -> tuple[float, float, float]:
    matrix = obj.matrix_world
    return (float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2]))


def light_from_object(obj: Any) -> AnyLight | None:
    """Read an NRP light from a Blender object, or None if it isn't one."""
    kind = obj.get("nrp_light_type")
    position = tuple(float(v) for v in obj.location)
    color = tuple(float(v) for v in obj.get("nrp_color", (1.0, 1.0, 1.0)))
    intensity = float(obj.get("nrp_intensity", 1.0))
    if kind == "sphere":
        return SphereLight(
            position=position,
            radius=float(obj.get("nrp_radius", 0.25)),
            color=color,
            intensity=intensity,
        )
    if kind == "quad":
        return QuadLight(
            position=position,
            normal=_object_z_axis(obj),
            width=float(obj.get("nrp_width", 1.0)),
            height=float(obj.get("nrp_height", 1.0)),
            color=color,
            intensity=intensity,
        )
    return None


def collect_rig_lights(objects: Any) -> list[AnyLight]:
    """All NRP lights among `objects`, in iteration order."""
    lights = []
    for obj in objects:
        light = light_from_object(obj)
        if light is not None:
            lights.append(light)
    return lights


def apply_light_to_object(obj: Any, light: AnyLight) -> None:
    """Write solved light parameters back onto the Blender object.

    Positions and scalar/color parameters are written; a quad's *normal* is not
    (re-aiming would require solving for a rotation — the solver treats quad normals
    as fixed, see operators/optimize_lights.py).
    """
    obj.location = light.position
    obj["nrp_color"] = list(light.color)
    obj["nrp_intensity"] = float(light.intensity)
    if isinstance(light, SphereLight):
        obj["nrp_radius"] = float(light.radius)
    else:
        obj["nrp_width"] = float(light.width)
        obj["nrp_height"] = float(light.height)
