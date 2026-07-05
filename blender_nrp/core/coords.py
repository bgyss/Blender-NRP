"""Coordinate-system conversion for light rigs (interop debt 2).

Blender scenes are right-handed Z-up (`blender_z_up`: x right, y into the screen,
z up). ComfyUI-NeuralRenderProxy defaults to `right_handed_y_up` (x right, y up,
z toward the viewer). Both are right-handed, so one proper rotation maps between
them and applies identically to positions and normals:

    blender_z_up -> right_handed_y_up:  (x, y, z) -> (x, z, -y)
    right_handed_y_up -> blender_z_up:  (x, y, z) -> (x, -z, y)

V1 only *labeled* the `coordinate_system` field; V2 actually converts on
import/export via `convert_rig`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .lights import AnyLight, LightRig, QuadLight

BLENDER_Z_UP = "blender_z_up"
RIGHT_HANDED_Y_UP = "right_handed_y_up"

Vec3 = tuple[float, float, float]

_CONVERSIONS: dict[tuple[str, str], Callable[[Vec3], Vec3]] = {
    (BLENDER_Z_UP, RIGHT_HANDED_Y_UP): lambda v: (v[0], v[2], -v[1]),
    (RIGHT_HANDED_Y_UP, BLENDER_Z_UP): lambda v: (v[0], -v[2], v[1]),
}


def convert_vector(vector: Vec3, source: str, target: str) -> Vec3:
    """Rotate one 3-vector (position or normal) between coordinate systems."""
    if source == target:
        return (float(vector[0]), float(vector[1]), float(vector[2]))
    try:
        convert = _CONVERSIONS[(source, target)]
    except KeyError:
        raise ValueError(
            f"unsupported coordinate conversion: {source!r} -> {target!r}"
        ) from None
    return convert((float(vector[0]), float(vector[1]), float(vector[2])))


def convert_light(light: AnyLight, source: str, target: str) -> AnyLight:
    converted = replace(light, position=convert_vector(light.position, source, target))
    if isinstance(light, QuadLight):
        converted = replace(converted, normal=convert_vector(light.normal, source, target))
    return converted


def convert_rig(rig: LightRig, target: str) -> LightRig:
    """Return a rig expressed in `target` coordinates (no-op when already there)."""
    source = rig.coordinate_system
    if source == target:
        return rig
    return LightRig(
        tuple(convert_light(light, source, target) for light in rig.lights),
        scene_id=rig.scene_id,
        camera_id=rig.camera_id,
        coordinate_system=target,
    )
