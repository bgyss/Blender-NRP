"""Import NRP light JSON into Blender (spheres and quads, with coordinate conversion)."""

from __future__ import annotations

from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

from ..core.coords import BLENDER_Z_UP, convert_rig
from ..core.lights import LightRig, QuadLight

if bpy is not None:
    from mathutils import Vector

    from .. import light_build
    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_import_lights(bpy.types.Operator):
        bl_idname = "blender_nrp.import_lights"
        bl_label = "Import NRP Lights JSON"
        bl_description = (
            "Import NRP lights as Blender objects, converting from the rig's "
            "coordinate system into Blender's"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.light_json_path:
                return cancel_with_status(self, context, "No light JSON path selected")
            try:
                rig = LightRig.load(Path(bpy.path.abspath(settings.light_json_path)))
                rig = convert_rig(rig, BLENDER_Z_UP)
            except Exception as exc:
                return cancel_with_status(self, context, f"Light import failed: {exc}")

            scene_id = rig.scene_id or ""
            camera_id = rig.camera_id or ""
            for index, light in enumerate(rig.lights, start=1):
                if isinstance(light, QuadLight):
                    obj = light_build.create_quad_light(
                        context,
                        name=f"NRP_Quad_{index:03d}",
                        width=light.width,
                        height=light.height,
                        location=tuple(light.position),
                        color=tuple(light.color),
                        intensity=light.intensity,
                        scene_id=scene_id,
                        camera_id=camera_id,
                    )
                    obj.rotation_mode = "QUATERNION"
                    obj.rotation_quaternion = Vector(light.normal).to_track_quat("Z", "Y")
                else:
                    light_build.create_sphere_light(
                        context,
                        name=f"NRP_Sphere_{index:03d}",
                        radius=light.radius,
                        location=tuple(light.position),
                        color=tuple(light.color),
                        intensity=light.intensity,
                        scene_id=scene_id,
                        camera_id=camera_id,
                    )
            return finish_with_status(self, context, f"Imported {len(rig.lights)} NRP lights")


CLASSES = (BLENDER_NRP_OT_import_lights,) if bpy is not None else ()


def register() -> None:
    if bpy is None:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is None:
        return
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
