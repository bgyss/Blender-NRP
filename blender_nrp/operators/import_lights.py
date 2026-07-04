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
                return cancel_with_status(context, "No light JSON path selected")
            try:
                rig = LightRig.load(Path(bpy.path.abspath(settings.light_json_path)))
                rig = convert_rig(rig, BLENDER_Z_UP)
            except Exception as exc:
                return cancel_with_status(context, f"Light import failed: {exc}")

            for index, light in enumerate(rig.lights, start=1):
                if isinstance(light, QuadLight):
                    bpy.ops.mesh.primitive_plane_add(size=1.0, location=light.position)
                    obj = context.object
                    obj.name = f"NRP_Quad_{index:03d}"
                    obj.scale = (light.width, light.height, 1.0)
                    obj.rotation_mode = "QUATERNION"
                    obj.rotation_quaternion = Vector(light.normal).to_track_quat("Z", "Y")
                    obj["nrp_light_type"] = "quad"
                    obj["nrp_width"] = light.width
                    obj["nrp_height"] = light.height
                else:
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        segments=32,
                        ring_count=16,
                        radius=light.radius,
                        location=light.position,
                    )
                    obj = context.object
                    obj.name = f"NRP_Sphere_{index:03d}"
                    obj["nrp_light_type"] = "sphere"
                    obj["nrp_radius"] = light.radius
                obj["nrp_scene_id"] = rig.scene_id or ""
                obj["nrp_camera_id"] = rig.camera_id or ""
                obj["nrp_coordinate_system"] = BLENDER_Z_UP
                obj["nrp_color"] = list(light.color)
                obj["nrp_intensity"] = light.intensity
            return finish_with_status(context, f"Imported {len(rig.lights)} NRP lights")


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
