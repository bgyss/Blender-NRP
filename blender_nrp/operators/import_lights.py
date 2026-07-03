"""Import NRP light JSON into Blender."""

from __future__ import annotations

from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

from blender_nrp.core.lights import LightRig

if bpy is not None:
    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_import_lights(bpy.types.Operator):
        bl_idname = "blender_nrp.import_lights"
        bl_label = "Import NRP Lights JSON"
        bl_description = "Import NRP sphere lights as Blender objects"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.light_json_path:
                return cancel_with_status(context, "No light JSON path selected")
            try:
                rig = LightRig.load(Path(bpy.path.abspath(settings.light_json_path)))
            except Exception as exc:
                return cancel_with_status(context, f"Light import failed: {exc}")

            for index, light in enumerate(rig.lights, start=1):
                bpy.ops.mesh.primitive_uv_sphere_add(
                    segments=32,
                    ring_count=16,
                    radius=light.radius,
                    location=light.position,
                )
                obj = context.object
                obj.name = f"NRP_Sphere_{index:03d}"
                obj["nrp_light_type"] = "sphere"
                obj["nrp_scene_id"] = rig.scene_id or ""
                obj["nrp_camera_id"] = rig.camera_id or ""
                obj["nrp_coordinate_system"] = rig.coordinate_system
                obj["nrp_radius"] = light.radius
                obj["nrp_color"] = light.color
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

