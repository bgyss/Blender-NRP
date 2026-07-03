"""Export selected NRP lights from Blender."""

from __future__ import annotations

from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

from blender_nrp.core.lights import LightRig, SphereLight

if bpy is not None:
    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_export_lights(bpy.types.Operator):
        bl_idname = "blender_nrp.export_lights"
        bl_label = "Export Selected NRP Lights JSON"
        bl_description = "Export selected NRP sphere-light objects as JSON"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.light_json_path:
                return cancel_with_status(context, "No light JSON path selected")
            lights: list[SphereLight] = []
            for obj in context.selected_objects:
                if obj.get("nrp_light_type") != "sphere":
                    continue
                lights.append(
                    SphereLight(
                        position=tuple(float(v) for v in obj.location),
                        radius=float(obj.get("nrp_radius", 0.25)),
                        color=tuple(float(v) for v in obj.get("nrp_color", (1.0, 1.0, 1.0))),
                        intensity=float(obj.get("nrp_intensity", 1.0)),
                    )
                )
            if not lights:
                return cancel_with_status(context, "No selected NRP sphere lights")
            rig = LightRig(
                tuple(lights),
                scene_id=settings.scene_id,
                camera_id=settings.camera.name if settings.camera else None,
            )
            rig.save(Path(bpy.path.abspath(settings.light_json_path)))
            return finish_with_status(context, f"Exported {len(lights)} NRP lights")


CLASSES = (BLENDER_NRP_OT_export_lights,) if bpy is not None else ()


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

