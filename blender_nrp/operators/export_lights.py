"""Export selected NRP lights from Blender, converting coordinates on the way out."""

from __future__ import annotations

from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

from ..core.coords import BLENDER_Z_UP, convert_rig
from ..core.light_objects import collect_rig_lights
from ..core.lights import LightRig

if bpy is not None:
    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_export_lights(bpy.types.Operator):
        bl_idname = "blender_nrp.export_lights"
        bl_label = "Export Selected NRP Lights JSON"
        bl_description = (
            "Export selected NRP light objects as JSON in the configured "
            "target coordinate system"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.light_json_path:
                return cancel_with_status(context, "No light JSON path selected")
            lights = collect_rig_lights(context.selected_objects)
            if not lights:
                return cancel_with_status(context, "No selected NRP lights")
            rig = LightRig(
                tuple(lights),
                scene_id=settings.scene_id,
                camera_id=settings.camera.name if settings.camera else None,
                coordinate_system=BLENDER_Z_UP,
            )
            try:
                rig = convert_rig(rig, settings.export_coordinate_system)
            except ValueError as exc:
                return cancel_with_status(context, f"Light export failed: {exc}")
            rig.save(Path(bpy.path.abspath(settings.light_json_path)))
            return finish_with_status(
                context,
                f"Exported {len(lights)} NRP lights ({rig.coordinate_system})",
            )


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
