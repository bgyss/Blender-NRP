"""Bake path-cache operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from blender_nrp.backends.interface import BakeSettings
    from blender_nrp.backends.stock_blender_hemi import bake

    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_bake_cache(bpy.types.Operator):
        bl_idname = "blender_nrp.bake_cache"
        bl_label = "Bake Path Cache"
        bl_description = "Bake an NRP path cache for the selected fixed camera shot"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            original_camera = context.scene.camera
            if settings.camera:
                context.scene.camera = settings.camera
            try:
                result = bake(
                    context,
                    BakeSettings(
                        scene_id=settings.scene_id,
                        output_dir=Path(bpy.path.abspath(settings.output_dir)),
                        width=settings.resolution_x,
                        height=settings.resolution_y,
                        segment_count=settings.segment_count,
                        max_segment_distance=settings.max_segment_distance,
                        camera_id=settings.camera.name if settings.camera else "Camera",
                    ),
                )
            except Exception as exc:
                return cancel_with_status(context, f"Path-cache bake failed: {exc}")
            finally:
                context.scene.camera = original_camera
            settings.cache_path = str(result.cache_path)
            return finish_with_status(context, f"Baked {result.cache_path}")


CLASSES = (BLENDER_NRP_OT_bake_cache,) if bpy is not None else ()


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
