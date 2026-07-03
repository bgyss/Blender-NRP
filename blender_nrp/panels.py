"""Blender UI panels."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None


if bpy is not None:

    class BLENDER_NRP_PT_main(bpy.types.Panel):
        bl_label = "Blender-NRP"
        bl_idname = "BLENDER_NRP_PT_main"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "scene"

        def draw(self, context: bpy.types.Context) -> None:
            layout = self.layout
            settings = context.scene.blender_nrp

            layout.prop(settings, "scene_id")
            layout.prop(settings, "camera")
            row = layout.row(align=True)
            row.prop(settings, "resolution_x")
            row.prop(settings, "resolution_y")
            layout.prop(settings, "output_dir")

            box = layout.box()
            box.label(text="Bake")
            box.prop(settings, "backend")
            if settings.backend == "cycles_capture":
                row = box.row(align=True)
                row.prop(settings, "paths_per_pixel")
                row.prop(settings, "max_bounces")
                box.prop(settings, "packed_cache")
            else:
                box.prop(settings, "segment_count")
            box.prop(settings, "max_segment_distance")
            box.operator("blender_nrp.bake_cache")
            box.operator("blender_nrp.validate_cache")
            box.prop(settings, "cache_path")

            box = layout.box()
            box.label(text="Proxy")
            row = box.row(align=True)
            row.prop(settings, "train_iterations")
            row.prop(settings, "train_device")
            row = box.row(align=True)
            row.operator("blender_nrp.train_proxy")
            row.operator("blender_nrp.cancel_train", text="", icon="X")
            box.operator("blender_nrp.load_proxy")
            box.prop(settings, "model_path")

            box = layout.box()
            box.label(text="Relight")
            row = box.row(align=True)
            row.operator("blender_nrp.create_sphere_light")
            row.operator("blender_nrp.create_quad_light")
            box.operator("blender_nrp.relight_preview")
            row = box.row(align=True)
            row.prop(settings, "live_preview")
            row.prop(settings, "preview_exposure")
            box.prop(settings, "target_image_path")
            row = box.row(align=True)
            row.prop(settings, "optimize_steps")
            row.operator("blender_nrp.optimize_lights", text="Solve")

            box = layout.box()
            box.label(text="Interchange")
            box.operator("blender_nrp.import_lights")
            box.operator("blender_nrp.export_lights")
            box.prop(settings, "light_json_path")
            box.prop(settings, "export_coordinate_system")

            layout.label(text=settings.status)


CLASSES = (BLENDER_NRP_PT_main,) if bpy is not None else ()


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
