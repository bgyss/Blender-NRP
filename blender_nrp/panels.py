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
            layout.prop(settings, "segment_count")
            layout.prop(settings, "max_segment_distance")
            layout.prop(settings, "output_dir")

            box = layout.box()
            box.label(text="Bake")
            box.operator("blender_nrp.bake_cache")
            box.operator("blender_nrp.validate_cache")
            box.prop(settings, "cache_path")

            box = layout.box()
            box.label(text="Proxy")
            box.operator("blender_nrp.train_proxy")
            box.operator("blender_nrp.load_proxy")
            box.prop(settings, "model_path")

            box = layout.box()
            box.label(text="Relight")
            box.operator("blender_nrp.create_sphere_light")
            box.operator("blender_nrp.relight_preview")
            box.operator("blender_nrp.optimize_lights")

            box = layout.box()
            box.label(text="Interchange")
            box.operator("blender_nrp.import_lights")
            box.operator("blender_nrp.export_lights")
            box.prop(settings, "light_json_path")

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

