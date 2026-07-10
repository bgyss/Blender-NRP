"""Relight preview and NRP light-creation operators."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from .. import light_build, preview
    from ._helpers import cancel_with_status, finish_with_status

    def _scene_ids(context: bpy.types.Context) -> tuple[str, str]:
        settings = context.scene.blender_nrp
        camera_id = settings.camera.name if settings.camera else ""
        return settings.scene_id, camera_id

    class BLENDER_NRP_OT_create_sphere_light(bpy.types.Operator):
        bl_idname = "blender_nrp.create_sphere_light"
        bl_label = "Add NRP Sphere Light"
        bl_description = "Create a visible NRP sphere emitter object at the 3D cursor"

        def execute(self, context: bpy.types.Context) -> set[str]:
            scene_id, camera_id = _scene_ids(context)
            obj = light_build.create_sphere_light(
                context,
                location=tuple(context.scene.cursor.location),
                scene_id=scene_id,
                camera_id=camera_id,
            )
            return finish_with_status(self, context, f"Added NRP sphere light '{obj.name}'")

    class BLENDER_NRP_OT_create_quad_light(bpy.types.Operator):
        bl_idname = "blender_nrp.create_quad_light"
        bl_label = "Add NRP Quad Light"
        bl_description = (
            "Create a visible NRP rectangle emitter at the 3D cursor (its local +Z "
            "axis is the emission normal — rotate the object to aim it)"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            scene_id, camera_id = _scene_ids(context)
            obj = light_build.create_quad_light(
                context,
                location=tuple(context.scene.cursor.location),
                scene_id=scene_id,
                camera_id=camera_id,
            )
            return finish_with_status(self, context, f"Added NRP quad light '{obj.name}'")

    class BLENDER_NRP_OT_relight_preview(bpy.types.Operator):
        bl_idname = "blender_nrp.relight_preview"
        bl_label = "Preview Relight"
        bl_description = (
            "Relight the fixed camera view into the 'NRP Relight Preview' image "
            "(uses the loaded proxy when available, exact cache gather otherwise)"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            ok, message = preview.update_preview(context)
            if not ok:
                return cancel_with_status(self, context, message)
            return finish_with_status(self, context, message)

    class BLENDER_NRP_OT_select_light(bpy.types.Operator):
        bl_idname = "blender_nrp.select_light"
        bl_label = "Select NRP Light"
        object_name: bpy.props.StringProperty()

        def execute(self, context: bpy.types.Context) -> set[str]:
            obj = context.scene.objects.get(self.object_name)
            if obj is None:
                return cancel_with_status(self, context, f"Light not found: {self.object_name}")
            for selected in context.selected_objects:
                selected.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            return {"FINISHED"}

    class BLENDER_NRP_OT_duplicate_light(bpy.types.Operator):
        bl_idname = "blender_nrp.duplicate_light"
        bl_label = "Duplicate NRP Light"
        object_name: bpy.props.StringProperty()

        def execute(self, context: bpy.types.Context) -> set[str]:
            source = context.scene.objects.get(self.object_name)
            if source is None:
                return cancel_with_status(self, context, f"Light not found: {self.object_name}")
            copy = source.copy()
            if source.data is not None:
                copy.data = source.data.copy()
            copy.name = f"{source.name}_copy"
            context.collection.objects.link(copy)
            copy.location.x += 0.25
            for selected in context.selected_objects:
                selected.select_set(False)
            copy.select_set(True)
            context.view_layer.objects.active = copy
            return finish_with_status(self, context, f"Duplicated NRP light '{copy.name}'")

    class BLENDER_NRP_OT_delete_light(bpy.types.Operator):
        bl_idname = "blender_nrp.delete_light"
        bl_label = "Delete NRP Light"
        object_name: bpy.props.StringProperty()

        def execute(self, context: bpy.types.Context) -> set[str]:
            obj = context.scene.objects.get(self.object_name)
            if obj is None:
                return cancel_with_status(self, context, f"Light not found: {self.object_name}")
            name = obj.name
            bpy.data.objects.remove(obj, do_unlink=True)
            return finish_with_status(self, context, f"Deleted NRP light '{name}'")


CLASSES = (
    (
        BLENDER_NRP_OT_create_sphere_light,
        BLENDER_NRP_OT_create_quad_light,
        BLENDER_NRP_OT_relight_preview,
        BLENDER_NRP_OT_select_light,
        BLENDER_NRP_OT_duplicate_light,
        BLENDER_NRP_OT_delete_light,
    )
    if bpy is not None
    else ()
)


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
