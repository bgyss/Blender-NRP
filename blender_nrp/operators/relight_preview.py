"""Relight preview and NRP light-creation operators."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from blender_nrp import preview
    from blender_nrp.core.coords import BLENDER_Z_UP

    from ._helpers import cancel_with_status, finish_with_status

    def _stamp_light_props(context: bpy.types.Context, obj: bpy.types.Object) -> None:
        obj["nrp_scene_id"] = context.scene.blender_nrp.scene_id
        obj["nrp_camera_id"] = (
            context.scene.blender_nrp.camera.name if context.scene.blender_nrp.camera else ""
        )
        obj["nrp_coordinate_system"] = BLENDER_Z_UP
        obj["nrp_color"] = (1.0, 1.0, 1.0)
        obj["nrp_intensity"] = 1.0

    class BLENDER_NRP_OT_create_sphere_light(bpy.types.Operator):
        bl_idname = "blender_nrp.create_sphere_light"
        bl_label = "Create NRP Sphere Light"
        bl_description = "Create a visible NRP sphere emitter object"

        def execute(self, context: bpy.types.Context) -> set[str]:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.25)
            obj = context.object
            obj.name = "NRP_Sphere_001"
            obj["nrp_light_type"] = "sphere"
            obj["nrp_radius"] = 0.25
            _stamp_light_props(context, obj)
            return finish_with_status(context, "Created NRP sphere light")

    class BLENDER_NRP_OT_create_quad_light(bpy.types.Operator):
        bl_idname = "blender_nrp.create_quad_light"
        bl_label = "Create NRP Quad Light"
        bl_description = (
            "Create a visible NRP rectangle emitter (its local +Z axis is the "
            "emission normal — rotate the object to aim it)"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            bpy.ops.mesh.primitive_plane_add(size=1.0)
            obj = context.object
            obj.name = "NRP_Quad_001"
            obj["nrp_light_type"] = "quad"
            obj["nrp_width"] = 1.0
            obj["nrp_height"] = 1.0
            _stamp_light_props(context, obj)
            return finish_with_status(context, "Created NRP quad light")

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
                return cancel_with_status(context, message)
            return finish_with_status(context, message)


CLASSES = (
    (
        BLENDER_NRP_OT_create_sphere_light,
        BLENDER_NRP_OT_create_quad_light,
        BLENDER_NRP_OT_relight_preview,
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
