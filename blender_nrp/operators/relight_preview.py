"""Relight preview operators."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from ._helpers import cancel_with_status, finish_with_status

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
            obj["nrp_color"] = (1.0, 1.0, 1.0)
            obj["nrp_intensity"] = 1.0
            return finish_with_status(context, "Created NRP sphere light")

    class BLENDER_NRP_OT_relight_preview(bpy.types.Operator):
        bl_idname = "blender_nrp.relight_preview"
        bl_label = "Preview Relight"
        bl_description = "Preview fixed-camera relighting from the active proxy"

        def execute(self, context: bpy.types.Context) -> set[str]:
            return cancel_with_status(context, "Relight preview is not implemented yet")


CLASSES = (
    BLENDER_NRP_OT_create_sphere_light,
    BLENDER_NRP_OT_relight_preview,
) if bpy is not None else ()


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

