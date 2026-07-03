"""Bake path-cache operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from ._helpers import cancel_with_status

    class BLENDER_NRP_OT_bake_cache(bpy.types.Operator):
        bl_idname = "blender_nrp.bake_cache"
        bl_label = "Bake Path Cache"
        bl_description = "Bake an NRP path cache for the selected fixed camera shot"

        def execute(self, context: bpy.types.Context) -> set[str]:
            return cancel_with_status(context, "Path-cache baking is not implemented yet")


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

