"""Optimize lights operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from ._helpers import cancel_with_status

    class BLENDER_NRP_OT_optimize_lights(bpy.types.Operator):
        bl_idname = "blender_nrp.optimize_lights"
        bl_label = "Optimize Lights From Target"
        bl_description = "Solve NRP light parameters against a target image"

        def execute(self, context: bpy.types.Context) -> set[str]:
            return cancel_with_status(context, "Light optimization is not implemented yet")


CLASSES = (BLENDER_NRP_OT_optimize_lights,) if bpy is not None else ()


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

