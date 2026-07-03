"""Load proxy operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from ._helpers import cancel_with_status

    class BLENDER_NRP_OT_load_proxy(bpy.types.Operator):
        bl_idname = "blender_nrp.load_proxy"
        bl_label = "Load Proxy"
        bl_description = "Load an existing NRP proxy model"

        def execute(self, context: bpy.types.Context) -> set[str]:
            return cancel_with_status(context, "Proxy loading is not implemented yet")


CLASSES = (BLENDER_NRP_OT_load_proxy,) if bpy is not None else ()


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

