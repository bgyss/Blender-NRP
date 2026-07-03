"""Train proxy operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from ._helpers import cancel_with_status

    class BLENDER_NRP_OT_train_proxy(bpy.types.Operator):
        bl_idname = "blender_nrp.train_proxy"
        bl_label = "Train Proxy"
        bl_description = "Train a compact proxy from the selected path cache"

        def execute(self, context: bpy.types.Context) -> set[str]:
            return cancel_with_status(context, "Proxy training is not implemented yet")


CLASSES = (BLENDER_NRP_OT_train_proxy,) if bpy is not None else ()


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

