"""Load proxy operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from blender_nrp.core.proxy import load_basic_proxy

    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_load_proxy(bpy.types.Operator):
        bl_idname = "blender_nrp.load_proxy"
        bl_label = "Load Proxy"
        bl_description = "Load an existing NRP proxy model"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.model_path:
                return cancel_with_status(context, "No model path selected")
            model_path = Path(bpy.path.abspath(settings.model_path))
            try:
                payload = load_basic_proxy(model_path)
            except Exception as exc:
                return cancel_with_status(context, f"Proxy load failed: {exc}")
            proxy_format = payload.get("format")
            if proxy_format is None:
                return cancel_with_status(context, "Proxy load failed: missing format")
            return finish_with_status(context, f"Loaded proxy {model_path}")


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
