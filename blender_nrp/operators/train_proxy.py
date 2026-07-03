"""Train proxy operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from blender_nrp.core.proxy import train_basic_proxy

    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_train_proxy(bpy.types.Operator):
        bl_idname = "blender_nrp.train_proxy"
        bl_label = "Train Proxy"
        bl_description = "Train a compact proxy from the selected path cache"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.cache_path:
                return cancel_with_status(context, "No cache path selected")
            cache_path = Path(bpy.path.abspath(settings.cache_path))
            output_dir = cache_path.parent
            model_path = output_dir / "model.pt"
            try:
                train_basic_proxy(cache_path, model_path, output_dir / "train_report.json")
            except Exception as exc:
                return cancel_with_status(context, f"Proxy training failed: {exc}")
            settings.model_path = str(model_path)
            return finish_with_status(context, f"Trained proxy {model_path}")


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
