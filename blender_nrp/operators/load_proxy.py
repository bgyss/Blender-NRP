"""Load proxy operator: torch TorchNRP artifacts, with V1 numpy-summary detection."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from blender_nrp import proxy_runtime
    from blender_nrp.core.torch_proxy import torch_status

    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_load_proxy(bpy.types.Operator):
        bl_idname = "blender_nrp.load_proxy"
        bl_label = "Load Proxy"
        bl_description = "Load a trained NRP proxy model for preview and optimization"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.model_path:
                return cancel_with_status(context, "No model path selected")
            model_path = Path(bpy.path.abspath(settings.model_path))
            if not model_path.exists():
                return cancel_with_status(context, f"Model not found: {model_path}")
            available, detail = torch_status()
            if not available:
                proxy_runtime.clear()
                return cancel_with_status(context, detail)
            try:
                from blender_nrp.core.torch_proxy.model import TorchNRP

                model = TorchNRP.load(str(model_path))
            except Exception as exc:
                proxy_runtime.clear()
                # V1 wrote a numpy-summary artifact under the same name.
                try:
                    import numpy as np

                    with np.load(model_path) as npz:
                        if "format" in npz.files:
                            return cancel_with_status(
                                context,
                                "This model.pt is a V1 numpy-summary stub, not a torch "
                                "proxy — re-train with Train Proxy",
                            )
                except Exception:
                    pass
                return cancel_with_status(context, f"Proxy load failed: {exc}")
            proxy_runtime.set_model(model, str(model_path), model.light_type)
            return finish_with_status(
                context,
                f"Loaded torch proxy ({model.light_type}, "
                f"{model.parameter_count} params)",
            )


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
