"""Train proxy operator: real torch training on a background thread.

The worker thread only touches numpy/torch state; all bpy access (status updates,
report writing) happens on the main thread via `bpy.app.timers`. In background mode
(no event loop) training runs synchronously. Without torch installed the operator
reports the missing dependency clearly and leaves the gather preview path intact.
"""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    import threading
    import traceback
    from pathlib import Path

    from blender_nrp.core.path_cache import load_arrays
    from blender_nrp.core.reports import write_json_report
    from blender_nrp.core.torch_proxy import torch_status

    from ._helpers import cancel_with_status, finish_with_status

    _state: dict = {"thread": None, "progress": "", "report": None, "error": None, "cancel": False}

    def _run_training(arrays, model_path: Path, iterations: int, device: str) -> None:
        from blender_nrp.core.torch_proxy.train import train_proxy

        try:
            report = train_proxy(
                arrays,
                model_path,
                iterations=iterations,
                device=device,
                progress=lambda it, total, loss: _state.__setitem__(
                    "progress", f"Training… {it}/{total} (loss {loss:.4f})"
                ),
                should_cancel=lambda: _state["cancel"],
            )
            _state["report"] = report
        except Exception:
            _state["error"] = traceback.format_exc(limit=2)

    def _poll_training() -> float | None:
        """Main-thread timer: mirror worker progress into the scene status."""
        scene = bpy.context.scene
        if scene is None:
            return 0.5
        settings = scene.blender_nrp
        if _state["error"] is not None:
            settings.status = f"Proxy training failed: {_state['error'].splitlines()[-1]}"
            _state["thread"] = None
            return None
        if _state["report"] is not None:
            report = _state["report"]
            model_path = Path(report["model_path"])
            write_json_report(model_path.parent / "train_report.json", report)
            settings.model_path = str(model_path)
            if report.get("cancelled"):
                settings.status = "Training cancelled (partial model saved)"
            else:
                settings.status = (
                    f"Trained proxy on {report['device']} in {report['train_seconds']:.1f}s "
                    f"(val PSNR {report['val_psnr_db_mean']:.1f} dB)"
                )
            _state["thread"] = None
            return None
        if _state["progress"]:
            settings.status = _state["progress"]
        return 0.25

    class BLENDER_NRP_OT_train_proxy(bpy.types.Operator):
        bl_idname = "blender_nrp.train_proxy"
        bl_label = "Train Proxy"
        bl_description = "Train a torch neural proxy from the selected path cache"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if _state["thread"] is not None and _state["thread"].is_alive():
                return cancel_with_status(context, "Training already running")
            if not settings.cache_path:
                return cancel_with_status(context, "No cache path selected")
            available, detail = torch_status()
            if not available:
                return cancel_with_status(context, detail)
            cache_path = Path(bpy.path.abspath(settings.cache_path))
            try:
                arrays = load_arrays(cache_path).arrays
            except Exception as exc:
                return cancel_with_status(context, f"Cache load failed: {exc}")
            model_path = cache_path.parent / "model.pt"

            if bpy.app.background:
                from blender_nrp.core.torch_proxy.train import train_proxy

                try:
                    report = train_proxy(
                        arrays,
                        model_path,
                        iterations=settings.train_iterations,
                        device=settings.train_device,
                    )
                except Exception as exc:
                    return cancel_with_status(context, f"Proxy training failed: {exc}")
                write_json_report(model_path.parent / "train_report.json", report)
                settings.model_path = str(model_path)
                return finish_with_status(
                    context,
                    f"Trained proxy on {report['device']} "
                    f"(val PSNR {report['val_psnr_db_mean']:.1f} dB)",
                )

            _state.update({"progress": "", "report": None, "error": None, "cancel": False})
            thread = threading.Thread(
                target=_run_training,
                args=(arrays, model_path, settings.train_iterations, settings.train_device),
                daemon=True,
            )
            _state["thread"] = thread
            thread.start()
            bpy.app.timers.register(_poll_training, first_interval=0.25)
            return finish_with_status(context, "Training started in background…")

    class BLENDER_NRP_OT_cancel_train(bpy.types.Operator):
        bl_idname = "blender_nrp.cancel_train"
        bl_label = "Cancel Training"
        bl_description = "Stop the background proxy training after the current iteration"

        def execute(self, context: bpy.types.Context) -> set[str]:
            if _state["thread"] is None or not _state["thread"].is_alive():
                return cancel_with_status(context, "No training in progress")
            _state["cancel"] = True
            return finish_with_status(context, "Cancelling training…")


CLASSES = (
    (BLENDER_NRP_OT_train_proxy, BLENDER_NRP_OT_cancel_train) if bpy is not None else ()
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
