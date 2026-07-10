"""Bake path-cache operator (modal with progress + cancel for cycles_capture)."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from ..backends import cycles_capture, stock_blender_hemi
    from ..backends.interface import BakeSettings
    from ._helpers import cancel_with_status, finish_with_status

    def _settings_from_scene(context: bpy.types.Context) -> BakeSettings:
        settings = context.scene.blender_nrp
        return BakeSettings(
            scene_id=settings.scene_id,
            output_dir=Path(bpy.path.abspath(settings.output_dir)),
            width=settings.resolution_x,
            height=settings.resolution_y,
            segment_count=settings.segment_count,
            max_segment_distance=settings.max_segment_distance,
            camera_id=settings.camera.name if settings.camera else "Camera",
            paths_per_pixel=settings.paths_per_pixel,
            max_bounces=settings.max_bounces,
            packed=settings.packed_cache,
            reference_check=True,
            tracer_engine=settings.tracer_engine,
        )

    class BLENDER_NRP_OT_bake_cache(bpy.types.Operator):
        bl_idname = "blender_nrp.bake_cache"
        bl_label = "Bake Path Cache"
        bl_description = (
            "Bake an NRP path cache for the selected fixed camera shot "
            "(Esc cancels a running Cycles capture)"
        )

        _timer = None
        _steps = None
        _original_camera = None

        def _finish(self, context: bpy.types.Context, result) -> set[str]:
            context.scene.blender_nrp.cache_path = str(result.cache_path)
            # Auto-validate: a bake that quietly produced a malformed cache is worse
            # than a loud failure, so fold validation straight into the bake result.
            from ..core.path_cache import validate_npz

            name = Path(result.cache_path).name
            try:
                report = validate_npz(Path(result.cache_path))
            except Exception as exc:
                return finish_with_status(
                    self, context, f"Baked {name}; validation errored: {exc}", level="WARNING"
                )
            if not report.ok:
                return finish_with_status(
                    self,
                    context,
                    f"Baked {name} but validation FAILED: {'; '.join(report.errors)}",
                    level="ERROR",
                )
            layout = "packed" if report.packed else "default"
            return finish_with_status(
                self,
                context,
                f"Baked + validated {name}: {report.width}x{report.height}, "
                f"{report.segment_count} segments, {layout} layout",
            )

        def _cleanup(self, context: bpy.types.Context) -> None:
            if self._timer is not None:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            if self._original_camera is not None:
                context.scene.camera = self._original_camera
                self._original_camera = None
            self._steps = None

        def execute(self, context: bpy.types.Context) -> set[str]:
            scene_settings = context.scene.blender_nrp
            self._original_camera = context.scene.camera
            if scene_settings.camera:
                context.scene.camera = scene_settings.camera
            bake_settings = _settings_from_scene(context)

            if scene_settings.backend == "stock_blender_hemi":
                try:
                    result = stock_blender_hemi.bake(context, bake_settings)
                except Exception as exc:
                    return cancel_with_status(self, context, f"Path-cache bake failed: {exc}")
                finally:
                    context.scene.camera = self._original_camera
                    self._original_camera = None
                return self._finish(context, result)

            if bpy.app.background:
                # No event loop in background mode: run the capture synchronously.
                try:
                    result = cycles_capture.bake(context, bake_settings)
                except Exception as exc:
                    return cancel_with_status(self, context, f"Path-cache bake failed: {exc}")
                finally:
                    context.scene.camera = self._original_camera
                    self._original_camera = None
                return self._finish(context, result)

            self._steps = cycles_capture.bake_steps(context, bake_settings)
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.01, window=context.window)
            wm.modal_handler_add(self)
            context.scene.blender_nrp.status = "Baking… 0% (Esc to cancel)"
            return {"RUNNING_MODAL"}

        def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
            if event.type == "ESC":
                self._cleanup(context)
                return cancel_with_status(self, context, "Bake cancelled")
            if event.type != "TIMER":
                return {"PASS_THROUGH"}
            try:
                fraction, message, result = next(self._steps)
            except StopIteration:
                self._cleanup(context)
                return cancel_with_status(self, context, "Bake ended without a result")
            except Exception as exc:
                self._cleanup(context)
                return cancel_with_status(self, context, f"Path-cache bake failed: {exc}")
            if result is not None:
                self._cleanup(context)
                return self._finish(context, result)
            context.scene.blender_nrp.status = (
                f"Baking… {fraction * 100.0:.0f}% — {message} (Esc to cancel)"
            )
            return {"RUNNING_MODAL"}


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
