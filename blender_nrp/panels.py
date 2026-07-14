"""Blender UI panel.

The panel is organised as three numbered stages — Path Cache, Neural Proxy,
Relight — each with a right-aligned status chip (checkmark when the stage is
complete, dot when it isn't) so it's obvious at a glance where you are in the
workflow. Cache validation runs automatically after a bake and the proxy loads
automatically after training, so those buttons are no longer surfaced here; the
persistent status line at the bottom mirrors the last operator's toast.
"""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None


if bpy is not None:
    import json
    from pathlib import Path

    from . import proxy_runtime
    from .core.cost import estimate_bake_seconds, estimate_cost_usd
    from .core.execution import ExecutionQueue
    from .preview import PREVIEW_IMAGE_NAME

    def _exists(path_str: str) -> bool:
        if not path_str:
            return False
        try:
            return Path(bpy.path.abspath(path_str)).exists()
        except Exception:
            return False

    def _draw_pipeline_details(layout, settings) -> None:
        if not settings.last_report_path and not settings.last_error_details:
            return
        layout.prop(
            settings,
            "show_details",
            toggle=True,
            icon="DISCLOSURE_TRI_DOWN" if settings.show_details else "DISCLOSURE_TRI_RIGHT",
        )
        if not settings.show_details:
            return
        details = layout.box()
        if settings.last_report_path:
            report_path = Path(bpy.path.abspath(settings.last_report_path))
            details.label(text=f"Report: {report_path.name}", icon="TEXT")
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                details.label(text=f"Could not read report: {exc}", icon="ERROR")
            else:
                if "ok" in report:
                    details.label(
                        text=f"Result: {'OK' if report['ok'] else 'FAILED'}",
                        icon="CHECKMARK" if report["ok"] else "ERROR",
                    )
                if report.get("error"):
                    details.label(text=str(report["error"])[:180], icon="ERROR")
                for key in ("warnings", "limitations", "approximation_limits"):
                    values = report.get(key) or []
                    if values:
                        details.label(text=key.replace("_", " ").title())
                        for value in values[:6]:
                            details.label(text=str(value)[:180], icon="DOT")
        if settings.last_error_details:
            details.label(text="Worker details", icon="CONSOLE")
            lines = [
                line.strip()
                for line in settings.last_error_details.splitlines()
                if line.strip()
            ]
            for line in lines[-8:]:
                details.label(text=line[:180])

    class BLENDER_NRP_PT_main(bpy.types.Panel):
        bl_label = "Blender-NRP"
        bl_idname = "BLENDER_NRP_PT_main"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "scene"

        @staticmethod
        def _stage(layout, title: str, done: bool, done_text: str, todo_text: str) -> None:
            row = layout.row()
            row.label(text=title)
            chip = row.row()
            chip.alignment = "RIGHT"
            chip.label(
                text=done_text if done else todo_text,
                icon="CHECKMARK" if done else "DOT",
            )

        def draw(self, context: bpy.types.Context) -> None:
            layout = self.layout
            settings = context.scene.blender_nrp

            cache_ready = _exists(settings.cache_path)
            model_ready = _exists(settings.model_path)
            proxy_loaded = proxy_runtime.model is not None
            preview_ready = bpy.data.images.get(PREVIEW_IMAGE_NAME) is not None

            box = layout.box()
            box.label(text="Make Scene Relightable", icon="LIGHT")
            box.prop(settings, "compute", text="Compute")
            box.prop(settings, "quality_preset", text="Quality")
            box.prop(settings, "use_existing_cache")
            if settings.compute == "runpod":
                prefs = context.preferences.addons["blender_nrp"].preferences
                seconds = estimate_bake_seconds(
                    settings.resolution_x,
                    settings.resolution_y,
                    settings.paths_per_pixel,
                    settings.max_bounces,
                )
                cost = estimate_cost_usd(
                    settings.resolution_x,
                    settings.resolution_y,
                    settings.paths_per_pixel,
                    settings.max_bounces,
                    prefs.runpod_hourly_rate,
                )
                box.label(
                    text=f"Estimated bake: {seconds / 60.0:.1f} min / ${cost:.3f}",
                    icon="TIME",
                )
            row = box.row(align=True)
            row.scale_y = 1.4
            row.operator("blender_nrp.make_relightable", icon="PLAY")
            row.operator("blender_nrp.cancel_make_relightable", text="", icon="X")
            box.operator("blender_nrp.reconcile_jobs", icon="FILE_REFRESH")
            queue_dir = Path(bpy.path.abspath(settings.output_dir)) / ".nrp_jobs"
            try:
                queued_jobs = ExecutionQueue(queue_dir).load()
            except (OSError, ValueError):
                queued_jobs = []
                box.label(text="Persisted job queue is unreadable", icon="ERROR")
            if queued_jobs:
                box.label(text="Active / unresolved jobs", icon="SORTTIME")
                for queued in queued_jobs:
                    queued_row = box.row(align=True)
                    queued_row.label(
                        text=f"{queued.backend_id}: {queued.job_id[:12]}", icon="RENDER_ANIMATION"
                    )
                    fetch = queued_row.operator(
                        "blender_nrp.fetch_persisted_job", text="Fetch & Stop", icon="IMPORT"
                    )
                    fetch.job_id = queued.job_id
                    fetch.backend_id = queued.backend_id
                    cancel = queued_row.operator(
                        "blender_nrp.cancel_persisted_job", text="", icon="CANCEL"
                    )
                    cancel.job_id = queued.job_id
                    cancel.backend_id = queued.backend_id
            box.prop(settings, "show_advanced", toggle=True)

            if not settings.show_advanced:
                layout.separator()
                layout.label(text="Advanced controls are hidden", icon="PREFERENCES")
                layout.separator()
                layout.label(text=settings.status, icon="INFO")
                _draw_pipeline_details(layout, settings)
                return

            # --- Scene setup ------------------------------------------------
            col = layout.column(align=True)
            col.prop(settings, "scene_id")
            col.prop(settings, "camera")
            row = col.row(align=True)
            row.prop(settings, "resolution_x")
            row.prop(settings, "resolution_y")
            col.prop(settings, "output_dir")

            # --- Stage 1: Path Cache ---------------------------------------
            box = layout.box()
            self._stage(box, "1 · Path Cache", cache_ready, "baked", "not baked")
            box.prop(settings, "backend", text="")
            box.prop(settings, "tracer_engine")
            if settings.backend == "cycles_capture":
                row = box.row(align=True)
                row.prop(settings, "paths_per_pixel")
                row.prop(settings, "max_bounces")
                box.prop(settings, "packed_cache")
            else:
                box.prop(settings, "segment_count")
            box.prop(settings, "max_segment_distance")
            box.operator("blender_nrp.bake_cache", icon="RENDER_STILL")
            box.prop(settings, "cache_path", text="Cache")

            # --- Stage 2: Neural Proxy -------------------------------------
            box = layout.box()
            self._stage(
                box,
                "2 · Neural Proxy",
                proxy_loaded,
                "loaded",
                "trained · load below" if model_ready else "not trained",
            )
            row = box.row(align=True)
            row.prop(settings, "train_iterations")
            row.prop(settings, "train_device", text="")
            row = box.row(align=True)
            row.enabled = cache_ready
            row.operator("blender_nrp.train_proxy", icon="PLAY")
            row.operator("blender_nrp.cancel_train", text="", icon="X")
            if model_ready and not proxy_loaded:
                box.operator("blender_nrp.load_proxy", icon="IMPORT")
            box.prop(settings, "model_path", text="Model")

            # --- Stage 3: Relight ------------------------------------------
            box = layout.box()
            self._stage(box, "3 · Relight", preview_ready, "preview ready", "no preview yet")
            row = box.row(align=True)
            row.operator("blender_nrp.create_sphere_light", text="Sphere", icon="LIGHT_POINT")
            row.operator("blender_nrp.create_quad_light", text="Quad", icon="LIGHT_AREA")
            box.operator("blender_nrp.relight_preview", icon="IMAGE_RGB")
            if preview_ready:
                box.label(text=f"Image Editor → '{PREVIEW_IMAGE_NAME}'", icon="IMAGE_DATA")
            else:
                box.label(text="Add lights, then Preview to create the image", icon="INFO")
            row = box.row(align=True)
            row.prop(settings, "live_preview", toggle=True)
            row.prop(settings, "preview_exposure")
            sub = box.column(align=True)
            sub.prop(settings, "target_image_path", text="Target")
            row = sub.row(align=True)
            row.prop(settings, "optimize_steps")
            row.operator("blender_nrp.optimize_lights", text="Solve", icon="SHADERFX")
            row = sub.row(align=True)
            row.operator(
                "blender_nrp.match_reference", text="Match Reference", icon="IMAGE_REFERENCE"
            )
            if settings.match_pending_path:
                sub.operator(
                    "blender_nrp.review_match_reference", text="Review Wipe", icon="IMAGE_DATA"
                )
                row = sub.row(align=True)
                row.operator("blender_nrp.apply_match_reference", text="Apply", icon="CHECKMARK")
                row.operator("blender_nrp.discard_match_reference", text="Discard", icon="X")
            sub = box.column(align=True)
            sub.label(text="Rig Snapshots")
            row = sub.row(align=True)
            row.prop(settings, "snapshot_name", text="")
            row.operator("blender_nrp.save_rig_snapshot", text="Save", icon="FILE_TICK")
            row = sub.row(align=True)
            row.prop(settings, "snapshot_a", text="A")
            apply_a = row.operator("blender_nrp.apply_rig_snapshot", text="Apply")
            apply_a.slot = "a"
            row = sub.row(align=True)
            row.prop(settings, "snapshot_b", text="B")
            apply_b = row.operator("blender_nrp.apply_rig_snapshot", text="Apply")
            apply_b.slot = "b"

            # --- Interchange -----------------------------------------------
            box = layout.box()
            box.label(text="Interchange")
            row = box.row(align=True)
            row.operator("blender_nrp.import_lights", text="Import", icon="IMPORT")
            row.operator("blender_nrp.export_lights", text="Export", icon="EXPORT")
            box.prop(settings, "light_json_path", text="JSON")
            box.prop(settings, "export_coordinate_system", text="Coords")

            # --- Status ----------------------------------------------------
            layout.separator()
            layout.label(text=settings.status, icon="INFO")
            _draw_pipeline_details(layout, settings)


CLASSES = (BLENDER_NRP_PT_main,) if bpy is not None else ()


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
