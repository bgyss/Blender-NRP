"""Reconcile durable local, LAN, and cloud jobs after Blender restarts."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from ..core.execution import (
        ExecutionQueue,
        LocalSubprocessBackend,
        RunPodExecutionBackend,
        SshExecutionBackend,
    )
    from ..core.validation import validate_cache_bundle
    from ._helpers import cancel_with_status, finish_with_status

    def _configured_backends(context, queue_dir: Path):
        backends = {
            "local": LocalSubprocessBackend(queue_dir, blender_binary=bpy.app.binary_path)
        }
        prefs = context.preferences.addons["blender_nrp"].preferences
        if prefs.ssh_host and prefs.ssh_remote_root and prefs.ssh_worker_root:
            backends["ssh"] = SshExecutionBackend(
                queue_dir,
                host=prefs.ssh_host,
                remote_root=prefs.ssh_remote_root,
                worker_root=prefs.ssh_worker_root,
                blender_binary=prefs.ssh_blender_binary,
                python_binary=prefs.ssh_python_binary,
            )
        if prefs.runpod_api_key and prefs.runpod_image and prefs.runpod_worker_root:
            backends["runpod"] = RunPodExecutionBackend(
                queue_dir,
                api_key=prefs.runpod_api_key,
                image_name=prefs.runpod_image,
                worker_root=prefs.runpod_worker_root,
                gpu_type=prefs.runpod_gpu_type,
            )
        return backends

    def _reconcile(context):
        settings = context.scene.blender_nrp
        queue_dir = Path(bpy.path.abspath(settings.output_dir)) / ".nrp_jobs"
        queue = ExecutionQueue(queue_dir)
        return queue.reconcile(_configured_backends(context, queue_dir))

    def _startup_reconcile() -> float | None:
        context = bpy.context
        if context.scene is None:
            return 1.0
        try:
            statuses = _reconcile(context)
        except Exception as exc:
            context.scene.blender_nrp.status = f"Startup job reconciliation failed: {exc}"
            return None
        if statuses:
            active = sum(status.state in {"queued", "running"} for status in statuses.values())
            failed = sum(status.state == "failed" for status in statuses.values())
            context.scene.blender_nrp.status = (
                f"Reconciled {len(statuses)} persisted jobs: "
                f"{active} active, {failed} failed/orphaned"
            )
        return None

    class BLENDER_NRP_OT_reconcile_jobs(bpy.types.Operator):
        bl_idname = "blender_nrp.reconcile_jobs"
        bl_label = "Reconcile Jobs"
        bl_description = "Find running worker jobs after a Blender restart and surface their status"

        def execute(self, context: bpy.types.Context) -> set[str]:
            try:
                statuses = _reconcile(context)
            except Exception as exc:
                return cancel_with_status(self, context, f"Job reconciliation failed: {exc}")
            active = sum(status.state in {"queued", "running"} for status in statuses.values())
            failures = sum(status.state == "failed" for status in statuses.values())
            return finish_with_status(
                self,
                context,
                f"Reconciled {len(statuses)} jobs: {active} active, {failures} failed/orphaned",
            )

    class BLENDER_NRP_OT_cancel_persisted_job(bpy.types.Operator):
        bl_idname = "blender_nrp.cancel_persisted_job"
        bl_label = "Cancel Persisted Job"
        bl_description = "Cancel a reconciled local, LAN, or cloud job"

        job_id: bpy.props.StringProperty()
        backend_id: bpy.props.StringProperty()

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            queue_dir = Path(bpy.path.abspath(settings.output_dir)) / ".nrp_jobs"
            queue = ExecutionQueue(queue_dir)
            backend = _configured_backends(context, queue_dir).get(self.backend_id)
            if backend is None:
                return cancel_with_status(
                    self,
                    context,
                    f"Cannot cancel {self.job_id[:12]}: configure {self.backend_id} first",
                )
            try:
                backend.cancel(self.job_id)
            except Exception as exc:
                return cancel_with_status(
                    self, context, f"Could not cancel {self.job_id[:12]}: {exc}"
                )
            queue.remove(self.job_id)
            return finish_with_status(
                self, context, f"Cancelled {self.backend_id} job {self.job_id[:12]}"
            )

    class BLENDER_NRP_OT_fetch_persisted_job(bpy.types.Operator):
        bl_idname = "blender_nrp.fetch_persisted_job"
        bl_label = "Fetch Persisted Job"
        bl_description = "Fetch completed artifacts and stop any billing cloud pod"

        job_id: bpy.props.StringProperty()
        backend_id: bpy.props.StringProperty()

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            queue_dir = Path(bpy.path.abspath(settings.output_dir)) / ".nrp_jobs"
            queue = ExecutionQueue(queue_dir)
            backend = _configured_backends(context, queue_dir).get(self.backend_id)
            if backend is None:
                return cancel_with_status(
                    self,
                    context,
                    f"Cannot fetch {self.job_id[:12]}: configure {self.backend_id} first",
                )
            try:
                progress = backend.status(self.job_id)
                if progress.state != "succeeded":
                    return cancel_with_status(
                        self,
                        context,
                        f"Job {self.job_id[:12]} is {progress.state}; fetch when complete",
                    )
                artifacts = backend.fetch(self.job_id)
                if "path_cache" in artifacts:
                    metadata = artifacts.get("metadata")
                    if metadata is None:
                        raise RuntimeError("completed bake has no metadata.json")
                    report = validate_cache_bundle(artifacts["path_cache"], metadata)
                    if not report.ok:
                        raise RuntimeError("; ".join(report.errors))
                    settings.cache_path = str(artifacts["path_cache"])
                if "model" in artifacts:
                    settings.model_path = str(artifacts["model"])
                    if bpy.ops.blender_nrp.load_proxy() != {"FINISHED"}:
                        raise RuntimeError(settings.status)
                for name, path in artifacts.items():
                    if name.endswith("_report") or name == "bake_report":
                        settings.last_report_path = str(path)
            except Exception as exc:
                settings.last_error_details = str(exc)
                return cancel_with_status(
                    self, context, f"Could not fetch {self.job_id[:12]}: {exc}"
                )
            queue.remove(self.job_id)
            return finish_with_status(
                self,
                context,
                f"Fetched {self.backend_id} job {self.job_id[:12]} and released its worker",
            )


CLASSES = (
    (
        BLENDER_NRP_OT_reconcile_jobs,
        BLENDER_NRP_OT_fetch_persisted_job,
        BLENDER_NRP_OT_cancel_persisted_job,
    )
    if bpy is not None
    else ()
)


def register() -> None:
    if bpy is not None:
        for cls in CLASSES:
            bpy.utils.register_class(cls)
        bpy.app.timers.register(_startup_reconcile, first_interval=1.0)


def unregister() -> None:
    if bpy is not None:
        if bpy.app.timers.is_registered(_startup_reconcile):
            bpy.app.timers.unregister(_startup_reconcile)
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)
