"""Non-destructive reference matching with explicit review/apply/discard."""

from __future__ import annotations

import json
import time
import uuid

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    import numpy as np

    from ..core.execution import (
        ExecutionQueue,
        LocalSubprocessBackend,
        QueuedJob,
        RunPodExecutionBackend,
        SshExecutionBackend,
    )
    from ..core.gather import gather_hdr
    from ..core.images import write_png_rgb
    from ..core.jobs import SolveJob
    from ..core.light_objects import apply_light_to_object, light_from_object
    from ..core.lights import LightRig
    from ..core.path_cache import load_arrays
    from ._helpers import cancel_with_status, finish_with_status
    from .optimize_lights import _load_target

    MATCH_WIPE_IMAGE = "NRP Match Reference Wipe"
    _match_state: dict = {}

    def _match_backend(context, queue_dir: Path):
        settings = context.scene.blender_nrp
        if settings.compute == "runpod":
            prefs = context.preferences.addons["blender_nrp"].preferences
            return RunPodExecutionBackend(
                queue_dir,
                api_key=prefs.runpod_api_key,
                image_name=prefs.runpod_image,
                worker_root=prefs.runpod_worker_root,
                gpu_type=prefs.runpod_gpu_type,
            )
        if settings.compute == "ssh":
            prefs = context.preferences.addons["blender_nrp"].preferences
            return SshExecutionBackend(
                queue_dir,
                host=prefs.ssh_host,
                remote_root=prefs.ssh_remote_root,
                worker_root=prefs.ssh_worker_root,
                blender_binary=prefs.ssh_blender_binary,
                python_binary=prefs.ssh_python_binary,
            )
        return LocalSubprocessBackend(
            queue_dir,
            python_binary=getattr(bpy.app, "binary_path_python", None),
        )

    def _match_job_path(backend, queue: ExecutionQueue, job_id: str) -> str:
        if backend.id == "runpod":
            return str(queue.directory / "runpod" / job_id / "job.json")
        if backend.id == "ssh":
            return str(queue.directory / job_id / "job.json")
        return str((queue.directory / job_id).with_suffix(".json"))

    def _match_finish(context, message: str, *, error: bool = False, remove_queue: bool = True):
        settings = context.scene.blender_nrp
        settings.status = message
        if error:
            settings.last_error_details = message
        queue = _match_state.get("queue")
        job_id = _match_state.get("job_id")
        if remove_queue and queue is not None and job_id:
            queue.remove(job_id)
        _match_state.clear()
        try:
            bpy.ops.blender_nrp.pipeline_notice(message=message, is_error=error)
        except RuntimeError:
            pass

    def _match_poll() -> float | None:
        context = _match_state.get("context")
        backend = _match_state.get("backend")
        job_id = _match_state.get("job_id")
        if context is None or backend is None or job_id is None:
            return None
        settings = context.scene.blender_nrp
        try:
            progress = backend.status(job_id)
        except Exception as exc:
            _match_finish(
                context,
                f"Match Reference status failed: {exc}",
                error=True,
                remove_queue=False,
            )
            return None
        settings.status = (
            f"{progress.stage}: {progress.fraction * 100:.0f}% {progress.message}"
        ).strip()
        if progress.state in {"queued", "running"}:
            return 0.25
        if progress.state != "succeeded":
            _match_finish(
                context,
                f"Match Reference failed: {progress.message}",
                error=True,
                remove_queue=False,
            )
            return None
        try:
            artifacts = backend.fetch(job_id)
            solved_path = artifacts["solved_lights"]
            solved = LightRig.from_dict(json.loads(solved_path.read_text(encoding="utf-8")))
            original = _match_state["original"]
            arrays = _match_state["arrays"]
            pending, before_path, after_path, wipe_path = _paths(_match_state["cache_path"])
            pending.write_text(
                json.dumps(
                    {
                        "original": original.to_dict(),
                        "solved": solved.to_dict(),
                        "report": json.loads(
                            Path(artifacts["solve_report"]).read_text(encoding="utf-8")
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            before = np.clip(gather_hdr(arrays, original.lights), 0.0, 1.0)
            after = np.clip(gather_hdr(arrays, solved.lights), 0.0, 1.0)
            write_png_rgb(before_path, before)
            write_png_rgb(after_path, after)
            split = before.shape[1] // 2
            wipe = before.copy()
            wipe[:, split:] = after[:, split:]
            write_png_rgb(wipe_path, wipe)
            _show_wipe(context, wipe_path)
            settings.match_pending_path = str(pending)
            settings.last_report_path = str(artifacts["solve_report"])
        except Exception as exc:
            _match_finish(
                context,
                f"Could not fetch Match Reference artifacts: {exc}",
                error=True,
                remove_queue=False,
            )
            return None
        _match_finish(context, "Match ready — review the wipe, then Apply or Discard")
        return None

    def _paths(cache_path: Path) -> tuple[Path, Path, Path, Path]:
        return (
            cache_path.parent / "match_reference_pending.json",
            cache_path.parent / "match_reference_before.png",
            cache_path.parent / "match_reference_after.png",
            cache_path.parent / "match_reference_wipe.png",
        )

    def _show_wipe(context: bpy.types.Context, wipe_path: Path):
        existing = bpy.data.images.get(MATCH_WIPE_IMAGE)
        if existing is not None:
            bpy.data.images.remove(existing)
        image = bpy.data.images.load(str(wipe_path), check_existing=False)
        image.name = MATCH_WIPE_IMAGE
        if context.screen is not None:
            for area in context.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.spaces.active.image = image
        return image

    class BLENDER_NRP_OT_match_reference(bpy.types.Operator):
        bl_idname = "blender_nrp.match_reference"
        bl_label = "Match Reference"
        bl_description = (
            "Solve a reference match, review a before/after result, then apply or discard"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if _match_state.get("job_id") is not None:
                return cancel_with_status(self, context, "A Match Reference job is already running")
            if not settings.cache_path or not settings.target_image_path:
                return cancel_with_status(self, context, "Select a cache and reference image first")
            pairs = [
                (obj, light)
                for obj in context.scene.objects
                if (light := light_from_object(obj)) is not None
            ]
            if not pairs:
                return cancel_with_status(self, context, "No NRP lights to match")
            cache_path = Path(bpy.path.abspath(settings.cache_path))
            try:
                arrays = load_arrays(cache_path).arrays
                height, width, _ = arrays["albedo"].shape
                target = _load_target(
                    Path(bpy.path.abspath(settings.target_image_path)), height, width
                )
                objects = [obj for obj, _light in pairs]
                lights = tuple(light for _obj, light in pairs)
                locks = tuple(
                    {
                        field
                        for field in (
                            "position", "color", "intensity", "radius",
                            "width", "height", "normal",
                        )
                        if bool(obj.get(f"nrp_lock_{field}", False))
                    }
                    for obj in objects
                )
                original_rig = LightRig(lights, scene_id=settings.scene_id)
                work_dir = cache_path.parent / ".nrp_match" / uuid.uuid4().hex
                work_dir.mkdir(parents=True, exist_ok=True)
                lights_path = work_dir / "lights.json"
                target_path = work_dir / "target.npy"
                original_rig.save(lights_path)
                np.save(target_path, target)
                model_path = (
                    Path(bpy.path.abspath(settings.model_path))
                    if settings.model_path
                    else None
                )
                job = SolveJob(
                    str(cache_path),
                    str(lights_path),
                    str(target_path),
                    str(work_dir / "artifacts"),
                    steps=settings.optimize_steps,
                    torch_device=settings.train_device,
                    model_path=str(model_path) if model_path and model_path.exists() else None,
                    locks=locks,
                )
                queue = ExecutionQueue(cache_path.parent / ".nrp_jobs")
                backend = _match_backend(context, queue.directory)
                job_id = backend.submit(job)
                queue.add(
                    QueuedJob(
                        job_id,
                        backend.id,
                        time.time(),
                        _match_job_path(backend, queue, job_id),
                    )
                )
                _match_state.update(
                    {
                        "context": context,
                        "backend": backend,
                        "job_id": job_id,
                        "queue": queue,
                        "cache_path": cache_path,
                        "arrays": arrays,
                        "original": original_rig,
                    }
                )
                settings.status = "Match Reference… submitted"
                bpy.app.timers.register(_match_poll, first_interval=0.1)
                return finish_with_status(self, context, "Match Reference submitted — solving…")
            except Exception as exc:
                return cancel_with_status(self, context, f"Match Reference failed: {exc}")

    class BLENDER_NRP_OT_cancel_match_reference(bpy.types.Operator):
        bl_idname = "blender_nrp.cancel_match_reference"
        bl_label = "Cancel Match Reference"

        def execute(self, context: bpy.types.Context) -> set[str]:
            if not _match_state.get("backend"):
                return cancel_with_status(self, context, "No Match Reference job is running")
            try:
                _match_state["backend"].cancel(_match_state["job_id"])
            except Exception as exc:
                return cancel_with_status(self, context, f"Could not cancel Match Reference: {exc}")
            _match_finish(context, "Match Reference cancelled")
            return {"FINISHED"}

    class BLENDER_NRP_OT_review_match_reference(bpy.types.Operator):
        bl_idname = "blender_nrp.review_match_reference"
        bl_label = "Review Wipe"
        bl_description = "Load the pending before/after center wipe into Blender's Image Editor"

        def execute(self, context: bpy.types.Context) -> set[str]:
            pending_path = context.scene.blender_nrp.match_pending_path
            if not pending_path:
                return cancel_with_status(self, context, "No pending reference match")
            wipe_path = Path(bpy.path.abspath(pending_path)).with_name(
                "match_reference_wipe.png"
            )
            if not wipe_path.exists():
                return cancel_with_status(self, context, "Match review wipe is missing")
            _show_wipe(context, wipe_path)
            return finish_with_status(
                self, context, f"Review '{MATCH_WIPE_IMAGE}' in an Image Editor"
            )

    class BLENDER_NRP_OT_apply_match_reference(bpy.types.Operator):
        bl_idname = "blender_nrp.apply_match_reference"
        bl_label = "Apply Match"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.match_pending_path:
                return cancel_with_status(self, context, "No pending reference match")
            pending = Path(bpy.path.abspath(settings.match_pending_path))
            try:
                solved = LightRig.from_dict(json.loads(pending.read_text())["solved"])
                objects = [
                    obj for obj in context.scene.objects if light_from_object(obj) is not None
                ]
                if len(objects) != len(solved.lights):
                    raise ValueError("pending match light count differs from current rig")
                for obj, light in zip(objects, solved.lights, strict=True):
                    apply_light_to_object(obj, light)
                pending.unlink(missing_ok=True)
            except Exception as exc:
                return cancel_with_status(self, context, f"Could not apply match: {exc}")
            settings.match_pending_path = ""
            bpy.ops.blender_nrp.relight_preview()
            return finish_with_status(self, context, "Reference match applied")

    class BLENDER_NRP_OT_discard_match_reference(bpy.types.Operator):
        bl_idname = "blender_nrp.discard_match_reference"
        bl_label = "Discard Match"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if settings.match_pending_path:
                Path(bpy.path.abspath(settings.match_pending_path)).unlink(missing_ok=True)
            settings.match_pending_path = ""
            return finish_with_status(self, context, "Reference match discarded")


CLASSES = (
    (
        BLENDER_NRP_OT_match_reference,
        BLENDER_NRP_OT_cancel_match_reference,
        BLENDER_NRP_OT_review_match_reference,
        BLENDER_NRP_OT_apply_match_reference,
        BLENDER_NRP_OT_discard_match_reference,
    )
    if bpy is not None
    else ()
)


def register() -> None:
    if bpy is not None:
        for cls in CLASSES:
            bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is not None:
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)
