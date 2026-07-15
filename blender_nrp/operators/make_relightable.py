"""The V3 default workflow: submit, monitor, validate, train, and preview."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    import time
    from pathlib import Path

    from ..core.execution import (
        ExecutionQueue,
        LocalSubprocessBackend,
        QueuedJob,
        RunPodExecutionBackend,
        SshExecutionBackend,
    )
    from ..core.jobs import BakeJob, TrainJob
    from ..core.pipeline import resolve_preset
    from ..core.staleness import stable_hash
    from ..core.validation import validate_cache_bundle
    from ._helpers import cancel_with_status

    _state: dict = {
        "backend": None,
        "job_id": None,
        "stage": None,
        "scene": None,
        "queue": None,
        "window_manager": None,
    }

    def _settings_hash(settings) -> str:
        return stable_hash(
            {
                "width": settings.resolution_x,
                "height": settings.resolution_y,
                "paths": settings.paths_per_pixel,
                "bounces": settings.max_bounces,
                "backend": settings.backend,
                "packed": settings.packed_cache,
                "preset": settings.quality_preset,
                "tracer_engine": settings.tracer_engine,
                "camera": settings.camera.name if settings.camera else None,
                "scene_id": settings.scene_id,
                "max_segment_distance": settings.max_segment_distance,
            }
        )

    def _scene_hash(scene) -> str:
        content = []
        for obj in scene.objects:
            record = {
                "name": obj.name,
                "type": obj.type,
                "matrix": [[float(value) for value in row] for row in obj.matrix_world],
            }
            if obj.type == "MESH" and obj.data is not None:
                record["vertices"] = [
                    [float(value) for value in vertex.co] for vertex in obj.data.vertices
                ]
                record["polygons"] = [list(polygon.vertices) for polygon in obj.data.polygons]
                record["materials"] = [
                    {
                        "name": slot.material.name,
                        "diffuse": [float(value) for value in slot.material.diffuse_color],
                    }
                    for slot in obj.material_slots
                    if slot.material is not None
                ]
            content.append(record)
        return stable_hash(content)

    def _queue_job_path(backend, job_id: str) -> str:
        if isinstance(backend, RunPodExecutionBackend):
            return str(backend.queue_dir / "runpod" / job_id / "job.json")
        if isinstance(backend, SshExecutionBackend):
            return str(backend.queue_dir / job_id / "job.json")
        return str((backend.queue_dir / job_id).with_suffix(".json"))

    def _finish_chain(
        scene, message: str, *, is_error: bool = False, remove_queue: bool = True
    ) -> None:
        scene.blender_nrp.status = message
        window_manager = _state.get("window_manager")
        if window_manager is not None:
            window_manager.progress_end()
        if (
            remove_queue
            and _state.get("queue") is not None
            and _state.get("job_id") is not None
        ):
            _state["queue"].remove(_state["job_id"])
        try:
            bpy.ops.blender_nrp.pipeline_notice(message=message, is_error=is_error)
        except RuntimeError:
            pass
        _state.update(
            {
                "backend": None,
                "job_id": None,
                "stage": None,
                "scene": None,
                "window_manager": None,
            }
        )

    def _poll() -> float | None:
        scene = _state.get("scene")
        backend = _state.get("backend")
        job_id = _state.get("job_id")
        if scene is None or backend is None or job_id is None:
            return None
        settings = scene.blender_nrp
        try:
            progress = backend.status(job_id)
        except Exception as exc:
            settings.last_error_details = str(exc)
            _finish_chain(
                scene,
                f"Make Scene Relightable failed during {_state['stage']}: {exc}",
                is_error=True,
                remove_queue=False,
            )
            return None
        for name, path in progress.artifacts.items():
            if name.endswith("_report") or name == "bake_report":
                settings.last_report_path = str(path)
        if progress.error:
            settings.last_error_details = progress.error
        cost_suffix = ""
        if progress.accrued_cost is not None:
            hourly = progress.cost_per_hour or 0.0
            cost_suffix = f" — ${progress.accrued_cost:.3f} accrued (${hourly:.2f}/h)"
        scene.blender_nrp.status = (
            f"{progress.stage}: {progress.fraction * 100:.0f}% {progress.message}{cost_suffix}"
        ).strip()
        window_manager = _state.get("window_manager")
        if window_manager is not None:
            overall = (
                progress.fraction * 60.0
                if _state["stage"] == "bake"
                else 60.0 + progress.fraction * 35.0
            )
            window_manager.progress_update(overall)
        if progress.state in {"queued", "running"}:
            return 0.25
        if progress.state != "succeeded":
            _finish_chain(
                scene,
                f"Make Scene Relightable failed during {_state['stage']}: {progress.message}",
                is_error=True,
                remove_queue=False,
            )
            return None
        try:
            artifacts = backend.fetch(job_id)
        except Exception as exc:
            settings.last_error_details = str(exc)
            _finish_chain(
                scene,
                f"Could not fetch {_state['stage']} artifacts: {exc}",
                is_error=True,
                remove_queue=False,
            )
            return None
        if _state["stage"] == "bake":
            cache = artifacts["path_cache"]
            metadata = artifacts.get("metadata")
            if metadata is None:
                _finish_chain(scene, "Bake completed without metadata.json", is_error=True)
                return None
            report = validate_cache_bundle(cache, metadata)
            if not report.ok:
                _finish_chain(
                    scene,
                    f"Bake completed but validation failed: {'; '.join(report.errors)}",
                    is_error=True,
                )
                return None
            settings.cache_path = str(cache)
            if "bake_report" in artifacts:
                settings.last_report_path = str(artifacts["bake_report"])
            train = TrainJob(
                str(cache), str(cache.parent), settings.train_iterations, settings.train_device
            )
            if _state.get("queue") is not None:
                _state["queue"].remove(job_id)
            try:
                train_job_id = backend.submit(train)
            except Exception as exc:
                settings.last_error_details = str(exc)
                _finish_chain(scene, f"Could not submit training: {exc}", is_error=True)
                return None
            if _state.get("queue") is not None:
                _state["queue"].add(
                    QueuedJob(
                        train_job_id,
                        backend.id if hasattr(backend, "id") else "local",
                        time.time(),
                        _queue_job_path(backend, train_job_id),
                    )
                )
            _state["job_id"] = train_job_id
            _state["stage"] = "train"
            settings.status = "Validated cache — training proxy…"
            return 0.25
        settings.model_path = str(artifacts["model"])
        if "train_report" in artifacts:
            settings.last_report_path = str(artifacts["train_report"])
        if bpy.ops.blender_nrp.load_proxy() != {"FINISHED"}:
            _finish_chain(scene, f"Proxy load failed: {settings.status}", is_error=True)
            return None
        if not any(obj.get("nrp_light_type") for obj in scene.objects):
            if bpy.ops.blender_nrp.create_sphere_light() != {"FINISHED"}:
                _finish_chain(scene, f"Starter light failed: {settings.status}", is_error=True)
                return None
        if bpy.ops.blender_nrp.relight_preview() != {"FINISHED"}:
            _finish_chain(scene, f"Preview failed: {settings.status}", is_error=True)
            return None
        settings.pipeline_settings_hash = _settings_hash(settings)
        settings.pipeline_scene_hash = _scene_hash(scene)
        _finish_chain(
            scene, "Scene relightable — validated cache, trained proxy, and preview ready"
        )
        return None

    class BLENDER_NRP_OT_pipeline_notice(bpy.types.Operator):
        bl_idname = "blender_nrp.pipeline_notice"
        bl_label = "Blender-NRP Pipeline Notice"
        bl_options = {"INTERNAL"}

        message: bpy.props.StringProperty()
        is_error: bpy.props.BoolProperty(default=False)

        def execute(self, _context: bpy.types.Context) -> set[str]:
            self.report({"ERROR" if self.is_error else "INFO"}, self.message)
            return {"FINISHED"}

    class BLENDER_NRP_OT_make_relightable(bpy.types.Operator):
        bl_idname = "blender_nrp.make_relightable"
        bl_label = "Make Scene Relightable"
        bl_description = (
            "Bake, validate, train, load a proxy, add a starter light, and open preview"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if _state["job_id"] is not None:
                return cancel_with_status(
                    self, context, "A relightable-scene job is already running"
                )
            if not bpy.data.filepath:
                return cancel_with_status(
                    self, context, "Save the .blend before submitting a worker job"
                )
            settings.last_report_path = ""
            settings.last_error_details = ""
            settings.show_details = False
            if not settings.scene_id:
                settings.scene_id = Path(bpy.data.filepath).stem
            camera = settings.camera or context.scene.camera
            if camera is None:
                return cancel_with_status(self, context, "Set an active scene camera first")
            settings.camera = camera
            if not settings.show_advanced:
                width, height, budget = resolve_preset(
                    settings.quality_preset,
                    context.scene.render.resolution_x,
                    context.scene.render.resolution_y,
                )
                settings.resolution_x = width
                settings.resolution_y = height
                settings.paths_per_pixel = budget.paths_per_pixel
                settings.max_bounces = budget.max_bounces
                settings.train_iterations = budget.train_iterations
            output_dir = Path(bpy.path.abspath(settings.output_dir))
            if settings.compute == "runpod":
                prefs = context.preferences.addons["blender_nrp"].preferences
                try:
                    backend = RunPodExecutionBackend(
                        output_dir / ".nrp_jobs",
                        api_key=prefs.runpod_api_key,
                        image_name=prefs.runpod_image,
                        worker_root=prefs.runpod_worker_root,
                        gpu_type=prefs.runpod_gpu_type,
                    )
                except ValueError as exc:
                    return cancel_with_status(
                        self, context, f"Configure RunPod in add-on preferences: {exc}"
                    )
            elif settings.compute == "ssh":
                prefs = context.preferences.addons["blender_nrp"].preferences
                try:
                    backend = SshExecutionBackend(
                        output_dir / ".nrp_jobs",
                        host=prefs.ssh_host,
                        remote_root=prefs.ssh_remote_root,
                        worker_root=prefs.ssh_worker_root,
                        blender_binary=prefs.ssh_blender_binary,
                        python_binary=prefs.ssh_python_binary,
                    )
                except ValueError as exc:
                    return cancel_with_status(
                        self, context, f"Configure SSH node in add-on preferences: {exc}"
                    )
            else:
                backend = LocalSubprocessBackend(
                    output_dir / ".nrp_jobs", blender_binary=bpy.app.binary_path
                )
            if settings.compute != "local_subprocess":
                # Remote workers receive the saved .blend over rsync. Pack
                # external images/libraries first so a remote success cannot
                # silently render a scene with missing local-only assets.
                packed_result = bpy.ops.file.pack_all()
                if packed_result != {"FINISHED"}:
                    return cancel_with_status(
                        self,
                        context,
                        "Could not pack external scene assets for remote compute",
                    )
                if bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath) != {"FINISHED"}:
                    return cancel_with_status(
                        self, context, "Could not save the packed scene for remote compute"
                    )
            queue = ExecutionQueue(output_dir / ".nrp_jobs")
            job = BakeJob(
                settings.scene_id,
                bpy.data.filepath,
                str(output_dir),
                camera.name,
                settings.resolution_x,
                settings.resolution_y,
                settings.paths_per_pixel,
                settings.max_bounces,
                settings.backend,
                packed=settings.packed_cache,
                torch_device=settings.train_device,
                tracer_engine=settings.tracer_engine,
            )
            stale = (
                bool(settings.cache_path)
                and Path(bpy.path.abspath(settings.cache_path)).exists()
                and bool(settings.pipeline_settings_hash)
                and (
                    settings.pipeline_settings_hash != _settings_hash(settings)
                    or settings.pipeline_scene_hash != _scene_hash(context.scene)
                )
            )
            if stale and settings.use_existing_cache:
                cache = Path(bpy.path.abspath(settings.cache_path))
                job = TrainJob(
                    str(cache), str(cache.parent), settings.train_iterations, settings.train_device
                )
                stage = "train"
                settings.status = "Cache is stale — using existing cache by request"
            else:
                stage = "bake"
                if stale:
                    settings.status = "Cache is stale — re-baking because Use Existing Cache is off"
            try:
                job_id = backend.submit(job)
            except Exception as exc:
                settings.last_error_details = str(exc)
                return cancel_with_status(self, context, f"Could not submit {stage}: {exc}")
            queue.add(
                QueuedJob(
                    job_id,
                    backend.id if hasattr(backend, "id") else "local",
                    time.time(),
                    _queue_job_path(backend, job_id),
                )
            )
            _state.update(
                {
                    "backend": backend,
                    "job_id": job_id,
                    "stage": stage,
                    "scene": context.scene,
                    "queue": queue,
                    "window_manager": context.window_manager,
                }
            )
            compute_label = {
                "local_subprocess": "This Machine",
                "ssh": "SSH / LAN Node",
                "runpod": "RunPod Cloud",
            }.get(settings.compute, settings.compute)
            settings.status = f"{stage.title()}… submitted to {compute_label}"
            context.window_manager.progress_begin(0.0, 100.0)
            bpy.app.timers.register(_poll, first_interval=0.1)
            return {"FINISHED"}

        def invoke(self, context: bpy.types.Context, _event: bpy.types.Event) -> set[str]:
            result = self.execute(context)
            if result == {"FINISHED"} and _state.get("job_id") is not None:
                context.window_manager.modal_handler_add(self)
                return {"RUNNING_MODAL"}
            return result

        def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
            if event.type == "ESC" and event.value == "PRESS":
                if _state.get("backend") is not None and _state.get("job_id") is not None:
                    try:
                        _state["backend"].cancel(_state["job_id"])
                    except Exception as exc:
                        _finish_chain(
                            context.scene,
                            f"Could not cancel Make Scene Relightable: {exc}",
                            is_error=True,
                            remove_queue=False,
                        )
                        return {"CANCELLED"}
                    _finish_chain(context.scene, "Make Scene Relightable cancelled")
                return {"CANCELLED"}
            if _state.get("job_id") is None:
                return {"FINISHED"}
            return {"PASS_THROUGH"}

    class BLENDER_NRP_OT_cancel_make_relightable(bpy.types.Operator):
        bl_idname = "blender_nrp.cancel_make_relightable"
        bl_label = "Cancel Relightable Job"

        def execute(self, context: bpy.types.Context) -> set[str]:
            if _state.get("backend") is None or _state.get("job_id") is None:
                return cancel_with_status(self, context, "No relightable-scene job is running")
            _state["backend"].cancel(_state["job_id"])
            _finish_chain(context.scene, "Make Scene Relightable cancelled")
            return {"FINISHED"}


CLASSES = (
    (
        BLENDER_NRP_OT_pipeline_notice,
        BLENDER_NRP_OT_make_relightable,
        BLENDER_NRP_OT_cancel_make_relightable,
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
