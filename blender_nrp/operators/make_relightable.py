"""The V3 default workflow: submit, monitor, validate, train, and preview."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from ..core.execution import LocalSubprocessBackend
    from ..core.jobs import BakeJob, TrainJob
    from ..core.path_cache import validate_npz
    from ..core.staleness import stable_hash
    from ._helpers import cancel_with_status

    _state: dict = {"backend": None, "job_id": None, "stage": None, "scene": None}

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
            }
        )

    def _scene_hash(scene) -> str:
        return stable_hash([(obj.name, obj.type, tuple(obj.matrix_world)) for obj in scene.objects])

    def _finish_chain(scene, message: str) -> None:
        scene.blender_nrp.status = message
        _state.update({"backend": None, "job_id": None, "stage": None, "scene": None})

    def _poll() -> float | None:
        scene = _state.get("scene")
        backend = _state.get("backend")
        job_id = _state.get("job_id")
        if scene is None or backend is None or job_id is None:
            return None
        progress = backend.status(job_id)
        scene.blender_nrp.status = (
            f"{progress.stage}: {progress.fraction * 100:.0f}% {progress.message}"
        ).strip()
        if progress.state in {"queued", "running"}:
            return 0.25
        if progress.state != "succeeded":
            _finish_chain(
                scene, f"Make Scene Relightable failed during {_state['stage']}: {progress.message}"
            )
            return None
        artifacts = backend.fetch(job_id)
        settings = scene.blender_nrp
        if _state["stage"] == "bake":
            cache = artifacts["path_cache"]
            report = validate_npz(cache)
            if not report.ok:
                _finish_chain(
                    scene, f"Bake completed but validation failed: {'; '.join(report.errors)}"
                )
                return None
            settings.cache_path = str(cache)
            train = TrainJob(
                str(cache), str(cache.parent), settings.train_iterations, settings.train_device
            )
            _state["job_id"] = backend.submit(train)
            _state["stage"] = "train"
            settings.status = "Validated cache — training proxy…"
            return 0.25
        settings.model_path = str(artifacts["model"])
        bpy.ops.blender_nrp.load_proxy()
        if not any(obj.get("blender_nrp_light", False) for obj in scene.objects):
            bpy.ops.blender_nrp.create_sphere_light()
        bpy.ops.blender_nrp.relight_preview()
        settings.pipeline_settings_hash = _settings_hash(settings)
        settings.pipeline_scene_hash = _scene_hash(scene)
        _finish_chain(
            scene, "Scene relightable — validated cache, trained proxy, and preview ready"
        )
        return None

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
            if settings.compute != "local_subprocess":
                return cancel_with_status(self, context, "Only This Machine is available in v0.3")
            if not bpy.data.filepath:
                return cancel_with_status(
                    self, context, "Save the .blend before submitting a worker job"
                )
            if not settings.scene_id:
                settings.scene_id = Path(bpy.data.filepath).stem
            output_dir = Path(bpy.path.abspath(settings.output_dir))
            backend = LocalSubprocessBackend(
                output_dir / ".nrp_jobs", blender_binary=bpy.app.binary_path
            )
            job = BakeJob(
                settings.scene_id,
                bpy.data.filepath,
                str(output_dir),
                settings.camera.name if settings.camera else "Camera",
                settings.resolution_x,
                settings.resolution_y,
                settings.paths_per_pixel,
                settings.max_bounces,
                settings.backend,
                packed=settings.packed_cache,
                torch_device=settings.train_device,
            )
            _state.update(
                {
                    "backend": backend,
                    "job_id": backend.submit(job),
                    "stage": "bake",
                    "scene": context.scene,
                }
            )
            settings.status = "Baking… submitted to This Machine"
            bpy.app.timers.register(_poll, first_interval=0.1)
            return {"FINISHED"}

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
    (BLENDER_NRP_OT_make_relightable, BLENDER_NRP_OT_cancel_make_relightable)
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
