"""Non-destructive reference matching with explicit review/apply/discard."""

from __future__ import annotations

import json

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    import numpy as np

    from .. import proxy_runtime
    from ..core.gather import gather_hdr
    from ..core.images import write_png_rgb
    from ..core.light_objects import apply_light_to_object, light_from_object
    from ..core.lights import LightRig, light_from_dict
    from ..core.path_cache import load_arrays
    from ..core.torch_proxy import torch_status
    from ._helpers import cancel_with_status, finish_with_status
    from .optimize_lights import _load_target

    MATCH_WIPE_IMAGE = "NRP Match Reference Wipe"

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
                torch_ok, _detail = torch_status()
                use_proxy = (
                    torch_ok
                    and proxy_runtime.model is not None
                    and {light.light_type for light in lights}
                    == {proxy_runtime.model_light_type}
                )
                if use_proxy:
                    from ..core.torch_proxy.optimize import optimize_lights

                    report = optimize_lights(
                        proxy_runtime.model,
                        arrays,
                        lights,
                        target,
                        steps=settings.optimize_steps,
                        locks=locks,
                    )
                else:
                    from ..core.optimize_fallback import optimize_lights_fallback

                    report = optimize_lights_fallback(arrays, lights, target, locks=locks)
                solved = tuple(light_from_dict(item) for item in report["optimized_lights"])
                pending, before_path, after_path, wipe_path = _paths(cache_path)
                original_rig = LightRig(lights, scene_id=settings.scene_id)
                solved_rig = LightRig(solved, scene_id=settings.scene_id)
                pending.write_text(
                    json.dumps(
                        {
                            "original": original_rig.to_dict(),
                            "solved": solved_rig.to_dict(),
                            "report": report,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                before = np.clip(gather_hdr(arrays, lights), 0.0, 1.0)
                after = np.clip(gather_hdr(arrays, solved), 0.0, 1.0)
                write_png_rgb(before_path, before)
                write_png_rgb(after_path, after)
                split = before.shape[1] // 2
                wipe = before.copy()
                wipe[:, split:] = after[:, split:]
                write_png_rgb(wipe_path, wipe)
                _show_wipe(context, wipe_path)
                settings.match_pending_path = str(pending)
                return finish_with_status(
                    self,
                    context,
                    "Match ready — review 'NRP Match Reference Wipe', then Apply or Discard",
                )
            except Exception as exc:
                return cancel_with_status(self, context, f"Match Reference failed: {exc}")

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
