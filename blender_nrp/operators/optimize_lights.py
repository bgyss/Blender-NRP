"""Inverse light optimization operator: solve the rig against a target image.

Uses the loaded torch proxy as the differentiable forward model when available;
otherwise falls back to coordinate descent over the numpy gather. Solved parameters
are written back onto the Blender light objects (positions, radii/sizes, colors,
intensities), and both `solved_lights.json` and `solve_report.json` land next to
the cache with before/after loss.
"""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    import numpy as np

    from .. import proxy_runtime
    from ..core.light_objects import apply_light_to_object, light_from_object
    from ..core.lights import LightRig, light_from_dict
    from ..core.path_cache import load_arrays
    from ..core.reports import write_json_report
    from ..core.torch_proxy import torch_status
    from ._helpers import cancel_with_status, finish_with_status

    def _load_target(path: Path, height: int, width: int) -> np.ndarray:
        if path.suffix == ".npy":
            target = np.load(path)
            if target.shape != (height, width, 3):
                raise ValueError(f"target shape {target.shape} != cache {(height, width, 3)}")
            return np.asarray(target, dtype=np.float64)
        image = bpy.data.images.load(str(path))
        try:
            if image.size[0] != width or image.size[1] != height:
                raise ValueError(
                    f"target resolution {image.size[0]}x{image.size[1]} != cache {width}x{height}"
                )
            pixels = np.array(image.pixels[:], dtype=np.float64).reshape(height, width, 4)
        finally:
            bpy.data.images.remove(image)
        return pixels[::-1, :, :3]  # Blender stores bottom row first

    class BLENDER_NRP_OT_optimize_lights(bpy.types.Operator):
        bl_idname = "blender_nrp.optimize_lights"
        bl_label = "Optimize Lights From Target"
        bl_description = (
            "Solve NRP light parameters against the target image and write the "
            "result back onto the light objects"
        )

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.cache_path:
                return cancel_with_status(context, "No cache path selected")
            if not settings.target_image_path:
                return cancel_with_status(context, "No target image selected")

            pairs = [
                (obj, light)
                for obj in context.scene.objects
                if (light := light_from_object(obj)) is not None
            ]
            if not pairs:
                return cancel_with_status(context, "No NRP lights to optimize")
            objects = [obj for obj, _ in pairs]
            lights = tuple(light for _, light in pairs)

            cache_path = Path(bpy.path.abspath(settings.cache_path))
            try:
                arrays = load_arrays(cache_path).arrays
            except Exception as exc:
                return cancel_with_status(context, f"Cache load failed: {exc}")
            height, width, _ = arrays["albedo"].shape
            try:
                target = _load_target(
                    Path(bpy.path.abspath(settings.target_image_path)), height, width
                )
            except Exception as exc:
                return cancel_with_status(context, f"Target load failed: {exc}")

            torch_ok, _detail = torch_status()
            use_proxy = (
                torch_ok
                and proxy_runtime.model is not None
                and {light.light_type for light in lights} == {proxy_runtime.model_light_type}
            )
            try:
                if use_proxy:
                    from ..core.torch_proxy.optimize import optimize_lights

                    report = optimize_lights(
                        proxy_runtime.model,
                        arrays,
                        lights,
                        target,
                        steps=settings.optimize_steps,
                    )
                else:
                    from ..core.optimize_fallback import optimize_lights_fallback

                    report = optimize_lights_fallback(arrays, lights, target)
                    if not torch_ok:
                        report["limitations"].append(
                            "Install torch and train/load a proxy for the "
                            "gradient-based solver."
                        )
            except Exception as exc:
                return cancel_with_status(context, f"Light optimization failed: {exc}")

            solved = [light_from_dict(entry) for entry in report["optimized_lights"]]
            for obj, light in zip(objects, solved, strict=False):
                apply_light_to_object(obj, light)

            output_dir = cache_path.parent
            rig = LightRig(tuple(solved), scene_id=settings.scene_id)
            rig.save(output_dir / "solved_lights.json")
            write_json_report(output_dir / "solve_report.json", report)
            return finish_with_status(
                context,
                f"Solved {len(solved)} lights via {report['solver']}: gather MSE "
                f"{report['gather_mse_vs_target_initial']:.4g} -> "
                f"{report['gather_mse_vs_target_final']:.4g}",
            )


CLASSES = (BLENDER_NRP_OT_optimize_lights,) if bpy is not None else ()


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
