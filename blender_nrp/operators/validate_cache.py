"""Validate cache operator."""

from __future__ import annotations

from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

from ..core.path_cache import validate_npz

if bpy is not None:
    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_validate_cache(bpy.types.Operator):
        bl_idname = "blender_nrp.validate_cache"
        bl_label = "Validate Cache"
        bl_description = "Validate the selected NRP path-cache npz"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.cache_path:
                return cancel_with_status(context, "No cache path selected")
            cache_path = Path(bpy.path.abspath(settings.cache_path))
            try:
                report = validate_npz(cache_path)
            except Exception as exc:
                return cancel_with_status(context, f"Cache validation failed: {exc}")
            if not report.ok:
                return cancel_with_status(context, "; ".join(report.errors))
            layout = "packed" if report.packed else "default"
            message = (
                f"Cache OK: {report.width}x{report.height}, {report.segment_count} "
                f"segments, schema v{report.schema_version}, {layout} layout"
            )
            if report.medium is not None:
                message += f", medium sigma_t={report.medium['sigma_t']}"
            return finish_with_status(context, message)


CLASSES = (BLENDER_NRP_OT_validate_cache,) if bpy is not None else ()


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

