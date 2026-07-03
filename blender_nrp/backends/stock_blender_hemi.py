"""Stock Blender hemisphere backend placeholder."""

from __future__ import annotations

from pathlib import Path

from .interface import BakeSettings

id = "stock_blender_hemi"
label = "Stock Blender Hemisphere"


def bake(context: object, settings: BakeSettings) -> Path:
    """Bake a cache with Blender ray casts.

    The real implementation must run inside Blender because it depends on `bpy` and
    the evaluated dependency graph. This placeholder makes the backend explicit while
    preventing silent fake cache generation.
    """
    _ = context
    _ = settings
    raise NotImplementedError("stock_blender_hemi baking is not implemented yet")

