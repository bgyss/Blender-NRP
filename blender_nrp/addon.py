"""Registration orchestration for Blender-NRP."""

from __future__ import annotations

from . import panels, preview, properties
from .operators import (
    bake_cache,
    export_lights,
    import_lights,
    load_proxy,
    optimize_lights,
    relight_preview,
    train_proxy,
    validate_cache,
)

MODULES = (
    properties,
    bake_cache,
    validate_cache,
    train_proxy,
    load_proxy,
    relight_preview,
    optimize_lights,
    import_lights,
    export_lights,
    preview,
    panels,
)


def register() -> None:
    for module in MODULES:
        module.register()


def unregister() -> None:
    for module in reversed(MODULES):
        module.unregister()
