"""Registration orchestration for Blender-NRP."""

from __future__ import annotations

from . import panels, preferences, preview, properties
from .operators import (
    bake_cache,
    export_lights,
    import_lights,
    load_proxy,
    make_relightable,
    match_reference,
    optimize_lights,
    relight_preview,
    rig_snapshots,
    train_proxy,
    validate_cache,
)
from .ui import light_panel

MODULES = (
    preferences,
    properties,
    bake_cache,
    validate_cache,
    train_proxy,
    load_proxy,
    make_relightable,
    match_reference,
    relight_preview,
    rig_snapshots,
    optimize_lights,
    import_lights,
    export_lights,
    preview,
    light_panel,
    panels,
)


def register() -> None:
    for module in MODULES:
        module.register()


def unregister() -> None:
    for module in reversed(MODULES):
        module.unregister()
