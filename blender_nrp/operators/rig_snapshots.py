"""Blender persistence and application operators for named NRP rig snapshots."""

from __future__ import annotations

import json

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from ..core.light_objects import apply_light_to_object, collect_rig_lights, light_from_object
    from ..core.lights import LightRig
    from ..core.rig_snapshots import (
        RigSnapshot,
        replace_snapshot,
        snapshots_from_json,
        snapshots_to_json,
    )
    from ._helpers import cancel_with_status, finish_with_status

    _KEY = "blender_nrp_rig_snapshots_json"

    def _load(scene) -> list[RigSnapshot]:
        raw = str(scene.get(_KEY, "[]"))
        return snapshots_from_json(json.loads(raw))

    def _save(scene, snapshots: list[RigSnapshot]) -> None:
        scene[_KEY] = json.dumps(snapshots_to_json(snapshots), sort_keys=True)

    def _apply(context, name: str) -> str:
        snapshots = _load(context.scene)
        snapshot = next((item for item in snapshots if item.name == name), None)
        if snapshot is None:
            raise ValueError(f"No snapshot named {name!r}")
        objects = [obj for obj in context.scene.objects if light_from_object(obj) is not None]
        if len(objects) != len(snapshot.rig.lights):
            raise ValueError("Snapshot light count differs from the current rig")
        for obj, light in zip(objects, snapshot.rig.lights, strict=True):
            apply_light_to_object(obj, light)
        return snapshot.name

    class BLENDER_NRP_OT_save_rig_snapshot(bpy.types.Operator):
        bl_idname = "blender_nrp.save_rig_snapshot"
        bl_label = "Save Rig Snapshot"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            try:
                snapshot = RigSnapshot(
                    settings.snapshot_name,
                    LightRig(
                        tuple(collect_rig_lights(context.scene.objects)),
                        scene_id=settings.scene_id,
                        camera_id=settings.camera.name if settings.camera else None,
                    ),
                )
                _save(context.scene, replace_snapshot(_load(context.scene), snapshot))
            except Exception as exc:
                return cancel_with_status(self, context, f"Could not save rig snapshot: {exc}")
            settings.snapshot_a = snapshot.name
            return finish_with_status(self, context, f"Saved rig snapshot '{snapshot.name}'")

    class BLENDER_NRP_OT_apply_rig_snapshot(bpy.types.Operator):
        bl_idname = "blender_nrp.apply_rig_snapshot"
        bl_label = "Apply Rig Snapshot"
        slot: bpy.props.EnumProperty(items=(("a", "A", "A"), ("b", "B", "B")))

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            name = settings.snapshot_a if self.slot == "a" else settings.snapshot_b
            if not name:
                return cancel_with_status(
                    self, context, f"Snapshot {self.slot.upper()} is not selected"
                )
            try:
                applied = _apply(context, name)
            except Exception as exc:
                return cancel_with_status(self, context, f"Could not apply snapshot: {exc}")
            bpy.ops.blender_nrp.relight_preview()
            return finish_with_status(self, context, f"Applied rig snapshot '{applied}'")


CLASSES = (
    (BLENDER_NRP_OT_save_rig_snapshot, BLENDER_NRP_OT_apply_rig_snapshot) if bpy is not None else ()
)


def register() -> None:
    if bpy is not None:
        for cls in CLASSES:
            bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is not None:
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)
