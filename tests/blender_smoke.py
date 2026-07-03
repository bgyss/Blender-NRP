"""Blender background-mode smoke test for the Blender-NRP add-on.

Run from the repository root with:

    blender --background --factory-startup --python tests/blender_smoke.py

The test creates a tiny scene, registers the add-on, executes the UI operators,
and validates the generated artifacts. It is intentionally script-based so it can
run in CI or on a local workstation without opening Blender's UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import blender_nrp
from blender_nrp.core.lights import LightRig
from blender_nrp.core.validation import validate_cache_bundle

SCENE_ID = "blender_smoke_001"


def assert_finished(result: set[str], label: str) -> None:
    if result != {"FINISHED"}:
        status = getattr(bpy.context.scene.blender_nrp, "status", "")
        raise AssertionError(f"{label} failed with {result}: {status}")


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_scene() -> bpy.types.Object:
    reset_scene()
    floor_mat = make_material("NRP Smoke Floor", (0.8, 0.7, 0.55, 1.0))
    cube_mat = make_material("NRP Smoke Cube", (0.45, 0.65, 0.9, 1.0))

    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "NRP_Smoke_Floor"
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0.0, 0.0, 0.45))
    cube = bpy.context.object
    cube.name = "NRP_Smoke_Cube"
    cube.data.materials.append(cube_mat)

    bpy.ops.object.light_add(type="POINT", location=(1.5, -2.0, 3.0))
    light = bpy.context.object
    light.name = "NRP_Smoke_Blender_Light"
    light.data.energy = 250.0

    bpy.ops.object.camera_add(location=(0.0, -4.0, 2.2))
    camera = bpy.context.object
    look_at(camera, Vector((0.0, 0.0, 0.4)))
    bpy.context.scene.camera = camera
    return camera


def configure_addon(camera: bpy.types.Object) -> Path:
    settings = bpy.context.scene.blender_nrp
    output_dir = ROOT / "build" / "blender_smoke"
    settings.scene_id = SCENE_ID
    settings.output_dir = str(output_dir)
    settings.camera = camera
    settings.resolution_x = 16
    settings.resolution_y = 16
    settings.segment_count = 4
    settings.max_segment_distance = 10.0
    settings.light_json_path = str(output_dir / SCENE_ID / "exported_lights.json")
    return output_dir / SCENE_ID


def run() -> None:
    blender_nrp.register()
    try:
        camera = build_scene()
        artifact_dir = configure_addon(camera)
        settings = bpy.context.scene.blender_nrp

        assert_finished(bpy.ops.blender_nrp.bake_cache(), "Bake Path Cache")
        cache_path = Path(settings.cache_path)
        metadata_path = artifact_dir / "metadata.json"
        report = validate_cache_bundle(cache_path, metadata_path)
        if not report.ok:
            raise AssertionError(f"cache validation failed: {report.errors}")
        if report.segment_count != 16 * 16 * 4:
            raise AssertionError(f"unexpected segment count: {report.segment_count}")

        assert_finished(bpy.ops.blender_nrp.validate_cache(), "Validate Cache")
        assert_finished(bpy.ops.blender_nrp.train_proxy(), "Train Proxy")
        assert Path(settings.model_path).exists(), settings.model_path
        assert_finished(bpy.ops.blender_nrp.load_proxy(), "Load Proxy")

        assert_finished(bpy.ops.blender_nrp.create_sphere_light(), "Create NRP Sphere Light")
        light = bpy.context.object
        light.location = (0.0, -1.5, 2.0)
        light["nrp_radius"] = 0.35
        light["nrp_color"] = (1.0, 0.85, 0.65)
        light["nrp_intensity"] = 6.0
        assert_finished(bpy.ops.blender_nrp.relight_preview(), "Preview Relight")
        assert (artifact_dir / "relight_preview.png").exists()

        bpy.ops.object.select_all(action="DESELECT")
        light.select_set(True)
        bpy.context.view_layer.objects.active = light
        assert_finished(bpy.ops.blender_nrp.export_lights(), "Export Lights")
        exported = Path(settings.light_json_path)
        rig = LightRig.load(exported)
        if len(rig.lights) != 1:
            raise AssertionError(f"expected one exported light, got {len(rig.lights)}")

        assert_finished(bpy.ops.blender_nrp.import_lights(), "Import Lights")
        imported = [
            obj for obj in bpy.context.scene.objects if obj.get("nrp_light_type") == "sphere"
        ]
        if len(imported) < 2:
            raise AssertionError("expected imported NRP sphere light")

        assert_finished(bpy.ops.blender_nrp.optimize_lights(), "Optimize Lights")
        assert (artifact_dir / "solved_lights.json").exists()
        print("BLENDER_NRP_SMOKE_OK")
    finally:
        blender_nrp.unregister()


if __name__ == "__main__":
    run()
