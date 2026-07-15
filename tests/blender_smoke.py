"""Blender background-mode smoke test for the Blender-NRP add-on (V2 chain).

Run from the repository root with:

    blender --background --factory-startup --python-exit-code 7 --python tests/blender_smoke.py

Creates a tiny scene, registers the add-on, and executes the whole operator chain:
stock + cycles_capture bakes (the latter with escape segments and a Cycles A/B PSNR
in bake_report.json), cache validation, proxy training (real torch when Blender's
Python has it, a clearly reported degradation otherwise), sphere + quad light
creation, relight preview into the Image datablock, light JSON export/import with
coordinate conversion, and inverse light optimization writing back onto objects.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

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
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
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
    settings.paths_per_pixel = 8
    settings.max_bounces = 3
    settings.light_json_path = str(output_dir / SCENE_ID / "exported_lights.json")
    return output_dir / SCENE_ID


def torch_available_in_blender() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def run() -> None:
    blender_nrp.register()
    try:
        camera = build_scene()
        artifact_dir = configure_addon(camera)
        settings = bpy.context.scene.blender_nrp

        # --- Stock backend still works (V1 fallback path).
        settings.backend = "stock_blender_hemi"
        assert_finished(bpy.ops.blender_nrp.bake_cache(), "Bake (stock)")
        if "validated" not in settings.status.lower():
            raise AssertionError(f"stock bake did not auto-validate: {settings.status}")
        report = validate_cache_bundle(Path(settings.cache_path), artifact_dir / "metadata.json")
        if not report.ok:
            raise AssertionError(f"stock cache validation failed: {report.errors}")
        if not 0 < report.segment_count <= 16 * 16 * 4 or report.segment_count % 4:
            raise AssertionError(f"unexpected stock segment count: {report.segment_count}")

        # --- Cycles capture backend: multi-bounce, escape segments, A/B PSNR.
        settings.backend = "cycles_capture"
        assert_finished(bpy.ops.blender_nrp.bake_cache(), "Bake (cycles_capture)")
        cache_path = Path(settings.cache_path)
        report = validate_cache_bundle(cache_path, artifact_dir / "metadata.json")
        if not report.ok:
            raise AssertionError(f"cycles cache validation failed: {report.errors}")
        if "validated" not in settings.status.lower():
            raise AssertionError(f"cycles bake did not auto-validate: {settings.status}")
        bake_report = json.loads((artifact_dir / "bake_report.json").read_text())
        if bake_report["backend"] != "cycles_capture":
            raise AssertionError("bake_report backend mismatch")
        if bake_report["escape_segments"] <= 0:
            raise AssertionError("cycles_capture bake recorded no escape segments")
        if "reference_check" in bake_report:
            psnr = bake_report["reference_check"]["psnr_db"]
            print(f"cycles A/B reference PSNR: {psnr:.2f} dB")
            if psnr < 5.0:
                raise AssertionError(f"reference PSNR implausibly low: {psnr}")
        else:
            print("WARNING: reference_check missing from bake_report:", bake_report["warnings"])
        assert_finished(bpy.ops.blender_nrp.validate_cache(), "Validate Cache")

        # --- Proxy training: real torch or a clearly reported degradation.
        has_torch = torch_available_in_blender()
        settings.train_iterations = 40
        settings.train_device = "cpu"
        train_result = bpy.ops.blender_nrp.train_proxy()
        if has_torch:
            assert_finished(train_result, "Train Proxy")
            assert Path(settings.model_path).exists(), settings.model_path
            train_report = json.loads((artifact_dir / "train_report.json").read_text())
            if train_report["training_backend"] != "torch":
                raise AssertionError("expected a torch training run")
            # Training now auto-loads the proxy into the shared runtime.
            from blender_nrp import proxy_runtime

            if proxy_runtime.model is None:
                raise AssertionError("proxy was not auto-loaded after training")
            if "auto-loaded" not in settings.status:
                raise AssertionError(f"train status omitted auto-load: {settings.status}")
            # The explicit Load Proxy operator still works (idempotent re-load).
            assert_finished(bpy.ops.blender_nrp.load_proxy(), "Load Proxy")
        else:
            if train_result != {"CANCELLED"} or "PyTorch" not in settings.status:
                raise AssertionError(
                    f"expected a clear missing-torch report, got {train_result}: "
                    f"{settings.status}"
                )
            print("torch not available in Blender Python; degradation path verified")

        # --- Sphere + quad lights and the relight preview image.
        assert_finished(bpy.ops.blender_nrp.create_sphere_light(), "Create NRP Sphere Light")
        sphere = bpy.context.object
        # Data-API creation (works from any editor) must produce a marked mesh object.
        if sphere is None or sphere.type != "MESH" or sphere.get("nrp_light_type") != "sphere":
            raise AssertionError(f"sphere light not created correctly: {sphere}")
        sphere.location = (0.0, -1.5, 2.0)
        sphere["nrp_radius"] = 0.35
        sphere["nrp_color"] = (1.0, 0.85, 0.65)
        sphere["nrp_intensity"] = 6.0

        assert_finished(bpy.ops.blender_nrp.create_quad_light(), "Create NRP Quad Light")
        quad = bpy.context.object
        quad.location = (1.0, -1.0, 1.8)
        quad["nrp_width"] = 1.2
        quad["nrp_height"] = 0.8

        assert_finished(bpy.ops.blender_nrp.relight_preview(), "Preview Relight")
        assert (artifact_dir / "relight_preview.png").exists()
        preview_image = bpy.data.images.get("NRP Relight Preview")
        if preview_image is None or preview_image.size[0] != 16:
            raise AssertionError("preview Image datablock missing or wrong size")
        if "gather" not in settings.status and "proxy" not in settings.status:
            raise AssertionError(f"preview status does not label its source: {settings.status}")

        # --- Export (converted to y-up) / import (converted back) round trip.
        bpy.ops.object.select_all(action="DESELECT")
        sphere.select_set(True)
        quad.select_set(True)
        bpy.context.view_layer.objects.active = sphere
        settings.export_coordinate_system = "right_handed_y_up"
        assert_finished(bpy.ops.blender_nrp.export_lights(), "Export Lights")
        rig = LightRig.load(Path(settings.light_json_path))
        if rig.coordinate_system != "right_handed_y_up":
            raise AssertionError("export did not convert to right_handed_y_up")
        if sorted(light.light_type for light in rig.lights) != ["quad", "sphere"]:
            raise AssertionError("expected one sphere + one quad in the export")
        exported_sphere = next(li for li in rig.lights if li.light_type == "sphere")
        if not np.allclose(exported_sphere.position, (0.0, 2.0, 1.5)):
            raise AssertionError(f"y-up conversion wrong: {exported_sphere.position}")

        assert_finished(bpy.ops.blender_nrp.import_lights(), "Import Lights")
        back_sphere = next(
            o
            for o in bpy.context.scene.objects
            if o.get("nrp_light_type") == "sphere" and o.name != sphere.name
        )
        if not np.allclose(tuple(back_sphere.location), (0.0, -1.5, 2.0), atol=1e-6):
            raise AssertionError(
                f"import did not convert back to Blender coords: {tuple(back_sphere.location)}"
            )

        # --- Inverse optimization against a target derived from a known rig.
        from blender_nrp.core.gather import gather_hdr
        from blender_nrp.core.light_objects import collect_rig_lights
        from blender_nrp.core.path_cache import load_arrays

        arrays = load_arrays(cache_path).arrays
        target = gather_hdr(arrays, tuple(collect_rig_lights([sphere])))
        target_path = artifact_dir / "target.npy"
        np.save(target_path, target)
        settings.target_image_path = str(target_path)
        settings.optimize_steps = 60
        prior_intensity = float(sphere["nrp_intensity"])
        assert_finished(bpy.ops.blender_nrp.optimize_lights(), "Optimize Lights")
        assert (artifact_dir / "solved_lights.json").exists()
        solve_report = json.loads((artifact_dir / "solve_report.json").read_text())
        if solve_report["solver"] not in ("torch_proxy_adam", "numpy_coordinate_descent"):
            raise AssertionError(f"unexpected solver: {solve_report['solver']}")
        if "gather_mse_vs_target_final" not in solve_report:
            raise AssertionError("solve_report missing gather-space loss")
        if float(sphere["nrp_intensity"]) == prior_intensity and solve_report[
            "gather_mse_vs_target_final"
        ] == solve_report["gather_mse_vs_target_initial"]:
            print("NOTE: solver made no change (already at optimum?)")

        # V3 Match Reference uses the same durable SolveJob rails and remains
        # asynchronous while its before/after review assets are produced.
        from blender_nrp.operators import match_reference

        settings.match_pending_path = ""
        assert_finished(bpy.ops.blender_nrp.match_reference(), "Match Reference submit")
        if bpy.app.timers.is_registered(match_reference._match_poll):
            bpy.app.timers.unregister(match_reference._match_poll)
        deadline = time.monotonic() + 120.0
        while match_reference._match_state.get("job_id") is not None:
            match_reference._match_poll()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Match Reference timed out: {settings.status}")
            time.sleep(0.1)
        if not settings.match_pending_path:
            raise AssertionError(
                f"Match Reference did not produce a pending review: {settings.status}"
            )
        assert_finished(
            bpy.ops.blender_nrp.discard_match_reference(), "Discard Match Reference"
        )

        # --- V3 one-button local-subprocess chain.
        # This is intentionally last: it saves the scene, then launches a second
        # background Blender for baking and training through the public operator.
        if has_torch:
            from blender_nrp.operators import make_relightable

            for obj in list(bpy.context.scene.objects):
                if obj.get("nrp_light_type"):
                    bpy.data.objects.remove(obj, do_unlink=True)
            one_button_root = ROOT / "build" / "blender_one_button_smoke"
            one_button_root.mkdir(parents=True, exist_ok=True)
            saved_scene = one_button_root / "one_button_scene.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(saved_scene))
            settings.scene_id = "one_button_scene"
            settings.output_dir = str(one_button_root)
            settings.compute = "local_subprocess"
            settings.show_advanced = True
            settings.quality_preset = "draft"
            settings.resolution_x = 8
            settings.resolution_y = 8
            settings.paths_per_pixel = 2
            settings.max_bounces = 2
            settings.train_iterations = 2
            settings.train_device = "cpu"
            settings.tracer_engine = "torch_mesh"
            settings.cache_path = ""
            settings.model_path = ""
            assert_finished(
                bpy.ops.blender_nrp.make_relightable(), "V3 Make Scene Relightable"
            )
            if bpy.app.timers.is_registered(make_relightable._poll):
                bpy.app.timers.unregister(make_relightable._poll)
            deadline = time.monotonic() + 180.0
            while make_relightable._state["job_id"] is not None:
                make_relightable._poll()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"one-button pipeline timed out: {settings.status}")
                time.sleep(0.1)
            if not settings.status.startswith("Scene relightable"):
                raise AssertionError(f"one-button chain failed: {settings.status}")
            if not Path(settings.cache_path).exists() or not Path(settings.model_path).exists():
                raise AssertionError("one-button chain omitted cache or model artifact")
            one_button_artifacts = one_button_root / settings.scene_id
            for name in (
                "path_cache.npz",
                "metadata.json",
                "bake_report.json",
                "model.pt",
                "train_report.json",
                "relight_preview.png",
            ):
                if not (one_button_artifacts / name).exists():
                    raise AssertionError(f"one-button chain omitted {name}")
            if not settings.pipeline_settings_hash or not settings.pipeline_scene_hash:
                raise AssertionError("one-button chain omitted persisted staleness hashes")
            if not any(obj.get("nrp_light_type") for obj in bpy.context.scene.objects):
                raise AssertionError("one-button chain did not create a starter light")
            from blender_nrp import proxy_runtime

            if proxy_runtime.model is None:
                raise AssertionError("one-button chain did not auto-load the proxy")
            if bpy.data.images.get("NRP Relight Preview") is None:
                raise AssertionError("one-button chain did not open the preview")
        else:
            print("torch not available in Blender Python; V3 one-button success path skipped")

        print("BLENDER_NRP_SMOKE_OK")
    finally:
        blender_nrp.unregister()


if __name__ == "__main__":
    run()
