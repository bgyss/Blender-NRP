from blender_nrp.backends.interface import BakeSettings
from blender_nrp.backends.stock_blender_hemi import bake
from blender_nrp.core.gather import write_relight_preview
from blender_nrp.core.lights import LightRig, SphereLight
from blender_nrp.core.proxy import train_basic_proxy
from blender_nrp.core.validation import validate_cache_bundle


def test_fixture_workflow_writes_cache_proxy_preview_and_lights(tmp_path):
    result = bake(
        None,
        BakeSettings(
            scene_id="fixture_room_001",
            output_dir=tmp_path,
            width=8,
            height=6,
            segment_count=4,
            max_segment_distance=10.0,
            camera_id="Camera",
            seed=3,
        ),
    )
    report = validate_cache_bundle(result.cache_path, result.metadata_path)
    assert report.ok
    assert report.segment_count == 8 * 6 * 4
    assert (result.output_dir / "preview_albedo.png").exists()

    model = train_basic_proxy(
        result.cache_path,
        result.output_dir / "model.pt",
        result.output_dir / "train_report.json",
    )
    assert model.exists()

    rig = LightRig(
        (SphereLight(position=(0.0, 2.0, 2.0), radius=0.25, color=(1.0, 0.9, 0.7), intensity=4.0),),
        scene_id="fixture_room_001",
        camera_id="Camera",
    )
    rig.save(result.output_dir / "solved_lights.json")
    write_relight_preview(result.cache_path, rig, result.output_dir / "relight_preview.png")
    assert (result.output_dir / "relight_preview.png").exists()
    assert LightRig.load(result.output_dir / "solved_lights.json") == rig
