from blender_nrp.core.metadata import NRPMetadata


def test_metadata_round_trip(tmp_path):
    metadata = NRPMetadata(
        scene_id="fixture_room_001",
        camera_id="Camera",
        resolution=(64, 32),
        light_type="sphere",
        aux_features=["albedo", "normal", "depth"],
        bbox_min=[-1.0, -1.0, 0.0],
        bbox_max=[1.0, 1.0, 2.0],
        model_width=64,
        model_depth=4,
    )
    path = tmp_path / "metadata.json"
    metadata.save(path)
    assert NRPMetadata.load(path) == metadata

