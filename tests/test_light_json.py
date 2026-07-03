from blender_nrp.core.lights import LightRig, SphereLight


def test_light_rig_round_trip(tmp_path):
    rig = LightRig(
        (
            SphereLight(
                position=(0.0, 1.0, 2.0),
                radius=0.25,
                color=(1.0, 0.85, 0.65),
                intensity=4.0,
            ),
        ),
        scene_id="fixture_room_001",
        camera_id="Camera",
    )
    path = tmp_path / "lights.json"
    rig.save(path)
    assert LightRig.load(path) == rig

