from __future__ import annotations

import pytest

from blender_nrp.core.lights import LightRig, SphereLight
from blender_nrp.core.rig_snapshots import (
    RigSnapshot,
    replace_snapshot,
    snapshots_from_json,
    snapshots_to_json,
)


def test_named_snapshots_round_trip_and_replace():
    rig = LightRig((SphereLight((0, 1, 2), 0.2, (1, 1, 1), 4.0),), scene_id="s")
    first = RigSnapshot("Warm key", rig)
    assert snapshots_from_json(snapshots_to_json([first])) == [first]
    changed_rig = LightRig((SphereLight((0, 1, 2), 0.2, (1, 1, 1), 8.0),), scene_id="s")
    changed = RigSnapshot("Warm key", changed_rig)
    assert replace_snapshot([first], changed) == [changed]
    with pytest.raises(ValueError, match="name"):
        RigSnapshot("", rig)
