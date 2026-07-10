from __future__ import annotations

import pytest

from blender_nrp.core.lighting import intensity_to_stops, kelvin_to_rgb, stops_to_intensity


def test_stops_round_trip():
    assert stops_to_intensity(intensity_to_stops(8.0)) == pytest.approx(8.0)


def test_kelvin_rgb_is_warm_to_cool_and_valid():
    warm, cool = kelvin_to_rgb(2700), kelvin_to_rgb(6500)
    assert warm[0] > warm[2]
    assert cool[2] >= warm[2]
    with pytest.raises(ValueError):
        kelvin_to_rgb(500)
