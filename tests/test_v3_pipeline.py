from __future__ import annotations

import pytest

from blender_nrp.core.pipeline import resolve_preset


def test_presets_scale_render_resolution_and_increase_budgets():
    draft = resolve_preset("draft", 1920, 1080)
    standard = resolve_preset("standard", 1920, 1080)
    final = resolve_preset("final", 1920, 1080)
    assert draft[:2] == (480, 270)
    assert standard[:2] == (960, 540)
    assert final[:2] == (1920, 1080)
    assert draft[2].paths_per_pixel < standard[2].paths_per_pixel < final[2].paths_per_pixel
    with pytest.raises(ValueError):
        resolve_preset("unknown", 1, 1)
