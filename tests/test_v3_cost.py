from __future__ import annotations

import pytest

from blender_nrp.core.cost import estimate_bake_seconds, estimate_cost_usd


def test_cost_estimate_scales_with_work_and_rate():
    baseline = estimate_bake_seconds(512, 512, 256, 4)
    assert baseline == pytest.approx(180.0)
    assert estimate_bake_seconds(1024, 512, 256, 4) == pytest.approx(360.0)
    assert estimate_cost_usd(512, 512, 256, 4, 0.69) == pytest.approx(0.0345)
    with pytest.raises(ValueError):
        estimate_cost_usd(1, 1, 1, 1, -1)
