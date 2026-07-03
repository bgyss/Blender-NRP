"""Multi-bounce tracer + analytic-room capture tests (cycles_capture core)."""

from __future__ import annotations

import json

import numpy as np

from blender_nrp.backends import cycles_capture
from blender_nrp.backends.interface import BakeSettings
from blender_nrp.core.path_cache import load_arrays, validate_npz
from blender_nrp.core.path_tracer import AnalyticRoomCaster, cosine_sample_hemisphere


def test_cosine_sample_hemisphere_stays_in_hemisphere():
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(200, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    dirs = cosine_sample_hemisphere(normals, rng)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    cosines = np.einsum("ij,ij->i", dirs, normals)
    assert np.all(cosines >= 0.0)
    # Cosine-weighted sampling has E[cos] = 2/3.
    assert abs(cosines.mean() - 2.0 / 3.0) < 0.05


def test_analytic_room_caster_geometry():
    caster = AnalyticRoomCaster()
    origins = np.array(
        [
            [0.0, 0.0, 1.5],  # toward the -X wall
            [-1.5, 0.0, 0.8],  # toward the sphere at (0.5, 0, 0.8)
            [0.0, 0.0, 1.5],  # toward the open +Y wall -> escape
        ]
    )
    dirs = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hit, t, position, normal, albedo = caster.cast(origins, dirs)
    assert hit.tolist() == [True, True, False]
    assert abs(t[0] - 2.0) < 1e-9
    np.testing.assert_allclose(normal[0], [1.0, 0.0, 0.0])  # interior-facing
    assert abs(t[1] - 1.4) < 1e-9  # sphere front face at x = 0.5 - 0.6 = -0.1
    np.testing.assert_allclose(albedo[1], caster.sphere_albedo)


def _bake(tmp_path, **overrides) -> tuple:
    settings = BakeSettings(
        scene_id="s",
        output_dir=tmp_path,
        width=12,
        height=10,
        segment_count=1,
        max_segment_distance=100.0,
        paths_per_pixel=8,
        max_bounces=3,
        **overrides,
    )
    result = cycles_capture.bake(None, settings)
    report = json.loads(result.bake_report_path.read_text())
    return result, report


def test_analytic_capture_writes_escape_segments_and_valid_cache(tmp_path):
    result, report = _bake(tmp_path)
    assert report["ok"]
    assert report["backend"] == "cycles_capture"
    assert report["escape_segments"] > 0
    assert report["throughput_normalization"] == "n_paths"
    assert report["approximation_limits"]
    validation = validate_npz(result.cache_path)
    assert validation.ok, validation.errors
    arrays = load_arrays(result.cache_path).arrays
    assert np.any(np.isinf(arrays["seg_tmax"]))
    # Camera-first segments carry unit throughput.
    assert np.isclose(arrays["seg_throughput"].max(), 1.0)
    # Bounced throughput never exceeds 1 for albedo <= 1 materials.
    assert np.all(arrays["seg_throughput"] <= 1.0 + 1e-9)


def test_analytic_capture_packed_layout(tmp_path):
    result, report = _bake(tmp_path, packed=True)
    assert report["cache_layout"] == "packed"
    validation = validate_npz(result.cache_path)
    assert validation.ok and validation.packed


def test_bake_steps_reports_monotonic_progress(tmp_path):
    settings = BakeSettings(
        scene_id="s",
        output_dir=tmp_path,
        width=8,
        height=8,
        segment_count=1,
        max_segment_distance=100.0,
        paths_per_pixel=4,
        max_bounces=2,
    )
    fractions = []
    result = None
    for fraction, _message, maybe in cycles_capture.bake_steps(None, settings):
        fractions.append(fraction)
        if maybe is not None:
            result = maybe
    assert result is not None
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
