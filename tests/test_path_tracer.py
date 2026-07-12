"""Multi-bounce tracer + analytic-room capture tests (cycles_capture core)."""

from __future__ import annotations

import json

import numpy as np
import pytest

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


def test_torch_analytic_tracer_matches_python_first_hit_and_gather_scale():
    pytest.importorskip("torch")
    from blender_nrp.core.gather import gather_hdr
    from blender_nrp.core.lights import SphereLight
    from blender_nrp.core.path_tracer import trace_camera_paths
    from blender_nrp.core.torch_path_tracer import trace_analytic_room_paths

    origin = np.array([0.0, 3.5, 1.5])
    corners = {
        "top_left": np.array([-0.7, -1.0, 0.7]),
        "top_right": np.array([0.7, -1.0, 0.7]),
        "bottom_left": np.array([-0.7, -1.0, -0.7]),
        "bottom_right": np.array([0.7, -1.0, -0.7]),
    }
    python_result = None
    for _progress, result in trace_camera_paths(
        AnalyticRoomCaster(), origin, corners, 8, 6, paths_per_pixel=4, max_bounces=3, seed=7
    ):
        if result is not None:
            python_result = result
    assert python_result is not None
    torch_result = trace_analytic_room_paths(
        origin, corners, 8, 6, paths_per_pixel=4, max_bounces=3, seed=7
    )
    np.testing.assert_allclose(torch_result.position, python_result.position, atol=1e-9)
    np.testing.assert_allclose(torch_result.normal, python_result.normal, atol=1e-9)
    np.testing.assert_allclose(torch_result.albedo, python_result.albedo, atol=1e-9)
    assert np.any(np.isinf(torch_result.seg_tmax))
    light = SphereLight((0.0, 0.0, 1.4), 0.35, (1.0, 0.9, 0.8), 5.0)
    py_mean = gather_hdr(python_result.as_arrays(), (light,)).mean()
    torch_mean = gather_hdr(torch_result.as_arrays(), (light,)).mean()
    assert torch_mean == pytest.approx(py_mean, rel=0.45, abs=1e-8)


def test_torch_triangle_caster_hits_device_mesh():
    torch = pytest.importorskip("torch")
    from blender_nrp.core.torch_path_tracer import TorchTriangleCaster

    caster = TorchTriangleCaster(
        vertices=np.array([[-1, -1, 2], [1, -1, 2], [0, 1, 2]], dtype=np.float64),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        normals=np.array([[0, 0, -1]], dtype=np.float64),
        albedos=np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        torch=torch,
        device="cpu",
    )
    hit, t, position, normal, albedo = caster.cast(
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
    )
    assert hit.tolist() == [True]
    assert float(t[0]) == pytest.approx(2.0)
    assert position[0].tolist() == pytest.approx([0.0, 0.0, 2.0])
    assert normal[0].tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert albedo[0].tolist() == pytest.approx([0.4, 0.5, 0.6])
    many_vertices = []
    many_triangles = []
    for index in range(9):
        offset = len(many_vertices)
        x = float(index * 4)
        many_vertices.extend([[-1 + x, -1, 2], [1 + x, -1, 2], [x, 1, 2]])
        many_triangles.append([offset, offset + 1, offset + 2])
    many = TorchTriangleCaster(
        many_vertices,
        many_triangles,
        np.tile([[0, 0, -1]], (9, 1)),
        np.tile([[0.4, 0.5, 0.6]], (9, 1)),
        torch=torch,
        device="cpu",
    )
    assert many.bvh_node_count > 1
    many_hit, many_t, *_ = many.cast(
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
    )
    assert many_hit.tolist() == [True]
    assert float(many_t[0]) == pytest.approx(2.0)
    from blender_nrp.core.torch_path_tracer import trace_mesh_paths

    traced = trace_mesh_paths(
        caster,
        np.array([0.0, 0.0, 0.0]),
        {
            "top_left": np.array([-0.5, 0.5, 1.0]),
            "top_right": np.array([0.5, 0.5, 1.0]),
            "bottom_left": np.array([-0.5, -0.5, 1.0]),
            "bottom_right": np.array([0.5, -0.5, 1.0]),
        },
        2,
        2,
        paths_per_pixel=1,
        max_bounces=1,
        seed=2,
    )
    assert traced.seg_pixel.size > 0
