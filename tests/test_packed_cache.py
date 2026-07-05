"""Packed-layout (fp16 + rgb9e5) cache read/write, mirroring nrp's format."""

from __future__ import annotations

import numpy as np

from blender_nrp.core.path_cache import load_arrays, save_arrays, validate_npz
from blender_nrp.core.rgb9e5 import MAX_RGB9E5, rgb9e5_decode, rgb9e5_encode


def _demo_arrays(height: int = 2, width: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    pixels = height * width
    segments = pixels * 4
    dirs = rng.normal(size=(segments, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tmax = rng.random(segments) * 10.0 + 0.1
    tmax[::5] = np.inf  # escape segments must survive packing
    return {
        "n_paths": np.full(pixels, 4, dtype=np.int64),
        "seg_pixel": np.repeat(np.arange(pixels, dtype=np.int64), 4),
        "seg_origin": rng.normal(size=(segments, 3)),
        "seg_dir": dirs,
        "seg_tmax": tmax,
        "seg_throughput": rng.random((segments, 3)) * 3.0,
        "albedo": rng.random((height, width, 3)),
        "normal": np.tile([0.0, 0.0, 1.0], (height, width, 1)),
        "depth": rng.random((height, width)) + 0.5,
        "position": rng.normal(size=(height, width, 3)),
    }


def test_rgb9e5_round_trip_relative_error():
    rng = np.random.default_rng(0)
    values = rng.random((1000, 3)) * 100.0
    decoded = rgb9e5_decode(rgb9e5_encode(values))
    dominant = values.max(axis=1)
    err = np.abs(decoded - values).max(axis=1)
    # ~9 mantissa bits of relative precision on the dominant channel.
    assert np.all(err <= dominant * 2.0**-8 + 1e-12)


def test_rgb9e5_edge_cases():
    special = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 2.0], [-5.0, 1.0, 1.0], [1e9, 1.0, 1.0]])
    decoded = rgb9e5_decode(rgb9e5_encode(special))
    assert np.all(decoded[0] == 0.0)
    assert decoded[1, 0] == 0.0
    assert decoded[2, 0] == 0.0
    assert decoded[3, 0] <= MAX_RGB9E5


def test_packed_round_trip_preserves_semantics(tmp_path):
    arrays = _demo_arrays()
    packed_path = tmp_path / "packed.npz"
    plain_path = tmp_path / "plain.npz"
    save_arrays(packed_path, arrays, width=3, height=2, packed=True)
    save_arrays(plain_path, arrays, width=3, height=2)

    packed = load_arrays(packed_path)
    plain = load_arrays(plain_path)
    assert packed.packed and not plain.packed
    assert packed.schema_version == 2

    # Escape segments stay infinite, finite t_max stays positive and close.
    inf_mask = np.isinf(arrays["seg_tmax"])
    assert np.all(np.isinf(packed.arrays["seg_tmax"][inf_mask]))
    assert np.all(packed.arrays["seg_tmax"] > 0.0)
    # Directions renormalized to unit length.
    norms = np.linalg.norm(packed.arrays["seg_dir"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    # Throughput within rgb9e5 precision; geometry within fp16 precision.
    dominant = arrays["seg_throughput"].max(axis=1)
    err = np.abs(packed.arrays["seg_throughput"] - arrays["seg_throughput"]).max(axis=1)
    assert np.all(err <= dominant * 2.0**-8 + 1e-9)
    np.testing.assert_allclose(packed.arrays["seg_origin"], arrays["seg_origin"], atol=2e-2)


def test_packed_cache_is_smaller(tmp_path):
    arrays = _demo_arrays(height=16, width=16)
    packed_path = tmp_path / "packed.npz"
    plain_path = tmp_path / "plain.npz"
    save_arrays(packed_path, arrays, width=16, height=16, packed=True)
    save_arrays(plain_path, arrays, width=16, height=16)
    assert packed_path.stat().st_size < plain_path.stat().st_size


def test_validate_npz_accepts_packed_and_surfaces_medium(tmp_path):
    arrays = _demo_arrays()
    path = tmp_path / "packed.npz"
    save_arrays(
        path, arrays, width=3, height=2, packed=True, medium={"sigma_t": 0.5, "albedo": 0.9}
    )
    report = validate_npz(path)
    assert report.ok, report.errors
    assert report.packed
    assert report.schema_version == 2
    assert report.medium == {"sigma_t": 0.5, "albedo": 0.9}
    assert any("medium" in warning for warning in report.warnings)


def test_validate_npz_rejects_missing_arrays(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, packed_layout=1, n_paths=np.ones(1, dtype=np.int64))
    report = validate_npz(path)
    assert not report.ok
    assert "packed-layout cache missing arrays" in report.errors[0]
