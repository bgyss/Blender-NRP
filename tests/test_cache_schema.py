import numpy as np

from blender_nrp.core.path_cache import validate_arrays


def test_cache_schema_validation_accepts_minimal_valid_cache():
    arrays = {
        "n_paths": np.array([1], dtype=np.int64),
        "seg_pixel": np.array([0], dtype=np.int64),
        "seg_origin": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "seg_dir": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        "seg_tmax": np.array([1.0], dtype=np.float32),
        "seg_throughput": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        "albedo": np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32),
        "normal": np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32),
        "depth": np.array([[1.0]], dtype=np.float32),
        "position": np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32),
    }
    report = validate_arrays(arrays)
    assert report.ok
    assert report.width == 1
    assert report.height == 1
    assert report.segment_count == 1

