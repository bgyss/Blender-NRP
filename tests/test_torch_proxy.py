"""Torch proxy tests (skipped without torch installed)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from blender_nrp.backends import cycles_capture  # noqa: E402
from blender_nrp.backends.interface import BakeSettings  # noqa: E402
from blender_nrp.core.gather import gather_hdr  # noqa: E402
from blender_nrp.core.lights import QuadLight, SphereLight  # noqa: E402
from blender_nrp.core.optimize_fallback import optimize_lights_fallback  # noqa: E402
from blender_nrp.core.path_cache import load_arrays  # noqa: E402
from blender_nrp.core.torch_proxy import select_device, torch_status  # noqa: E402
from blender_nrp.core.torch_proxy.gather import TorchPathCache  # noqa: E402
from blender_nrp.core.torch_proxy.model import TorchNRP  # noqa: E402
from blender_nrp.core.torch_proxy.optimize import optimize_lights  # noqa: E402
from blender_nrp.core.torch_proxy.relight import proxy_relight  # noqa: E402
from blender_nrp.core.torch_proxy.train import train_proxy  # noqa: E402


@pytest.fixture(scope="module")
def traced_arrays(tmp_path_factory):
    settings = BakeSettings(
        scene_id="s",
        output_dir=tmp_path_factory.mktemp("cache"),
        width=16,
        height=12,
        segment_count=1,
        max_segment_distance=100.0,
        paths_per_pixel=12,
        max_bounces=3,
    )
    result = cycles_capture.bake(None, settings)
    return load_arrays(result.cache_path).arrays


def test_torch_status_and_device():
    available, detail = torch_status()
    assert available
    assert "torch" in detail
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in ("cpu", "mps", "cuda")


def test_torch_gather_matches_numpy_reference(traced_arrays):
    cache = TorchPathCache(traced_arrays, "cpu")
    sphere = SphereLight(position=(0.2, -0.1, 1.2), radius=0.4, color=(1, 0.5, 2), intensity=1.3)
    quad = QuadLight(
        position=(0.0, 0.5, 1.0),
        normal=(0.3, -0.7, 0.6),
        width=1.2,
        height=0.8,
        color=(1, 1, 1),
        intensity=2.0,
    )
    for light in (sphere, quad):
        reference = gather_hdr(traced_arrays, (light,))
        result = cache.gather_light(light).cpu().numpy()
        assert np.abs(reference - result).max() < 1e-8


def test_train_save_load_resume_round_trip(traced_arrays, tmp_path):
    model_path = tmp_path / "model.pt"
    report = train_proxy(
        traced_arrays,
        model_path,
        iterations=60,
        batch_size=1024,
        pool_size=6,
        n_val_lights=2,
        device="cpu",
        checkpoint_every=30,
    )
    assert report["ok"]
    assert report["training_backend"] == "torch"
    assert report["device"] == "cpu"
    assert report["iterations_run"] == 60
    assert (tmp_path / "checkpoint.pt").exists()
    assert report["limitations"]

    model = TorchNRP.load(str(model_path))
    assert model.light_type == "sphere"
    image = proxy_relight(
        model,
        traced_arrays,
        (SphereLight(position=(0, 0, 1), radius=0.3, color=(1, 1, 1), intensity=1.0),),
    )
    assert image.shape == traced_arrays["albedo"].shape
    assert np.all(image >= 0.0)

    resumed = train_proxy(
        traced_arrays,
        model_path,
        iterations=80,
        batch_size=1024,
        pool_size=6,
        n_val_lights=2,
        device="cpu",
        checkpoint_every=30,
        resume=True,
    )
    assert resumed["iterations_run"] == 80


def test_train_quad_proxy(traced_arrays, tmp_path):
    report = train_proxy(
        traced_arrays,
        tmp_path / "model.pt",
        light_type="quad",
        iterations=30,
        batch_size=512,
        pool_size=4,
        n_val_lights=2,
        device="cpu",
        checkpoint_every=0,
    )
    assert report["ok"] and report["light_type"] == "quad"
    model = TorchNRP.load(str(tmp_path / "model.pt"))
    quad = QuadLight(
        position=(0, 0, 1), normal=(0, 0, 1), width=1.0, height=1.0,
        color=(1, 1, 1), intensity=1.0,
    )
    image = proxy_relight(model, traced_arrays, (quad,))
    assert image.shape == traced_arrays["albedo"].shape
    with pytest.raises(ValueError):
        proxy_relight(
            model,
            traced_arrays,
            (SphereLight(position=(0, 0, 1), radius=0.3, color=(1, 1, 1), intensity=1.0),),
        )


def test_inverse_optimization_improves_gather_mse(traced_arrays, tmp_path):
    true_light = SphereLight(
        position=(0.0, 0.0, 1.5), radius=0.5, color=(1.0, 0.8, 0.6), intensity=3.0
    )
    target = gather_hdr(traced_arrays, (true_light,))
    train_proxy(
        traced_arrays,
        tmp_path / "model.pt",
        iterations=600,
        batch_size=2048,
        pool_size=16,
        n_val_lights=2,
        device="cpu",
        checkpoint_every=0,
        seed=1,
    )
    model = TorchNRP.load(str(tmp_path / "model.pt"))
    init = SphereLight(position=(-1.0, 0.5, 0.5), radius=0.2, color=(1, 1, 1), intensity=1.0)
    report = optimize_lights(model, traced_arrays, (init,), target, steps=300, device="cpu")
    assert report["ok"]
    assert report["gather_mse_vs_target_final"] < report["gather_mse_vs_target_initial"]
    assert len(report["optimized_lights"]) == 1
    assert report["limitations"]


def test_fallback_optimizer_improves_without_torch_api(traced_arrays):
    true_light = SphereLight(
        position=(0.0, 0.0, 1.5), radius=0.5, color=(1.0, 0.8, 0.6), intensity=3.0
    )
    target = gather_hdr(traced_arrays, (true_light,))
    init = SphereLight(position=(-1.0, 0.5, 0.5), radius=0.2, color=(1, 1, 1), intensity=1.0)
    report = optimize_lights_fallback(traced_arrays, (init,), target, sweeps=3)
    assert report["solver"] == "numpy_coordinate_descent"
    assert report["gather_mse_vs_target_final"] < report["gather_mse_vs_target_initial"]


def test_fallback_optimizer_honors_match_reference_locks(traced_arrays):
    target = gather_hdr(
        traced_arrays,
        (SphereLight(position=(0.0, 0.0, 1.5), radius=0.5, color=(1.0, 0.8, 0.6), intensity=3.0),),
    )
    initial = SphereLight(
        position=(-1.0, 0.5, 0.5), radius=0.2, color=(1, 1, 1), intensity=1.0
    )
    report = optimize_lights_fallback(
        traced_arrays, (initial,), target, sweeps=1, locks=({"intensity"},)
    )
    assert report["optimized_lights"][0]["intensity"] == initial.intensity
    assert report["locked_fields"] == [["intensity"]]
