"""Proxy training loop: pool-of-images scheme with cosine LR and checkpoint/resume.

Follows nrp's `torch_backend/train.py`: a pool of GATHERLIGHT target images (one per
random light configuration, rendered with the batched device gather), every training
pixel sampling its target uniformly from the pool, periodic pool replacement, the
relative-MSE loss with a stop-gradient denominator, cosine LR annealing, and
checkpoints carrying the full training state (model, optimizer, scheduler, RNG,
pool) so `resume=True` continues the exact trajectory. One documented deviation from
nrp: pool targets are raw gathers, not denoised (recorded in the report).

Designed to run on a worker thread inside Blender: `progress` is called with
(iteration, total, loss) and `should_cancel()` is polled every iteration.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from . import select_device
from .gather import TorchPathCache
from .model import LIGHT_PARAM_DIMS, TorchNRP, relative_mse_loss
from .sampling import default_bounds, light_param_vector, sample_light

LIMITATIONS = [
    "Pool targets are raw GATHERLIGHT gathers; nrp's denoised-target pool is not "
    "ported (quality deviation, not a format one).",
    "ComfyUI-NeuralRenderProxy's NRPProxy is a different architecture; this model.pt "
    "loads in nrp's TorchNRP but needs a ComfyUI-side change to load there.",
]


def pixel_tensors(
    arrays: dict[str, np.ndarray], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """((N,2) pixel xy in [0,1]^2, (N,7) aux features albedo+depth+normal)."""
    h, w, _ = arrays["albedo"].shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h], axis=1)
    aux = np.concatenate(
        [
            np.asarray(arrays["albedo"]).reshape(-1, 3),
            np.asarray(arrays["depth"]).reshape(-1, 1),
            np.asarray(arrays["normal"]).reshape(-1, 3),
        ],
        axis=1,
    )
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return to(xy), to(aux)


class ImagePool:
    """Pool of (light params, target image) rows with periodic replacement."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        torch_cache: TorchPathCache,
        *,
        light_type: str,
        bounds: dict,
        size: int,
        rng: np.random.Generator,
        device,
        fill: bool = True,
    ):
        self.arrays = arrays
        self.torch_cache = torch_cache
        self.light_type = light_type
        self.bounds = bounds
        self.rng = rng
        self.device = device
        self.size = size
        n_px = torch_cache.height * torch_cache.width
        self.params = torch.empty(
            (size, LIGHT_PARAM_DIMS[light_type]), dtype=torch.float32, device=device
        )
        self.targets = torch.empty((size, n_px, 3), dtype=torch.float32, device=device)
        self._next_replace = 0
        self.supervision_images = 0
        if fill:
            for i in range(size):
                self.fill(i)

    def fill(self, slot: int) -> None:
        light = sample_light(self.arrays, self.rng, self.light_type, self.bounds)
        vec = light_param_vector(light)
        self.params[slot] = torch.as_tensor(vec, dtype=torch.float32, device=self.device)
        target = self.torch_cache.gather_light(light).reshape(-1, 3)
        self.targets[slot] = target.to(torch.float32)
        self.supervision_images += 1

    def replace_round(self, count: int) -> None:
        for _ in range(count):
            self.fill(self._next_replace)
            self._next_replace = (self._next_replace + 1) % self.size

    def state_dict(self) -> dict:
        return {
            "params": self.params.cpu(),
            "targets": self.targets.cpu(),
            "next_replace": self._next_replace,
            "supervision_images": self.supervision_images,
        }

    def load_state_dict(self, state: dict) -> None:
        self.params = state["params"].to(self.device)
        self.targets = state["targets"].to(self.device)
        self._next_replace = state["next_replace"]
        self.supervision_images = state["supervision_images"]


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    peak = float(ref.max()) if ref.size and ref.max() > 0 else 1.0
    err = float(np.mean((np.asarray(pred, dtype=np.float64) - ref) ** 2))
    return float("inf") if err == 0.0 else float(10.0 * np.log10(peak**2 / err))


def train_proxy(
    arrays: dict[str, np.ndarray],
    model_path: str | Path,
    *,
    light_type: str = "sphere",
    iterations: int = 2000,
    batch_size: int = 8192,
    lr: float = 1e-2,
    lr_min: float | None = None,
    pool_size: int = 48,
    pool_replace_every: int = 5,
    pool_replace_count: int = 2,
    n_val_lights: int = 8,
    hidden_width: int = 64,
    hidden_layers: int = 3,
    encoding: dict | None = None,
    device: str = "auto",
    seed: int = 0,
    checkpoint_every: int = 500,
    resume: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Train a TorchNRP on the cache; returns the report dict (also what the
    operator writes as train_report.json). Raises on hard failures."""
    device = select_device(device)
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    torch_cache = TorchPathCache(arrays, dev, dtype=torch.float32)
    xy, aux = pixel_tensors(arrays, dev)
    n_px = xy.shape[0]
    bounds = default_bounds(arrays)

    t0 = time.perf_counter()
    model = TorchNRP(
        light_type=light_type,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        encoding=encoding,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=iterations, eta_min=lr_min if lr_min is not None else lr / 100.0
    )
    pool = ImagePool(
        arrays,
        torch_cache,
        light_type=light_type,
        bounds=bounds,
        size=pool_size,
        rng=rng,
        device=dev,
        fill=not resume,
    )

    model_path = Path(model_path)
    checkpoint_path = model_path.parent / "checkpoint.pt"
    start_iter = 0
    loss_curve: list[float] = []
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"no checkpoint to resume from at {checkpoint_path}")
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        gen.set_state(blob["torch_gen"])
        rng.bit_generator.state = blob["numpy_rng"]
        pool.load_state_dict(blob["pool"])
        start_iter = blob["iteration"]
        loss_curve = blob["loss_curve"]

    # Fixed held-out validation lights from a dedicated RNG (never perturbs training).
    val_rng = np.random.default_rng([seed, 0x5EED])
    val_lights = [sample_light(arrays, val_rng, light_type, bounds) for _ in range(n_val_lights)]
    val_targets = [
        torch_cache.gather_light(light).reshape(-1, 3).cpu().numpy() for light in val_lights
    ]

    pool_seconds = time.perf_counter() - t0
    t_train = time.perf_counter()
    cancelled = False
    model.train()
    for it in range(start_iter, iterations):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        px_idx = torch.randint(0, n_px, (batch_size,), generator=gen).to(dev)
        pool_idx = torch.randint(0, pool.size, (batch_size,), generator=gen).to(dev)
        pred = model(xy[px_idx], aux[px_idx], pool.params[pool_idx])
        target = pool.targets[pool_idx, px_idx]
        loss = relative_mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_curve.append(float(loss.detach().cpu()))
        if (it + 1) % pool_replace_every == 0:
            pool.replace_round(pool_replace_count)
        if progress is not None and (it + 1) % 25 == 0:
            progress(it + 1, iterations, loss_curve[-1])
        if checkpoint_every and (it + 1) % checkpoint_every == 0:
            torch.save(
                {
                    "iteration": it + 1,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "torch_gen": gen.get_state(),
                    "numpy_rng": rng.bit_generator.state,
                    "pool": pool.state_dict(),
                    "loss_curve": loss_curve,
                },
                checkpoint_path,
            )
    train_seconds = time.perf_counter() - t_train

    model.eval()
    val_metrics = []
    with torch.no_grad():
        for light, target in zip(val_lights, val_targets, strict=False):
            params = torch.as_tensor(
                light_param_vector(light), dtype=torch.float32, device=dev
            ).expand(n_px, -1)
            pred = model(xy, aux, params).cpu().numpy().astype(np.float64)
            val_metrics.append({"light": light.to_dict(), "psnr_db": psnr(pred, target)})

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    return {
        "ok": not cancelled,
        "cancelled": cancelled,
        "format": "torch_nrp",
        "training_backend": "torch",
        "device": device,
        "light_type": light_type,
        "iterations_run": len(loss_curve),
        "iterations_requested": iterations,
        "parameter_count": model.parameter_count,
        "pool_size": pool_size,
        "supervision_images": pool.supervision_images,
        "pool_build_seconds": pool_seconds,
        "train_seconds": train_seconds,
        "loss_first": loss_curve[0] if loss_curve else None,
        "loss_last": loss_curve[-1] if loss_curve else None,
        "loss_curve": loss_curve[:: max(1, len(loss_curve) // 100)],
        "val_lights": val_metrics,
        "val_psnr_db_mean": float(np.mean([m["psnr_db"] for m in val_metrics])),
        "model_path": str(model_path),
        "limitations": list(LIMITATIONS),
    }
