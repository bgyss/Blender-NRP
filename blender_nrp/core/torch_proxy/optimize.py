"""Differentiable multi-light inverse optimization through the trained proxy.

GATHERLIGHT itself has zero gradient almost everywhere in light shape parameters
(hard visibility indicator), so the smooth proxy is the differentiable forward
model — exactly the paper's point and nrp's approach. All lights in the rig are
optimized jointly: positions, radii (spheres), normals/sizes (quads), colors, and
intensities, by Adam over a per-light parameter set with box constraints derived
from the cache's visible extent.

The result is always *re-rendered through the reference numpy gather* as well, and
the report separates proxy-space loss from gather-space error — the proxy may be
wrong; the gather numbers are the physically grounded ones.
"""

from __future__ import annotations

import numpy as np
import torch

from ..gather import gather_hdr
from ..lights import AnyLight, QuadLight, SphereLight
from .model import TorchNRP, quad_params, sphere_params
from .train import pixel_tensors


def _bounds_from_cache(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    positions = arrays["position"].reshape(-1, 3)
    valid = np.asarray(arrays["n_paths"]) > 0
    pts = positions[valid] if np.any(valid) else positions
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = float(np.linalg.norm(hi - lo)) or 1.0
    margin = 0.5 * extent
    return lo - margin, hi + margin, extent


class _LightParams:
    """Trainable torch parameters for one rig light."""

    def __init__(self, light: AnyLight, device: torch.device, locks: set[str] | None = None):
        self.kind = light.light_type
        self.locks = locks or set()
        to = lambda v: torch.tensor(  # noqa: E731
            np.asarray(v, dtype=np.float64), dtype=torch.float32, device=device, requires_grad=True
        )
        self.position = to(light.position)
        self.color = to(light.color)
        self.intensity = to([light.intensity])
        if isinstance(light, SphereLight):
            self.radius = to([light.radius])
            self.shape_params = [self.radius]
            self.shape_fields = ["radius"]
        else:
            self.normal = to(light.normal)
            self.width = to([light.width])
            self.height = to([light.height])
            self.shape_params = [self.normal, self.width, self.height]
            self.shape_fields = ["normal", "width", "height"]

    def all_params(self) -> list[torch.Tensor]:
        params = []
        if "position" not in self.locks:
            params.append(self.position)
        if "color" not in self.locks:
            params.append(self.color)
        if "intensity" not in self.locks:
            params.append(self.intensity)
        params.extend(
            param for field, param in zip(self.shape_fields, self.shape_params, strict=True)
            if field not in self.locks
        )
        return params

    def light_param_block(self, n: int) -> torch.Tensor:
        if self.kind == "sphere":
            return sphere_params(self.position, self.radius, n)
        return quad_params(self.position, self.normal, self.width, self.height, n)

    def emission(self) -> torch.Tensor:
        return self.color * self.intensity

    @torch.no_grad()
    def clamp_(self, lo: torch.Tensor, hi: torch.Tensor, extent: float) -> None:
        self.position.clamp_(min=lo, max=hi)
        self.color.clamp_(min=0.0)
        self.intensity.clamp_(min=0.0)
        if self.kind == "sphere":
            self.radius.clamp_(min=1e-3 * extent, max=0.5 * extent)
        else:
            self.width.clamp_(min=1e-3 * extent, max=extent)
            self.height.clamp_(min=1e-3 * extent, max=extent)

    @torch.no_grad()
    def to_light(self) -> AnyLight:
        position = tuple(float(v) for v in self.position.cpu())
        color = tuple(float(v) for v in self.color.cpu())
        intensity = float(self.intensity.cpu())
        if self.kind == "sphere":
            return SphereLight(
                position=position, radius=float(self.radius.cpu()), color=color,
                intensity=intensity,
            )
        return QuadLight(
            position=position,
            normal=tuple(float(v) for v in self.normal.cpu()),
            width=float(self.width.cpu()),
            height=float(self.height.cpu()),
            color=color,
            intensity=intensity,
        )


def optimize_lights(
    model: TorchNRP,
    arrays: dict[str, np.ndarray],
    lights: tuple[AnyLight, ...],
    target: np.ndarray,
    *,
    steps: int = 300,
    lr: float = 2e-2,
    device: str = "cpu",
    locks: tuple[set[str], ...] | None = None,
) -> dict:
    """Optimize `lights` so the proxy image matches `target` (H, W, 3 linear HDR).

    Returns a report dict with `optimized_lights` (list of light dicts, same order
    as the input rig), before/after losses, and gather-space validation numbers.
    """
    mismatched = [light.light_type for light in lights if light.light_type != model.light_type]
    if mismatched:
        raise ValueError(
            f"proxy was trained for {model.light_type} lights; rig contains {mismatched}"
        )
    h, w, _ = arrays["albedo"].shape
    if target.shape != (h, w, 3):
        raise ValueError(f"target must be {(h, w, 3)}, got {target.shape}")

    dev = torch.device(device)
    model = model.to(dev).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    xy, aux = pixel_tensors(arrays, dev)
    n_px = xy.shape[0]
    target_t = torch.as_tensor(
        target.reshape(n_px, 3), dtype=torch.float32, device=dev
    )

    lo_np, hi_np, extent = _bounds_from_cache(arrays)
    lo = torch.as_tensor(lo_np, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(hi_np, dtype=torch.float32, device=dev)

    locks = locks or tuple(set() for _ in lights)
    if len(locks) != len(lights):
        raise ValueError("locks must contain one field set per light")
    params = [
        _LightParams(light, dev, field_locks)
        for light, field_locks in zip(lights, locks, strict=True)
    ]
    trainable = [p for lp in params for p in lp.all_params()]
    opt = torch.optim.Adam(trainable, lr=lr) if trainable else None

    def predict() -> torch.Tensor:
        image = torch.zeros((n_px, 3), dtype=torch.float32, device=dev)
        for lp in params:
            contribution = model(xy, aux, lp.light_param_block(n_px))
            image = image + contribution * lp.emission()
        return image

    loss_curve: list[float] = []
    for _step in range(steps):
        if opt is None:
            break
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((predict() - target_t) ** 2)
        loss.backward()
        opt.step()
        for lp in params:
            lp.clamp_(lo, hi, extent)
        loss_curve.append(float(loss.detach().cpu()))

    optimized = tuple(lp.to_light() for lp in params)
    with torch.no_grad():
        proxy_final = predict().cpu().numpy().reshape(h, w, 3)
    gather_initial = gather_hdr(arrays, lights)
    gather_final = gather_hdr(arrays, optimized)

    def mse(a, b) -> float:
        return float(np.mean((np.asarray(a, dtype=np.float64) - b) ** 2))

    return {
        "ok": True,
        "solver": "torch_proxy_adam",
        "steps": steps,
        "light_count": len(lights),
        "locked_fields": [sorted(fields) for fields in locks],
        "initial_lights": [light.to_dict() for light in lights],
        "optimized_lights": [light.to_dict() for light in optimized],
        "proxy_loss_first": loss_curve[0],
        "proxy_loss_last": loss_curve[-1],
        "proxy_loss_curve": loss_curve[:: max(1, steps // 50)],
        "gather_mse_vs_target_initial": mse(gather_initial, target),
        "gather_mse_vs_target_final": mse(gather_final, target),
        "proxy_mse_vs_target_final": mse(proxy_final, target),
        "limitations": [
            "Optimization descends through the smooth proxy, not GATHERLIGHT itself; "
            "gather-space numbers above are the physically grounded check.",
            "Quad normals are optimized through the proxy's normalized-normal "
            "parameterization; degenerate flips are prevented only by the loss.",
        ],
    }
