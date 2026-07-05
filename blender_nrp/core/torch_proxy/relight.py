"""Proxy inference: relight a rig through a trained TorchNRP."""

from __future__ import annotations

import numpy as np
import torch

from ..lights import AnyLight, QuadLight, SphereLight
from .model import TorchNRP
from .sampling import light_param_vector
from .train import pixel_tensors


def proxy_relight(
    model: TorchNRP,
    arrays: dict[str, np.ndarray],
    lights: tuple[AnyLight, ...],
    *,
    device: str = "cpu",
) -> np.ndarray:
    """(H, W, 3) float64 linear-HDR image: sum over lights of the proxy's
    pre-emission contribution scaled by each light's emission."""
    h, w, _ = arrays["albedo"].shape
    dev = torch.device(device)
    model = model.to(dev).eval()
    xy, aux = pixel_tensors(arrays, dev)
    n_px = xy.shape[0]
    image = np.zeros((n_px, 3), dtype=np.float64)
    with torch.no_grad():
        for light in lights:
            expected = model.light_type
            actual = "sphere" if isinstance(light, SphereLight) else "quad"
            if not isinstance(light, (SphereLight, QuadLight)):
                raise TypeError(f"unsupported light type: {type(light).__name__}")
            if actual != expected:
                raise ValueError(
                    f"proxy was trained for {expected} lights; rig contains a {actual} light"
                )
            params = torch.as_tensor(
                light_param_vector(light), dtype=torch.float32, device=dev
            ).expand(n_px, -1)
            contribution = model(xy, aux, params).cpu().numpy().astype(np.float64)
            emission = np.asarray(light.color, dtype=np.float64) * float(light.intensity)
            image += contribution * emission[None, :]
    return image.reshape(h, w, 3)
