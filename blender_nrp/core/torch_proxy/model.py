"""Torch neural render proxy — port of nrp's `torch_backend/model.py`.

The save format is byte-compatible with nrp's `TorchNRP.save`/`load`
(`torch.save({"config": ..., "state_dict": ...})` with identical config keys and
module names), so `model.pt` written here loads in the nrp reference and vice versa.
ComfyUI-NeuralRenderProxy's `NRPProxy` is a *different* architecture (13-input SiLU
MLP); loading TorchNRP artifacts there requires a ComfyUI-side change — recorded in
train_report.json rather than papered over.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .encoding import HashEncoding2D

LIGHT_PARAM_DIMS = {"sphere": 4, "quad": 8}


def relative_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Relative MSE (Eq. 4): denominator is the stop-gradient of the prediction."""
    return ((pred - target) ** 2 / (pred.detach() ** 2 + eps)).mean()


class TorchNRP(nn.Module):
    def __init__(
        self,
        light_type: str = "sphere",
        hidden_width: int = 128,
        hidden_layers: int = 4,
        encoding: dict | None = None,
        use_encoding: bool = True,
        use_aux: bool = True,
    ):
        super().__init__()
        if light_type not in LIGHT_PARAM_DIMS:
            raise ValueError(f"light_type must be one of {sorted(LIGHT_PARAM_DIMS)}")
        self.light_type = light_type
        self.use_encoding = use_encoding
        self.use_aux = use_aux
        self.config = {
            "light_type": light_type,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "encoding": encoding or {},
            "use_encoding": use_encoding,
            "use_aux": use_aux,
        }
        self.encoding = HashEncoding2D(**(encoding or {})) if use_encoding else None
        px_dim = self.encoding.output_dim if use_encoding else 2
        in_dim = px_dim + (7 if use_aux else 0) + LIGHT_PARAM_DIMS[light_type]
        layers: list[nn.Module] = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_width, hidden_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_width if hidden_layers else in_dim, 3))
        self.mlp = nn.Sequential(*layers)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, pixel_xy: torch.Tensor, aux: torch.Tensor, light_params: torch.Tensor
    ) -> torch.Tensor:
        """pixel_xy (N,2) in [0,1]^2, aux (N,7), light_params (N, 4 or 8) -> (N,3)."""
        px = self.encoding(pixel_xy) if self.encoding is not None else pixel_xy
        parts = [px, aux, light_params] if self.use_aux else [px, light_params]
        return F.softplus(self.mlp(torch.cat(parts, dim=1)))

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> TorchNRP:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


def sphere_params(center: torch.Tensor, radius: torch.Tensor, n: int) -> torch.Tensor:
    """Broadcast one sphere's (center, radius) to an (N, 4) light-parameter block."""
    return torch.cat([center.reshape(1, 3).expand(n, 3), radius.reshape(1, 1).expand(n, 1)], dim=1)


def quad_params(
    center: torch.Tensor,
    normal: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Broadcast one quad's parameters to an (N, 8) block (normal normalized here so
    gradients flow through the normalization during inverse optimization)."""
    unit = normal / torch.linalg.vector_norm(normal)
    return torch.cat(
        [
            center.reshape(1, 3).expand(n, 3),
            unit.reshape(1, 3).expand(n, 3),
            width.reshape(1, 1).expand(n, 1),
            height.reshape(1, 1).expand(n, 1),
        ],
        dim=1,
    )
