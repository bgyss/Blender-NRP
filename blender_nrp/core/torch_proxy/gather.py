"""Batched torch GATHERLIGHT over an arrays-dict cache (port of nrp's torch gather).

Mirrors `blender_nrp.core.gather` semantics exactly (same overlap predicates, same
n_paths normalization) with all segments tested in a handful of batched tensor ops
on whatever device the cache lives on. The numpy gather remains the authoritative
reference; unit tests assert agreement.
"""

from __future__ import annotations

import numpy as np
import torch

from ..gather import quad_tangent_frame
from ..lights import AnyLight, QuadLight, SphereLight


class TorchPathCache:
    """Device-resident copy of the cache's segment arrays for batched gathering."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ):
        device = torch.device(device)
        if dtype is None:
            dtype = torch.float32 if device.type in ("mps",) else torch.float64
        self.height, self.width, _ = arrays["albedo"].shape
        self.device = device
        self.dtype = dtype
        to = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        self.origin = to(arrays["seg_origin"])
        self.dir = to(arrays["seg_dir"])
        self.tmax = to(arrays["seg_tmax"])  # inf escape segments compare like numpy
        self.throughput = to(arrays["seg_throughput"])
        self.pixel = torch.as_tensor(
            np.asarray(arrays["seg_pixel"]), dtype=torch.long, device=device
        )
        self.inv_paths = to(1.0 / np.maximum(np.asarray(arrays["n_paths"]), 1))

    @property
    def segment_count(self) -> int:
        return int(self.pixel.shape[0])

    def _accumulate(self, hits: torch.Tensor) -> torch.Tensor:
        n_px = self.height * self.width
        contrib = torch.zeros((n_px, 3), dtype=self.dtype, device=self.device)
        weighted = self.throughput * hits.to(self.dtype).unsqueeze(-1)
        contrib.index_add_(0, self.pixel, weighted)
        contrib *= self.inv_paths.unsqueeze(-1)
        return contrib.reshape(self.height, self.width, 3)

    def gather_throughput(self, center, radius: float) -> torch.Tensor:
        """Sphere GATHERtype: (H,W,3) pre-emission contribution."""
        if not self.segment_count:
            return torch.zeros((self.height, self.width, 3), dtype=self.dtype, device=self.device)
        c = torch.as_tensor(center, dtype=self.dtype, device=self.device)
        oc = self.origin - c
        b = (oc * self.dir).sum(dim=1)
        cc = (oc * oc).sum(dim=1) - float(radius) ** 2
        disc = b * b - cc
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        t0 = -b - sq
        t1 = -b + sq
        hits = (disc >= 0.0) & (t0 <= self.tmax) & (t1 >= 0.0)
        return self._accumulate(hits)

    def gather_throughput_quad(self, center, normal, width: float, height: float) -> torch.Tensor:
        """Quad GATHERtype: (H,W,3) pre-emission contribution."""
        if not self.segment_count:
            return torch.zeros((self.height, self.width, 3), dtype=self.dtype, device=self.device)
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        u, v = quad_tangent_frame(n)
        to = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        c, n_t, u_t, v_t = to(center), to(n), to(u), to(v)

        denom = self.dir @ n_t
        parallel = denom.abs() < 1e-12
        safe = torch.where(parallel, torch.ones_like(denom), denom)
        t = ((c - self.origin) @ n_t) / safe
        p = self.origin + t.unsqueeze(-1) * self.dir
        local = p - c
        hits = (
            ~parallel
            & (t >= 0.0)
            & (t <= self.tmax)
            & ((local @ u_t).abs() <= width / 2.0)
            & ((local @ v_t).abs() <= height / 2.0)
        )
        return self._accumulate(hits)

    def gather_light(self, light: AnyLight) -> torch.Tensor:
        """Full contribution of one light: GATHERtype scaled by emission."""
        if isinstance(light, SphereLight):
            image = self.gather_throughput(light.position, light.radius)
        elif isinstance(light, QuadLight):
            image = self.gather_throughput_quad(
                light.position, light.normal, light.width, light.height
            )
        else:
            raise TypeError(f"unsupported light type: {type(light).__name__}")
        emission = np.asarray(light.color, dtype=np.float64) * float(light.intensity)
        return image * torch.as_tensor(emission, dtype=self.dtype, device=self.device)
