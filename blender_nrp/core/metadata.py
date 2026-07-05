"""NRP metadata contract.

V2 additions (all backward-compatible on load — older files simply lack the keys):

- `light_type` may be "sphere" or "quad" (V1 hard-coded "sphere").
- `throughput_normalization` makes the gather convention explicit (interop debt 1):
  "n_paths" means the cache stores raw per-segment throughput and gather divides
  per-pixel sums by n_paths (the nrp-main convention this repo follows);
  "pre_divided" means throughput was divided at export time, for consumers whose
  gather does not normalize (ComfyUI-NeuralRenderProxy).
- `schema_version` mirrors the cache npz's schema version.
- `medium` optionally records a homogeneous participating medium
  ({"sigma_t": float, "albedo": float}); surfaced on load/validate even though
  Blender-side volume capture is out of scope for V2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LIGHT_TYPES = ("sphere", "quad")
THROUGHPUT_NORMALIZATIONS = ("n_paths", "pre_divided")


@dataclass(frozen=True)
class NRPMetadata:
    scene_id: str
    camera_id: str
    resolution: tuple[int, int]
    light_type: str
    aux_features: list[str]
    bbox_min: list[float]
    bbox_max: list[float]
    model_width: int
    model_depth: int
    coordinate_system: str = "blender_z_up"
    throughput_normalization: str = "n_paths"
    schema_version: int = 2
    medium: dict[str, float] | None = field(default=None)

    def __post_init__(self) -> None:
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ValueError("resolution must contain positive width and height")
        if self.light_type not in LIGHT_TYPES:
            raise ValueError(f"light_type must be one of {LIGHT_TYPES}")
        if self.aux_features != ["albedo", "normal", "depth"]:
            raise ValueError("aux_features must be ['albedo', 'normal', 'depth']")
        if len(self.bbox_min) != 3 or len(self.bbox_max) != 3:
            raise ValueError("bbox_min and bbox_max must contain 3 values")
        if self.model_width <= 0 or self.model_depth <= 0:
            raise ValueError("model dimensions must be positive")
        if self.throughput_normalization not in THROUGHPUT_NORMALIZATIONS:
            raise ValueError(
                f"throughput_normalization must be one of {THROUGHPUT_NORMALIZATIONS}"
            )
        if self.medium is not None:
            if not float(self.medium["sigma_t"]) > 0.0:
                raise ValueError("medium sigma_t must be positive")
            if not 0.0 <= float(self.medium["albedo"]) <= 1.0:
                raise ValueError("medium albedo must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NRPMetadata:
        required = {
            "scene_id",
            "camera_id",
            "resolution",
            "light_type",
            "aux_features",
            "bbox_min",
            "bbox_max",
            "model_width",
            "model_depth",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"missing metadata fields: {', '.join(missing)}")
        known = required | {
            "coordinate_system",
            "throughput_normalization",
            "schema_version",
            "medium",
        }
        payload = {key: value for key, value in data.items() if key in known}
        payload["resolution"] = tuple(payload["resolution"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolution"] = list(self.resolution)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> NRPMetadata:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
