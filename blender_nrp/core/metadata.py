"""NRP metadata contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


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

    def __post_init__(self) -> None:
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ValueError("resolution must contain positive width and height")
        if self.light_type != "sphere":
            raise ValueError("V1 only supports sphere-light metadata")
        if self.aux_features != ["albedo", "normal", "depth"]:
            raise ValueError("aux_features must be ['albedo', 'normal', 'depth']")
        if len(self.bbox_min) != 3 or len(self.bbox_max) != 3:
            raise ValueError("bbox_min and bbox_max must contain 3 values")
        if self.model_width <= 0 or self.model_depth <= 0:
            raise ValueError("model dimensions must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NRPMetadata":
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
        payload = dict(data)
        payload["resolution"] = tuple(payload["resolution"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolution"] = list(self.resolution)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "NRPMetadata":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

