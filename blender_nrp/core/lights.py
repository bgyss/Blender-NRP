"""NRP sphere-light JSON contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


def _vec3(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain 3 numeric values")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class SphereLight:
    position: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]
    intensity: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("sphere light radius must be positive")
        if self.intensity < 0:
            raise ValueError("sphere light intensity must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SphereLight":
        if data.get("type") != "sphere":
            raise ValueError(f"unsupported light type: {data.get('type')!r}")
        for field in ("position", "radius", "color", "intensity"):
            if field not in data:
                raise ValueError(f"missing light field: {field}")
        return cls(
            position=_vec3(data["position"], name="position"),
            radius=float(data["radius"]),
            color=_vec3(data["color"], name="color"),
            intensity=float(data["intensity"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sphere",
            "position": list(self.position),
            "radius": self.radius,
            "color": list(self.color),
            "intensity": self.intensity,
        }


@dataclass(frozen=True)
class LightRig:
    lights: tuple[SphereLight, ...]
    scene_id: str | None = None
    camera_id: str | None = None
    coordinate_system: str = "blender_z_up"

    def __post_init__(self) -> None:
        if not self.lights:
            raise ValueError("light rig must contain at least one light")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LightRig":
        lights = data.get("lights")
        if not isinstance(lights, list):
            raise ValueError("light rig JSON must contain a lights list")
        return cls(
            tuple(SphereLight.from_dict(light) for light in lights),
            scene_id=data.get("scene_id"),
            camera_id=data.get("camera_id"),
            coordinate_system=data.get("coordinate_system", "blender_z_up"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "camera_id": self.camera_id,
            "coordinate_system": self.coordinate_system,
            "lights": [light.to_dict() for light in self.lights],
        }

    @classmethod
    def load(cls, path: str | Path) -> "LightRig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

