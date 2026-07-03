"""NRP light JSON contract (sphere + quad).

The rig JSON stays wire-compatible with `nrp` and `ComfyUI-NeuralRenderProxy`:
`position`/`radius`/`color`/`intensity` for spheres, plus `"type": "quad"` entries with
`normal`/`width`/`height`. Dispatch follows nrp's `light_from_dict`: specs without a
`"type"` key are quads when they carry a `width` field and spheres otherwise, so V1
sphere JSON stays loadable unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    light_type = "sphere"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("sphere light radius must be positive")
        if self.intensity < 0:
            raise ValueError("sphere light intensity must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SphereLight:
        if data.get("type", "sphere") != "sphere":
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
class QuadLight:
    """Rectangle emitter: center position, unit normal, width x height extent."""

    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    width: float
    height: float
    color: tuple[float, float, float]
    intensity: float

    light_type = "quad"

    def __post_init__(self) -> None:
        norm = sum(component * component for component in self.normal) ** 0.5
        if norm <= 0.0:
            raise ValueError("quad light normal must be nonzero")
        object.__setattr__(
            self, "normal", tuple(float(component) / norm for component in self.normal)
        )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("quad light width/height must be positive")
        if self.intensity < 0:
            raise ValueError("quad light intensity must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuadLight:
        if data.get("type", "quad") != "quad":
            raise ValueError(f"unsupported light type: {data.get('type')!r}")
        for field in ("position", "normal", "width", "height", "color", "intensity"):
            if field not in data:
                raise ValueError(f"missing light field: {field}")
        return cls(
            position=_vec3(data["position"], name="position"),
            normal=_vec3(data["normal"], name="normal"),
            width=float(data["width"]),
            height=float(data["height"]),
            color=_vec3(data["color"], name="color"),
            intensity=float(data["intensity"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "quad",
            "position": list(self.position),
            "normal": list(self.normal),
            "width": self.width,
            "height": self.height,
            "color": list(self.color),
            "intensity": self.intensity,
        }


AnyLight = SphereLight | QuadLight


def light_from_dict(data: dict[str, Any]) -> AnyLight:
    """Dispatch on the optional "type" key; untyped specs with a width are quads,
    all other untyped specs remain spheres (matches nrp's `light_from_dict`)."""
    kind = data.get("type", "quad" if "width" in data else "sphere")
    if kind == "sphere":
        return SphereLight.from_dict(data)
    if kind == "quad":
        return QuadLight.from_dict(data)
    raise ValueError(f"unknown light type {kind!r}")


@dataclass(frozen=True)
class LightRig:
    lights: tuple[AnyLight, ...]
    scene_id: str | None = None
    camera_id: str | None = None
    coordinate_system: str = "blender_z_up"

    def __post_init__(self) -> None:
        if not self.lights:
            raise ValueError("light rig must contain at least one light")

    @property
    def light_types(self) -> tuple[str, ...]:
        return tuple(sorted({light.light_type for light in self.lights}))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LightRig:
        lights = data.get("lights")
        if not isinstance(lights, list):
            raise ValueError("light rig JSON must contain a lights list")
        return cls(
            tuple(light_from_dict(light) for light in lights),
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
    def load(cls, path: str | Path) -> LightRig:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
