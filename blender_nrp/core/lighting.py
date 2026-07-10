"""Lighting-friendly, Blender-independent color and exposure conversions."""

from __future__ import annotations

import math

import numpy as np


def intensity_to_stops(intensity: float, reference: float = 1.0) -> float:
    if intensity <= 0 or reference <= 0:
        raise ValueError("intensity and reference must be positive")
    return math.log2(intensity / reference)


def stops_to_intensity(stops: float, reference: float = 1.0) -> float:
    if reference <= 0:
        raise ValueError("reference must be positive")
    return reference * (2.0**stops)


def kelvin_to_rgb(kelvin: float, tint: float = 0.0) -> tuple[float, float, float]:
    """Approximate display-linear RGB; tint shifts green/magenta without clipping HDR."""
    if not 1000.0 <= kelvin <= 40000.0:
        raise ValueError("kelvin must be between 1000 and 40000")
    temp = kelvin / 100.0
    if temp <= 66.0:
        red = 1.0
        green = np.clip((99.4708025861 * math.log(temp) - 161.1195681661) / 255.0, 0.0, 1.0)
        blue = (
            0.0
            if temp <= 19.0
            else np.clip(
                (138.5177312231 * math.log(temp - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0
            )
        )
    else:
        red = np.clip((329.698727446 * (temp - 60.0) ** -0.1332047592) / 255.0, 0.0, 1.0)
        green = np.clip((288.1221695283 * (temp - 60.0) ** -0.0755148492) / 255.0, 0.0, 1.0)
        blue = 1.0
    rgb = np.array([red, green * (1.0 + tint), blue], dtype=np.float64)
    return tuple(np.clip(rgb, 0.0, None))
