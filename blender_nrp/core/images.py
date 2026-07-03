"""Small image writers used by Blender and pure-Python fixture paths."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png_rgb(path: str | Path, rgb: np.ndarray) -> None:
    """Write a uint8 or float RGB array as a PNG without optional dependencies."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0 + 0.5).astype(np.uint8)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(_png_chunk(b"IHDR", payload))
        handle.write(_png_chunk(b"IDAT", zlib.compress(raw)))
        handle.write(_png_chunk(b"IEND", b""))


def normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(normal) * 0.5 + 0.5, 0.0, 1.0)


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        scaled = np.zeros_like(values, dtype=np.float32)
    else:
        min_depth = float(finite.min())
        max_depth = float(finite.max())
        denom = max(max_depth - min_depth, 1e-6)
        scaled = np.where(values > 0.0, 1.0 - ((values - min_depth) / denom), 0.0)
    return np.repeat(np.clip(scaled[..., None], 0.0, 1.0), 3, axis=2)
