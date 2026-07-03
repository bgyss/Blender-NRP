"""Backend protocol for path-cache baking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class BakeSettings:
    scene_id: str
    output_dir: Path
    width: int
    height: int
    segment_count: int
    max_segment_distance: float
    camera_id: str = "Camera"
    model_width: int = 64
    model_depth: int = 4
    seed: int = 0


@dataclass(frozen=True)
class BakeResult:
    output_dir: Path
    cache_path: Path
    metadata_path: Path
    bake_report_path: Path
    preview_paths: dict[str, Path]
    warnings: tuple[str, ...] = ()


class PathCacheBackend(Protocol):
    id: str
    label: str

    def bake(self, context: Any, settings: BakeSettings) -> BakeResult:
        """Bake a path cache and return the cache path."""
