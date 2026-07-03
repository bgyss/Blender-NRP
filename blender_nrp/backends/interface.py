"""Backend protocol for path-cache baking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BakeSettings:
    scene_id: str
    output_dir: Path
    width: int
    height: int
    segment_count: int
    max_segment_distance: float
    seed: int = 0


class PathCacheBackend(Protocol):
    id: str
    label: str

    def bake(self, context: object, settings: BakeSettings) -> Path:
        """Bake a path cache and return the cache path."""

