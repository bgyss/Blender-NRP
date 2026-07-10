"""Versioned, Blender-independent descriptions for V3 worker jobs.

The JSON written by these types is a public interchange boundary: an add-on can
create it, while a local process, a LAN host, or a cloud worker can consume it.
There are deliberately no ``bpy`` imports in this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

JOB_SCHEMA_VERSION = 1


def _required(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


@dataclass(frozen=True)
class BakeJob:
    scene_id: str
    scene_path: str
    output_dir: str
    camera_id: str = "Camera"
    width: int = 256
    height: int = 256
    paths_per_pixel: int = 64
    max_bounces: int = 4
    backend: str = "cycles_capture"
    seed: int = 0
    packed: bool = False
    torch_device: str = "auto"
    output_manifest: tuple[str, ...] = ("path_cache.npz", "metadata.json", "bake_report.json")
    kind: Literal["bake"] = "bake"
    schema_version: int = JOB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required(self.scene_id, "scene_id")
        _required(self.scene_path, "scene_path")
        _required(self.output_dir, "output_dir")
        if self.schema_version != JOB_SCHEMA_VERSION:
            raise ValueError(f"unsupported bake job schema version: {self.schema_version}")
        if min(self.width, self.height, self.paths_per_pixel, self.max_bounces) < 1:
            raise ValueError("resolution, paths_per_pixel, and max_bounces must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"output_manifest": list(self.output_manifest)}


@dataclass(frozen=True)
class TrainJob:
    cache_path: str
    output_dir: str
    iterations: int = 2000
    torch_device: str = "auto"
    output_manifest: tuple[str, ...] = ("model.pt", "train_report.json")
    kind: Literal["train"] = "train"
    schema_version: int = JOB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required(self.cache_path, "cache_path")
        _required(self.output_dir, "output_dir")
        if self.schema_version != JOB_SCHEMA_VERSION or self.iterations < 1:
            raise ValueError("unsupported train job or non-positive iterations")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"output_manifest": list(self.output_manifest)}


@dataclass(frozen=True)
class SolveJob:
    cache_path: str
    lights_path: str
    target_path: str
    output_dir: str
    steps: int = 300
    torch_device: str = "auto"
    output_manifest: tuple[str, ...] = ("solved_lights.json", "solve_report.json")
    kind: Literal["solve"] = "solve"
    schema_version: int = JOB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all((self.cache_path, self.lights_path, self.target_path, self.output_dir)):
            raise ValueError("cache_path, lights_path, target_path, and output_dir are required")
        if self.schema_version != JOB_SCHEMA_VERSION or self.steps < 1:
            raise ValueError("unsupported solve job or non-positive steps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"output_manifest": list(self.output_manifest)}


Job = BakeJob | TrainJob | SolveJob


def job_from_dict(payload: dict[str, Any]) -> Job:
    version = payload.get("schema_version")
    if version != JOB_SCHEMA_VERSION:
        raise ValueError(f"unsupported job schema version: {version}")
    values = dict(payload)
    values["output_manifest"] = tuple(values.get("output_manifest", ()))
    kind = values.get("kind")
    if kind == "bake":
        return BakeJob(**values)
    if kind == "train":
        return TrainJob(**values)
    if kind == "solve":
        return SolveJob(**values)
    raise ValueError(f"unknown job kind: {kind!r}")


def write_job(path: str | Path, job: Job) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_job(path: str | Path) -> Job:
    return job_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class JobProgress:
    job_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    fraction: float = 0.0
    stage: str = "Queued"
    message: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobProgress:
        return cls(**payload)


def write_progress(path: str | Path, progress: JobProgress) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(progress.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def read_progress(path: str | Path) -> JobProgress:
    return JobProgress.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
