"""Versioned, JSON-safe light-rig snapshots for live A/B lighting work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lights import LightRig

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RigSnapshot:
    name: str
    rig: LightRig
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("snapshot name is required")
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported rig snapshot schema: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "name": self.name, "rig": self.rig.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RigSnapshot:
        return cls(
            name=str(value["name"]),
            rig=LightRig.from_dict(value["rig"]),
            schema_version=int(value.get("schema_version", 0)),
        )


def replace_snapshot(snapshots: list[RigSnapshot], snapshot: RigSnapshot) -> list[RigSnapshot]:
    """Replace same-named snapshot in order, or append a new named version."""
    result = list(snapshots)
    for index, existing in enumerate(result):
        if existing.name == snapshot.name:
            result[index] = snapshot
            return result
    result.append(snapshot)
    return result


def snapshots_to_json(snapshots: list[RigSnapshot]) -> list[dict[str, Any]]:
    return [snapshot.to_dict() for snapshot in snapshots]


def snapshots_from_json(values: list[dict[str, Any]]) -> list[RigSnapshot]:
    return [RigSnapshot.from_dict(value) for value in values]
