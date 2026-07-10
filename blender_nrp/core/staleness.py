"""Deterministic settings/content hashes used by the one-button workflow."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(value: Any) -> str:
    """Hash JSON-compatible data independent of mapping insertion order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_stale(previous_settings: str, previous_scene: str, settings: Any, scene: Any) -> bool:
    return previous_settings != stable_hash(settings) or previous_scene != stable_hash(scene)
