"""High-level validation entrypoints."""

from __future__ import annotations

from pathlib import Path

from .metadata import NRPMetadata
from .path_cache import CacheValidationReport, validate_npz


def validate_cache_bundle(cache_path: str | Path, metadata_path: str | Path) -> CacheValidationReport:
    metadata = NRPMetadata.load(metadata_path)
    report = validate_npz(cache_path)
    if not report.ok:
        return report
    if metadata.resolution != (report.width, report.height):
        return CacheValidationReport(
            report.width,
            report.height,
            report.segment_count,
            (f"metadata resolution {metadata.resolution} does not match cache",),
        )
    return report

