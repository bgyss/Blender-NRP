"""Shared bake-output writer: cache npz, metadata, previews, bake_report.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.images import depth_to_rgb, normal_to_rgb, write_png_rgb
from ..core.metadata import NRPMetadata
from ..core.path_cache import SCHEMA_VERSION, save_arrays, validate_arrays
from .interface import BakeResult, BakeSettings


def write_bake_outputs(
    arrays: dict[str, np.ndarray],
    settings: BakeSettings,
    *,
    camera_id: str,
    backend_id: str,
    backend_version: str,
    approximation_limits: list[str],
    warnings: tuple[str, ...] = (),
    blender_file_name: str | None = None,
    packed: bool = False,
    extra_report: dict | None = None,
) -> BakeResult:
    output_dir = Path(settings.output_dir) / settings.scene_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "path_cache.npz"
    save_arrays(cache_path, arrays, width=settings.width, height=settings.height, packed=packed)

    finite_positions = arrays["position"].reshape(-1, 3)
    valid = arrays["n_paths"] > 0
    if np.any(valid):
        bbox_min = finite_positions[valid].min(axis=0).astype(float).tolist()
        bbox_max = finite_positions[valid].max(axis=0).astype(float).tolist()
    else:
        bbox_min = [0.0, 0.0, 0.0]
        bbox_max = [0.0, 0.0, 0.0]
    metadata = NRPMetadata(
        scene_id=settings.scene_id,
        camera_id=camera_id,
        resolution=(settings.width, settings.height),
        light_type="sphere",
        aux_features=["albedo", "normal", "depth"],
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        model_width=settings.model_width,
        model_depth=settings.model_depth,
    )
    metadata_path = output_dir / "metadata.json"
    metadata.save(metadata_path)

    preview_paths = {
        "albedo": output_dir / "preview_albedo.png",
        "normal": output_dir / "preview_normal.png",
        "depth": output_dir / "preview_depth.png",
    }
    write_png_rgb(preview_paths["albedo"], arrays["albedo"])
    write_png_rgb(preview_paths["normal"], normal_to_rgb(arrays["normal"]))
    write_png_rgb(preview_paths["depth"], depth_to_rgb(arrays["depth"]))

    validation = validate_arrays(arrays)
    bake_report = {
        "ok": validation.ok,
        "scene_id": settings.scene_id,
        "camera_id": camera_id,
        "backend": backend_id,
        "backend_version": backend_version,
        "cache_schema_version": SCHEMA_VERSION,
        "cache_layout": "packed" if packed else "default",
        "throughput_normalization": "n_paths",
        "resolution": [settings.width, settings.height],
        "segment_count": validation.segment_count,
        "warnings": list(warnings),
        "validation_errors": list(validation.errors),
        "approximation_limits": list(approximation_limits),
    }
    if blender_file_name:
        bake_report["blender_file_name"] = blender_file_name
    if extra_report:
        bake_report.update(extra_report)
    bake_report_path = output_dir / "bake_report.json"
    bake_report_path.write_text(
        json.dumps(bake_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BakeResult(
        output_dir,
        cache_path,
        metadata_path,
        bake_report_path,
        preview_paths,
        warnings,
    )
