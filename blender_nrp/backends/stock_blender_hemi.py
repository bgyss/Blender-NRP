"""Stock Blender hemisphere path-cache backend."""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

import numpy as np

from blender_nrp.core.images import depth_to_rgb, normal_to_rgb, write_png_rgb
from blender_nrp.core.metadata import NRPMetadata
from blender_nrp.core.path_cache import validate_arrays

from .interface import BakeResult, BakeSettings

id = "stock_blender_hemi"
label = "Stock Blender Hemisphere"
SCHEMA_VERSION = "1.0"


def _hemisphere_dirs(normal: np.ndarray, count: int, seed: int) -> np.ndarray:
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(normal, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tangent = np.cross(up, normal)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    bitangent = np.cross(normal, tangent)
    dirs = []
    offset = (seed % 997) / 997.0
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        u = (index + 0.5) / count
        phi = (index + offset) * golden
        r = math.sqrt(max(0.0, 1.0 - u * u))
        local = tangent * (math.cos(phi) * r) + bitangent * (math.sin(phi) * r) + normal * u
        dirs.append(local / max(float(np.linalg.norm(local)), 1e-8))
    return np.asarray(dirs, dtype=np.float32)


def _write_outputs(
    arrays: dict[str, np.ndarray],
    settings: BakeSettings,
    *,
    camera_id: str,
    warnings: tuple[str, ...],
    blender_file_name: str | None,
) -> BakeResult:
    output_dir = Path(settings.output_dir) / settings.scene_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "path_cache.npz"
    np.savez_compressed(cache_path, **arrays)

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
        "backend": id,
        "backend_version": SCHEMA_VERSION,
        "cache_schema_version": SCHEMA_VERSION,
        "resolution": [settings.width, settings.height],
        "segment_count": validation.segment_count,
        "warnings": list(warnings),
        "validation_errors": list(validation.errors),
        "approximation_limits": [
            "Uses one first-hit camera ray per pixel.",
            (
                "Stores deterministic normal-oriented hemisphere spokes, "
                "not true Cycles path vertices."
            ),
            "Diffuse albedo is approximated from Blender material viewport/base color.",
        ],
    }
    if blender_file_name:
        bake_report["blender_file_name"] = blender_file_name
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


def _synthetic_arrays(settings: BakeSettings) -> dict[str, np.ndarray]:
    height, width = settings.height, settings.width
    yy, xx = np.mgrid[0:height, 0:width]
    u = (xx + 0.5) / width * 2.0 - 1.0
    v = (yy + 0.5) / height * 2.0 - 1.0
    position = np.stack([u, -np.ones_like(u), 1.0 - v], axis=2).astype(np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    normal[..., 1] = 1.0
    albedo = np.stack(
        [0.55 + 0.35 * (u > 0), 0.45 + 0.25 * (v > 0), np.full_like(u, 0.42)],
        axis=2,
    ).astype(np.float32)
    depth = np.sqrt(np.sum(position * position, axis=2)).astype(np.float32)
    hit_mask = np.ones(height * width, dtype=bool)
    return _segments_from_hits(position, normal, albedo, depth, hit_mask, settings)


def _segments_from_hits(
    position: np.ndarray,
    normal: np.ndarray,
    albedo: np.ndarray,
    depth: np.ndarray,
    hit_mask: np.ndarray,
    settings: BakeSettings,
) -> dict[str, np.ndarray]:
    pixels = settings.width * settings.height
    n_paths = np.zeros(pixels, dtype=np.int64)
    seg_pixel: list[int] = []
    seg_origin: list[np.ndarray] = []
    seg_dir: list[np.ndarray] = []
    seg_tmax: list[float] = []
    seg_throughput: list[np.ndarray] = []
    flat_pos = position.reshape(-1, 3)
    flat_normal = normal.reshape(-1, 3)
    flat_albedo = albedo.reshape(-1, 3)
    for pixel in range(pixels):
        if not hit_mask[pixel]:
            continue
        dirs = _hemisphere_dirs(flat_normal[pixel], settings.segment_count, settings.seed + pixel)
        n_paths[pixel] = settings.segment_count
        for direction in dirs:
            cosine = max(float(np.dot(direction, flat_normal[pixel])), 0.0)
            seg_pixel.append(pixel)
            seg_origin.append(flat_pos[pixel] + flat_normal[pixel] * 1e-4)
            seg_dir.append(direction)
            seg_tmax.append(settings.max_segment_distance)
            seg_throughput.append(flat_albedo[pixel] * (cosine / max(settings.segment_count, 1)))
    return {
        "n_paths": n_paths,
        "seg_pixel": np.asarray(seg_pixel, dtype=np.int64),
        "seg_origin": np.asarray(seg_origin, dtype=np.float32).reshape((-1, 3)),
        "seg_dir": np.asarray(seg_dir, dtype=np.float32).reshape((-1, 3)),
        "seg_tmax": np.asarray(seg_tmax, dtype=np.float32),
        "seg_throughput": np.asarray(seg_throughput, dtype=np.float32).reshape((-1, 3)),
        "albedo": albedo.astype(np.float32),
        "normal": normal.astype(np.float32),
        "depth": depth.astype(np.float32),
        "position": position.astype(np.float32),
    }


def _material_color(obj: Any) -> np.ndarray:
    if getattr(obj, "active_material", None) is not None:
        material = obj.active_material
        color = getattr(material, "diffuse_color", None)
        if color is not None:
            return np.asarray(color[:3], dtype=np.float32)
    return np.array([0.8, 0.8, 0.8], dtype=np.float32)


def _blender_arrays(
    context: Any,
    settings: BakeSettings,
) -> tuple[dict[str, np.ndarray], str, str | None, tuple[str, ...]]:
    import bpy
    from mathutils import Vector

    scene = context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("No camera selected for Blender-NRP bake")
    depsgraph = context.evaluated_depsgraph_get()
    frame = scene.frame_current
    scene.frame_set(frame)
    height, width = settings.height, settings.width
    position = np.zeros((height, width, 3), dtype=np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    albedo = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float32)
    hit_mask = np.zeros(width * height, dtype=bool)

    frame_corners = camera.data.view_frame(scene=scene)
    top_right, bottom_right, bottom_left, top_left = frame_corners
    origin = camera.matrix_world.translation
    for y in range(height):
        ty = (y + 0.5) / height
        left = top_left.lerp(bottom_left, ty)
        right = top_right.lerp(bottom_right, ty)
        for x in range(width):
            tx = (x + 0.5) / width
            target = left.lerp(right, tx)
            direction = (camera.matrix_world.to_3x3() @ target).normalized()
            ok, location, hit_normal, _index, obj, _matrix = scene.ray_cast(
                depsgraph,
                origin,
                direction,
            )
            pixel = y * width + x
            if not ok:
                continue
            hit_mask[pixel] = True
            position[y, x] = np.asarray(location, dtype=np.float32)
            world_normal = Vector(hit_normal).normalized()
            normal[y, x] = np.asarray(world_normal, dtype=np.float32)
            depth[y, x] = float((location - origin).length)
            albedo[y, x] = _material_color(obj)

    warnings = []
    if not np.any(hit_mask):
        warnings.append("No camera rays hit scene geometry")
    arrays = _segments_from_hits(position, normal, albedo, depth, hit_mask, settings)
    return arrays, camera.name, bpy.data.filepath or None, tuple(warnings)


def bake(context: Any, settings: BakeSettings) -> BakeResult:
    """Bake a cache using Blender ray casts or a deterministic pure-Python fixture."""
    try:
        import bpy  # noqa: F401
    except ModuleNotFoundError:
        arrays = _synthetic_arrays(settings)
        return _write_outputs(
            arrays,
            settings,
            camera_id=settings.camera_id,
            warnings=("Synthetic fallback cache generated outside Blender.",),
            blender_file_name=None,
        )
    arrays, camera_id, blender_file_name, warnings = _blender_arrays(context, settings)
    return _write_outputs(
        arrays,
        settings,
        camera_id=camera_id,
        warnings=warnings,
        blender_file_name=blender_file_name,
    )
