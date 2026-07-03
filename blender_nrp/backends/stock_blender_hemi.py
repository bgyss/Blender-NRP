"""Stock Blender hemisphere path-cache backend."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._output import write_bake_outputs
from .interface import BakeResult, BakeSettings

id = "stock_blender_hemi"
label = "Stock Blender Hemisphere"
BACKEND_VERSION = "1.0"
CACHE_SCHEMA_VERSION = 2


def _hemisphere_dirs_batch(normals: np.ndarray, count: int, seeds: np.ndarray) -> np.ndarray:
    """Deterministic normal-oriented hemisphere directions for a batch of pixels.

    normals (P, 3), seeds (P,) per-pixel integer seeds. Returns (P, count, 3) unit dirs.
    """
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (normals.shape[0], 1))
    flip = np.abs(normals[:, 2]) > 0.95
    up[flip] = [1.0, 0.0, 0.0]
    tangent = np.cross(up, normals)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    bitangent = np.cross(normals, tangent)

    index = np.arange(count, dtype=np.float64)
    u = (index + 0.5) / count
    offset = (seeds % 997).astype(np.float64) / 997.0
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = (index[None, :] + offset[:, None]) * golden
    r = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    local = (
        tangent[:, None, :] * (np.cos(phi) * r[None, :])[..., None]
        + bitangent[:, None, :] * (np.sin(phi) * r[None, :])[..., None]
        + normals[:, None, :] * u[None, :, None]
    )
    local /= np.maximum(np.linalg.norm(local, axis=2, keepdims=True), 1e-8)
    return local


def _write_outputs(
    arrays: dict[str, np.ndarray],
    settings: BakeSettings,
    *,
    camera_id: str,
    warnings: tuple[str, ...],
    blender_file_name: str | None,
) -> BakeResult:
    return write_bake_outputs(
        arrays,
        settings,
        camera_id=camera_id,
        backend_id=id,
        backend_version=BACKEND_VERSION,
        approximation_limits=[
            "Uses one first-hit camera ray per pixel.",
            (
                "Stores deterministic normal-oriented hemisphere spokes, "
                "not true Cycles path vertices."
            ),
            "Diffuse albedo is approximated from Blender material viewport/base color.",
        ],
        warnings=warnings,
        blender_file_name=blender_file_name,
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
    count = settings.segment_count
    n_paths = np.zeros(pixels, dtype=np.int64)
    flat_pos = position.reshape(-1, 3).astype(np.float64)
    flat_normal = normal.reshape(-1, 3).astype(np.float64)
    flat_albedo = albedo.reshape(-1, 3).astype(np.float64)

    hit_idx = np.flatnonzero(hit_mask)
    n_paths[hit_idx] = count
    hit_normals = flat_normal[hit_idx]
    dirs = _hemisphere_dirs_batch(hit_normals, count, settings.seed + hit_idx)
    # Throughput carries only albedo * cosine; per-pixel averaging is n_paths'
    # job at gather time (matches the nrp reference GATHERLIGHT normalization).
    cosine = np.maximum(np.einsum("pkj,pj->pk", dirs, hit_normals), 0.0)
    origins = flat_pos[hit_idx] + hit_normals * 1e-4
    throughput = flat_albedo[hit_idx, None, :] * cosine[..., None]
    return {
        "n_paths": n_paths,
        "seg_pixel": np.repeat(hit_idx, count).astype(np.int64),
        "seg_origin": np.repeat(origins, count, axis=0).astype(np.float32),
        "seg_dir": dirs.reshape((-1, 3)).astype(np.float32),
        "seg_tmax": np.full(hit_idx.size * count, settings.max_segment_distance, dtype=np.float32),
        "seg_throughput": throughput.reshape((-1, 3)).astype(np.float32),
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
