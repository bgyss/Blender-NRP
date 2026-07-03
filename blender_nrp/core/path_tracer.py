"""Monte Carlo multi-bounce path capture, independent of the ray-cast provider.

This is the light-agnostic tracer behind the `cycles_capture` backend: per-pixel
camera paths with cosine-weighted diffuse bounces, true segment origins/directions/
lengths, accumulated BSDF throughput, and escape segments recorded with
t_max = inf. The actual intersection queries go through the `RayCaster` protocol so
the same tracer runs against Blender's `scene.ray_cast` (see
`backends/cycles_capture.py`) or the analytic test room below (pure numpy, used by
pytest and the cross-repo round-trip script).

Semantics match the nrp cache contract: `seg_throughput` is the path throughput
accumulated *before* each segment (camera segments carry 1; after a Lambertian
bounce sampled cosine-weighted, throughput multiplies by albedo exactly since
f·cosθ/pdf = albedo), and per-pixel averaging is the gather's job via `n_paths`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RayCaster(Protocol):
    """Batched nearest-hit query against static scene geometry."""

    def cast(
        self, origins: np.ndarray, dirs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(N,3) origins + unit dirs -> (hit (N,) bool, t (N,), position (N,3),
        normal (N,3) unit outward, albedo (N,3))."""
        ...


@dataclass
class TraceResult:
    """Flattened segment arrays plus first-hit G-buffer from the tracer itself."""

    n_paths: np.ndarray
    seg_pixel: np.ndarray
    seg_origin: np.ndarray
    seg_dir: np.ndarray
    seg_tmax: np.ndarray
    seg_throughput: np.ndarray
    albedo: np.ndarray
    normal: np.ndarray
    depth: np.ndarray
    position: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "n_paths": self.n_paths,
            "seg_pixel": self.seg_pixel,
            "seg_origin": self.seg_origin,
            "seg_dir": self.seg_dir,
            "seg_tmax": self.seg_tmax,
            "seg_throughput": self.seg_throughput,
            "albedo": self.albedo,
            "normal": self.normal,
            "depth": self.depth,
            "position": self.position,
        }


def cosine_sample_hemisphere(normals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N,3) cosine-weighted directions about (N,3) unit normals."""
    n = normals.shape[0]
    u1 = rng.random(n)
    u2 = rng.random(n)
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(1.0 - u1, 0.0))], axis=1)

    up = np.tile([0.0, 0.0, 1.0], (n, 1))
    flip = np.abs(normals[:, 2]) > 0.95
    up[flip] = [1.0, 0.0, 0.0]
    tangent = np.cross(up, normals)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals, tangent)
    dirs = (
        tangent * local[:, 0:1] + bitangent * local[:, 1:2] + normals * local[:, 2:3]
    )
    return dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)


@dataclass
class _SegmentBuffer:
    pixel: list
    origin: list
    dir: list
    tmax: list
    throughput: list

    def append(self, pixel, origin, dirs, tmax, throughput) -> None:
        self.pixel.append(pixel)
        self.origin.append(origin)
        self.dir.append(dirs)
        self.tmax.append(tmax)
        self.throughput.append(throughput)


def trace_pixel_block(
    caster: RayCaster,
    pixel_indices: np.ndarray,
    cam_origin: np.ndarray,
    cam_dirs: np.ndarray,
    *,
    paths_per_pixel: int,
    max_bounces: int,
    rng: np.random.Generator,
    offset_epsilon: float = 1e-4,
) -> tuple[_SegmentBuffer, dict[str, np.ndarray]]:
    """Trace all paths for one block of pixels.

    pixel_indices (P,) flat pixel ids; cam_dirs (P, p_per_px, 3) jittered primary
    directions. Returns the block's segments plus per-pixel first-hit G-buffer
    (from the first path of each pixel).
    """
    n_px = pixel_indices.shape[0]
    n_rays = n_px * paths_per_pixel
    pixels = np.repeat(pixel_indices, paths_per_pixel)
    origins = np.tile(cam_origin, (n_rays, 1))
    dirs = cam_dirs.reshape(n_rays, 3)
    throughput = np.ones((n_rays, 3), dtype=np.float64)
    alive = np.ones(n_rays, dtype=bool)

    buffer = _SegmentBuffer([], [], [], [], [])
    gbuffer = {
        "position": np.zeros((n_px, 3), dtype=np.float64),
        "normal": np.zeros((n_px, 3), dtype=np.float64),
        "albedo": np.zeros((n_px, 3), dtype=np.float64),
        "depth": np.zeros(n_px, dtype=np.float64),
        "hit": np.zeros(n_px, dtype=bool),
    }
    first_path_rows = np.arange(n_px) * paths_per_pixel

    for bounce in range(max_bounces):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        hit, t, position, normal, albedo = caster.cast(origins[idx], dirs[idx])

        # Escape segments: recorded with t_max = inf, then the path ends.
        esc = ~hit
        if np.any(esc):
            rows = idx[esc]
            buffer.append(
                pixels[rows],
                origins[rows],
                dirs[rows],
                np.full(rows.size, np.inf),
                throughput[rows].copy(),
            )
            alive[rows] = False

        if np.any(hit):
            rows = idx[hit]
            buffer.append(
                pixels[rows],
                origins[rows],
                dirs[rows],
                t[hit].copy(),
                throughput[rows].copy(),
            )
            if bounce == 0:
                # First-hit G-buffer from each pixel's first path.
                in_first = np.isin(rows, first_path_rows)
                # rows are block-local ray rows; map back to this block's pixel slots.
                slots = np.searchsorted(first_path_rows, rows[in_first])
                gbuffer["position"][slots] = position[hit][in_first]
                gbuffer["normal"][slots] = normal[hit][in_first]
                gbuffer["albedo"][slots] = albedo[hit][in_first]
                gbuffer["depth"][slots] = t[hit][in_first]
                gbuffer["hit"][slots] = True

            if bounce + 1 < max_bounces:
                # Lambertian bounce: cosine-weighted sample, throughput *= albedo.
                n_vec = normal[hit]
                new_dirs = cosine_sample_hemisphere(n_vec, rng)
                origins[rows] = position[hit] + n_vec * offset_epsilon
                dirs[rows] = new_dirs
                throughput[rows] *= np.clip(albedo[hit], 0.0, 1.0)
            else:
                alive[rows] = False

    return buffer, gbuffer


def generate_camera_dirs(
    corners: dict[str, np.ndarray],
    width: int,
    height: int,
    row: int,
    *,
    paths_per_pixel: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Jittered primary directions for one image row: (W, paths_per_pixel, 3).

    `corners` holds the camera frustum's world-space corner directions
    (top_left/top_right/bottom_left/bottom_right as unnormalized vectors from the
    camera origin); pixel directions bilinearly interpolate between them. The first
    path of each pixel is un-jittered (pixel center) so the G-buffer is stable.
    """
    jx = rng.random((width, paths_per_pixel))
    jy = rng.random((width, paths_per_pixel))
    jx[:, 0] = 0.5
    jy[:, 0] = 0.5
    xs = (np.arange(width)[:, None] + jx) / width
    ys = (row + jy) / height
    tl, tr = corners["top_left"], corners["top_right"]
    bl, br = corners["bottom_left"], corners["bottom_right"]
    left = tl[None, None, :] + (bl - tl)[None, None, :] * ys[..., None]
    right = tr[None, None, :] + (br - tr)[None, None, :] * ys[..., None]
    target = left + (right - left) * xs[..., None]
    return target / np.maximum(np.linalg.norm(target, axis=2, keepdims=True), 1e-12)


def trace_camera_paths(
    caster: RayCaster,
    cam_origin: np.ndarray,
    corners: dict[str, np.ndarray],
    width: int,
    height: int,
    *,
    paths_per_pixel: int,
    max_bounces: int,
    seed: int = 0,
) -> Iterator[tuple[float, TraceResult | None]]:
    """Row-chunked tracing generator.

    Yields (progress_fraction, None) after each row and finally (1.0, TraceResult).
    Chunked so a modal operator can interleave UI updates and cancellation.
    """
    rng = np.random.default_rng(seed)
    buffers: list[_SegmentBuffer] = []
    position = np.zeros((height, width, 3), dtype=np.float64)
    normal = np.zeros((height, width, 3), dtype=np.float64)
    albedo = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.zeros((height, width), dtype=np.float64)
    n_paths = np.full(height * width, paths_per_pixel, dtype=np.int64)

    for row in range(height):
        dirs = generate_camera_dirs(
            corners, width, height, row, paths_per_pixel=paths_per_pixel, rng=rng
        )
        pixel_indices = row * width + np.arange(width, dtype=np.int64)
        buffer, gbuffer = trace_pixel_block(
            caster,
            pixel_indices,
            cam_origin,
            dirs,
            paths_per_pixel=paths_per_pixel,
            max_bounces=max_bounces,
            rng=rng,
        )
        buffers.append(buffer)
        position[row] = gbuffer["position"]
        normal[row] = gbuffer["normal"]
        albedo[row] = gbuffer["albedo"]
        depth[row] = gbuffer["depth"]
        if row + 1 < height:
            yield (row + 1) / height, None

    def _concat(parts: list, dtype, columns: int | None = None) -> np.ndarray:
        if not parts:
            shape = (0,) if columns is None else (0, columns)
            return np.zeros(shape, dtype=dtype)
        return np.concatenate([np.asarray(p, dtype=dtype) for p in parts])

    result = TraceResult(
        n_paths=n_paths,
        seg_pixel=_concat([p for b in buffers for p in b.pixel], np.int64),
        seg_origin=_concat([p for b in buffers for p in b.origin], np.float64, 3),
        seg_dir=_concat([p for b in buffers for p in b.dir], np.float64, 3),
        seg_tmax=_concat([p for b in buffers for p in b.tmax], np.float64),
        seg_throughput=_concat([p for b in buffers for p in b.throughput], np.float64, 3),
        albedo=albedo,
        normal=normal,
        depth=depth,
        position=position,
    )
    yield 1.0, result


class AnalyticRoomCaster:
    """Pure-numpy test scene: an axis-aligned box room (interior) with one sphere.

    Used by pytest and the cross-repo round-trip script to exercise the tracer
    without Blender: rays start inside the room, the open +Y wall lets paths escape
    (producing real t_max = inf segments).
    """

    def __init__(
        self,
        room_min=(-2.0, -2.0, 0.0),
        room_max=(2.0, 2.0, 3.0),
        open_wall: str | None = "y_max",
        sphere_center=(0.5, 0.0, 0.8),
        sphere_radius: float = 0.6,
    ):
        self.room_min = np.asarray(room_min, dtype=np.float64)
        self.room_max = np.asarray(room_max, dtype=np.float64)
        self.open_wall = open_wall
        self.sphere_center = np.asarray(sphere_center, dtype=np.float64)
        self.sphere_radius = float(sphere_radius)
        self.wall_albedo = np.array([0.7, 0.7, 0.7])
        self.floor_albedo = np.array([0.6, 0.5, 0.4])
        self.sphere_albedo = np.array([0.8, 0.2, 0.2])

    def _sphere_hit(self, origins, dirs):
        oc = origins - self.sphere_center
        b = np.einsum("ij,ij->i", oc, dirs)
        c = np.einsum("ij,ij->i", oc, oc) - self.sphere_radius**2
        disc = b * b - c
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = -b - sq
        t1 = -b + sq
        t = np.where(t0 > 1e-9, t0, t1)
        valid = (disc > 0.0) & (t > 1e-9)
        return np.where(valid, t, np.inf)

    def _walls_hit(self, origins, dirs):
        """Nearest interior wall crossing (slab exit point), inf on the open wall."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (self.room_min - origins) / dirs
            t_hi = (self.room_max - origins) / dirs
        t_exit = np.where(dirs > 0, t_hi, np.where(dirs < 0, t_lo, np.inf))
        axis = np.argmin(t_exit, axis=1)
        t = t_exit[np.arange(origins.shape[0]), axis]
        sign_positive = dirs[np.arange(origins.shape[0]), axis] > 0
        wall = np.where(sign_positive, 1, 0)  # 1 = max face, 0 = min face
        open_axis = {"x": 0, "y": 1, "z": 2}.get((self.open_wall or "  ")[0], -1)
        open_max = (self.open_wall or "").endswith("max")
        is_open = (axis == open_axis) & (wall == (1 if open_max else 0))
        t = np.where(is_open | (t <= 1e-9), np.inf, t)
        return t, axis, wall

    def cast(self, origins, dirs):
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        n = origins.shape[0]
        t_sphere = self._sphere_hit(origins, dirs)
        t_wall, axis, wall_face = self._walls_hit(origins, dirs)
        t = np.minimum(t_sphere, t_wall)
        hit = np.isfinite(t)
        position = origins + np.where(hit, t, 0.0)[:, None] * dirs

        normal = np.zeros((n, 3), dtype=np.float64)
        albedo = np.zeros((n, 3), dtype=np.float64)
        sphere_first = hit & (t_sphere <= t_wall)
        if np.any(sphere_first):
            sn = position[sphere_first] - self.sphere_center
            sn /= np.maximum(np.linalg.norm(sn, axis=1, keepdims=True), 1e-12)
            normal[sphere_first] = sn
            albedo[sphere_first] = self.sphere_albedo
        wall_first = hit & ~sphere_first
        if np.any(wall_first):
            rows = np.flatnonzero(wall_first)
            for row in rows:
                normal_vec = np.zeros(3)
                # Interior normals point back into the room.
                normal_vec[axis[row]] = -1.0 if wall_face[row] == 1 else 1.0
                normal[row] = normal_vec
                is_floor = axis[row] == 2 and wall_face[row] == 0
                albedo[row] = self.floor_albedo if is_floor else self.wall_albedo
        t_out = np.where(hit, t, 0.0)
        return hit, t_out, position, normal, albedo
