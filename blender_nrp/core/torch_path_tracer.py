"""Batched device-side analytic-room path tracing for V3 worker verification.

This is deliberately separate from the Blender ``scene.ray_cast`` fallback: the
analytic room has a compact torch intersection kernel, so CI can prove the
wavefront segment contract on CPU while CUDA/MPS workers exercise the same kernel.
Arbitrary Blender meshes still use the V2 ray-cast backend until a scene-BVH
adapter is available; callers must record that distinction in their bake report.
"""

from __future__ import annotations

import numpy as np

from .path_tracer import AnalyticRoomCaster, TraceResult, generate_camera_dirs


def _sample_hemisphere(normals, torch):
    u1 = torch.rand((normals.shape[0],), dtype=normals.dtype, device=normals.device)
    u2 = torch.rand((normals.shape[0],), dtype=normals.dtype, device=normals.device)
    r = torch.sqrt(u1)
    phi = 2.0 * torch.pi * u2
    local = torch.stack((r * torch.cos(phi), r * torch.sin(phi), torch.sqrt(1.0 - u1)), dim=1)
    up = torch.zeros_like(normals)
    up[:, 2] = 1.0
    flip = normals[:, 2].abs() > 0.95
    up[flip] = torch.tensor((1.0, 0.0, 0.0), dtype=normals.dtype, device=normals.device)
    tangent = torch.linalg.cross(up, normals)
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp_min(1e-12)
    bitangent = torch.linalg.cross(normals, tangent)
    return tangent * local[:, :1] + bitangent * local[:, 1:2] + normals * local[:, 2:]


def _cast_analytic(origins, dirs, caster: AnalyticRoomCaster, torch):
    """Torch nearest-hit equivalent of :meth:`AnalyticRoomCaster.cast`."""
    lo = torch.as_tensor(caster.room_min, dtype=origins.dtype, device=origins.device)
    hi = torch.as_tensor(caster.room_max, dtype=origins.dtype, device=origins.device)
    center = torch.as_tensor(caster.sphere_center, dtype=origins.dtype, device=origins.device)
    oc = origins - center
    b = (oc * dirs).sum(dim=1)
    c = (oc * oc).sum(dim=1) - caster.sphere_radius**2
    disc = b * b - c
    sq = torch.sqrt(disc.clamp_min(0.0))
    t0, t1 = -b - sq, -b + sq
    sphere_t = torch.where(t0 > 1e-9, t0, t1)
    sphere_t = torch.where((disc > 0.0) & (sphere_t > 1e-9), sphere_t, torch.inf)

    t_lo, t_hi = (lo - origins) / dirs, (hi - origins) / dirs
    exits = torch.where(dirs > 0, t_hi, torch.where(dirs < 0, t_lo, torch.inf))
    wall_t, axis = exits.min(dim=1)
    wall_face = dirs.gather(1, axis[:, None]).squeeze(1) > 0
    open_axis = {"x": 0, "y": 1, "z": 2}.get((caster.open_wall or " ")[0], -1)
    open_max = (caster.open_wall or "").endswith("max")
    is_open = (axis == open_axis) & (wall_face == open_max)
    wall_t = torch.where(is_open | (wall_t <= 1e-9), torch.inf, wall_t)

    t = torch.minimum(sphere_t, wall_t)
    hit = torch.isfinite(t)
    position = origins + torch.where(hit, t, torch.zeros_like(t))[:, None] * dirs
    normal = torch.zeros_like(origins)
    albedo = torch.zeros_like(origins)
    sphere_first = hit & (sphere_t <= wall_t)
    if sphere_first.any():
        sn = position[sphere_first] - center
        normal[sphere_first] = sn / torch.linalg.vector_norm(
            sn, dim=1, keepdim=True
        ).clamp_min(1e-12)
        albedo[sphere_first] = torch.as_tensor(
            caster.sphere_albedo, dtype=origins.dtype, device=origins.device
        )
    wall_first = hit & ~sphere_first
    if wall_first.any():
        rows = torch.nonzero(wall_first, as_tuple=False).squeeze(1)
        normal[rows, axis[rows]] = torch.where(
            wall_face[rows], -torch.ones_like(t[rows]), torch.ones_like(t[rows])
        )
        wall_albedo = torch.as_tensor(
            caster.wall_albedo, dtype=origins.dtype, device=origins.device
        )
        floor_albedo = torch.as_tensor(
            caster.floor_albedo, dtype=origins.dtype, device=origins.device
        )
        albedo[rows] = wall_albedo
        is_floor = (axis[rows] == 2) & ~wall_face[rows]
        albedo[rows[is_floor]] = floor_albedo
    return hit, t, position, normal, albedo


def trace_analytic_room_paths(
    cam_origin: np.ndarray,
    corners: dict[str, np.ndarray],
    width: int,
    height: int,
    *,
    paths_per_pixel: int,
    max_bounces: int,
    seed: int = 0,
    device: str = "cpu",
    caster: AnalyticRoomCaster | None = None,
) -> TraceResult:
    """Trace the analytic fixture with device-resident wavefront intersections."""
    import torch

    caster = caster or AnalyticRoomCaster()
    rng = np.random.default_rng(seed)
    camera_dirs = [
        generate_camera_dirs(corners, width, height, row, paths_per_pixel=paths_per_pixel, rng=rng)
        for row in range(height)
    ]
    dirs_np = np.concatenate(camera_dirs, axis=0).reshape(-1, 3)
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    dev = torch.device(device)
    dtype = torch.float64 if dev.type == "cpu" else torch.float32
    torch.manual_seed(seed)
    pixels = torch.arange(width * height, device=dev).repeat_interleave(paths_per_pixel)
    origins = torch.as_tensor(cam_origin, dtype=dtype, device=dev).repeat(pixels.numel(), 1)
    dirs = torch.as_tensor(dirs_np, dtype=dtype, device=dev)
    throughput = torch.ones((pixels.numel(), 3), dtype=dtype, device=dev)
    alive = torch.ones(pixels.numel(), dtype=torch.bool, device=dev)
    segments: list[tuple] = []
    position = torch.zeros((height * width, 3), dtype=dtype, device=dev)
    normal = torch.zeros_like(position)
    albedo = torch.zeros_like(position)
    depth = torch.zeros(height * width, dtype=dtype, device=dev)
    first = torch.arange(width * height, device=dev) * paths_per_pixel

    for bounce in range(max_bounces):
        rows = torch.nonzero(alive, as_tuple=False).squeeze(1)
        if not rows.numel():
            break
        hit, t, hit_pos, hit_normal, hit_albedo = _cast_analytic(
            origins[rows], dirs[rows], caster, torch
        )
        tmax = torch.where(hit, t, torch.full_like(t, torch.inf))
        segments.append((pixels[rows], origins[rows], dirs[rows], tmax, throughput[rows]))
        if bounce == 0:
            is_first = torch.isin(rows, first)
            slots = rows[is_first] // paths_per_pixel
            first_hits = hit[is_first]
            position[slots[first_hits]] = hit_pos[is_first][first_hits]
            normal[slots[first_hits]] = hit_normal[is_first][first_hits]
            albedo[slots[first_hits]] = hit_albedo[is_first][first_hits]
            depth[slots[first_hits]] = t[is_first][first_hits]
        escaped = rows[~hit]
        alive[escaped] = False
        if bounce + 1 < max_bounces and hit.any():
            hit_rows = rows[hit]
            new_dirs = _sample_hemisphere(hit_normal[hit], torch)
            origins[hit_rows] = hit_pos[hit] + hit_normal[hit] * 1e-4
            dirs[hit_rows] = new_dirs
            throughput[hit_rows] *= hit_albedo[hit].clamp(0.0, 1.0)
        else:
            alive[rows[hit]] = False

    def cpu(parts, index):
        return torch.cat([part[index] for part in segments]).detach().cpu().numpy()

    return TraceResult(
        n_paths=np.full(width * height, paths_per_pixel, dtype=np.int64),
        seg_pixel=cpu(segments, 0).astype(np.int64),
        seg_origin=cpu(segments, 1).astype(np.float64),
        seg_dir=cpu(segments, 2).astype(np.float64),
        seg_tmax=cpu(segments, 3).astype(np.float64),
        seg_throughput=cpu(segments, 4).astype(np.float64),
        albedo=albedo.reshape(height, width, 3).detach().cpu().numpy(),
        normal=normal.reshape(height, width, 3).detach().cpu().numpy(),
        depth=depth.reshape(height, width).detach().cpu().numpy(),
        position=position.reshape(height, width, 3).detach().cpu().numpy(),
    )
