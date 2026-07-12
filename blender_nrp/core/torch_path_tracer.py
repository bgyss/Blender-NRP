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


class TorchTriangleCaster:
    """Device-resident flat-triangle caster for worker-exported Blender meshes."""

    def __init__(self, vertices, triangles, normals, albedos, torch, device="auto"):
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        to = lambda value: torch.as_tensor(value, dtype=self.dtype, device=self.device)  # noqa: E731
        vertex_np = np.asarray(vertices, dtype=np.float64)
        triangle_np = np.asarray(triangles, dtype=np.int64)
        tri_vertices = vertex_np[triangle_np]
        tri_min = tri_vertices.min(axis=1)
        tri_max = tri_vertices.max(axis=1)
        centers = (tri_min + tri_max) * 0.5
        nodes: list[tuple] = []
        order: list[int] = []

        def build(indices: np.ndarray) -> int:
            node_index = len(nodes)
            bounds_min = tri_min[indices].min(axis=0)
            bounds_max = tri_max[indices].max(axis=0)
            nodes.append((bounds_min, bounds_max, -1, -1, 0, 0))
            if indices.size <= 8:
                start = len(order)
                order.extend(int(index) for index in indices)
                nodes[node_index] = (
                    bounds_min, bounds_max, -1, -1, start, int(indices.size)
                )
                return node_index
            axis = int(np.argmax(bounds_max - bounds_min))
            sorted_indices = indices[np.argsort(centers[indices, axis], kind="stable")]
            middle = sorted_indices.size // 2
            left = build(sorted_indices[:middle])
            right = build(sorted_indices[middle:])
            nodes[node_index] = (bounds_min, bounds_max, left, right, 0, 0)
            return node_index

        build(np.arange(triangle_np.shape[0], dtype=np.int64))
        triangle_np = triangle_np[np.asarray(order, dtype=np.int64)]
        self.v0 = to(vertex_np[triangle_np[:, 0]])
        self.e1 = to(vertex_np[triangle_np[:, 1]] - vertex_np[triangle_np[:, 0]])
        self.e2 = to(vertex_np[triangle_np[:, 2]] - vertex_np[triangle_np[:, 0]])
        order_np = np.asarray(order, dtype=np.int64)
        self.normals = to(np.asarray(normals, dtype=np.float64)[order_np])
        self.albedos = to(np.asarray(albedos, dtype=np.float64)[order_np])
        self.node_min = to(np.asarray([node[0] for node in nodes]))
        self.node_max = to(np.asarray([node[1] for node in nodes]))
        self.node_left = torch.as_tensor(
            [node[2] for node in nodes], dtype=torch.long, device=self.device
        )
        self.node_right = torch.as_tensor(
            [node[3] for node in nodes], dtype=torch.long, device=self.device
        )
        self.node_start = torch.as_tensor(
            [node[4] for node in nodes], dtype=torch.long, device=self.device
        )
        self.node_count = torch.as_tensor(
            [node[5] for node in nodes], dtype=torch.long, device=self.device
        )
        self.bvh_node_count = len(nodes)

    def cast(self, origins, dirs):
        torch = self.torch
        count = origins.shape[0]
        best_t = torch.full((count,), torch.inf, dtype=self.dtype, device=self.device)
        best_index = torch.zeros(count, dtype=torch.long, device=self.device)
        eps = 1e-8
        stack = torch.full((count, 64), -1, dtype=torch.long, device=self.device)
        top = torch.ones(count, dtype=torch.long, device=self.device)
        stack[:, 0] = 0

        def box_hit(rows, node_ids):
            inv = torch.where(dirs[rows].abs() > eps, 1.0 / dirs[rows], torch.inf)
            t0 = (self.node_min[node_ids] - origins[rows]) * inv
            t1 = (self.node_max[node_ids] - origins[rows]) * inv
            near = torch.minimum(t0, t1).amax(dim=1)
            far = torch.maximum(t0, t1).amin(dim=1)
            return (far >= near.clamp_min(0.0)) & (near < best_t[rows])

        def intersect_leaf(rows, start, stop):
            e1, e2, v0 = self.e1[start:stop], self.e2[start:stop], self.v0[start:stop]
            pvec = torch.linalg.cross(dirs[rows, None, :], e2[None, :, :])
            det = (e1[None, :, :] * pvec).sum(dim=2)
            inv_det = torch.where(det.abs() > eps, 1.0 / det, torch.zeros_like(det))
            tvec = origins[rows, None, :] - v0[None, :, :]
            u = (tvec * pvec).sum(dim=2) * inv_det
            qvec = torch.linalg.cross(tvec, e1[None, :, :])
            v = (dirs[rows, None, :] * qvec).sum(dim=2) * inv_det
            t = (e2[None, :, :] * qvec).sum(dim=2) * inv_det
            valid = (
                (det.abs() > eps) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0)
                & (u + v <= 1.0) & (t > eps)
            )
            t = torch.where(valid, t, torch.inf)
            chunk_t, chunk_index = t.min(dim=1)
            improve = chunk_t < best_t[rows]
            selected_rows = rows[improve]
            best_t[selected_rows] = chunk_t[improve]
            best_index[selected_rows] = chunk_index[improve] + start

        while bool((top > 0).any()):
            rows = torch.nonzero(top > 0, as_tuple=False).squeeze(1)
            top[rows] -= 1
            node_ids = stack[rows, top[rows]]
            for node_id in torch.unique(node_ids).tolist():
                node_rows = rows[node_ids == node_id]
                node_tensor = torch.full_like(node_rows, int(node_id))
                visible = box_hit(node_rows, node_tensor)
                node_rows = node_rows[visible]
                if not node_rows.numel():
                    continue
                start = int(self.node_start[node_id])
                leaf_count = int(self.node_count[node_id])
                if leaf_count:
                    intersect_leaf(node_rows, start, start + leaf_count)
                    continue
                children = (int(self.node_left[node_id]), int(self.node_right[node_id]))
                if int(top[node_rows].max()) + len(children) >= stack.shape[1]:
                    raise RuntimeError("torch BVH traversal stack overflow")
                for child in children:
                    stack[node_rows, top[node_rows]] = child
                    top[node_rows] += 1
        hit = torch.isfinite(best_t)
        position = origins + torch.where(hit, best_t, torch.zeros_like(best_t))[:, None] * dirs
        normal = self.normals[best_index]
        normal = torch.where((normal * dirs).sum(dim=1, keepdim=True) > 0.0, -normal, normal)
        albedo = self.albedos[best_index]
        normal = torch.where(hit[:, None], normal, torch.zeros_like(normal))
        albedo = torch.where(hit[:, None], albedo, torch.zeros_like(albedo))
        return hit, best_t, position, normal, albedo


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


def trace_mesh_paths(
    caster: TorchTriangleCaster,
    cam_origin: np.ndarray,
    corners: dict[str, np.ndarray],
    width: int,
    height: int,
    *,
    paths_per_pixel: int,
    max_bounces: int,
    seed: int = 0,
) -> TraceResult:
    """Trace a triangle scene in torch wavefront batches."""
    import torch

    rng = np.random.default_rng(seed)
    dirs_np = np.concatenate(
        [
            generate_camera_dirs(
                corners, width, height, row, paths_per_pixel=paths_per_pixel, rng=rng
            )
            for row in range(height)
        ],
        axis=0,
    ).reshape(-1, 3)
    torch.manual_seed(seed)
    device, dtype = caster.device, caster.dtype
    pixels = torch.arange(width * height, device=device).repeat_interleave(paths_per_pixel)
    origins = torch.as_tensor(cam_origin, dtype=dtype, device=device).repeat(pixels.numel(), 1)
    dirs = torch.as_tensor(dirs_np, dtype=dtype, device=device)
    throughput = torch.ones((pixels.numel(), 3), dtype=dtype, device=device)
    alive = torch.ones(pixels.numel(), dtype=torch.bool, device=device)
    segments: list[tuple] = []
    position = torch.zeros((height * width, 3), dtype=dtype, device=device)
    normal = torch.zeros_like(position)
    albedo = torch.zeros_like(position)
    depth = torch.zeros(height * width, dtype=dtype, device=device)
    first = torch.arange(width * height, device=device) * paths_per_pixel
    for bounce in range(max_bounces):
        rows = torch.nonzero(alive, as_tuple=False).squeeze(1)
        if not rows.numel():
            break
        hit, t, hit_pos, hit_normal, hit_albedo = caster.cast(origins[rows], dirs[rows])
        segments.append(
            (
                pixels[rows], origins[rows], dirs[rows],
                torch.where(hit, t, torch.full_like(t, torch.inf)), throughput[rows],
            )
        )
        if bounce == 0:
            first_rows = torch.isin(rows, first)
            slots = rows[first_rows] // paths_per_pixel
            valid = hit[first_rows]
            position[slots[valid]] = hit_pos[first_rows][valid]
            normal[slots[valid]] = hit_normal[first_rows][valid]
            albedo[slots[valid]] = hit_albedo[first_rows][valid]
            depth[slots[valid]] = t[first_rows][valid]
        escaped = rows[~hit]
        alive[escaped] = False
        if bounce + 1 < max_bounces and hit.any():
            hit_rows = rows[hit]
            origins[hit_rows] = hit_pos[hit] + hit_normal[hit] * 1e-4
            dirs[hit_rows] = _sample_hemisphere(hit_normal[hit], torch)
            throughput[hit_rows] *= hit_albedo[hit].clamp(0.0, 1.0)
        else:
            alive[rows[hit]] = False

    def cpu(index):
        return torch.cat([part[index] for part in segments]).detach().cpu().numpy()

    return TraceResult(
        n_paths=np.full(width * height, paths_per_pixel, dtype=np.int64),
        seg_pixel=cpu(0).astype(np.int64),
        seg_origin=cpu(1).astype(np.float64),
        seg_dir=cpu(2).astype(np.float64),
        seg_tmax=cpu(3).astype(np.float64),
        seg_throughput=cpu(4).astype(np.float64),
        albedo=albedo.reshape(height, width, 3).cpu().numpy(),
        normal=normal.reshape(height, width, 3).cpu().numpy(),
        depth=depth.reshape(height, width).cpu().numpy(),
        position=position.reshape(height, width, 3).cpu().numpy(),
    )
