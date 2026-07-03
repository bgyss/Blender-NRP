"""Cycles-instrumented multi-bounce path-capture backend.

Practical route per the V2 goal prompt: sampling is driven from Python — camera rays
through the fixed camera's frustum, nearest-hit queries via `scene.ray_cast` over the
evaluated depsgraph, cosine-weighted Lambertian bounces with albedo read from each
material's Principled BSDF Base Color — while the G-buffer aux (albedo/normal/depth/
position) comes from real Cycles render passes (denoising albedo + Normal/Depth/
Position) when a render is possible, falling back to the tracer's own first-hit
buffers otherwise. A true Cycles-kernel hook remains a stretch goal; the report names
every approximation.

Escape segments are recorded with t_max = inf (V1 never wrote escape segments).

Verification: with `reference_check`, the bake also renders the same scene in Cycles
with a real emissive validation sphere (all other lights disabled) and reports the
PSNR between that render and GATHERLIGHT over the captured cache — reported, never
claimed exact.

The backend is a generator (`bake_steps`) so the modal bake operator can interleave
progress updates and cancellation; `bake()` simply drains it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from blender_nrp.core.gather import gather_hdr
from blender_nrp.core.lights import SphereLight
from blender_nrp.core.path_tracer import AnalyticRoomCaster, trace_camera_paths

from ._output import write_bake_outputs
from .interface import BakeResult, BakeSettings

id = "cycles_capture"
label = "Cycles Capture"
BACKEND_VERSION = "1.0"

APPROXIMATION_LIMITS = [
    "BSDF model is Lambertian-diffuse only (cosine-weighted sampling with Principled "
    "Base Color albedo); glossy/transmissive transport is not captured.",
    "Sampling is Python-driven over scene.ray_cast, not a Cycles kernel hook; "
    "path budgets are paper-scale (preview resolutions), not production renders.",
    "No next-event estimation or MIS in the capture; agreement with Cycles is "
    "reported as PSNR, not claimed exact.",
]


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """PSNR in dB with the reference max as peak (HDR convention, matches nrp)."""
    ref = np.asarray(ref, dtype=np.float64)
    peak = float(ref.max()) if ref.size and ref.max() > 0 else 1.0
    err = float(np.mean((np.asarray(pred, dtype=np.float64) - ref) ** 2))
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak**2 / err))


# ---------------------------------------------------------------------------
# Blender adapters


class BlenderRayCaster:
    """Batched RayCaster over `scene.ray_cast` (per-ray Python loop inside)."""

    def __init__(self, scene: Any, depsgraph: Any):
        self.scene = scene
        self.depsgraph = depsgraph
        self._albedo_cache: dict[str, np.ndarray] = {}

    def _albedo(self, obj: Any) -> np.ndarray:
        key = obj.name if obj is not None else ""
        cached = self._albedo_cache.get(key)
        if cached is not None:
            return cached
        albedo = np.array([0.8, 0.8, 0.8], dtype=np.float64)
        material = getattr(obj, "active_material", None) if obj is not None else None
        if material is not None:
            found = False
            if material.use_nodes and material.node_tree is not None:
                for node in material.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        albedo = np.asarray(
                            node.inputs["Base Color"].default_value[:3], dtype=np.float64
                        )
                        found = True
                        break
            if not found:
                color = getattr(material, "diffuse_color", None)
                if color is not None:
                    albedo = np.asarray(color[:3], dtype=np.float64)
        self._albedo_cache[key] = albedo
        return albedo

    def cast(self, origins: np.ndarray, dirs: np.ndarray):
        from mathutils import Vector

        n = origins.shape[0]
        hit = np.zeros(n, dtype=bool)
        t = np.zeros(n, dtype=np.float64)
        position = np.zeros((n, 3), dtype=np.float64)
        normal = np.zeros((n, 3), dtype=np.float64)
        albedo = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            origin = Vector(origins[i])
            ok, location, hit_normal, _index, obj, _matrix = self.scene.ray_cast(
                self.depsgraph, origin, Vector(dirs[i])
            )
            if not ok:
                continue
            hit[i] = True
            position[i] = location[:]
            face_normal = Vector(hit_normal).normalized()
            # Orient the shading normal against the incoming ray.
            if face_normal.dot(Vector(dirs[i])) > 0.0:
                face_normal = -face_normal
            normal[i] = face_normal[:]
            t[i] = (location - origin).length
            albedo[i] = self._albedo(obj)
        return hit, t, position, normal, albedo


def _camera_frame(context: Any, settings: BakeSettings):
    """(origin (3,), corner-direction dict) for the active camera in world space."""
    scene = context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("No camera selected for Blender-NRP bake")
    top_right, bottom_right, bottom_left, top_left = camera.data.view_frame(scene=scene)
    rotation = camera.matrix_world.to_3x3()
    corners = {
        "top_left": np.asarray(rotation @ top_left, dtype=np.float64),
        "top_right": np.asarray(rotation @ top_right, dtype=np.float64),
        "bottom_left": np.asarray(rotation @ bottom_left, dtype=np.float64),
        "bottom_right": np.asarray(rotation @ bottom_right, dtype=np.float64),
    }
    origin = np.asarray(camera.matrix_world.translation, dtype=np.float64)
    return origin, corners, camera.name


def _read_exr_pixels(filepath: str, width: int, height: int) -> np.ndarray:
    """(H, W, 4) float pixels of a rendered EXR, top row first."""
    import bpy

    image = bpy.data.images.load(filepath)
    try:
        pixels = np.array(image.pixels[:], dtype=np.float64)
    finally:
        bpy.data.images.remove(image)
    pixels = pixels.reshape(height, width, 4)
    return pixels[::-1]  # Blender stores bottom row first


def _cycles_gbuffer(
    context: Any, settings: BakeSettings
) -> dict[str, np.ndarray] | None:
    """Render denoising-albedo/normal/depth/position passes with Cycles.

    Returns None (caller falls back to tracer G-buffer) if anything goes wrong.
    """
    import tempfile

    import bpy

    scene = context.scene
    view_layer = context.view_layer

    import glob

    # Blender 5.x moved the compositor to a scene-assigned node group and the File
    # Output node to `file_output_items`; 4.x uses scene.node_tree + file_slots.
    legacy_compositor = hasattr(scene, "node_tree")

    saved = {
        "engine": scene.render.engine,
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "res_pct": scene.render.resolution_percentage,
        "samples": None,
        "denoising_store": None,
        "pass_z": view_layer.use_pass_z,
        "pass_normal": view_layer.use_pass_normal,
        "pass_position": getattr(view_layer, "use_pass_position", None),
        "filepath": scene.render.filepath,
        "use_nodes": getattr(scene, "use_nodes", None),
        "node_group": getattr(scene, "compositing_node_group", None),
    }
    tmpdir = tempfile.mkdtemp(prefix="blender_nrp_gbuffer_")
    added_nodes: list[Any] = []
    added_group = None
    tree = None
    try:
        scene.render.engine = "CYCLES"
        saved["samples"] = scene.cycles.samples
        saved["denoising_store"] = view_layer.cycles.denoising_store_passes
        scene.cycles.samples = 16
        scene.render.resolution_x = settings.width
        scene.render.resolution_y = settings.height
        scene.render.resolution_percentage = 100
        view_layer.use_pass_z = True
        view_layer.use_pass_normal = True
        if hasattr(view_layer, "use_pass_position"):
            view_layer.use_pass_position = True
        view_layer.cycles.denoising_store_passes = True

        if legacy_compositor:
            scene.use_nodes = True
            tree = scene.node_tree
            render_node = tree.nodes.new("CompositorNodeRLayers")
            out_node = tree.nodes.new("CompositorNodeOutputFile")
            added_nodes = [render_node, out_node]
            out_node.base_path = tmpdir
            out_node.format.file_format = "OPEN_EXR"
            out_node.format.color_depth = "32"
            out_node.file_slots.clear()
        else:
            tree = bpy.data.node_groups.new("NRP GBuffer", "CompositorNodeTree")
            added_group = tree
            scene.compositing_node_group = tree
            render_node = tree.nodes.new("CompositorNodeRLayers")
            render_node.scene = scene
            out_node = tree.nodes.new("CompositorNodeOutputFile")
            out_node.directory = tmpdir
            out_node.format.media_type = "IMAGE"
            out_node.format.file_format = "OPEN_EXR"
            out_node.format.color_depth = "32"
            out_node.file_output_items.clear()

        wanted = {
            "albedo": "Denoising Albedo",
            "normal": "Normal",
            "depth": "Depth",
            "position": "Position",
        }
        available = {output.name for output in render_node.outputs}
        slots = {}
        for slot_name, pass_name in wanted.items():
            if pass_name not in available:
                continue
            if legacy_compositor:
                out_node.file_slots.new(slot_name)
            else:
                item = out_node.file_output_items.new("RGBA", slot_name)
                item.save_as_render = False
            tree.links.new(render_node.outputs[pass_name], out_node.inputs[slot_name])
            slots[slot_name] = pass_name
        if not slots:
            return None

        bpy.ops.render.render(write_still=False)

        buffers: dict[str, np.ndarray] = {}
        for slot_name in slots:
            matches = sorted(glob.glob(f"{tmpdir}/*{slot_name}*.exr"))
            if not matches:
                return None
            pixels = _read_exr_pixels(matches[-1], settings.width, settings.height)
            if slot_name == "depth":
                buffers[slot_name] = pixels[..., 0]
            else:
                buffers[slot_name] = pixels[..., :3]
        return buffers or None
    except Exception:
        return None
    finally:
        try:
            if legacy_compositor:
                if tree is not None:
                    for node in added_nodes:
                        tree.nodes.remove(node)
                scene.use_nodes = saved["use_nodes"]
            else:
                scene.compositing_node_group = saved["node_group"]
                if added_group is not None:
                    bpy.data.node_groups.remove(added_group)
            scene.render.engine = saved["engine"]
            scene.render.resolution_x = saved["res_x"]
            scene.render.resolution_y = saved["res_y"]
            scene.render.resolution_percentage = saved["res_pct"]
            scene.render.filepath = saved["filepath"]
            view_layer.use_pass_z = saved["pass_z"]
            view_layer.use_pass_normal = saved["pass_normal"]
            if saved["pass_position"] is not None:
                view_layer.use_pass_position = saved["pass_position"]
            if saved["samples"] is not None:
                scene.cycles.samples = saved["samples"]
            if saved["denoising_store"] is not None:
                view_layer.cycles.denoising_store_passes = saved["denoising_store"]
        except Exception:
            pass


def _cycles_reference_render(
    context: Any, settings: BakeSettings, light: SphereLight
) -> np.ndarray | None:
    """Cycles render of the scene lit *only* by an emissive validation sphere.

    Returns (H, W, 3) linear radiance, or None if rendering fails.
    """
    import tempfile

    import bpy

    scene = context.scene
    hidden: list[tuple[Any, bool]] = []
    sphere_obj = None
    saved = {
        "engine": scene.render.engine,
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "res_pct": scene.render.resolution_percentage,
        "filepath": scene.render.filepath,
        "file_format": scene.render.image_settings.file_format,
        "color_depth": scene.render.image_settings.color_depth,
        "samples": None,
        "world": scene.world,
    }
    tmpdir = tempfile.mkdtemp(prefix="blender_nrp_reference_")
    try:
        # Disable every existing light and emissive world background.
        for obj in scene.objects:
            if obj.type == "LIGHT":
                hidden.append((obj, obj.hide_render))
                obj.hide_render = True
        scene.world = None

        mesh = bpy.data.meshes.new("NRP_ReferenceLight")
        sphere_obj = bpy.data.objects.new("NRP_ReferenceLight", mesh)
        scene.collection.objects.link(sphere_obj)
        import bmesh

        bm = bmesh.new()
        bmesh.ops.create_uvsphere(
            bm, u_segments=32, v_segments=16, radius=light.radius
        )
        bm.to_mesh(mesh)
        bm.free()
        sphere_obj.location = light.position
        material = bpy.data.materials.new("NRP_ReferenceEmission")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*light.color, 1.0)
        emission.inputs["Strength"].default_value = light.intensity
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        mesh.materials.append(material)

        scene.render.engine = "CYCLES"
        saved["samples"] = scene.cycles.samples
        scene.cycles.samples = settings.reference_spp
        scene.render.resolution_x = settings.width
        scene.render.resolution_y = settings.height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = f"{tmpdir}/reference.exr"
        bpy.ops.render.render(write_still=True)
        pixels = _read_exr_pixels(scene.render.filepath, settings.width, settings.height)
        return pixels[..., :3]
    except Exception:
        return None
    finally:
        try:
            for obj, prior in hidden:
                obj.hide_render = prior
            scene.world = saved["world"]
            if sphere_obj is not None:
                bpy.data.objects.remove(sphere_obj, do_unlink=True)
            scene.render.engine = saved["engine"]
            scene.render.resolution_x = saved["res_x"]
            scene.render.resolution_y = saved["res_y"]
            scene.render.resolution_percentage = saved["res_pct"]
            scene.render.filepath = saved["filepath"]
            scene.render.image_settings.file_format = saved["file_format"]
            scene.render.image_settings.color_depth = saved["color_depth"]
            if saved["samples"] is not None:
                scene.cycles.samples = saved["samples"]
        except Exception:
            pass


def _validation_light(arrays: dict[str, np.ndarray]) -> SphereLight:
    """Deterministic validation light at a point recorded paths actually reach.

    Positions are sampled along recorded segments (so the light is inside the
    space the cache can see — a light floating outside a closed room would make
    the A/B comparison vacuous: two black images). Among a handful of candidates
    the one with the largest gathered contribution wins.
    """
    positions = arrays["position"].reshape(-1, 3)
    valid = arrays["depth"].reshape(-1) > 0
    pts = positions[valid] if np.any(valid) else positions
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
    radius = max(0.08 * extent, 1e-2)

    rng = np.random.default_rng(0xA0B1)
    seg_count = int(arrays["seg_pixel"].shape[0])
    tmax = arrays["seg_tmax"]
    finite = tmax[np.isfinite(tmax)]
    span = float(finite.max()) if finite.size else 1.0
    best = None
    best_mean = -1.0
    for _ in range(8):
        i = int(rng.integers(0, seg_count))
        t = float(rng.random()) * float(min(tmax[i], span))
        point = arrays["seg_origin"][i] + t * arrays["seg_dir"][i]
        light = SphereLight(
            position=tuple(float(v) for v in point),
            radius=radius,
            color=(1.0, 1.0, 1.0),
            intensity=5.0,
        )
        mean = float(gather_hdr(arrays, (light,)).mean())
        if mean > best_mean:
            best, best_mean = light, mean
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Bake driver


def bake_steps(
    context: Any, settings: BakeSettings
) -> Iterator[tuple[float, str, BakeResult | None]]:
    """Generator: yields (fraction, message, None) then (1.0, message, BakeResult)."""
    try:
        import bpy  # noqa: F401

        in_blender = True
    except ModuleNotFoundError:
        in_blender = False

    warnings: list[str] = []
    if in_blender:
        origin, corners, camera_id = _camera_frame(context, settings)
        caster = BlenderRayCaster(context.scene, context.evaluated_depsgraph_get())
        blender_file_name = bpy.data.filepath or None
    else:
        caster = AnalyticRoomCaster()
        origin = np.array([0.0, 3.5, 1.5])
        # Simple pinhole looking into the room along -Y.
        corners = {
            "top_left": np.array([-0.7, -1.0, 0.7]),
            "top_right": np.array([0.7, -1.0, 0.7]),
            "bottom_left": np.array([-0.7, -1.0, -0.7]),
            "bottom_right": np.array([0.7, -1.0, -0.7]),
        }
        camera_id = settings.camera_id
        blender_file_name = None
        warnings.append("Synthetic analytic-room capture generated outside Blender.")

    trace = trace_camera_paths(
        caster,
        origin,
        corners,
        settings.width,
        settings.height,
        paths_per_pixel=settings.paths_per_pixel,
        max_bounces=settings.max_bounces,
        seed=settings.seed,
    )
    result = None
    for fraction, maybe_result in trace:
        if maybe_result is not None:
            result = maybe_result
            break
        yield 0.9 * fraction, f"Tracing paths {fraction * 100.0:.0f}%", None
    assert result is not None
    arrays = result.as_arrays()

    gbuffer_source = "path_tracer_first_hit"
    if in_blender:
        yield 0.9, "Rendering Cycles G-buffer passes", None
        buffers = _cycles_gbuffer(context, settings)
        if buffers is None:
            warnings.append(
                "Cycles G-buffer render unavailable; using tracer first-hit buffers"
            )
        else:
            gbuffer_source = "cycles_passes:" + ",".join(sorted(buffers))
            if "albedo" in buffers:
                arrays["albedo"] = buffers["albedo"]
            if "normal" in buffers:
                arrays["normal"] = buffers["normal"]
            if "depth" in buffers:
                depth = buffers["depth"]
                # Cycles writes large sentinel depth for miss pixels; zero them
                # to match the cache convention.
                miss = ~np.isfinite(depth) | (depth > 1e9)
                arrays["depth"] = np.where(miss, 0.0, depth)
            if "position" in buffers:
                arrays["position"] = buffers["position"]

    extra_report: dict[str, Any] = {
        "paths_per_pixel": settings.paths_per_pixel,
        "max_bounces": settings.max_bounces,
        "gbuffer_source": gbuffer_source,
        "escape_segments": int(np.sum(~np.isfinite(arrays["seg_tmax"]))),
    }

    if settings.reference_check:
        light = _validation_light(arrays)
        gathered = gather_hdr(arrays, (light,))
        if in_blender:
            yield 0.95, "Rendering Cycles A/B reference", None
            reference = _cycles_reference_render(context, settings, light)
        else:
            reference = None
        if reference is None and in_blender:
            warnings.append("Cycles reference render failed; PSNR not computed")
        if reference is not None:
            extra_report["reference_check"] = {
                "light": light.to_dict(),
                "reference_spp": settings.reference_spp,
                "psnr_db": psnr(gathered, reference),
                "gather_mean": float(gathered.mean()),
                "reference_mean": float(reference.mean()),
            }

    yield 0.98, "Writing cache and reports", None
    bake_result = write_bake_outputs(
        arrays,
        settings,
        camera_id=camera_id,
        backend_id=id,
        backend_version=BACKEND_VERSION,
        approximation_limits=APPROXIMATION_LIMITS,
        warnings=tuple(warnings),
        blender_file_name=blender_file_name,
        packed=settings.packed,
        extra_report=extra_report,
    )
    yield 1.0, f"Baked {bake_result.cache_path}", bake_result


def bake(context: Any, settings: BakeSettings) -> BakeResult:
    """Synchronous bake: drain the generator (used in background mode and tests)."""
    result: BakeResult | None = None
    for _fraction, _message, maybe in bake_steps(context, settings):
        if maybe is not None:
            result = maybe
    assert result is not None
    return result
