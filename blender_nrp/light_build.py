"""Construct NRP light objects with the bpy *data* API (not operators).

Why the data API: ``bpy.ops.mesh.primitive_*_add`` depends on an active
3D-viewport context and reports its result only through ``context.object``. When
a button in the Properties editor triggers creation there is no viewport in
context, so ``context.object`` comes back ``None`` and the caller crashes with
``AttributeError: 'NoneType' object has no attribute 'name'``. Building the mesh
with :mod:`bmesh` / ``mesh.from_pydata`` and linking the object ourselves works
from any editor and never touches ``context.object``.

Both the create operators and the light-JSON importer route through here so the
two paths stay identical.
"""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised only inside Blender.
    bpy = None

if bpy is not None:
    import bmesh

    from .core.coords import BLENDER_Z_UP

    def _link_new_object(
        context: bpy.types.Context, name: str, mesh: bpy.types.Mesh
    ) -> bpy.types.Object:
        """Create an object for ``mesh``, link it, and make it the active selection."""
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)
        for other in tuple(context.selected_objects):
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    def _sphere_mesh(name: str, radius: float) -> bpy.types.Mesh:
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=radius)
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    def _plane_mesh(name: str, width: float, height: float) -> bpy.types.Mesh:
        mesh = bpy.data.meshes.new(name)
        half_w, half_h = width / 2.0, height / 2.0
        verts = [
            (-half_w, -half_h, 0.0),
            (half_w, -half_h, 0.0),
            (half_w, half_h, 0.0),
            (-half_w, half_h, 0.0),
        ]
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        mesh.update()
        return mesh

    def _stamp_common(
        obj: bpy.types.Object,
        *,
        scene_id: str,
        camera_id: str,
        color: tuple[float, float, float],
        intensity: float,
    ) -> None:
        obj["nrp_scene_id"] = scene_id
        obj["nrp_camera_id"] = camera_id
        obj["nrp_coordinate_system"] = BLENDER_Z_UP
        obj["nrp_color"] = list(color)
        obj["nrp_intensity"] = float(intensity)
        obj["nrp_enabled"] = True
        obj["nrp_muted"] = False
        obj["nrp_solo"] = False
        obj["nrp_kelvin"] = 6500.0
        obj["nrp_tint"] = 0.0

    def create_sphere_light(
        context: bpy.types.Context,
        *,
        name: str = "NRP_Sphere",
        radius: float = 0.25,
        location: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        scene_id: str = "",
        camera_id: str = "",
    ) -> bpy.types.Object:
        obj = _link_new_object(context, name, _sphere_mesh(name, radius))
        obj.location = location
        obj["nrp_light_type"] = "sphere"
        obj["nrp_radius"] = float(radius)
        _stamp_common(
            obj, scene_id=scene_id, camera_id=camera_id, color=color, intensity=intensity
        )
        return obj

    def create_quad_light(
        context: bpy.types.Context,
        *,
        name: str = "NRP_Quad",
        width: float = 1.0,
        height: float = 1.0,
        location: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        scene_id: str = "",
        camera_id: str = "",
    ) -> bpy.types.Object:
        obj = _link_new_object(context, name, _plane_mesh(name, width, height))
        obj.location = location
        obj["nrp_light_type"] = "quad"
        obj["nrp_width"] = float(width)
        obj["nrp_height"] = float(height)
        _stamp_common(
            obj, scene_id=scene_id, camera_id=camera_id, color=color, intensity=intensity
        )
        return obj
