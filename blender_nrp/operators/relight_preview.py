"""Relight preview operators."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    from pathlib import Path

    from blender_nrp.core.gather import write_relight_preview
    from blender_nrp.core.lights import LightRig, SphereLight

    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_create_sphere_light(bpy.types.Operator):
        bl_idname = "blender_nrp.create_sphere_light"
        bl_label = "Create NRP Sphere Light"
        bl_description = "Create a visible NRP sphere emitter object"

        def execute(self, context: bpy.types.Context) -> set[str]:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.25)
            obj = context.object
            obj.name = "NRP_Sphere_001"
            obj["nrp_light_type"] = "sphere"
            obj["nrp_scene_id"] = context.scene.blender_nrp.scene_id
            obj["nrp_camera_id"] = (
                context.scene.blender_nrp.camera.name if context.scene.blender_nrp.camera else ""
            )
            obj["nrp_coordinate_system"] = "blender_z_up"
            obj["nrp_radius"] = 0.25
            obj["nrp_color"] = (1.0, 1.0, 1.0)
            obj["nrp_intensity"] = 1.0
            return finish_with_status(context, "Created NRP sphere light")

    class BLENDER_NRP_OT_relight_preview(bpy.types.Operator):
        bl_idname = "blender_nrp.relight_preview"
        bl_label = "Preview Relight"
        bl_description = "Preview fixed-camera relighting from the active proxy"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            if not settings.cache_path:
                return cancel_with_status(context, "No cache path selected")
            lights = []
            for obj in context.scene.objects:
                if obj.get("nrp_light_type") == "sphere":
                    lights.append(
                        SphereLight(
                            position=tuple(float(v) for v in obj.location),
                            radius=float(obj.get("nrp_radius", 0.25)),
                            color=tuple(float(v) for v in obj.get("nrp_color", (1.0, 1.0, 1.0))),
                            intensity=float(obj.get("nrp_intensity", 1.0)),
                        )
                    )
            if not lights:
                return cancel_with_status(context, "No NRP sphere lights in scene")
            cache_path = Path(bpy.path.abspath(settings.cache_path))
            target = cache_path.parent / "relight_preview.png"
            rig = LightRig(tuple(lights), scene_id=settings.scene_id)
            try:
                write_relight_preview(cache_path, rig, target)
            except Exception as exc:
                return cancel_with_status(context, f"Relight preview failed: {exc}")
            return finish_with_status(context, f"Wrote {target}")


CLASSES = (
    BLENDER_NRP_OT_create_sphere_light,
    BLENDER_NRP_OT_relight_preview,
) if bpy is not None else ()


def register() -> None:
    if bpy is None:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is None:
        return
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
