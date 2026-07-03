"""Blender property groups for add-on state."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised only inside Blender.
    bpy = None


if bpy is not None:

    class BlenderNRPSettings(bpy.types.PropertyGroup):
        scene_id: bpy.props.StringProperty(
            name="Scene ID",
            default="fixture_room_001",
            description="Stable ID used in NRP cache metadata and light JSON",
        )
        output_dir: bpy.props.StringProperty(
            name="Output Directory",
            subtype="DIR_PATH",
            default="//output",
        )
        camera: bpy.props.PointerProperty(
            name="Camera",
            type=bpy.types.Object,
            poll=lambda _self, obj: obj.type == "CAMERA",
        )
        resolution_x: bpy.props.IntProperty(name="Width", default=256, min=1)
        resolution_y: bpy.props.IntProperty(name="Height", default=256, min=1)
        segment_count: bpy.props.IntProperty(name="Hemisphere Segments", default=16, min=1)
        max_segment_distance: bpy.props.FloatProperty(
            name="Max Segment Distance",
            default=100.0,
            min=0.001,
        )
        cache_path: bpy.props.StringProperty(name="Cache Path", subtype="FILE_PATH")
        model_path: bpy.props.StringProperty(name="Model Path", subtype="FILE_PATH")
        light_json_path: bpy.props.StringProperty(name="Light JSON", subtype="FILE_PATH")
        status: bpy.props.StringProperty(name="Status", default="Ready")


def register() -> None:
    if bpy is None:
        return
    bpy.utils.register_class(BlenderNRPSettings)
    bpy.types.Scene.blender_nrp = bpy.props.PointerProperty(type=BlenderNRPSettings)


def unregister() -> None:
    if bpy is None:
        return
    del bpy.types.Scene.blender_nrp
    bpy.utils.unregister_class(BlenderNRPSettings)

