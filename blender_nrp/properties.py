"""Blender property groups for add-on state."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised only inside Blender.
    bpy = None


if bpy is not None:
    from .core.pipeline import resolve_preset

    def _apply_quality_preset(settings, context) -> None:
        if context is None or context.scene is None:
            return
        render = context.scene.render
        width, height, budget = resolve_preset(
            settings.quality_preset, render.resolution_x, render.resolution_y
        )
        settings.resolution_x = width
        settings.resolution_y = height
        settings.paths_per_pixel = budget.paths_per_pixel
        settings.max_bounces = budget.max_bounces
        settings.train_iterations = budget.train_iterations

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
        backend: bpy.props.EnumProperty(
            name="Backend",
            description="Path-cache capture backend",
            items=(
                (
                    "cycles_capture",
                    "Cycles Capture",
                    "Multi-bounce path capture with Cycles G-buffer passes",
                ),
                (
                    "stock_blender_hemi",
                    "Stock Hemisphere",
                    "Fast first-hit + deterministic hemisphere spokes (V1 fallback)",
                ),
            ),
            default="cycles_capture",
        )
        compute: bpy.props.EnumProperty(
            name="Compute",
            items=(
                ("local_subprocess", "This Machine", "Run the worker outside Blender"),
                ("ssh", "SSH / LAN Node", "Submit the same job bundle through SSH and rsync"),
            ),
            default="local_subprocess",
        )
        quality_preset: bpy.props.EnumProperty(
            name="Quality",
            items=(
                ("draft", "Draft", "Fast validation settings"),
                ("standard", "Standard", "Balanced local settings"),
                ("final", "Final", "Higher budget; use remote compute when available"),
            ),
            default="draft",
            update=_apply_quality_preset,
        )
        tracer_engine: bpy.props.EnumProperty(
            name="Tracer",
            items=(
                ("auto", "Auto", "Use torch analytic fixture tracing when available"),
                ("python", "Python", "Use the V2 CPU ray-cast tracer"),
                (
                    "torch_analytic",
                    "Torch Analytic",
                    "Require the device-side analytic fixture tracer",
                ),
            ),
            default="auto",
        )
        show_advanced: bpy.props.BoolProperty(name="Advanced", default=False)
        pipeline_settings_hash: bpy.props.StringProperty(name="Pipeline Settings Hash", default="")
        pipeline_scene_hash: bpy.props.StringProperty(name="Pipeline Scene Hash", default="")
        snapshot_name: bpy.props.StringProperty(name="Snapshot", default="Look 01")
        snapshot_a: bpy.props.StringProperty(name="A", default="")
        snapshot_b: bpy.props.StringProperty(name="B", default="")
        segment_count: bpy.props.IntProperty(
            name="Hemisphere Segments",
            default=16,
            min=1,
            description="Spokes per pixel (stock hemisphere backend only)",
        )
        paths_per_pixel: bpy.props.IntProperty(
            name="Paths / Pixel",
            default=64,
            min=1,
            description="Monte Carlo paths traced per pixel (Cycles capture backend)",
        )
        max_bounces: bpy.props.IntProperty(
            name="Max Bounces",
            default=4,
            min=1,
            max=32,
            description="Maximum path depth (Cycles capture backend)",
        )
        max_segment_distance: bpy.props.FloatProperty(
            name="Max Segment Distance",
            default=100.0,
            min=0.001,
        )
        packed_cache: bpy.props.BoolProperty(
            name="Packed Cache",
            default=False,
            description="Write the ~4x smaller fp16 + rgb9e5 packed cache layout",
        )
        cache_path: bpy.props.StringProperty(name="Cache Path", subtype="FILE_PATH")
        model_path: bpy.props.StringProperty(name="Model Path", subtype="FILE_PATH")
        light_json_path: bpy.props.StringProperty(name="Light JSON", subtype="FILE_PATH")
        export_coordinate_system: bpy.props.EnumProperty(
            name="Export Coords",
            description="Coordinate system written into exported light JSON",
            items=(
                ("right_handed_y_up", "Y Up (ComfyUI)", "right_handed_y_up, ComfyUI default"),
                ("blender_z_up", "Z Up (Blender)", "blender_z_up, no conversion"),
            ),
            default="right_handed_y_up",
        )
        train_iterations: bpy.props.IntProperty(
            name="Train Iterations",
            default=2000,
            min=1,
            description="Torch proxy training iterations",
        )
        train_device: bpy.props.EnumProperty(
            name="Device",
            description="Torch device for proxy training/inference",
            items=(
                ("auto", "Auto", "mps/cuda when available, else cpu"),
                ("cpu", "CPU", "cpu"),
                ("mps", "MPS", "Apple Metal"),
                ("cuda", "CUDA", "NVIDIA GPU"),
            ),
            default="auto",
        )
        target_image_path: bpy.props.StringProperty(
            name="Target Image",
            subtype="FILE_PATH",
            description="Reference image the light optimizer matches (PNG or .npy)",
        )
        optimize_steps: bpy.props.IntProperty(
            name="Solver Steps",
            default=300,
            min=1,
            description="Gradient steps for inverse light optimization",
        )
        live_preview: bpy.props.BoolProperty(
            name="Live Preview",
            default=False,
            description="Auto-refresh the relight preview image when NRP lights change",
        )
        preview_exposure: bpy.props.FloatProperty(
            name="Exposure",
            default=1.0,
            min=0.0,
            description="Linear exposure applied to the relight preview",
        )
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
