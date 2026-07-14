"""Add-on preferences for execution backends; never serialized into scene data."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None


if bpy is not None:

    class BlenderNRPPreferences(bpy.types.AddonPreferences):
        bl_idname = "blender_nrp"

        ssh_host: bpy.props.StringProperty(name="SSH Host", default="")
        ssh_remote_root: bpy.props.StringProperty(
            name="Remote Job Root", default="~/blender-nrp-jobs"
        )
        ssh_worker_root: bpy.props.StringProperty(
            name="Remote Worker Root", default="~/Blender-NRP"
        )
        ssh_blender_binary: bpy.props.StringProperty(name="Remote Blender", default="blender")
        ssh_python_binary: bpy.props.StringProperty(name="Remote Python", default="python3")
        runpod_api_key: bpy.props.StringProperty(
            name="RunPod API Key", subtype="PASSWORD", default=""
        )
        runpod_image: bpy.props.StringProperty(name="RunPod Image", default="")
        runpod_worker_root: bpy.props.StringProperty(
            name="RunPod Worker Root", default="/opt/Blender-NRP"
        )
        runpod_gpu_type: bpy.props.StringProperty(
            name="RunPod GPU", default="NVIDIA GeForce RTX 4090"
        )
        runpod_hourly_rate: bpy.props.FloatProperty(
            name="Estimated $/hour", default=0.69, min=0.0
        )

        def draw(self, context: bpy.types.Context) -> None:
            layout = self.layout
            layout.label(text="SSH / LAN render node")
            layout.prop(self, "ssh_host")
            layout.prop(self, "ssh_remote_root")
            layout.prop(self, "ssh_worker_root")
            layout.prop(self, "ssh_blender_binary")
            layout.prop(self, "ssh_python_binary")
            layout.label(text="Credentials remain in your SSH agent or config, never in a .blend.")
            layout.separator()
            layout.label(text="RunPod GPU pods")
            layout.prop(self, "runpod_api_key")
            layout.prop(self, "runpod_image")
            layout.prop(self, "runpod_worker_root")
            layout.prop(self, "runpod_gpu_type")
            layout.prop(self, "runpod_hourly_rate")


CLASSES = (BlenderNRPPreferences,) if bpy is not None else ()


def register() -> None:
    if bpy is not None:
        for cls in CLASSES:
            bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is not None:
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)
