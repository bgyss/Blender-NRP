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

        def draw(self, context: bpy.types.Context) -> None:
            layout = self.layout
            layout.label(text="SSH / LAN render node")
            layout.prop(self, "ssh_host")
            layout.prop(self, "ssh_remote_root")
            layout.prop(self, "ssh_worker_root")
            layout.prop(self, "ssh_blender_binary")
            layout.label(text="Credentials remain in your SSH agent or config, never in a .blend.")


CLASSES = (BlenderNRPPreferences,) if bpy is not None else ()


def register() -> None:
    if bpy is not None:
        for cls in CLASSES:
            bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is not None:
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)
