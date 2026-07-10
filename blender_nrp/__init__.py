"""Blender-NRP add-on entrypoint.

This module intentionally keeps Blender-specific imports inside register-time modules so
the package can still be imported by normal Python tests outside Blender.
"""

bl_info = {
    "name": "Blender-NRP",
    "author": "Blender-NRP contributors",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "Properties > Scene > Blender-NRP",
    "description": "Bake and relight Neural Render Proxy scenes inside Blender",
    "category": "Render",
}


def register() -> None:
    from . import addon

    addon.register()


def unregister() -> None:
    from . import addon

    addon.unregister()
