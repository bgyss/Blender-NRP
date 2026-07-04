"""Live in-Blender relight preview.

The relit image lands in an Image datablock (`NRP Relight Preview`) updated in
place, so any Image Editor showing it refreshes live; a PNG is still written next
to the cache for parity with V1. The image is produced by the trained proxy when
one is loaded (fast path) and by the exact cache gather when not, and the status
line labels which one it was.

Auto-update: a `depsgraph_update_post` handler watches for transform/data changes
on NRP light objects and schedules a debounced refresh through `bpy.app.timers`
(0.3 s of quiet before recomputing), gated by the scene's `live_preview` toggle.
The explicit Preview button remains the fallback path.
"""

from __future__ import annotations

try:
    import bpy
    from bpy.app.handlers import persistent
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

import time
from pathlib import Path

import numpy as np

PREVIEW_IMAGE_NAME = "NRP Relight Preview"
_DEBOUNCE_SECONDS = 0.3

_cache_state: dict = {"path": None, "mtime": None, "arrays": None}
_pending: dict = {"deadline": None}


def _load_cache_arrays(cache_path: Path):
    """mtime-cached arrays so dragging a light doesn't re-read the npz every tick."""
    from .core.path_cache import load_arrays

    mtime = cache_path.stat().st_mtime
    if _cache_state["path"] == str(cache_path) and _cache_state["mtime"] == mtime:
        return _cache_state["arrays"]
    arrays = load_arrays(cache_path).arrays
    _cache_state.update({"path": str(cache_path), "mtime": mtime, "arrays": arrays})
    return arrays


def _write_image_datablock(image_float: np.ndarray) -> None:
    """Write an (H, W, 3) [0,1] image into the preview datablock in place."""
    height, width, _ = image_float.shape
    image = bpy.data.images.get(PREVIEW_IMAGE_NAME)
    if image is None:
        image = bpy.data.images.new(
            PREVIEW_IMAGE_NAME, width=width, height=height, float_buffer=True
        )
    elif image.size[0] != width or image.size[1] != height:
        image.scale(width, height)
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[..., :3] = image_float[::-1]  # Blender stores bottom row first
    image.pixels.foreach_set(rgba.reshape(-1))
    image.update()
    # Nudge open Image Editors to redraw.
    wm = bpy.context.window_manager
    if wm is not None:
        for window in wm.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.tag_redraw()


def update_preview(context) -> tuple[bool, str]:
    """Recompute the relight preview. Returns (ok, status message)."""
    from . import proxy_runtime
    from .core.gather import gather_hdr
    from .core.images import write_png_rgb
    from .core.light_objects import collect_rig_lights

    settings = context.scene.blender_nrp
    if not settings.cache_path:
        return False, "No cache path selected"
    cache_path = Path(bpy.path.abspath(settings.cache_path))
    if not cache_path.exists():
        return False, f"Cache not found: {cache_path}"
    lights = tuple(collect_rig_lights(context.scene.objects))
    if not lights:
        return False, "No NRP lights in scene"
    try:
        arrays = _load_cache_arrays(cache_path)
    except Exception as exc:
        return False, f"Cache load failed: {exc}"

    source = "cache gather (exact)"
    hdr = None
    if proxy_runtime.model is not None:
        rig_types = {light.light_type for light in lights}
        if rig_types == {proxy_runtime.model_light_type}:
            try:
                from .core.torch_proxy.relight import proxy_relight

                hdr = proxy_relight(proxy_runtime.model, arrays, lights)
                source = "trained proxy (fast)"
            except Exception as exc:
                return False, f"Proxy relight failed: {exc}"
        else:
            source = (
                "cache gather (exact; proxy skipped: trained for "
                f"{proxy_runtime.model_light_type} lights)"
            )
    if hdr is None:
        hdr = gather_hdr(arrays, lights)

    image = np.clip(hdr * float(settings.preview_exposure), 0.0, 1.0).astype(np.float32)
    _write_image_datablock(image)
    png_path = cache_path.parent / "relight_preview.png"
    write_png_rgb(png_path, image)
    return True, f"Preview updated via {source} — image '{PREVIEW_IMAGE_NAME}' + {png_path.name}"


# ---------------------------------------------------------------------------
# Debounced auto-update


def _timer_tick() -> float | None:
    deadline = _pending["deadline"]
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining > 0:
        return remaining
    _pending["deadline"] = None
    context = bpy.context
    if context.scene is None or not context.scene.blender_nrp.live_preview:
        return None
    ok, message = update_preview(context)
    context.scene.blender_nrp.status = message if ok else f"Live preview: {message}"
    return None


if bpy is not None:

    @persistent
    def _on_depsgraph_update(scene, depsgraph) -> None:
        settings = getattr(scene, "blender_nrp", None)
        if settings is None or not settings.live_preview:
            return
        for update in depsgraph.updates:
            obj = update.id
            if isinstance(obj, bpy.types.Object) and obj.get("nrp_light_type"):
                break
        else:
            return
        _pending["deadline"] = time.monotonic() + _DEBOUNCE_SECONDS
        if not bpy.app.timers.is_registered(_timer_tick):
            bpy.app.timers.register(_timer_tick, first_interval=_DEBOUNCE_SECONDS)


def register() -> None:
    if bpy is None:
        return
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister() -> None:
    if bpy is None:
        return
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _pending["deadline"] = None
