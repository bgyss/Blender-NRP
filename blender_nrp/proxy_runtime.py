"""Session-scoped holder for the currently loaded proxy model.

Blender operators are stateless between invocations; the loaded TorchNRP lives here
so Load Proxy, Preview, and Optimize share it. Not persisted with the .blend file —
reload the proxy after reopening a scene.
"""

from __future__ import annotations

from typing import Any

model: Any | None = None
model_path: str | None = None
model_light_type: str | None = None


def set_model(new_model: Any, path: str, light_type: str) -> None:
    global model, model_path, model_light_type
    model = new_model
    model_path = path
    model_light_type = light_type


def clear() -> None:
    global model, model_path, model_light_type
    model = None
    model_path = None
    model_light_type = None
