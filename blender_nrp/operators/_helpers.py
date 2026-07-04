"""Shared operator helpers.

Every operator reports its outcome two ways: it writes the message onto
``scene.blender_nrp.status`` (shown persistently in the panel) *and* calls
``op.report(...)`` so Blender raises its usual header/toast feedback — otherwise
button presses feel like they did nothing. Pass the operator (``self``) as the
first argument from inside ``execute``/``modal``/``invoke``.
"""

from __future__ import annotations


def finish_with_status(
    op: object, context: object, message: str, *, level: str = "INFO"
) -> set[str]:
    context.scene.blender_nrp.status = message
    if op is not None:
        op.report({level}, message)
    return {"FINISHED"}


def cancel_with_status(
    op: object, context: object, message: str, *, level: str = "WARNING"
) -> set[str]:
    context.scene.blender_nrp.status = message
    if op is not None:
        op.report({level}, message)
    return {"CANCELLED"}
