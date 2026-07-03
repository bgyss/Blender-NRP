"""Shared operator helpers."""

from __future__ import annotations


def finish_with_status(context: object, message: str) -> set[str]:
    context.scene.blender_nrp.status = message
    return {"FINISHED"}


def cancel_with_status(context: object, message: str) -> set[str]:
    context.scene.blender_nrp.status = message
    return {"CANCELLED"}

