"""V3 one-button preset resolution, independent of Blender UI state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineBudget:
    scale: float
    paths_per_pixel: int
    max_bounces: int
    train_iterations: int


PRESETS = {
    "draft": PipelineBudget(0.25, 16, 3, 500),
    "standard": PipelineBudget(0.5, 64, 4, 2000),
    "final": PipelineBudget(1.0, 256, 6, 8000),
}


def resolve_preset(
    name: str, render_width: int, render_height: int
) -> tuple[int, int, PipelineBudget]:
    """Return preview-scaled resolution and budgets for Draft/Standard/Final."""
    try:
        budget = PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown quality preset: {name!r}") from exc
    return (
        max(1, round(render_width * budget.scale)),
        max(1, round(render_height * budget.scale)),
        budget,
    )
