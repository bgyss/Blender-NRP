"""Transparent compute-time/cost estimates for the remote progress UI."""

from __future__ import annotations


def estimate_bake_seconds(
    width: int,
    height: int,
    paths_per_pixel: int,
    max_bounces: int,
    *,
    reference_seconds: float = 180.0,
) -> float:
    """Estimate GPU bake wall time from a documented calibration point.

    This is intentionally a heuristic until a worker reports historical timings;
    callers should label it as an estimate, never as a guarantee.
    """
    work = width * height * paths_per_pixel * max_bounces
    reference_work = 512 * 512 * 256 * 4
    return max(1.0, reference_seconds * work / reference_work)


def estimate_cost_usd(
    width: int,
    height: int,
    paths_per_pixel: int,
    max_bounces: int,
    hourly_rate: float,
) -> float:
    if hourly_rate < 0:
        raise ValueError("hourly_rate must be non-negative")
    return estimate_bake_seconds(width, height, paths_per_pixel, max_bounces) / 3600.0 * hourly_rate
