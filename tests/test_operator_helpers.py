"""Pure-Python tests for the operator status helpers.

These run outside Blender with a fake operator + context, pinning the contract
every operator relies on: the message is written to ``scene.blender_nrp.status``
*and* surfaced through ``op.report`` (the toast feedback users were missing), and
the right operator return set comes back.
"""

from __future__ import annotations

import types

from blender_nrp.operators._helpers import cancel_with_status, finish_with_status


class _FakeOp:
    def __init__(self) -> None:
        self.reports: list[tuple[set[str], str]] = []

    def report(self, level: set[str], message: str) -> None:
        self.reports.append((level, message))


def _fake_context() -> types.SimpleNamespace:
    settings = types.SimpleNamespace(status="Ready")
    scene = types.SimpleNamespace(blender_nrp=settings)
    return types.SimpleNamespace(scene=scene)


def test_finish_sets_status_reports_and_returns_finished() -> None:
    op, context = _FakeOp(), _fake_context()
    result = finish_with_status(op, context, "done")
    assert result == {"FINISHED"}
    assert context.scene.blender_nrp.status == "done"
    assert op.reports == [({"INFO"}, "done")]


def test_cancel_sets_status_reports_warning_and_returns_cancelled() -> None:
    op, context = _FakeOp(), _fake_context()
    result = cancel_with_status(op, context, "nope")
    assert result == {"CANCELLED"}
    assert context.scene.blender_nrp.status == "nope"
    assert op.reports == [({"WARNING"}, "nope")]


def test_finish_honours_explicit_level() -> None:
    op, context = _FakeOp(), _fake_context()
    finish_with_status(op, context, "baked but invalid", level="ERROR")
    assert op.reports == [({"ERROR"}, "baked but invalid")]


def test_helpers_tolerate_missing_operator() -> None:
    context = _fake_context()
    # Non-operator callers (e.g. timers) pass op=None and must not crash.
    assert finish_with_status(None, context, "x") == {"FINISHED"}
    assert cancel_with_status(None, context, "y") == {"CANCELLED"}
