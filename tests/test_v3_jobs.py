from __future__ import annotations

import json

import pytest

from blender_nrp.core.execution import LocalSubprocessBackend
from blender_nrp.core.jobs import BakeJob, JobProgress, read_job, write_job, write_progress
from blender_nrp.core.staleness import is_stale, stable_hash


def test_bake_job_round_trip_and_version_gate(tmp_path):
    job = BakeJob("scene", "scene.blend", str(tmp_path), width=32, height=16)
    path = write_job(tmp_path / "bake_job.json", job)
    assert read_job(path) == job
    payload = json.loads(path.read_text())
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema version"):
        read_job(path)


def test_staleness_hash_is_deterministic_and_detects_change():
    settings = {"width": 64, "paths": 8}
    scene = {"objects": ["Cube"]}
    assert stable_hash(settings) == stable_hash({"paths": 8, "width": 64})
    assert not is_stale(stable_hash(settings), stable_hash(scene), settings, scene)
    assert is_stale(stable_hash(settings), stable_hash(scene), {"width": 128, "paths": 8}, scene)


def test_local_backend_reads_durable_progress_and_fetches(tmp_path):
    backend = LocalSubprocessBackend(tmp_path)
    job = BakeJob("scene", "scene.blend", str(tmp_path))
    job_id = backend.submit(job)
    _job_path, status_path = backend._paths(job_id)
    artifact = tmp_path / "cache.npz"
    artifact.touch()
    write_progress(
        status_path, JobProgress(job_id, "succeeded", artifacts={"path_cache": str(artifact)})
    )
    assert backend.status(job_id).state == "succeeded"
    assert backend.fetch(job_id) == {"path_cache": artifact}
    backend.cancel(job_id)
