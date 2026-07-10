from __future__ import annotations

import json
import time

import pytest

from blender_nrp.core.execution import (
    ExecutionQueue,
    LocalSubprocessBackend,
    QueuedJob,
    SshExecutionBackend,
)
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


class _Backend:
    def __init__(self, progress):
        self.progress = progress

    def submit(self, job):  # pragma: no cover - Protocol completeness for the test fake.
        return "unused"

    def status(self, job_id):
        return self.progress

    def fetch(self, job_id):  # pragma: no cover
        return {}

    def cancel(self, job_id):  # pragma: no cover
        return None


def test_queue_persists_and_reconciles_active_jobs(tmp_path):
    queue = ExecutionQueue(tmp_path)
    queued = QueuedJob("live", "local", time.time(), str(tmp_path / "live.json"))
    done = QueuedJob("done", "local", time.time(), str(tmp_path / "done.json"))
    queue.add(queued)
    queue.add(done)
    assert [item.job_id for item in queue.load()] == ["live", "done"]

    class _MixedBackend(_Backend):
        def status(self, job_id):
            return JobProgress(job_id, "running" if job_id == "live" else "succeeded")

    statuses = queue.reconcile({"local": _MixedBackend(JobProgress("x", "running"))})
    assert statuses["done"].state == "succeeded"
    assert [item.job_id for item in queue.load()] == ["live"]


def test_ssh_backend_stages_job_and_fetches_reported_artifacts(tmp_path):
    commands = []

    def runner(command):
        commands.append(command)
        if command[:3] == ["ssh", "renderbox", "cat"]:
            progress = JobProgress(
                "j", "succeeded", artifacts={"path_cache": "/remote/cache.npz"}
            )
            return type("Result", (), {"stdout": json.dumps(progress.to_dict())})()
        return type("Result", (), {"stdout": ""})()

    backend = SshExecutionBackend(
        tmp_path, host="renderbox", remote_root="/jobs", worker_root="/worker", runner=runner
    )
    scene = tmp_path / "scene.blend"
    scene.touch()
    job_id = backend.submit(BakeJob("scene", str(scene), str(tmp_path)))
    assert any(command[0] == "rsync" for command in commands)
    assert any(command[0] == "ssh" and "nohup" in command[-1] for command in commands)
    # The fake status carries a different worker id, as a real remote worker does;
    # fetching is keyed by the submitted id and artifact names remain stable.
    progress = backend.status(job_id)
    assert progress.state == "succeeded"
    assert backend.fetch(job_id)["path_cache"].name == "cache.npz"
