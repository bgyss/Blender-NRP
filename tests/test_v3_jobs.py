from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from blender_nrp.core.execution import (
    ExecutionQueue,
    LocalSubprocessBackend,
    QueuedJob,
    RunPodExecutionBackend,
    SshExecutionBackend,
)
from blender_nrp.core.jobs import (
    BakeJob,
    JobProgress,
    SolveJob,
    TrainJob,
    read_job,
    read_progress,
    write_job,
    write_progress,
)
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


@pytest.mark.parametrize(
    "job",
    [
        TrainJob("cache.npz", "out", iterations=3, torch_device="cpu"),
        SolveJob("cache.npz", "lights.json", "target.npy", "out", steps=4),
    ],
)
def test_train_and_solve_job_round_trip(tmp_path, job):
    path = write_job(tmp_path / f"{job.kind}_job.json", job)
    assert read_job(path) == job


def test_failed_worker_writes_honest_machine_readable_report(tmp_path):
    job_path = write_job(
        tmp_path / "train_job.json",
        TrainJob(str(tmp_path / "missing_cache.npz"), str(tmp_path / "artifacts"), iterations=1),
    )
    status_path = tmp_path / "status.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_train_job.py",
            str(job_path),
            "--status",
            str(status_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    progress = read_progress(status_path)
    assert progress.state == "failed"
    report_path = tmp_path / "artifacts" / "train_report.json"
    report = json.loads(report_path.read_text())
    assert report["ok"] is False
    assert report["limitations"]
    assert progress.artifacts["train_report"] == str(report_path)


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


def test_local_backend_recovers_running_pid_and_cancels(tmp_path, monkeypatch):
    backend = LocalSubprocessBackend(tmp_path)
    job_id = "recovered-local"
    backend._pid_path(job_id).write_text("12345\n")
    signals = []

    def fake_kill(pid, sig):
        signals.append((pid, sig))

    monkeypatch.setattr("blender_nrp.core.execution.os.kill", fake_kill)
    assert backend.status(job_id).state == "running"
    backend.cancel(job_id)
    assert signals[-1][0] == 12345


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


def test_mock_execution_backend_exercises_full_lifecycle():
    calls = []

    class MockExecutionBackend:
        def submit(self, job):
            calls.append(("submit", job.kind))
            return "mock-job"

        def status(self, job_id):
            calls.append(("status", job_id))
            return JobProgress(job_id, "succeeded", 1.0)

        def fetch(self, job_id):
            calls.append(("fetch", job_id))
            return {"model": "model.pt"}

        def cancel(self, job_id):
            calls.append(("cancel", job_id))

    backend = MockExecutionBackend()
    job_id = backend.submit(TrainJob("cache.npz", "out"))
    assert backend.status(job_id).state == "succeeded"
    assert backend.fetch(job_id) == {"model": "model.pt"}
    backend.cancel(job_id)
    assert [name for name, _value in calls] == ["submit", "status", "fetch", "cancel"]


def test_queue_surfaces_mock_backend_failure_until_explicit_dismissal(tmp_path):
    queue = ExecutionQueue(tmp_path)
    queue.add(QueuedJob("failed", "mock", time.time(), "failed.json"))
    failure = JobProgress("failed", "failed", stage="Training", message="GPU OOM")
    statuses = queue.reconcile({"mock": _Backend(failure)})
    assert statuses["failed"].message == "GPU OOM"
    assert [item.job_id for item in queue.load()] == ["failed"]
    queue.remove("failed")
    assert queue.load() == []


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
    assert [item.job_id for item in queue.load()] == ["live", "done"]


def test_ssh_backend_stages_job_and_fetches_reported_artifacts(tmp_path):
    commands = []

    def runner(command):
        commands.append(command)
        if "cat" in command:
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


def test_ssh_backend_stable_id_guards_launch_and_cancels_after_restart(tmp_path):
    commands = []

    def runner(command):
        commands.append(command)
        return type("Result", (), {"stdout": ""})()

    scene = tmp_path / "scene.blend"
    scene.touch()
    backend = SshExecutionBackend(
        tmp_path, host="renderbox", remote_root="/jobs", worker_root="/worker", runner=runner
    )
    assert backend.submit(
        BakeJob("scene", str(scene), str(tmp_path)), job_id="stable-delegate"
    ) == "stable-delegate"
    launch = next(command[-1] for command in commands if "nohup" in command[-1])
    assert "/jobs/stable-delegate/.worker-launched" in launch

    restarted = SshExecutionBackend(
        tmp_path, host="renderbox", remote_root="/jobs", worker_root="/worker", runner=runner
    )
    restarted.cancel("stable-delegate")
    assert commands[-1][-2:] == ["-f", "/jobs/stable-delegate/job.json"]


def test_ssh_backend_runs_torch_jobs_with_plain_python(tmp_path):
    commands = []

    def runner(command):
        commands.append(command)
        return type("Result", (), {"stdout": ""})()

    cache = tmp_path / "path_cache.npz"
    cache.touch()
    backend = SshExecutionBackend(
        tmp_path,
        host="renderbox",
        remote_root="/jobs",
        worker_root="/worker",
        blender_binary="/opt/blender/blender",
        python_binary="/venv/bin/python",
        runner=runner,
    )
    backend.submit(TrainJob(str(cache), str(tmp_path)))
    launch = next(command[-1] for command in commands if "nohup" in command[-1])
    assert "nohup /venv/bin/python /worker/scripts/run_train_job.py" in launch
    assert "/opt/blender/blender" not in launch


def test_runpod_adapter_creates_polls_fetches_and_deletes_pod(tmp_path):
    api_calls = []

    def request(method, path, payload):
        api_calls.append((method, path, payload))
        if method == "POST" and path == "/pods":
            return {"id": "pod-1", "costPerHr": 0.42}
        if method == "GET":
            return {
                "id": "pod-1",
                "desiredStatus": "RUNNING",
                "publicIp": "203.0.113.10",
                "portMappings": {"22": 2222},
                "costPerHr": 0.42,
            }
        return {}

    def runner(command):
        if "cat" in command:
            progress = JobProgress(
                "delegate", "succeeded", artifacts={"path_cache": "/remote/cache.npz"}
            )
            return type("Result", (), {"stdout": json.dumps(progress.to_dict())})()
        return type("Result", (), {"stdout": ""})()

    scene = tmp_path / "scene.blend"
    scene.touch()
    backend = RunPodExecutionBackend(
        tmp_path,
        api_key="secret",
        image_name="registry/blender-nrp:latest",
        worker_root="/opt/Blender-NRP",
        requester=request,
    )
    backend.ssh_runner = runner
    pod_id = backend.submit(BakeJob("scene", str(scene), str(tmp_path)))
    progress = backend.status(pod_id)
    assert progress.state == "succeeded"
    assert progress.cost_per_hour == 0.42
    assert backend.fetch(pod_id)["path_cache"].name == "cache.npz"
    backend.cancel(pod_id)
    assert ("POST", "/pods",) == api_calls[0][:2]
    assert ("DELETE", "/pods/pod-1") in [call[:2] for call in api_calls]


def test_runpod_adapter_recovers_persisted_pod_after_backend_restart(tmp_path):
    commands = []

    def request(method, path, payload):
        if method == "POST":
            return {"id": "pod-recover", "costPerHr": 0.25}
        return {
            "desiredStatus": "RUNNING",
            "publicIp": "203.0.113.11",
            "portMappings": {"22": 2233},
            "costPerHr": 0.25,
        }

    def runner(command):
        commands.append(command)
        if "cat" in command:
            progress = JobProgress(
                "delegate", "running", stage="Remote worker", message="still running"
            )
            return type("Result", (), {"stdout": json.dumps(progress.to_dict())})()
        return type("Result", (), {"stdout": ""})()

    first = RunPodExecutionBackend(
        tmp_path,
        api_key="secret",
        image_name="registry/blender-nrp:latest",
        worker_root="/opt/Blender-NRP",
        requester=request,
        ssh_runner=runner,
    )
    job_id = first.submit(BakeJob("scene", "scene.blend", str(tmp_path)))
    assert first.status(job_id).state == "running"
    metadata = json.loads((tmp_path / "runpod" / job_id / "pod.json").read_text())
    assert metadata["delegate_id"] == "pod-recover-bake"
    assert metadata["handoff_complete"] is True
    initial_launches = sum("nohup" in command[-1] for command in commands)
    initial_rsyncs = sum(command[0] == "rsync" for command in commands)
    restarted = RunPodExecutionBackend(
        tmp_path,
        api_key="secret",
        image_name="registry/blender-nrp:latest",
        worker_root="/opt/Blender-NRP",
        requester=request,
        ssh_runner=runner,
    )
    progress = restarted.status(job_id)
    assert progress.state == "running"
    assert progress.cost_per_hour == 0.25
    assert sum("nohup" in command[-1] for command in commands) == initial_launches
    assert sum(command[0] == "rsync" for command in commands) == initial_rsyncs


def test_runpod_adapter_retries_interrupted_handoff_with_same_guarded_id(tmp_path):
    def request(method, path, payload):
        if method == "POST":
            return {"id": "pod-interrupted", "costPerHr": 0.25}
        return {
            "desiredStatus": "RUNNING",
            "publicIp": "203.0.113.12",
            "portMappings": {"22": 2244},
        }

    commands = []

    def runner(command):
        commands.append(command)
        if "cat" in command:
            return type("Result", (), {"stdout": ""})()
        return type("Result", (), {"stdout": ""})()

    first = RunPodExecutionBackend(
        tmp_path,
        api_key="secret",
        image_name="registry/blender-nrp:latest",
        worker_root="/opt/Blender-NRP",
        requester=request,
        ssh_runner=runner,
    )
    job_id = first.submit(BakeJob("scene", "scene.blend", str(tmp_path)))
    metadata_path = tmp_path / "runpod" / job_id / "pod.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "delegate_id": "pod-interrupted-bake",
            "ssh_host": "203.0.113.12",
            "ssh_port": 2244,
            "handoff_complete": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata))

    restarted = RunPodExecutionBackend(
        tmp_path,
        api_key="secret",
        image_name="registry/blender-nrp:latest",
        worker_root="/opt/Blender-NRP",
        requester=request,
        ssh_runner=runner,
    )
    progress = restarted.status(job_id)
    assert progress.state == "running"
    launch = next(command[-1] for command in commands if "nohup" in command[-1])
    assert "/pod-interrupted-bake/.worker-launched" in launch
    assert json.loads(metadata_path.read_text())["handoff_complete"] is True
