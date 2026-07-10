"""Execution backend contract and local-subprocess implementation.

This layer only knows files and processes.  Blender-specific polling belongs in
operators, which allows the exact same job bundle to move to SSH/cloud backends.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Protocol

from .jobs import BakeJob, Job, JobProgress, SolveJob, TrainJob, read_progress, write_job


class ExecutionBackend(Protocol):
    def submit(self, job: Job) -> str: ...
    def status(self, job_id: str) -> JobProgress: ...
    def fetch(self, job_id: str) -> dict[str, Path]: ...
    def cancel(self, job_id: str) -> None: ...


@dataclass(frozen=True)
class QueuedJob:
    """Persistent, backend-agnostic queue record.

    Queue records contain no credentials. Backend configuration stays in add-on
    preferences or the invoking process; a reopened Blender reconciles records
    by asking that configured backend for their latest status.
    """

    job_id: str
    backend_id: str
    submitted_at: float
    job_path: str


class ExecutionQueue:
    """Small durable queue that survives Blender restarts."""

    FILE_NAME = "queue.json"

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / self.FILE_NAME

    def load(self) -> list[QueuedJob]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [QueuedJob(**item) for item in payload]

    def save(self, records: list[QueuedJob]) -> None:
        self.path.write_text(
            json.dumps([asdict(record) for record in records], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def add(self, record: QueuedJob) -> None:
        records = [item for item in self.load() if item.job_id != record.job_id]
        records.append(record)
        self.save(records)

    def remove(self, job_id: str) -> None:
        self.save([item for item in self.load() if item.job_id != job_id])

    def reconcile(self, backends: dict[str, ExecutionBackend]) -> dict[str, JobProgress]:
        """Return fresh statuses and retain only work that is not terminal."""
        remaining: list[QueuedJob] = []
        statuses: dict[str, JobProgress] = {}
        for record in self.load():
            backend = backends.get(record.backend_id)
            if backend is None:
                statuses[record.job_id] = JobProgress(
                    record.job_id,
                    "failed",
                    stage="Reconciliation",
                    message=f"configured execution backend {record.backend_id!r} is unavailable",
                )
                remaining.append(record)
                continue
            progress = backend.status(record.job_id)
            statuses[record.job_id] = progress
            if progress.state in {"queued", "running"}:
                remaining.append(record)
        self.save(remaining)
        return statuses


class LocalSubprocessBackend:
    """Worker runner with durable job/progress files and explicit cancellation."""

    def __init__(self, queue_dir: str | Path, *, blender_binary: str | None = None):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.blender_binary = blender_binary
        self._processes: dict[str, subprocess.Popen] = {}

    def _paths(self, job_id: str) -> tuple[Path, Path]:
        base = self.queue_dir / job_id
        return base.with_suffix(".json"), base.with_suffix(".status.json")

    def submit(self, job: Job) -> str:
        job_id = uuid.uuid4().hex
        job_path, status_path = self._paths(job_id)
        write_job(job_path, job)
        script = Path(__file__).resolve().parents[2] / "scripts" / f"run_{job.kind}_job.py"
        command = [sys.executable, str(script), str(job_path), "--status", str(status_path)]
        # Blender's embedded interpreter is the only reliable local Python from an
        # add-on process. Worker scripts remain plain-Python compatible for CI and
        # remote hosts, but use Blender when the caller supplies its executable.
        if self.blender_binary:
            command = [self.blender_binary, "--background"]
            if job.kind == "bake":
                command.append(job.scene_path)
            else:
                command.append("--factory-startup")
            command += ["--python", str(script), "--", str(job_path), "--status", str(status_path)]
        self._processes[job_id] = subprocess.Popen(command, start_new_session=True)
        return job_id

    def status(self, job_id: str) -> JobProgress:
        _job, status_path = self._paths(job_id)
        if status_path.exists():
            return read_progress(status_path)
        proc = self._processes.get(job_id)
        if proc is not None and proc.poll() is not None:
            return JobProgress(
                job_id, "failed", stage="Worker", message="worker exited before reporting"
            )
        return JobProgress(job_id, "queued")

    def fetch(self, job_id: str) -> dict[str, Path]:
        progress = self.status(job_id)
        if progress.state != "succeeded":
            raise RuntimeError(f"job {job_id} is {progress.state}")
        return {name: Path(path) for name, path in progress.artifacts.items()}

    def cancel(self, job_id: str) -> None:
        proc = self._processes.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class SshExecutionBackend:
    """LAN worker adapter using only SSH and rsync.

    The remote machine receives the same JSON job and packed scene that a cloud
    adapter consumes. It needs only an SSH account, rsync, Blender, and the
    Blender-NRP checkout/image referenced by ``worker_root``.
    """

    id = "ssh"

    def __init__(
        self,
        queue_dir: str | Path,
        *,
        host: str,
        remote_root: str,
        worker_root: str,
        blender_binary: str = "blender",
        ssh_port: int = 22,
        runner: CommandRunner | None = None,
    ):
        if not host or not remote_root or not worker_root:
            raise ValueError("SSH host, remote_root, and worker_root are required")
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.remote_root = remote_root.rstrip("/")
        self.worker_root = worker_root.rstrip("/")
        self.blender_binary = blender_binary
        self.ssh_port = int(ssh_port)
        self.runner = runner or self._run
        self._remote: dict[str, dict[str, str]] = {}

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=True, text=True, capture_output=True)

    def _ssh(self, *args: str) -> list[str]:
        return ["ssh", "-p", str(self.ssh_port), self.host, *args]

    def _rsync(self, source: str, target: str) -> list[str]:
        return ["rsync", "-a", "-e", f"ssh -p {self.ssh_port}", source, target]

    def _paths(self, job_id: str) -> tuple[Path, Path, Path]:
        directory = self.queue_dir / job_id
        return directory / "job.json", directory / "status.json", directory / "artifacts"

    def _remote_info(self, job_id: str) -> dict[str, str]:
        return {
            "dir": f"{self.remote_root}/{job_id}",
            "output": f"{self.remote_root}/{job_id}/artifacts",
            "status": f"{self.remote_root}/{job_id}/status.json",
        }

    def submit(self, job: Job) -> str:
        job_id = uuid.uuid4().hex
        job_path, _status_path, _artifacts = self._paths(job_id)
        job_path.parent.mkdir(parents=True, exist_ok=True)
        remote_dir = self._remote_info(job_id)["dir"]
        remote_scene = (
            f"{remote_dir}/{Path(job.scene_path).name}" if isinstance(job, BakeJob) else ""
        )
        remote_output = f"{remote_dir}/artifacts"
        replacement: dict[str, str] = {"output_dir": remote_output}
        staged: list[tuple[str, str]] = []
        if isinstance(job, BakeJob):
            replacement["scene_path"] = remote_scene
            staged.append((job.scene_path, remote_scene))
        elif isinstance(job, TrainJob):
            remote_cache = f"{remote_dir}/{Path(job.cache_path).name}"
            replacement["cache_path"] = remote_cache
            staged.append((job.cache_path, remote_cache))
        elif isinstance(job, SolveJob):
            remote_cache = f"{remote_dir}/{Path(job.cache_path).name}"
            remote_lights = f"{remote_dir}/{Path(job.lights_path).name}"
            remote_target = f"{remote_dir}/{Path(job.target_path).name}"
            replacement.update(
                {
                    "cache_path": remote_cache,
                    "lights_path": remote_lights,
                    "target_path": remote_target,
                }
            )
            staged.extend(
                [
                    (job.cache_path, remote_cache),
                    (job.lights_path, remote_lights),
                    (job.target_path, remote_target),
                ]
            )
        remote_job = dataclass_replace(job, **replacement)
        write_job(job_path, remote_job)
        self.runner(self._ssh("mkdir", "-p", remote_dir, remote_output))
        self.runner(self._rsync(str(job_path), f"{self.host}:{remote_dir}/job.json"))
        for source, target in staged:
            self.runner(self._rsync(source, f"{self.host}:{target}"))
        worker = f"{self.worker_root}/scripts/run_{job.kind}_job.py"
        status = f"{remote_dir}/status.json"
        scene_arg = remote_scene if isinstance(job, BakeJob) else "--factory-startup"
        arguments = " ".join(
            (scene_arg, "--python", worker, "--", f"{remote_dir}/job.json", "--status", status)
        )
        command = (
            f"nohup {self.blender_binary} --background {arguments} "
            f"> {remote_dir}/worker.log 2>&1 &"
        )
        self.runner(self._ssh(command))
        self._remote[job_id] = {"dir": remote_dir, "output": remote_output, "status": status}
        return job_id

    def status(self, job_id: str) -> JobProgress:
        local_job, local_status, _artifacts = self._paths(job_id)
        remote = self._remote.setdefault(job_id, self._remote_info(job_id))
        try:
            result = self.runner(self._ssh("cat", remote["status"]))
            payload = result.stdout.strip()
            if payload:
                local_status.write_text(payload + "\n", encoding="utf-8")
                return read_progress(local_status)
        except subprocess.CalledProcessError:
            pass
        if local_job.exists():
            return JobProgress(job_id, "running", stage="Remote worker", message="awaiting status")
        return JobProgress(job_id, "failed", stage="Remote worker", message="job record is missing")

    def fetch(self, job_id: str) -> dict[str, Path]:
        progress = self.status(job_id)
        if progress.state != "succeeded":
            raise RuntimeError(f"job {job_id} is {progress.state}")
        remote = self._remote.setdefault(job_id, self._remote_info(job_id))
        _job_path, _status_path, artifacts = self._paths(job_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        self.runner(self._rsync(f"{self.host}:{remote['output']}/", f"{artifacts}/"))
        return {name: artifacts / Path(path).name for name, path in progress.artifacts.items()}

    def cancel(self, job_id: str) -> None:
        remote = self._remote.get(job_id)
        if remote is None:
            return
        self.runner(self._ssh("pkill", "-f", f"{remote['dir']}/job.json"))


class RunPodExecutionBackend:
    """RunPod REST pod lifecycle adapter, using SSH for bundle transfer.

    The configured RunPod image must contain Blender-NRP at ``worker_root`` and
    expose TCP/22.  The adapter creates an idle GPU pod, waits for its public SSH
    mapping, then sends the same job bundle through :class:`SshExecutionBackend`.
    """

    id = "runpod"

    def __init__(
        self,
        queue_dir: str | Path,
        *,
        api_key: str,
        image_name: str,
        worker_root: str,
        gpu_type: str = "NVIDIA GeForce RTX 4090",
        remote_root: str = "/workspace/blender-nrp-jobs",
        api_url: str = "https://rest.runpod.io/v1",
        requester: Callable[[str, str, dict | None], dict] | None = None,
        ssh_runner: CommandRunner | None = None,
    ):
        if not api_key or not image_name or not worker_root:
            raise ValueError("RunPod API key, image name, and worker root are required")
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.image_name = image_name
        self.worker_root = worker_root.rstrip("/")
        self.gpu_type = gpu_type
        self.remote_root = remote_root.rstrip("/")
        self.api_url = api_url.rstrip("/")
        self.requester = requester or self._request
        self.ssh_runner = ssh_runner
        self._jobs: dict[str, dict] = {}

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"RunPod API {exc.code}: {exc.read().decode(errors='replace')}"
            ) from exc
        return json.loads(raw) if raw else {}

    def submit(self, job: Job) -> str:
        pod = self.requester(
            "POST",
            "/pods",
            {
                "name": f"blender-nrp-{job.kind}",
                "imageName": self.image_name,
                "gpuTypeIds": [self.gpu_type],
                "gpuCount": 1,
                "containerDiskInGb": 50,
                "volumeInGb": 20,
                "volumeMountPath": "/workspace",
                "ports": ["22/tcp"],
                "supportPublicIp": True,
                "dockerStartCmd": ["sleep", "infinity"],
            },
        )
        pod_id = str(pod["id"])
        self._jobs[pod_id] = {
            "job": job,
            "submitted": time.monotonic(),
            "cost_per_hour": float(pod.get("costPerHr", 0.0) or 0.0),
            "pod": pod,
            "ssh": None,
            "delegate_id": None,
        }
        return pod_id

    def status(self, job_id: str) -> JobProgress:
        record = self._jobs.get(job_id)
        if record is None:
            return JobProgress(job_id, "failed", stage="RunPod", message="unknown pod")
        pod = self.requester("GET", f"/pods/{job_id}", None)
        record["pod"] = pod
        cost = record["cost_per_hour"]
        accrued = cost * ((time.monotonic() - record["submitted"]) / 3600.0)
        state = str(pod.get("desiredStatus", pod.get("status", ""))).upper()
        if state in {"TERMINATED", "EXITED"}:
            return JobProgress(
                job_id,
                "failed",
                stage="RunPod",
                message=f"pod state: {state}",
                cost_per_hour=cost,
                accrued_cost=accrued,
            )
        if record["ssh"] is None:
            ip = pod.get("publicIp")
            mappings = pod.get("portMappings") or {}
            port = mappings.get("22") or mappings.get(22)
            if not ip or not port:
                return JobProgress(
                    job_id,
                    "queued",
                    stage="RunPod",
                    message="waiting for GPU pod SSH endpoint",
                    cost_per_hour=cost,
                    accrued_cost=accrued,
                )
            record["ssh"] = SshExecutionBackend(
                self.queue_dir / "ssh",
                host=str(ip),
                ssh_port=int(port),
                remote_root=self.remote_root,
                worker_root=self.worker_root,
                blender_binary="blender",
                runner=self.ssh_runner,
            )
            record["delegate_id"] = record["ssh"].submit(record["job"])
        progress = record["ssh"].status(record["delegate_id"])
        progress.cost_per_hour = cost
        progress.accrued_cost = accrued
        return progress

    def fetch(self, job_id: str) -> dict[str, Path]:
        progress = self.status(job_id)
        record = self._jobs[job_id]
        if progress.state != "succeeded":
            raise RuntimeError(f"RunPod job {job_id} is {progress.state}")
        return record["ssh"].fetch(record["delegate_id"])

    def cancel(self, job_id: str) -> None:
        record = self._jobs.get(job_id)
        if record and record["ssh"] is not None:
            record["ssh"].cancel(record["delegate_id"])
        self.requester("DELETE", f"/pods/{job_id}", None)
