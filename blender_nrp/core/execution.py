"""Execution backend contract and local-subprocess implementation.

This layer only knows files and processes.  Blender-specific polling belongs in
operators, which allows the exact same job bundle to move to SSH/cloud backends.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path
from typing import Protocol

from .jobs import Job, JobProgress, read_progress, write_job


class ExecutionBackend(Protocol):
    def submit(self, job: Job) -> str: ...
    def status(self, job_id: str) -> JobProgress: ...
    def fetch(self, job_id: str) -> dict[str, Path]: ...
    def cancel(self, job_id: str) -> None: ...


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
