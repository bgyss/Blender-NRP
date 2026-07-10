"""Shared no-bpy worker mechanics for the V3 job entrypoints."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_nrp.core.jobs import JobProgress, read_job, write_progress


def parse_args(description: str) -> tuple[Path, Path]:
    argv = sys.argv[1:]
    if "--" in argv:  # Blender forwards script arguments after this separator.
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("job")
    parser.add_argument("--status", required=True)
    parsed = parser.parse_args(argv)
    return Path(parsed.job), Path(parsed.status)


def run_worker(description: str, execute) -> int:
    job_path, status_path = parse_args(description)
    job_id = job_path.stem
    try:
        job = read_job(job_path)
        write_progress(status_path, JobProgress(job_id, "running", 0.02, "Starting"))
        artifacts = execute(
            job,
            lambda fraction, stage, message="": write_progress(
                status_path, JobProgress(job_id, "running", fraction, stage, message)
            ),
        )
        write_progress(
            status_path,
            JobProgress(
                job_id,
                "succeeded",
                1.0,
                "Complete",
                artifacts={k: str(v) for k, v in artifacts.items()},
            ),
        )
        return 0
    except Exception as exc:
        write_progress(
            status_path,
            JobProgress(
                job_id,
                "failed",
                stage="Failed",
                message=str(exc),
                error=traceback.format_exc(limit=4),
            ),
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1
