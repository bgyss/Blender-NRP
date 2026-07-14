#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-blender-nrp-worker:smoke}"
TRACER_ENGINE="${2:-torch_mesh}"
OUTPUT="$ROOT/build/container_smoke"

mkdir -p "$OUTPUT"
cp "$ROOT/tests/fixtures/minimal_scene.blend" "$OUTPUT/minimal_scene.blend"
python3 -c '
import sys
from blender_nrp.core.jobs import BakeJob, write_job

target = sys.argv[1]
tracer_engine = sys.argv[2]
write_job(
    target,
    BakeJob(
        "container_fixture",
        "/work/minimal_scene.blend",
        "/work/artifacts",
        width=8,
        height=8,
        paths_per_pixel=2,
        max_bounces=2,
        packed=True,
        torch_device="cpu",
        tracer_engine=tracer_engine,
    ),
)
' "$OUTPUT/bake_job.json" "$TRACER_ENGINE"

docker run --rm --platform linux/amd64 \
    -v "$OUTPUT:/work" \
    "$IMAGE" \
    blender --background /work/minimal_scene.blend \
    --python /opt/Blender-NRP/scripts/run_bake_job.py -- \
    /work/bake_job.json --status /work/status.json

python3 "$ROOT/scripts/validate_cache.py" \
    "$OUTPUT/artifacts/container_fixture/path_cache.npz" \
    "$OUTPUT/artifacts/container_fixture/metadata.json"
python3 -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
expected_engine = sys.argv[2]
if not report.get("ok"):
    raise SystemExit(f"container bake report failed: {report}")
if not str(report.get("tracer_engine", "")).startswith(expected_engine):
    raise SystemExit(
        f"container used {report.get('tracer_engine')!r}, expected {expected_engine!r}: {report}"
    )
' "$OUTPUT/artifacts/container_fixture/bake_report.json" \
    "$(if [[ "$TRACER_ENGINE" == "torch_mesh" ]]; then echo 'torch_mesh:cpu'; else echo 'python_ray_cast'; fi)"

echo "CONTAINER_SMOKE_OK"
