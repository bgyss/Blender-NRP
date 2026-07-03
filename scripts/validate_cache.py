#!/usr/bin/env python
"""Validate a path-cache and metadata bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blender_nrp.core.validation import validate_cache_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("metadata")
    args = parser.parse_args()
    report = validate_cache_bundle(args.cache, args.metadata)
    if not report.ok:
        for error in report.errors:
            print(error, file=sys.stderr)
        return 1
    print(f"OK {report.width}x{report.height} {report.segment_count} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
