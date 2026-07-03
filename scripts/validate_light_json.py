#!/usr/bin/env python
"""Validate an NRP light-rig JSON file."""

from __future__ import annotations

from blender_nrp.core.lights import LightRig
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    args = parser.parse_args()
    try:
        rig = LightRig.load(args.json_path)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"OK {len(rig.lights)} lights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

