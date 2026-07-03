#!/usr/bin/env python
"""Package Blender-NRP as an installable add-on zip."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE_NAME = "Blender-NRP.zip"
INCLUDE_DIRS = ("blender_nrp",)
INCLUDE_FILES = ("blender_manifest.toml", "README.md", "LICENSE")


def main() -> None:
    DIST.mkdir(exist_ok=True)
    target = DIST / PACKAGE_NAME
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in INCLUDE_FILES:
            archive.write(ROOT / file_name, file_name)
        for dirname in INCLUDE_DIRS:
            for path in sorted((ROOT / dirname).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT))
    print(target)


if __name__ == "__main__":
    main()
