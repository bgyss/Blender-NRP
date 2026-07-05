# Development Guide

## Lineage

Blender-NRP implements a Blender-facing workflow derived from Sancho et al.'s Disney
Research paper
[*Neural Render Proxies for Interactive and Differentiable Lighting*](https://studios.disneyresearch.com/2026/07/01/neural-render-proxies-for-interactive-and-differentiable-lighting/).
Use [`bgyss/nrp`](https://github.com/bgyss/nrp) as the behavioral and file-format
reference for path caches, gather semantics, packed layouts, light JSON, torch proxy
artifacts, and inverse-light optimization behavior.

## Blender Add-on Practices

This repository follows a few constraints that make Blender add-ons easier to maintain:

- Keep `blender_nrp/__init__.py` small and use it only as the add-on entrypoint.
- Keep `bl_info` for normal add-on installs and `blender_manifest.toml` for Blender 4.x extension packaging.
- Put Blender-dependent code in registration, panel, property, operator, and backend modules.
- Keep schema, JSON, validation, and report logic independent of `bpy` so it can run in normal Python tests.
- Register and unregister classes in deterministic order, unregistering in reverse order.
- Make generated files live under ignored directories such as `build/`, `dist/`, and `output/`.
- Treat UI operators as thin orchestration layers; put reusable behavior in `blender_nrp/core`.
- Fail with clear user-facing status messages from operators instead of silent console-only errors.

## Validation Levels

Use three levels of validation as the project grows:

1. Pure-Python tests for metadata, light JSON, path-cache schema, and reports.
2. Blender background-mode scripts for fixture bake and relight flows.
3. Manual Blender UI checks for installation, panels, object creation, preview images, and import/export.

Run the full Blender operator smoke test with:

```bash
blender --background --factory-startup --python-exit-code 7 --python tests/blender_smoke.py
```

The smoke test creates a tiny scene, registers the add-on, runs the bake, validate,
train, load, create-light, preview, export, import, and optimize operators, then
checks the generated artifacts under `build/blender_smoke/`.

## Packaging

Build an installable add-on zip with:

```bash
python scripts/package_addon.py
```

The zip includes:

- `blender_manifest.toml`
- `README.md`
- `blender_nrp/`

Do not put generated zips, caches, models, previews, or production scene downloads under version control.

## Dependency Policy

The add-on should degrade gracefully when optional ML dependencies are unavailable inside Blender:

- cache and metadata validation should work with the basic runtime dependencies,
- training and proxy inference should report missing PyTorch clearly,
- Blender scene import/export should not require ComfyUI to be running,
- ComfyUI compatibility should be verified through files, not through a mandatory live server.
