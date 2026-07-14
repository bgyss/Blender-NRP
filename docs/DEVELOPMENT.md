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

Use the repository's validation tiers:

1. Pure-Python tests for metadata, light JSON, path-cache/job schemas, execution
   backends, torch parity, and reports.
2. Cross-repo round trips against the sibling `nrp` and ComfyUI implementations;
   pass `--artifact-dir` to verify worker-produced outputs directly.
3. Blender background fixture scripts and the full operator smoke.
4. Manual Blender UI checks, including one-button cancellation/restart reconciliation,
   gaffer masks, snapshots, and Match Reference apply/discard.
5. The optional worker-container smoke when a Docker/GPU runner is available.

Run the full Blender operator smoke test with:

```bash
blender --background --factory-startup --python-exit-code 7 --python tests/blender_smoke.py
```

The smoke test creates a tiny scene, registers the add-on, runs the granular V2 chain,
then drives the V3 one-button local-subprocess backend through cache validation,
training, proxy auto-load, starter-light creation, preview, and staleness hashes.

Build and smoke the worker container with:

```bash
docker build --build-arg BLENDER_URL="$BLENDER_URL" -f Dockerfile.worker -t blender-nrp-worker:smoke .
scripts/container_smoke.sh blender-nrp-worker:smoke
```

For an image-structure smoke on a runner that cannot install torch, add
`--build-arg INSTALL_TORCH=0` and invoke the smoke as
`scripts/container_smoke.sh blender-nrp-worker:smoke python`. This is an analytic
fallback check only; it does not replace the default torch-mesh container gate.

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
