# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`Blender-NRP` is a Blender add-on for baking Neural Render Proxy (NRP) path caches and relighting
fixed-camera scenes inside Blender. V1 goal is functional correctness, not real-time performance:
bake a light-agnostic path cache, train/load a proxy, create/edit NRP sphere lights as Blender
objects, preview fixed-camera relighting, and import/export ComfyUI-compatible light rigs.

The full build goal/spec lives in `docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md`.
User-facing install/usage docs are in `docs/USER_GUIDE.md`.

## Commands

```bash
python -m pip install -e ".[dev]"   # dev environment
python -m pytest                     # pure-Python tests
python -m ruff check .               # lint (line-length 100, py311, rules E,F,I,UP,B)
python scripts/package_addon.py      # build dist/Blender-NRP.zip for install in Blender
```

Run a single test file/case the normal pytest way, e.g. `python -m pytest tests/test_light_json.py -k foo`.

Tested fixture workflow (writes to `build/nrp/fixture_room_001/`, gitignored):

```bash
python scripts/bake_fixture.py
python scripts/validate_cache.py build/nrp/fixture_room_001/path_cache.npz build/nrp/fixture_room_001/metadata.json
python scripts/relight_fixture.py
python scripts/validate_light_json.py build/nrp/fixture_room_001/solved_lights.json
```

When Blender is available, run the same scripts against a real scene in background mode:

```bash
blender --background tests/fixtures/minimal_scene.blend --python scripts/bake_fixture.py
blender --background tests/fixtures/minimal_scene.blend --python scripts/relight_fixture.py
```

If `tests/fixtures/minimal_scene.blend` doesn't exist, create a small Cornell-style room with a
camera and one light in Blender and save it there. The pure-Python fixture path (no `.blend`
needed) still validates schema/report shape without Blender.

Full Blender operator smoke test (registers the add-on, runs the whole operator chain, checks
artifacts under `build/blender_smoke/`):

```bash
blender --background --factory-startup --python-exit-code 7 --python tests/blender_smoke.py
```

## Architecture

The core design constraint: **everything that doesn't strictly need `bpy` must be importable and
testable without Blender.** This is what makes `python -m pytest` possible outside the Blender
Python environment.

```
blender_nrp/
  __init__.py        add-on entrypoint only: bl_info + register()/unregister() that defer
                      all bpy-touching imports into addon.py (keeps this module importable
                      by plain pytest without Blender installed)
  addon.py            registration orchestration — MODULES tuple lists every bpy-dependent
                      module in registration order; register() walks forward, unregister()
                      walks in reverse
  core/               bpy-independent: path-cache schema, metadata, light JSON, path caching,
                      validation, report logic. This is what pure-Python tests exercise.
  backends/           PathCacheBackend Protocol (bake(context, BakeSettings) -> BakeResult) +
                      stock_blender_hemi.py, the V1 backend. It ray-casts one camera sample per
                      pixel, records the first visible hit, and writes deterministic
                      normal-oriented light-transport spokes — a workflow-validation backend,
                      not a physically exact Cycles path capture. New backends implement the
                      same Protocol.
  operators/          bpy.types.Operator classes — thin orchestration only; real logic belongs
                      in core/. _helpers.py has finish_with_status()/cancel_with_status() for
                      reporting status onto scene.blender_nrp.status instead of failing silently
                      to console only.
  panels.py           Scene properties UI panel ("Blender-NRP" panel under Scene properties)
  properties.py       bpy.props definitions attached to the scene
  ui/                 UI helper code
scripts/              CLI entry points for packaging and fixture bake/relight/validate flows
                      (these double as the non-Blender and Blender-background test harnesses)
tests/                pure-Python tests + blender_smoke.py (Blender-background operator chain test)
examples/scene_manifests/   production scene manifests: upstream URLs, license notes, camera/frame
                             settings, expected artifact names for manual QA against real shots
```

### Compatibility contracts

The add-on must stay wire-compatible with `nrp` and `ComfyUI-NeuralRenderProxy`:

- `path_cache.npz`: `n_paths`, `seg_pixel`, `seg_origin`, `seg_dir`, `seg_tmax`, `seg_throughput`,
  `albedo`, `normal`, `depth`, `position`, plus the scalar keys `schema_version`, `width`, `height`
  required by `nrp`'s `PathCache.load`. `seg_throughput` is raw (not pre-averaged); per-pixel
  averaging divides by `n_paths` at gather time, matching `nrp` main's GATHERLIGHT.
- `metadata.json`: fixed-scene, fixed-camera, sphere-light metadata.
- sphere-light rig JSON: `scene_id`, `camera_id`, `coordinate_system`, `lights`.

Changing these shapes breaks interop with the sibling projects — treat them as an external API.

### Conventions specific to this repo

- Keep `blender_nrp/__init__.py` minimal (entrypoint only); put real registration logic in `addon.py`.
- `bl_info` (in `__init__.py`) and `blender_manifest.toml` (Blender 4.x extension packaging) must
  both be kept in sync when bumping version/metadata.
- Register/unregister classes in deterministic order; unregister in reverse of register.
- Generated artifacts (caches, trained models, previews, reports, zips) always go under gitignored
  directories (`build/`, `dist/`) — never commit them.
- Training/proxy inference should degrade gracefully and report missing PyTorch clearly rather than
  hard-crashing; cache/metadata validation must work with only the base runtime deps; scene
  import/export must not require a running ComfyUI server.
- Use three validation levels as functionality grows: pure-Python tests (`core/` logic), Blender
  background-mode fixture scripts, and manual Blender UI checks (panels, object creation, previews,
  import/export).
