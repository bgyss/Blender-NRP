# Blender-NRP

`Blender-NRP` is a Blender add-on for baking Neural Render Proxy path caches and relighting fixed-camera scenes inside Blender.

The first implementation goal is functional correctness, not real-time performance:

- bake a light-agnostic path cache from a Blender scene,
- train or load a proxy,
- create and edit NRP sphere lights as Blender objects,
- preview fixed-camera relighting,
- import and export ComfyUI-compatible light rigs.

The detailed build goal lives in [docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md](docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md).

## Repository Layout

```text
blender_nrp/          Blender add-on package
  core/               Blender-independent cache, metadata, light, and validation logic
  backends/           Bake backend interface and stock Blender backend placeholder
  operators/          Blender operators
  ui/                 UI helpers
scripts/             Packaging and validation commands
tests/               Pure-Python tests
examples/            Scene manifests and small examples
docs/prompts/        Goal prompts and implementation notes
```

## Development

Create a development environment and run the pure-Python checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Package the add-on zip:

```bash
python scripts/package_addon.py
```

The package command writes `dist/Blender-NRP.zip`, which can be installed through Blender's add-on preferences.

## Blender Installation

For a local development install, either install the generated zip in Blender or place the repository on Blender's add-on search path so the `blender_nrp` package is importable.

The add-on exposes a `Blender-NRP` panel in Scene properties. V1 uses a stock-Blender hemisphere backend: it ray casts one camera sample per pixel, records the first visible hit, and writes deterministic normal-oriented light-transport spokes. This is useful for workflow validation and fixed-shot relighting iteration, but it is not a physically exact Cycles path capture.

## Tested Fixture Workflow

The fixture commands write generated artifacts under `build/nrp/fixture_room_001/`, which is ignored by git:

```bash
python scripts/bake_fixture.py
python scripts/validate_cache.py build/nrp/fixture_room_001/path_cache.npz build/nrp/fixture_room_001/metadata.json
python scripts/relight_fixture.py
python scripts/validate_light_json.py build/nrp/fixture_room_001/solved_lights.json
```

Expected outputs include:

- `path_cache.npz`
- `metadata.json`
- `bake_report.json`
- `preview_albedo.png`
- `preview_normal.png`
- `preview_depth.png`
- `model.pt`
- `train_report.json`
- `relight_preview.png`
- `solved_lights.json`
- `solve_report.json`

When Blender is available, run the same scripts in background mode against a local fixture scene:

```bash
blender --background tests/fixtures/minimal_scene.blend --python scripts/bake_fixture.py
blender --background tests/fixtures/minimal_scene.blend --python scripts/relight_fixture.py
```

If `tests/fixtures/minimal_scene.blend` is not present in a fresh checkout, open Blender, create a small Cornell-style room with a camera and one light, and save it at that path. The pure-Python fixture path remains available for schema and report validation without Blender.

## Manual Production Scene Path

For Spring or Sprite Fright, download the production files from Blender Studio, open the chosen shot in Blender, install `dist/Blender-NRP.zip`, and use the Scene properties `Blender-NRP` panel:

1. Set a stable scene ID, camera, resolution, output directory, segment count, and max segment distance.
2. Click `Bake Path Cache`, then `Validate Cache`.
3. Click `Train Proxy` or set `Model Path` to an existing compatible model.
4. Create or import NRP sphere lights.
5. Click `Preview Relight` to write `relight_preview.png`.
6. Click `Optimize Lights From Target` for the V1 deterministic solve path, then export the solved light rig JSON.

The production scene manifests in `examples/scene_manifests/` document upstream URLs, license notes, suggested camera/frame settings, and expected artifact names. Large `.blend` files, generated caches, trained weights, and preview images should stay outside git.

## Compatibility Contracts

The add-on targets the same core contracts as `nrp` and `ComfyUI-NeuralRenderProxy`:

- `path_cache.npz` with `n_paths`, `seg_pixel`, `seg_origin`, `seg_dir`, `seg_tmax`, `seg_throughput`, `albedo`, `normal`, `depth`, and `position`.
- `metadata.json` with fixed-scene, fixed-camera, sphere-light metadata.
- sphere-light rig JSON containing `scene_id`, `camera_id`, `coordinate_system`, and `lights`.

Generated caches, model files, previews, and reports should stay under ignored output directories.
