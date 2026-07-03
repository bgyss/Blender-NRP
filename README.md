# Blender-NRP

`Blender-NRP` is a Blender add-on for baking Neural Render Proxy path caches and relighting fixed-camera scenes inside Blender.

V2 targets parity with the current `nrp` reference, still prioritizing correctness over real-time performance:

- bake a light-agnostic multi-bounce path cache (Cycles G-buffer passes, escape segments, optional packed fp16 + rgb9e5 layout) with a modal, cancellable operator,
- train a real PyTorch neural proxy in the background (cosine LR, checkpoint/resume, MPS/CUDA) writing an `nrp`-loadable `model.pt`,
- create and edit NRP sphere *and quad* lights as Blender objects,
- preview fixed-camera relighting live inside Blender (Image datablock, debounced auto-update) via the proxy or the exact cache gather,
- solve a light rig against a target image with gradients through the proxy (coordinate-descent fallback without torch),
- import and export ComfyUI-compatible light rigs with real coordinate conversion.

The build goals live in [docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md](docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md) (V1) and [docs/prompts/blender-nrp-v2-goal-prompt.md](docs/prompts/blender-nrp-v2-goal-prompt.md) (V2).

For installation, usage, and a quick tutorial, read [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Repository Layout

```text
blender_nrp/          Blender add-on package
  core/               Blender-independent cache, metadata, light, and validation logic
  backends/           Bake backend interface, cycles_capture (V2), stock hemisphere (V1 fallback)
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

The add-on exposes a `Blender-NRP` panel in Scene properties with two capture backends: `cycles_capture` (V2 default — Python-driven multi-bounce Lambertian transport over `scene.ray_cast` with real Cycles G-buffer passes and an A/B PSNR against a Cycles reference render in `bake_report.json`) and the V1 `stock_blender_hemi` fallback (first-hit + deterministic hemisphere spokes). Neither is claimed to be an exact Cycles kernel capture; the reports state the approximations.

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
6. Set a target image and click `Solve` to run inverse light optimization (gradient descent through the torch proxy, or coordinate descent without torch), then export the solved light rig JSON.

The production scene manifests in `examples/scene_manifests/` document upstream URLs, license notes, suggested camera/frame settings, and expected artifact names. Large `.blend` files, generated caches, trained weights, and preview images should stay outside git.

## Compatibility Contracts

The add-on targets the same core contracts as `nrp` and `ComfyUI-NeuralRenderProxy`:

- `path_cache.npz` with `n_paths`, `seg_pixel`, `seg_origin`, `seg_dir`, `seg_tmax`, `seg_throughput`, `albedo`, `normal`, `depth`, and `position`.
- the packed cache layout (`packed_layout` key, fp16 geometry + rgb9e5 throughput) is read natively and optionally written.
- `metadata.json` with fixed-scene, fixed-camera light metadata, including `throughput_normalization` (this repo stores raw throughput normalized by `n_paths` at gather time; the ComfyUI export path pre-divides instead).
- light rig JSON containing `scene_id`, `camera_id`, `coordinate_system`, and `lights` (spheres and `"type": "quad"` entries; untyped entries stay spheres).
- `model.pt` in `nrp`'s `TorchNRP` format.

Run the cross-repo round-trip check against the actual sibling implementations with `python scripts/cross_repo_roundtrip.py`.

Generated caches, model files, previews, and reports should stay under ignored output directories.
