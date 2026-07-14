# Blender-NRP

`Blender-NRP` is a Blender add-on for baking Neural Render Proxy path caches and relighting fixed-camera scenes inside Blender.

The project is based on Sancho et al.'s Disney Research paper
[*Neural Render Proxies for Interactive and Differentiable Lighting*](https://studios.disneyresearch.com/2026/07/01/neural-render-proxies-for-interactive-and-differentiable-lighting/)
and derives its data contracts and torch proxy behavior from the
[`bgyss/nrp`](https://github.com/bgyss/nrp) sample reimplementation.

V3 (`0.4.0`) adds remote execution and the production workflow while retaining V2's compatibility
contracts and correctness-first approach:

- bake a light-agnostic multi-bounce path cache (Cycles G-buffer passes, escape segments, optional packed fp16 + rgb9e5 layout) with a modal, cancellable operator,
- train a real PyTorch neural proxy in the background (cosine LR, checkpoint/resume, MPS/CUDA) writing an `nrp`-loadable `model.pt`,
- create and edit NRP sphere *and quad* lights as Blender objects,
- preview fixed-camera relighting live inside Blender (Image datablock, debounced auto-update) via the proxy or the exact cache gather,
- solve a light rig against a target image with gradients through the proxy (coordinate-descent fallback without torch),
- import and export ComfyUI-compatible light rigs with real coordinate conversion.
- submit versioned bake/train/solve jobs to a local background worker, with durable
  progress files ready for LAN and cloud adapters,
- use **Make Scene Relightable** for the default local bake → validate → train →
  load → starter-light → preview path; detailed V2 settings remain under Advanced.
- use SSH/LAN or RunPod Cloud from the same Compute selector when a configured
  worker image/node is available, with cost and cancellation state kept in the job
  status rather than scene artifacts.

The build goals live in [docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md](docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md) (V1), [docs/prompts/blender-nrp-v2-goal-prompt.md](docs/prompts/blender-nrp-v2-goal-prompt.md) (V2), and [docs/prompts/blender-nrp-v3-goal-prompt.md](docs/prompts/blender-nrp-v3-goal-prompt.md) (V3).

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

## Lineage And Compatibility

The original Neural Render Proxies paper decouples rendering into a light-agnostic
path pass and a light-dependent gather pass, then trains a compact neural proxy for
interactive and differentiable relighting. `Blender-NRP` adapts that workflow into a
native Blender add-on.

The implementation follows [`bgyss/nrp`](https://github.com/bgyss/nrp) as the reference
for path-cache vocabulary, gather semantics, packed cache layout, light JSON shape,
and `TorchNRP` model artifacts. The add-on also keeps ComfyUI interchange support for
light rigs and exported cache bundles.

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

For Spring or Sprite Fright, download the production files from Blender Studio, open the
chosen shot in Blender, install `dist/Blender-NRP.zip`, and use the Scene Properties
`Blender-NRP` panel:

1. Set `Scene ID`, select the `Camera`, and choose a resolution and output directory.
   Start at `64 × 64` to validate the pipeline before committing to full resolution.
2. Click **Bake Path Cache** — validation runs automatically at the end; the
   `1 · Path Cache` chip flips to **✓ baked** and the status shows
   *"Baked + validated…"* with a toast.
3. Click **Train Proxy** — when training finishes the proxy is auto-loaded; the
   `2 · Neural Proxy` chip flips to **✓ loaded** and the status ends with
   *"proxy auto-loaded"*. Use `Load Proxy` only when reopening a scene with a
   pre-existing `model.pt`.
4. In Stage 3, click **Sphere** (appears at the 3D cursor) and position it, then
   click **Preview Relight** — the `3 · Relight` chip flips to **✓ preview ready**.
5. Split off an Image Editor — it already shows `NRP Relight Preview`. Enable
   **Live Preview** and drag the light; the preview refreshes ~0.3 s after you stop.
6. Set a target image and click **Solve** to run inverse light optimization (gradient
   descent through the torch proxy, or coordinate descent without torch), then export
   the solved light rig JSON.

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
