# Blender-NRP

`Blender-NRP` is a Blender add-on scaffold for baking Neural Render Proxy path caches and relighting fixed-camera scenes inside Blender.

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

The add-on exposes a `Blender-NRP` panel in Scene properties. Early operators are intentionally conservative placeholders until the bake, proxy, and relight internals land.

## Compatibility Contracts

The add-on targets the same core contracts as `nrp` and `ComfyUI-NeuralRenderProxy`:

- `path_cache.npz` with `n_paths`, `seg_pixel`, `seg_origin`, `seg_dir`, `seg_tmax`, `seg_throughput`, `albedo`, `normal`, `depth`, and `position`.
- `metadata.json` with fixed-scene, fixed-camera, sphere-light metadata.
- sphere-light rig JSON containing `scene_id`, `camera_id`, `coordinate_system`, and `lights`.

Generated caches, model files, previews, and reports should stay under ignored output directories.

