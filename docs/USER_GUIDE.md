# Blender-NRP User Guide

This guide covers installing the Blender-NRP add-on, running the built-in fixture workflow,
and using the Blender panel for a fixed-camera bake and relight pass.

Blender-NRP is currently a V1 workflow implementation. It prioritizes correct file
contracts and an end-to-end Blender loop over physical accuracy or real-time preview.
The stock backend records first-hit camera data and deterministic hemisphere spokes;
it does not capture true Cycles multi-bounce paths.

## What The Add-on Does

Blender-NRP helps you:

- bake a fixed Blender camera view into an NRP-style `path_cache.npz`,
- write strict `metadata.json` beside the cache,
- export albedo, normal, and depth preview images,
- create and edit NRP sphere lights as Blender objects,
- preview a fixed-camera cache-gather relight,
- import and export ComfyUI-compatible sphere-light JSON,
- write a lightweight V1 `model.pt` proxy artifact and training report,
- write solved light JSON and solve reports.

Generated files are written under an output directory such as
`build/nrp/fixture_room_001/`, `output/fixture_room_001/`, or a directory you choose
in the Blender panel.

## Requirements

For command-line fixture validation:

- Python 3.11 or newer
- NumPy

For the Blender add-on:

- A current Blender build with Python add-on support
- The packaged add-on zip: `dist/Blender-NRP.zip`

Optional for development:

- `pytest` for the pure-Python test suite
- `ruff` for lint checks

Install the development dependencies from the repository root when you want to run all checks locally:

```bash
python -m pip install -e ".[dev]"
```

## Build The Add-on Zip

From the repository root:

```bash
python scripts/package_addon.py
```

This writes:

```text
dist/Blender-NRP.zip
```

That zip is the normal file to install in Blender.

## Install In Blender

1. Open Blender.
2. Open `Edit > Preferences`.
3. Select `Add-ons`.
4. Click `Install...`.
5. Select `dist/Blender-NRP.zip`.
6. Enable the `Blender-NRP` add-on.
7. Close Preferences.

The add-on panel appears in `Scene Properties` as `Blender-NRP`.

## Panel Overview

Open `Scene Properties > Blender-NRP`.

The panel contains:

- Scene ID: stable identifier written into metadata and light JSON.
- Camera: Blender camera used for the fixed shot.
- Width and Height: cache resolution.
- Hemisphere Segments: number of outgoing spokes per hit pixel.
- Max Segment Distance: maximum distance stored for candidate light segments.
- Output Directory: parent directory for generated scene artifacts.
- Cache Path: selected or last baked `path_cache.npz`.
- Model Path: selected or last trained `model.pt`.
- Light JSON: import or export path for sphere-light rigs.
- Status: last operator result or error.

Buttons are grouped by workflow:

- Bake: `Bake Path Cache`, `Validate Cache`
- Proxy: `Train Proxy`, `Load Proxy`
- Relight: `Create NRP Sphere Light`, `Preview Relight`, `Optimize Lights From Target`
- Interchange: `Import NRP Lights JSON`, `Export Selected NRP Lights JSON`

## Quick Tutorial: Bake And Relight A Small Scene

This tutorial uses a simple Blender scene. You can use any fixed-camera scene with visible geometry.

1. Create or open a Blender scene.
2. Add a camera and point it at the geometry you want to bake.
3. Save the `.blend` file.
4. Open `Scene Properties > Blender-NRP`.
5. Set `Scene ID` to a stable value such as `fixture_room_001`.
6. Select the camera in the `Camera` field.
7. Set a small resolution for the first run, for example `64 x 64` or `128 x 128`.
8. Set `Hemisphere Segments` to `8` or `16`.
9. Set `Output Directory` to a project output folder, for example `//output`.
10. Click `Bake Path Cache`.
11. Click `Validate Cache`.
12. Click `Train Proxy`.
13. Click `Create NRP Sphere Light`.
14. Move the created `NRP_Sphere_001` object in the viewport.
15. Click `Preview Relight`.
16. Inspect the generated `relight_preview.png` in the scene output directory.
17. Click `Optimize Lights From Target` to run the V1 deterministic solve step.
18. Set `Light JSON` to a path ending in `.json`.
19. Select the NRP sphere light object.
20. Click `Export Selected NRP Lights JSON`.

For a scene ID of `fixture_room_001`, the default output layout is:

```text
output/
  fixture_room_001/
    path_cache.npz
    metadata.json
    bake_report.json
    preview_albedo.png
    preview_normal.png
    preview_depth.png
    model.pt
    train_report.json
    relight_preview.png
    solved_lights.json
    solve_report.json
```

Blender paths that start with `//` are relative to the current `.blend` file.

## Command-Line Fixture Tutorial

The repository includes scriptable fixture commands. These are useful for validating the
file contracts without using the UI.

From the repository root:

```bash
python scripts/bake_fixture.py
python scripts/validate_cache.py build/nrp/fixture_room_001/path_cache.npz build/nrp/fixture_room_001/metadata.json
python scripts/relight_fixture.py
python scripts/validate_light_json.py build/nrp/fixture_room_001/solved_lights.json
```

Expected validation output:

```text
OK 64x64 32768 segments
OK 1 lights
```

The generated files are ignored by git and live under:

```text
build/nrp/fixture_room_001/
```

## Blender Background Mode

If you have a local fixture `.blend`, you can run the same workflow in Blender background mode:

```bash
blender --background tests/fixtures/minimal_scene.blend --python scripts/bake_fixture.py
blender --background tests/fixtures/minimal_scene.blend --python scripts/relight_fixture.py
```

If `tests/fixtures/minimal_scene.blend` does not exist, create a small local scene with:

- a Cornell-style room or a few primitives,
- one camera,
- one conventional Blender light,
- visible diffuse materials.

Do not commit large generated `.blend` files, caches, previews, or model artifacts unless
the project explicitly adds a small hand-authored fixture.

## Import And Export Light JSON

Blender-NRP uses this sphere-light JSON shape:

```json
{
  "scene_id": "fixture_room_001",
  "camera_id": "Camera",
  "coordinate_system": "blender_z_up",
  "lights": [
    {
      "type": "sphere",
      "position": [0.0, 1.0, 2.0],
      "radius": 0.25,
      "color": [1.0, 0.85, 0.65],
      "intensity": 4.0
    }
  ]
}
```

To import lights:

1. Set `Light JSON` to an existing JSON file.
2. Click `Import NRP Lights JSON`.
3. Blender creates editable `NRP_Sphere_###` mesh objects.

To export lights:

1. Select one or more NRP sphere objects.
2. Set `Light JSON` to the target JSON path.
3. Click `Export Selected NRP Lights JSON`.

NRP sphere objects store custom properties such as:

- `nrp_light_type`
- `nrp_scene_id`
- `nrp_camera_id`
- `nrp_coordinate_system`
- `nrp_radius`
- `nrp_color`
- `nrp_intensity`

These properties allow the light rig to round trip between Blender and JSON.

## Output Files

`path_cache.npz` contains the NRP-compatible arrays:

- `n_paths`
- `seg_pixel`
- `seg_origin`
- `seg_dir`
- `seg_tmax`
- `seg_throughput`
- `albedo`
- `normal`
- `depth`
- `position`

`metadata.json` contains the strict compatibility fields:

- `scene_id`
- `camera_id`
- `resolution`
- `light_type`
- `aux_features`
- `bbox_min`
- `bbox_max`
- `model_width`
- `model_depth`
- `coordinate_system`

`bake_report.json`, `train_report.json`, and `solve_report.json` contain
Blender-NRP-specific status, limitations, warnings, and implementation details.

## Current V1 Limitations

- The stock backend is an approximation, not physically exact path tracing.
- It records one first-hit camera sample per pixel.
- Hemisphere spokes are deterministic candidate segments, not true renderer path vertices.
- The V1 proxy artifact validates the save/load workflow but is not a full PyTorch neural proxy.
- The V1 solve button performs a deterministic single-light update path.
- Animated cameras, animated geometry, and broad light-type coverage are out of scope for V1.
- ComfyUI compatibility is file-based; the basic Blender workflow does not require a running ComfyUI server.

## Troubleshooting

Bake fails with no camera:

- Select a camera in the Blender-NRP panel.
- Confirm the scene has a camera object.

Validation reports missing arrays:

- Re-run `Bake Path Cache`.
- Confirm `Cache Path` points to `path_cache.npz`, not a report or preview image.

Validation reports zero segments:

- Confirm the camera sees scene geometry.
- Lower the resolution for a quick test and rebake.
- Move the camera or geometry so camera rays hit visible surfaces.

Preview relight reports no NRP sphere lights:

- Click `Create NRP Sphere Light`, or import a valid light JSON file.

Light export reports no selected NRP sphere lights:

- Select one or more objects whose `nrp_light_type` custom property is `sphere`.

Python cannot import NumPy:

- Install the project dependencies:

```bash
python -m pip install -e .
```

Blender cannot find the panel:

- Confirm the add-on is enabled in Preferences.
- Confirm the installed zip was built after the latest code changes.
- Rebuild with `python scripts/package_addon.py` and reinstall `dist/Blender-NRP.zip`.

## Recommended First Settings

For a fast first bake:

- Resolution: `64 x 64`
- Hemisphere Segments: `8`
- Max Segment Distance: `20`
- Output Directory: `//output`

After the workflow is correct, increase resolution and segment count gradually.
