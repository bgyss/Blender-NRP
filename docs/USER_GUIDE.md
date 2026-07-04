# Blender-NRP User Guide

This guide covers installing the Blender-NRP add-on, running the built-in fixture workflow,
and using the Blender panel for a fixed-camera bake, proxy training, and relight pass.

Blender-NRP V2 replaces the V1 approximations with real implementations: multi-bounce
path capture with Cycles G-buffer passes, genuine PyTorch proxy training, gradient-based
inverse light optimization, quad lights, packed caches, and a live in-Blender preview.
Every remaining approximation is named in the corresponding report JSON, not hidden.

## What The Add-on Does

Blender-NRP helps you:

- bake a fixed Blender camera view into an NRP-style `path_cache.npz` with real
  multi-bounce segments and escape rays (default or ~4x-smaller packed layout),
- write strict `metadata.json` beside the cache (including the gather-normalization
  convention and coordinate system),
- export albedo, normal, and depth preview images from real Cycles passes,
- create and edit NRP sphere *and quad* lights as Blender objects,
- train a real PyTorch neural proxy in the background (checkpoint/resume, MPS/CUDA),
- preview relighting live inside Blender (Image datablock, auto-updating while you
  drag lights) via the trained proxy or the exact cache gather,
- solve a light rig against a target image and write results back onto the objects,
- import and export ComfyUI-compatible light JSON with coordinate conversion.

Generated files are written under an output directory such as
`build/nrp/fixture_room_001/`, `output/fixture_room_001/`, or a directory you choose
in the Blender panel.

## Requirements

For command-line fixture validation:

- Python 3.11 or newer
- NumPy

For the Blender add-on:

- A current Blender build (4.2+; tested on 5.1) with Python add-on support
- The packaged add-on zip: `dist/Blender-NRP.zip`

Optional:

- PyTorch, for proxy training/inference and the gradient light solver. Without torch,
  training and proxy preview report the missing dependency clearly; cache-gather
  preview and a coordinate-descent solver still work.
- `pytest` and `ruff` for development checks.

Install the development dependencies from the repository root when you want to run all
checks locally:

```bash
python -m pip install -e ".[dev]"        # numpy + pytest + ruff
python -m pip install -e ".[dev,torch]"  # additionally pulls in torch for proxy work
```

(The `torch` extra only affects the command-line environment — Blender's bundled
Python needs its own torch install, see Troubleshooting.)

## Build The Add-on Zip

From the repository root:

```bash
python scripts/package_addon.py
```

This writes `dist/Blender-NRP.zip` — the normal file to install in Blender.

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

Open `Scene Properties > Blender-NRP`. The panel is three numbered stages, each with a
right-aligned status chip — a **✓ checkmark** when the stage is complete, a dot when it
isn't — so you can see at a glance where you are. Every button also raises Blender's usual
info/error toast, and the bottom status line keeps the last message.

- Scene ID / Camera / Width / Height / Output Directory: shot identity and cache size.
- **1 · Path Cache** (chip: `baked` once a cache exists):
  - Backend: `Cycles Capture` (real multi-bounce transport, V2 default) or
    `Stock Hemisphere` (fast V1 fallback, no real bounces).
  - Paths / Pixel and Max Bounces: the Monte Carlo budget for Cycles Capture.
  - Packed Cache: write the fp16 + rgb9e5 packed layout (~4x smaller).
  - `Bake Path Cache` runs modally — the status line shows progress and **Esc cancels**.
    Validation now runs **automatically** at the end of the bake; the result (resolution,
    segment count, schema version, layout) is folded into the status, and a malformed
    cache is reported loudly instead of silently.
- **2 · Neural Proxy** (chip: `loaded` once a proxy is in memory):
  - Train Iterations / Device (`auto` picks MPS or CUDA when available). Training is
    disabled until a cache is baked.
  - `Train Proxy` runs on a background thread; the status line shows live loss. The X
    button cancels after the current iteration. `checkpoint.pt` is written periodically
    so long runs can resume. When training finishes the proxy is **auto-loaded** into
    memory — no separate Load step. `Load Proxy` appears only when a `model.pt` exists on
    disk but isn't loaded (e.g. after reopening the file).
- **3 · Relight** (chip: `preview ready` once the preview image exists):
  - `Sphere` / `Quad` add a visible NRP emitter at the 3D cursor. A quad emits along its
    local +Z axis — rotate the object to aim it; `nrp_width`/`nrp_height` set its extent.
  - `Preview Relight` renders into the `NRP Relight Preview` Image datablock and writes
    `relight_preview.png`. Any open Image Editor that isn't already showing an image is
    pointed at the preview automatically, and the panel prints where to find it. The
    status says whether the trained proxy (fast) or the exact cache gather produced it.
  - Live Preview (toggle) / Exposure: auto-refresh the preview ~0.3 s after you stop
    moving an NRP light; Exposure is a linear multiplier (raise it if the preview looks
    black at low light intensities).
  - Target + Solver Steps + `Solve`: inverse optimization (see below).
- Interchange box: light JSON import/export with an Export Coords selector
  (`right_handed_y_up` for ComfyUI, `blender_z_up` for no conversion). Import converts
  from the file's declared coordinate system into Blender's automatically.
- Status: last operator result or error.

## Quick Tutorial: Bake, Train, And Relight

This is the canonical manual test sequence. Follow it in order to confirm the whole
pipeline is wired correctly.

### Step 1 — Scene setup

1. Create or open a fixed-camera Blender scene and save the `.blend` file.
2. Open `Scene Properties > Blender-NRP`.
3. Set `Scene ID` to a short identifier (e.g. `fixture_room_001`), select the `Camera`,
   and set the resolution to `64 × 64` for the first run.

### Step 2 — Bake Path Cache (Stage 1)

1. Keep the Backend at `Cycles Capture`, `Max Bounces = 4`, and lower `Paths / Pixel`
   to `32` for a faster first run.
2. Click **Bake Path Cache**. The status line shows progress; **Esc** cancels.
3. When the bake finishes, validation runs automatically.
   - The `1 · Path Cache` chip flips to **✓ baked**.
   - The status line reads something like *"Baked + validated — 64×64, N segments, …"*.
   - A Blender info toast appears confirming success.
4. `bake_report.json` in the output directory includes an A/B PSNR against a real
   Cycles render — the honest agreement number, not a claim of exactness.

### Step 3 — Train Proxy (Stage 2)

1. Click **Train Proxy**. This requires torch in Blender's bundled Python (see
   Troubleshooting if torch is missing — the cache-gather preview still works without it).
2. Training runs on a background thread; the status line shows the live loss.
   The **X** button cancels after the current iteration.
3. When training finishes:
   - The proxy is **auto-loaded** into memory — no separate Load step.
   - The `2 · Neural Proxy` chip flips to **✓ loaded**.
   - The status line ends with *"… proxy auto-loaded"*.
   - `Load Proxy` only appears if a `model.pt` exists on disk but is not yet loaded
     (e.g. after reopening Blender).

### Step 4 — Preview Relight (Stage 3)

1. Click **Sphere** in the Stage 3 row. A sphere NRP light appears at the 3D cursor;
   move it where you want.
2. Click **Preview Relight**.
   - The `3 · Relight` chip flips to **✓ preview ready**.
   - The `NRP Relight Preview` Image datablock is created and written to
     `relight_preview.png` in the output directory.
   - Any open Image Editor that is not already showing an image is pointed at the
     preview automatically.

### Step 5 — Live Preview in the Image Editor

1. Split off (or open) an Image Editor area. If the area was empty it already shows
   `NRP Relight Preview`; otherwise pick it from the image dropdown in the header.
2. Enable **Live Preview** in the Blender-NRP panel (or in the Image Editor header
   if the toggle appears there).
3. Drag the sphere light in the 3D Viewport. After ~0.3 s of inactivity the preview
   refreshes automatically — no manual click needed.
4. Adjust **Exposure** (linear multiplier) if the preview looks black at low light
   intensities.

### Step 6 — Optional: Solve and Export

1. To solve a rig against a reference image: set `Target` to a PNG or `.npy` at
   cache resolution, click **Solve**. Solved positions/sizes/colors/intensities are
   written back onto the light objects; `solve_report.json` records before/after loss
   in both proxy space and physically-grounded gather space.
2. Set `Light JSON`, select your lights, choose Export Coords
   (`right_handed_y_up` for ComfyUI, `blender_z_up` for no conversion), and click
   **Export** for ComfyUI interchange.

For a scene ID of `fixture_room_001`, the output layout is:

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
    checkpoint.pt
    train_report.json
    relight_preview.png
    target.npy            (if you saved one there)
    solved_lights.json
    solve_report.json
```

Blender paths that start with `//` are relative to the current `.blend` file.

## Command-Line Fixture Tutorial

The repository includes scriptable fixture commands. Outside Blender they run against a
built-in analytic room (real multi-bounce tracing, real escape segments — no `.blend`
needed), which validates all file contracts:

```bash
python scripts/bake_fixture.py
python scripts/validate_cache.py build/nrp/fixture_room_001/path_cache.npz build/nrp/fixture_room_001/metadata.json
python scripts/relight_fixture.py
python scripts/validate_light_json.py build/nrp/fixture_room_001/solved_lights.json
```

Generated files are ignored by git and live under `build/nrp/fixture_room_001/`.

The cross-repo round-trip check loads the baked artifacts with the *actual* sibling
implementations (`nrp`'s `PathCache.load`/`gather_light`/`TorchNRP.load`, ComfyUI's
`NRPLightRig` parser and gather):

```bash
python scripts/cross_repo_roundtrip.py   # expects ../nrp and ../ComfyUI-NeuralRenderProxy
```

## Blender Background Mode

Run the same workflow against the committed fixture scene:

```bash
blender --background tests/fixtures/minimal_scene.blend --python scripts/bake_fixture.py
blender --background tests/fixtures/minimal_scene.blend --python scripts/relight_fixture.py
```

## Import And Export Light JSON

Sphere lights use the V1-compatible shape; quads add `"type": "quad"`:

```json
{
  "scene_id": "fixture_room_001",
  "camera_id": "Camera",
  "coordinate_system": "right_handed_y_up",
  "lights": [
    {
      "type": "sphere",
      "position": [0.0, 2.0, -1.0],
      "radius": 0.25,
      "color": [1.0, 0.85, 0.65],
      "intensity": 4.0
    },
    {
      "type": "quad",
      "position": [1.0, 1.8, 1.0],
      "normal": [0.0, -1.0, 0.0],
      "width": 1.2,
      "height": 0.8,
      "color": [0.7, 0.8, 1.0],
      "intensity": 2.0
    }
  ]
}
```

Entries without a `"type"` key remain spheres (matching `nrp`'s dispatch), so V1 JSON
loads unchanged. Positions **and quad normals** are converted between coordinate
systems on both import and export; the `coordinate_system` field is authoritative.

NRP light objects store custom properties (`nrp_light_type`, `nrp_radius` or
`nrp_width`/`nrp_height`, `nrp_color`, `nrp_intensity`, ...) that round trip between
Blender and JSON. A quad's emission normal is its local +Z axis in world space.

## Interop Conventions (Important For Sibling Projects)

- **Throughput normalization**: caches store *raw* per-segment throughput; gathering
  divides per-pixel sums by `n_paths` (the `nrp`-main convention). `metadata.json`
  records this as `"throughput_normalization": "n_paths"`. ComfyUI's gather does not
  normalize, so the ComfyUI export path (`core/comfy_export.py`) writes a bundle with
  *pre-divided* throughput labeled `"pre_divided"`.
- **Coordinate systems**: Blender-side data is `blender_z_up`; ComfyUI defaults to
  `right_handed_y_up`. V2 converts on import/export instead of only labeling.
- **Packed caches**: the fp16 + rgb9e5 layout from `nrp` (§4.2) is read natively and
  can be written with the Packed Cache toggle; escape segments survive the round trip.
- **model.pt**: the nrp `TorchNRP` format (`{"config", "state_dict"}`), loadable by
  `nrp` directly. ComfyUI's `NRPProxy` is a different architecture; loading TorchNRP
  artifacts there requires a ComfyUI-side change (recorded in `train_report.json`).

## Current V2 Limitations (also listed in the report files)

- The Cycles Capture backend samples a Lambertian-diffuse BSDF only (Principled Base
  Color albedo); glossy/transmissive transport is not captured, and sampling is
  Python-driven rather than a Cycles kernel hook. Agreement with Cycles is *reported*
  as PSNR in `bake_report.json`, never claimed exact.
- Proxy training uses raw gather targets (nrp's denoised-target pool is not ported).
- The solver descends through the smooth proxy; gather-space numbers in
  `solve_report.json` are the grounded check. Quad normals are optimized only by the
  torch solver, and are not written back onto object rotations.
- Volume capture is out of scope: `medium` metadata in caches from `nrp` is surfaced
  on load/validate, and gathering works unchanged, but Blender-side baking does not
  produce medium caches.
- Animated lights/cameras, dynamic geometry, per-layer compositing, and textured or
  environment lights remain V3+ scope.

## Troubleshooting

Bake fails with no camera:

- Select a camera in the Blender-NRP panel and confirm the scene has one.

Bake is slow:

- Cycles Capture traces paths from Python; use preview resolutions (64–256 px) and
  moderate budgets (16–128 paths/pixel). Esc cancels a running bake.

Training says PyTorch is not available:

- Install torch into Blender's bundled Python (e.g.
  `<blender>/python/bin/python -m pip install torch`), or keep using the exact
  cache-gather preview and the no-torch solver.

Validation reports missing arrays or zero segments:

- Re-run `Bake Path Cache`; confirm `Cache Path` points at `path_cache.npz`; confirm
  the camera sees geometry.

Preview reports no NRP lights:

- Create a sphere/quad light, or import a light JSON file.

Solve rejects the target image:

- The target must match the cache resolution exactly ((H, W, 3) `.npy`, or an image
  Blender can load).

## Recommended First Settings

- Resolution: `64 x 64`
- Backend: Cycles Capture, `Paths / Pixel = 32`, `Max Bounces = 4`
- Train Iterations: `2000` (device `auto`)
- Output Directory: `//output`

After the workflow is correct, increase resolution and budget gradually.
