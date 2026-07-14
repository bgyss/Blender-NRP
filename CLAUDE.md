# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`Blender-NRP` is a Blender add-on for baking Neural Render Proxy (NRP) path caches and relighting
fixed-camera scenes inside Blender. V3 (`0.4.0`) retains parity with the sibling `nrp` reference:
real multi-bounce path capture with Cycles G-buffer passes, genuine PyTorch proxy training,
gradient-based inverse light optimization, sphere + quad lights, packed caches, coordinate
conversion on interchange, and a live in-Blender preview. It adds versioned headless jobs,
local/SSH/RunPod execution, torch mesh tracing, a one-button pipeline, durable reconciliation,
and a lighter-facing gaffer/Match Reference workflow — still correctness-first, not real-time.

Build goals/specs live in `docs/prompts/blender-path-cache-bake-plugin-goal-prompt.md` (V1) and
`docs/prompts/blender-nrp-v2-goal-prompt.md` (V2). User-facing docs are in `docs/USER_GUIDE.md`.

## Commands

```bash
python -m pip install -e ".[dev]"   # dev environment (torch is optional, install separately)
python -m pytest                     # pure-Python tests (torch tests importorskip)
python -m ruff check .               # lint (line-length 100, py311, rules E,F,I,UP,B)
python scripts/package_addon.py      # build dist/Blender-NRP.zip for install in Blender
```

Run a single test file/case the normal pytest way, e.g. `python -m pytest tests/test_light_json.py -k foo`.

Tested fixture workflow (writes to `build/nrp/fixture_room_001/`, gitignored). Outside Blender it
traces the built-in analytic room (real multi-bounce + escape segments, no `.blend` needed):

```bash
python scripts/bake_fixture.py
python scripts/validate_cache.py build/nrp/fixture_room_001/path_cache.npz build/nrp/fixture_room_001/metadata.json
python scripts/relight_fixture.py
python scripts/validate_light_json.py build/nrp/fixture_room_001/solved_lights.json
```

Cross-repo round-trip (verification tier 2 — loads baked artifacts with the *actual* sibling
implementations; expects `../nrp` and `../ComfyUI-NeuralRenderProxy`, override via NRP_REPO /
COMFY_REPO; torch-dependent checks SKIP cleanly). Pass `--artifact-dir` to verify the exact output
of `run_bake_job.py` instead of creating another fixture:

```bash
python scripts/cross_repo_roundtrip.py
python scripts/cross_repo_roundtrip.py --artifact-dir build/worker/scene_id
```

When Blender is available, run the fixture scripts against the committed Cornell-style scene:

```bash
blender --background tests/fixtures/minimal_scene.blend --python scripts/bake_fixture.py
blender --background tests/fixtures/minimal_scene.blend --python scripts/relight_fixture.py
```

Full Blender operator smoke test (registers the add-on, runs the granular V2 operator chain and
the V3 one-button local-subprocess chain, including auto-validation/load/starter light/preview):

```bash
blender --background --factory-startup --python-exit-code 7 --python tests/blender_smoke.py
```

Worker-container smoke (requires a built image and Docker daemon):

```bash
scripts/container_smoke.sh blender-nrp-worker:smoke
```

`INSTALL_TORCH=0` with `scripts/container_smoke.sh blender-nrp-worker:smoke python`
is a reduced analytic image check for constrained runners; it does not replace the
default torch-mesh container gate.

## Architecture

The core design constraint: **everything that doesn't strictly need `bpy` must be importable and
testable without Blender.** This is what makes `python -m pytest` possible outside the Blender
Python environment. Torch is a second optional layer: nothing imports `core/torch_proxy` submodules
unless torch work is actually requested (`torch_status()` gates the operators).

```
blender_nrp/
  __init__.py        add-on entrypoint only: bl_info + register()/unregister() that defer
                      all bpy-touching imports into addon.py
  addon.py            registration orchestration — MODULES tuple lists every bpy-dependent
                      module in registration order; register() walks forward, unregister()
                      walks in reverse
  core/               bpy-independent: path-cache schema + packed layout (path_cache.py,
                      rgb9e5.py), metadata, light JSON (lights.py: sphere + quad),
                      gather (gather.py, reference GATHERLIGHT semantics), coordinate
                      conversion (coords.py), ComfyUI export bundle (comfy_export.py),
                      the multi-bounce Monte Carlo tracer behind cycles_capture
                      (path_tracer.py, RayCaster Protocol + AnalyticRoomCaster test scene),
                      object<->light mapping (light_objects.py, duck-typed), no-torch
                      solver (optimize_fallback.py), validation, reports; versioned jobs,
                      durable execution queue, local/SSH/RunPod adapters, cost/staleness/presets,
                      and torch analytic/triangle-BVH tracing
  core/torch_proxy/   optional-dep torch stack, nrp torch_backend parity: encoding.py
                      (hashgrid), model.py (TorchNRP — save format byte-compatible with
                      nrp's, keep it that way), gather.py (batched device gather),
                      sampling.py, train.py (pool training, cosine LR, checkpoint/resume),
                      optimize.py (differentiable multi-light solve), relight.py
  backends/           PathCacheBackend Protocol + _output.py (shared cache/metadata/report
                      writer) + cycles_capture.py (V2: Python-driven multi-bounce over
                      scene.ray_cast, Cycles G-buffer passes with 4.x/5.x compositor API
                      support, A/B PSNR vs a real Cycles emissive-sphere render, generator
                      bake_steps() for modal progress/cancel) + stock_blender_hemi.py
                      (V1 fallback: first-hit + deterministic hemisphere spokes)
  operators/          bpy.types.Operator classes — thin orchestration only; real logic in
                      core/. bake_cache is modal (Esc cancels; synchronous in background
                      mode). train_proxy runs a worker thread + bpy.app.timers polling
                      (worker touches numpy/torch only; all bpy access on the main thread), plus
                      one-button orchestration, restart reconciliation, snapshots, and Match Reference.
                      _helpers.py reports onto scene.blender_nrp.status.
  preview.py          live relight preview: Image datablock ("NRP Relight Preview") updated
                      in place, debounced depsgraph_update_post handler gated by the
                      live_preview toggle, proxy-vs-gather source labeling
  proxy_runtime.py    session-scoped holder for the loaded TorchNRP (shared by load/preview/solve)
  panels.py           Scene properties UI panel
  properties.py       bpy.props definitions attached to the scene
  ui/                 UI helper code
scripts/              packaging, fixture bake/relight/validate flows, cross_repo_roundtrip.py
tests/                pure-Python tests (torch ones importorskip) + blender_smoke.py
examples/scene_manifests/   production scene manifests for manual QA against real shots
```

### Compatibility contracts

The add-on must stay wire-compatible with `nrp` and `ComfyUI-NeuralRenderProxy`
(`scripts/cross_repo_roundtrip.py` checks this against the real sibling code):

- `path_cache.npz`: `n_paths`, `seg_pixel`, `seg_origin`, `seg_dir`, `seg_tmax`, `seg_throughput`,
  `albedo`, `normal`, `depth`, `position`, plus scalar `schema_version`/`width`/`height` required
  by `nrp`'s `PathCache.load`. `seg_throughput` is raw (not pre-averaged); per-pixel averaging
  divides by `n_paths` at gather time, matching `nrp` main's GATHERLIGHT — recorded in metadata as
  `throughput_normalization: "n_paths"`. Escape segments use `seg_tmax = inf`. The packed layout
  (`packed_layout` key, fp16 geometry + rgb9e5 throughput, int32 seg_pixel) must stay bit-level
  compatible with `nrp/path_cache.py` — `core/rgb9e5.py` is a synced port.
- ComfyUI's gather does **not** divide by n_paths and defaults to `right_handed_y_up`; the export
  path (`core/comfy_export.py`) pre-divides throughput and rotates geometry, labeling metadata
  `pre_divided`. Never change the primary cache convention to match ComfyUI.
- `metadata.json`: fixed-scene, fixed-camera light metadata; `light_type` is "sphere" or "quad".
- light rig JSON: `scene_id`, `camera_id`, `coordinate_system`, `lights`; sphere entries are
  V1-shaped, quads add `"type": "quad"` + `normal`/`width`/`height`; untyped entries dispatch like
  `nrp`'s `light_from_dict` (width present -> quad, else sphere).
- `model.pt`: nrp `TorchNRP` format `{"config", "state_dict"}` with identical config keys and
  module names — nrp must load and evaluate it identically. ComfyUI's `NRPProxy` is a different
  architecture; that gap is documented in train_report.json, not papered over.

Changing these shapes breaks interop with the sibling projects — treat them as an external API.

### Conventions specific to this repo

- Keep `blender_nrp/__init__.py` minimal (entrypoint only); put real registration logic in `addon.py`.
- `bl_info` (in `__init__.py`), `blender_manifest.toml`, and `pyproject.toml` versions bump together.
- Register/unregister classes in deterministic order; unregister in reverse of register.
- Generated artifacts (caches, trained models, previews, reports, zips) always go under gitignored
  directories (`build/`, `dist/`) — never commit them.
- Training/proxy inference degrades gracefully and reports missing PyTorch clearly rather than
  hard-crashing; cache/metadata validation works with numpy only; scene import/export must not
  require a running ComfyUI server. Every artifact gets a machine-readable JSON report with an
  `ok` flag and honest `limitations`/`approximation_limits` lists.
- Long-running operators never block the UI: bakes are modal generators with Esc cancel, training
  runs on a worker thread with timer polling. Worker threads must not touch bpy data.
- Blender 4.x vs 5.x API differences (compositor node tree, File Output node) are handled inside
  `cycles_capture.py` — test against both when touching that code.
- Four validation tiers: pure-Python pytest (`core/` + torch importorskip), the cross-repo
  round-trip script, Blender background fixture scripts + blender_smoke.py, and manual Blender UI
  checks (one-button cancel/restart reconciliation, live gaffer masks/snapshots, Match Reference).
  The worker-container smoke is an additional CI-optional gate.
