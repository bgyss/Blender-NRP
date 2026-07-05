# Goal Prompt: Blender-NRP V2 — Real Path Capture, Real Proxies, Current nrp Parity

## Context

V1 (complete) proved the workflow: an installable add-on that bakes a schema-compatible
path cache, writes metadata and light-rig JSON that round-trip with the sibling
projects, previews fixed-camera relighting with reference GATHERLIGHT semantics, and
runs three validation tiers (pure-Python pytest, Blender background scripts, UI smoke).
V1 deliberately shipped approximations, each labeled in its report JSON:

- The `stock_blender_hemi` backend records one first-hit camera ray per pixel plus
  deterministic normal-oriented hemisphere spokes — not real multi-bounce path
  transport. Albedo comes from viewport material color.
- "Training" writes a NumPy summary artifact named `model.pt`; there is no neural proxy.
- "Optimize Lights" is a deterministic intensity step, not a solver.
- Preview is a PNG written to disk, not a live viewport image.

Meanwhile the reference `nrp` project has moved well past the spec V1 was built
against. V2's job is to replace the approximations with the real thing and reach parity
with current `nrp` capabilities, while keeping the V1 architecture rule intact:
**everything that doesn't strictly need `bpy` stays importable and testable without
Blender.**

## Prior Art to Reuse

- Sancho et al., [*Neural Render Proxies for Interactive and Differentiable Lighting*](https://studios.disneyresearch.com/2026/07/01/neural-render-proxies-for-interactive-and-differentiable-lighting/) —
  the original Disney Research paper that defines the light-agnostic path pass,
  light-dependent gather pass, neural proxy, and differentiable inverse-lighting
  workflow this add-on adapts into Blender.
- [`bgyss/nrp`](https://github.com/bgyss/nrp) — now the authoritative implementation
  reference for this add-on. Since the V1 spec it
  added: schema v2 (`schema_version` key, optional homogeneous-medium metadata), a
  packed cache layout (fp16 geometry + rgb9e5 throughput, auto-detected on load),
  `QuadLight` + `segment_hits_quad`, batched torch GATHERLIGHT with device-resident
  (MPS/CUDA) training, paper-scale training with cosine LR and checkpoint/resume,
  multi-light inverse optimization, multi-view caches, per-layer compositing NRPs, and
  SSIM/FLIP metrics. Its `docs/extensions.md` roadmap (animated lights, light-aware
  sampling, textured/environment lights) signals where the format goes next.
- `ComfyUI-NeuralRenderProxy` — the interchange target for light-rig JSON
  (`position`/`radius`/`color`/`intensity` fields) and torch proxy artifacts.
- This repo's V1 code — the backend Protocol, operator/status conventions, and the
  three-tier validation harness all carry forward unchanged.

## Known Interop Debts (fix early, they are cheap and load-bearing)

1. **Normalization convention split.** `nrp` main divides gathered throughput by
   `n_paths` at gather time; `ComfyUI-NeuralRenderProxy`'s gather does not (it expects
   pre-averaged throughput). V1 follows `nrp` main (raw throughput + `n_paths`
   normalization). V2 must make this explicit in `metadata.json` (e.g.
   `"throughput_normalization": "n_paths"`), and the ComfyUI export path must either
   pre-divide on export or the ComfyUI repo gets a matching fix — decide once,
   document it, add a cross-repo round-trip test.
2. **Coordinate systems.** Blender writes `blender_z_up`; ComfyUI defaults to
   `right_handed_y_up`. V1 only labels the field. V2 must actually convert on
   import/export (positions, normals when quad lights land) with tests in both
   directions.
3. **Packed caches.** V1's validator detects `packed_layout` caches and refuses them
   with a pointer to `nrp`. V2 should read them natively (port `rgb9e5.py` decode into
   `core/`; it is small and dependency-free) and optionally write them (bakes at
   production resolution are large; §4.2 packing is ~4x smaller).

## V2 Objectives, in priority order

### 1. Cycles-instrumented path capture backend

Add a `cycles_capture` backend implementing the existing `PathCacheBackend` Protocol
beside `stock_blender_hemi` (which stays, as the fast/no-dependency fallback):

- Real multi-bounce light transport for the fixed camera: per-pixel paths with true
  segment origins/directions/lengths and accumulated BSDF throughput, escape segments
  recorded with `t_max = inf` (V1 never writes escape segments — a real capture must).
- Real G-buffer aux: denoising-albedo/normal/depth/position passes from Cycles rather
  than viewport colors.
- Practical route first: drive sampling from Python (BVH via
  `scene.ray_cast`/`bvhtree`, BSDF importance sampling from evaluated material data) at
  paper-style budgets (hundreds of paths/pixel at preview resolutions). A true
  Cycles-kernel hook is a stretch goal; do not block V2 on it.
- Bake must be a modal/background operator with progress reporting and cancel — V1's
  synchronous per-pixel loop blocking the UI is not acceptable at real budgets.
- Verify: `nrp`'s `compare_reference.py` style A/B — GATHERLIGHT over the captured
  cache vs a real Cycles render of the same scene with an emissive sphere, PSNR
  reported in `bake_report.json`; never claim exactness, report the number.

### 2. Real PyTorch proxy training and inference

Replace the NumPy-summary stub with `nrp`-parity training as an optional-dependency
feature (graceful degradation stays mandatory):

- Port/reuse `nrp`'s model + dataset + training loop: hashgrid-or-MLP proxy trained on
  (light params → gathered throughput) pairs sampled from the cache; cosine LR,
  checkpoint/resume, device-resident on MPS/CUDA when available.
- `model.pt` becomes a genuine torch artifact loadable by both sibling projects;
  train/load reports include device, iterations, wall-clock, and final loss/PSNR.
- Training runs in the background (thread or subprocess + timer polling) with progress
  in the panel status; never freeze the UI.
- Without torch installed: operators report the missing dependency clearly and the
  GATHERLIGHT preview path still works (as in V1).

### 3. Quad lights and the current light vocabulary

- Add `QuadLight` (center, normal, width, height, rgb/intensity) end-to-end: Blender
  object representation (empty/plane with custom props), gather support
  (`segment_hits_quad` port), JSON schema with `"type": "quad"`, import/export, and
  metadata `light_type` no longer hard-coded to `"sphere"`.
- Keep sphere JSON backward-compatible (untyped specs remain spheres, matching `nrp`'s
  `light_from_dict` dispatch).

### 4. Real inverse light optimization

Replace the intensity-step stub with `nrp`-parity multi-light optimization:

- Target = a rendered/loaded reference image; optimize light positions, radii, colors,
  intensities via gradient descent through the differentiable gather (torch) with the
  proxy as fast forward model.
- Write optimized parameters back onto the Blender light objects (not just JSON), with
  before/after loss in `solve_report.json`.
- No-torch fallback: coordinate-descent over the numpy gather at reduced resolution, or
  a clear "requires torch" status — never silently pretend to solve.

### 5. Live viewport preview

- Show the relit image inside Blender: an Image datablock updated in place (Image
  Editor) at minimum; a `gpu`-module viewport overlay as stretch.
- Debounced auto-update on light edits (`depsgraph_update_post` handler) with an
  explicit toggle; V1's button-press update remains the fallback path.
- Preview must use the trained proxy when loaded (fast path) and cache gather when not
  (exact path), and must label which one produced the image.

### 6. Format parity housekeeping

- Write `schema_version` 2 with correct semantics (V1 already writes the key); accept
  and surface `medium` metadata on load/validate even though Blender-side volume
  capture is out of scope for V2.
- Packed-layout read (and optional write) per Interop Debt 3.
- Multi-view groundwork only: keep `camera_id` plumbed through everything so a V3
  multi-camera cache does not require schema surgery; do not build multi-view UI in V2.

## Non-Goals for V2

- Animated lights/camera, dynamic geometry, per-layer compositing NRPs, textured or
  environment lights — these track `nrp`'s extensions roadmap and belong to V3+.
- Real-time performance guarantees. V2 targets *interactive* (seconds, not minutes)
  preview updates via the trained proxy, not frame-rate rendering.
- Running training inside ComfyUI or requiring a ComfyUI server for anything.

## Constraints Carried Forward from V1

- `blender_nrp/__init__.py` stays entrypoint-only; registration in `addon.py` with the
  ordered `MODULES` tuple; unregister in reverse.
- `core/` (and the new torch code) importable without Blender; torch imports lazy and
  optional; base runtime deps stay numpy-only for validation paths.
- Every operator reports through `scene.blender_nrp.status`; every artifact gets a
  machine-readable JSON report with an `ok` flag and honest `limitations`/
  `approximation_limits` lists.
- Generated artifacts under gitignored `build/`/`dist/` only.
- `bl_info` and `blender_manifest.toml` bump together (target `0.2.0`).

## Verification Plan (all three tiers must grow with the features)

1. **Pure-Python pytest:** quad-light hit tests mirroring `nrp`'s; packed-layout
   load round-trip against fixtures generated by `nrp` itself; coordinate-conversion
   round-trips; normalization-convention metadata tests; torch tests that
   `pytest.importorskip("torch")`.
2. **Cross-repo round-trip (new tier, scriptable):** a committed script that bakes a
   fixture cache, loads it with the *actual* `nrp` `PathCache.load`, compares
   `gather_light` output against `blender_nrp` gather (V1 already achieves ≤1e-8
   agreement — keep that bar), and round-trips a light rig through ComfyUI's
   `NRPLightRig` parser.
3. **Blender background:** `cycles_capture` bake + train + relight on
   `tests/fixtures/minimal_scene.blend`, PSNR-vs-Cycles reference asserted above a
   stated floor in `bake_report.json`.
4. **Manual UI:** modal bake progress + cancel; live preview updates while dragging a
   light; optimizer writes solved values back onto scene objects.

## Success Criteria

V2 is done when a user can: bake a real multi-bounce cache from a Cycles scene without
the UI freezing; train a genuine torch proxy in the background; drag sphere *and quad*
lights around a live in-Blender preview; solve a light rig against a target image; and
hand the resulting `path_cache.npz`, `model.pt`, and rig JSON to `nrp` and
`ComfyUI-NeuralRenderProxy` where they load and evaluate identically — with every
remaining approximation still named in a report file, not discovered by the user.
