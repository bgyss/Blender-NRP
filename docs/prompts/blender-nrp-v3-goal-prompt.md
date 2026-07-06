# Goal Prompt: Blender-NRP V3 — Remote GPU Compute, One-Button UX, a Lighter's Interface

## Context

V2 (complete, `v0.2.0-alpha`) reached parity with the `nrp` reference implementation:
real multi-bounce path capture with Cycles G-buffer passes, genuine PyTorch proxy
training with checkpoint/resume, gradient-based multi-light inverse optimization, quad
lights, packed caches, coordinate conversion on interchange, and a live in-Blender
preview. Every remaining approximation is named in a report JSON.

V2's honest ceiling, per its own reports:

- Path tracing is Python-driven over `scene.ray_cast` on the Blender main process.
  Practical budgets top out around 64–256 px at 16–128 paths/pixel — *minutes* on an
  Apple M1, and nothing about the bake uses a GPU at all. Production shots (1080p+,
  hundreds of paths/pixel, real BSDFs) are hours-to-days: not viable on artist
  hardware.
- The UI is a three-stage research harness with ~15 always-visible parameters. Correct
  and honest, but the default experience should be a single button, not a checklist.
- Lights are edited as scene objects through an Object Properties custom-prop panel.
  Fine for validation; wrong for a lighter doing lighting work.

V3's job: make production-quality scenes reachable by (1) moving heavy compute to a
fast GPU wherever it lives — cloud, LAN render node, or a local device — as
asynchronous jobs, (2) collapsing the default UX to one button with progressive
disclosure, and (3) reshaping the relight workflow into a lighter's tool. The V1/V2
architecture rule stays intact: **everything that doesn't strictly need `bpy` stays
importable and testable without Blender**, and remote workers are the ultimate proof
of that rule — they *are* the no-`bpy`-UI environment.

The full phased plan lives in `docs/ROADMAP.md` (Phases A–D); V3 is Phases A and B
plus the core of C.

## Prior Art to Reuse

- This repo's V2 code: the `PathCacheBackend` Protocol, `backends/_output.py` shared
  artifact writer, modal-generator bake and worker-thread training patterns, the
  four-tier validation harness, and the report-JSON discipline all carry forward.
- `scripts/bake_fixture.py` / `blender --background` flows — the embryo of the
  headless worker.
- `nrp`'s batched torch GATHERLIGHT and device-resident training — the model for how
  the tracing core vectorizes onto MPS/CUDA.
- Blender's own asset/render conventions: packed `.blend` files
  (`bpy.ops.file.pack_all` / `blend_paths`) for scene bundling; Draft/Standard/Final
  preset idioms; the F12 render-progress UX as the bar to meet for the one-button
  chain.
- Commodity GPU-rental APIs (RunPod / Lambda / Modal style: upload → start container →
  poll → download) for the cloud tier; plain SSH + rsync for the LAN tier.

## V3 Objectives, in priority order

### 1. Headless bake-job format + out-of-process execution (foundation)

Separate the *job* from the *machine* before adding any remote transport:

- Define versioned, deterministic `bake_job.json` / `train_job.json` /
  `solve_job.json` schemas in `core/` (no `bpy`): scene reference (packed `.blend` or
  scene bundle), camera, resolution, budget, backend, seed, torch device, output
  manifest. Treat these schemas as external API the moment they ship — same
  discipline as the cache format.
- `scripts/run_bake_job.py` (and train/solve equivalents): a worker consuming a job
  under `blender --background` (bake) or plain Python (train/solve), producing exactly
  the artifacts the add-on already validates: `path_cache.npz`, `metadata.json`,
  `model.pt`, and the report JSONs.
- Add-on side: an `ExecutionBackend` Protocol — `submit(job) -> job_id`,
  `status(job_id) -> progress`, `fetch(job_id) -> artifacts`, `cancel(job_id)` — with
  the first implementation being **local subprocess**. Poll via `bpy.app.timers`;
  ingest + auto-validate on completion exactly like a local bake finishing today.
- The V2 in-process modal bake remains as the zero-setup path.

### 2. GPU-vectorized tracing core

- Port the inner path loop to a batched, device-resident implementation using torch on
  MPS/CUDA (the dependency already exists for training): device-side BVH traversal (or
  an embree/optix binding available on the worker), batched BSDF sampling,
  wavefront-style segment accumulation writing the packed cache layout directly.
- Numeric parity gate: the vectorized tracer and the V2 Python tracer must agree on
  the analytic fixture room within stated tolerance, asserted in pytest (CPU-torch) so
  it runs in CI without a GPU.
- `bake_report.json` records which engine traced the cache and its wall-clock;
  performance target: 512×512 @ 256 paths/pixel in minutes on a single modern GPU.
- The Python `scene.ray_cast` tracer stays as the no-torch fallback, per the graceful
  degradation rule.

### 3. Remote execution backends (LAN first, then cloud)

- **SSH/LAN node** (`ssh` + `rsync`, no accounts): push the job bundle to a
  user-configured host, run the worker there, pull artifacts back. This is the
  studio-render-node path and shares 100% of the job format with the cloud tier.
- **Cloud GPU adapter** for at least one commodity rental API: upload bundle → start
  container → poll → download. Publish a container image from this repo (headless
  Blender + torch + worker scripts). Show estimated and accrued cost in the panel.
- Credentials/host config live in **add-on preferences**, never in the scene file or
  any exported artifact.
- Training and solving ride the same rails — a rented 4090 is 20–50× an M1 for the
  torch work and the artifacts are already machine-portable.
- Resilience: job queue persisted on disk across Blender restarts; reconcile on
  startup; a killed Blender must never silently orphan a billing cloud instance —
  surface anything still running and offer cancel.

### 4. One-button UX with progressive disclosure

- A single **"Make Scene Relightable"** operator chaining the pipeline: resolve
  settings → bake → auto-validate → train → auto-load proxy → create a starter light →
  open the preview. One progress bar with named sub-stages, one cancel, one final
  toast. Runs on whichever `ExecutionBackend` the single visible **Compute** dropdown
  selects (This Machine / <SSH node> / Cloud).
- Smart defaults: scene ID from the `.blend` name, resolution proportional to render
  settings, **Draft / Standard / Final** budget presets. The full V2 parameter surface
  moves behind an **Advanced** disclosure — re-layered, never removed.
- Staleness tracking: persist per-stage state (settings hash, scene-content hash) in
  the `.blend`, so the button can report "cache is stale — geometry changed" and offer
  re-bake vs use-existing rather than blindly re-running.
- Failures land in plain language at the failed stage with a "Details…" expando into
  the honest report JSON. Never a silent stall; never a fake success.

### 5. A lighter's interface (core)

- **NRP Lights gaffer list** (Viewport N-panel and/or Scene panel): every NRP light
  with name, type icon, color swatch, intensity, and per-light **enable / solo /
  mute**; inline rename; add/duplicate/delete; two-way selection sync with the
  viewport. Solo/mute are gather-time masks — the proxy makes this near-free.
- Lighting-native controls: intensity in **stops (±EV)** alongside linear; **kelvin +
  tint** as an alternative to raw RGB; preview exposure styled like Blender's color
  management.
- **Rig snapshots**: save named light-rig versions (the JSON format already exists),
  flip between them live, A/B two rigs in the preview.
- Reframe Solve as **"Match Reference"**: per-light/per-parameter lock toggles in the
  gaffer list choose what's free; run; review a before/after wipe; *apply* or
  *discard*. Same solver underneath, same write-back semantics.

## Non-Goals for V3

- Glossy/transmissive BSDF capture, textured/environment lights, 1080p+ default
  budgets, denoised training targets, multi-camera UI — Phase D of the roadmap,
  sequenced after remote compute exists because they multiply bake cost.
- Real-time frame-rate relighting; interactive-seconds via the proxy remains the bar.
- Building or hosting our own render farm; adapters target existing services/hosts.
- Viewport-placement gizmos and paint-over targets (Phase C stretch — take them only
  if the core lands early).

## Constraints Carried Forward

- `blender_nrp/__init__.py` entrypoint-only; registration via the ordered `MODULES`
  tuple in `addon.py`, unregister in reverse.
- `core/` + job schemas + workers importable without Blender; torch lazy and optional;
  numpy-only validation paths keep working.
- Wire compatibility with `nrp` and `ComfyUI-NeuralRenderProxy` is untouched: remote
  workers emit byte-identical artifact formats, and `scripts/cross_repo_roundtrip.py`
  must pass on worker-produced artifacts.
- Every operator reports through `scene.blender_nrp.status`; every artifact gets a
  machine-readable report with `ok` + honest `limitations`.
- Long operations never block the UI; worker threads never touch `bpy` data.
- Generated artifacts under gitignored `build/`/`dist/` only. No secrets in scene
  files or artifacts.
- Versions bump together (`bl_info`, `blender_manifest.toml`, `pyproject.toml`) —
  target `0.3.0` for objectives 1–2 + minimal one-button, `0.4.0` for remote backends
  and full UX, per the roadmap's release slices.

## Verification Plan (all four tiers grow)

1. **Pure-Python pytest:** job-schema round-trips and version gates; vectorized-vs-
   Python tracer parity on the analytic room (CPU torch, `importorskip`); EV/kelvin
   conversion math; staleness-hash unit tests; queue persistence round-trip; a
   **mocked `ExecutionBackend`** exercising submit/status/fetch/cancel and failure
   paths without any network.
2. **Cross-repo round-trip:** run against artifacts produced by the *worker* path
   (not just in-Blender bakes) to prove remote output is wire-identical.
3. **Blender background:** the worker itself is this tier — `run_bake_job.py` against
   `tests/fixtures/minimal_scene.blend`; plus a `blender_smoke.py` extension driving
   the one-button operator end-to-end on the local-subprocess backend and asserting
   the full artifact set + auto-loaded proxy.
4. **Manual UI:** one-button run from a fresh scene with zero pre-configuration;
   cancel mid-chain at each stage; kill Blender mid-remote-job and verify
   reconciliation on restart; gaffer solo/mute while the live preview updates; rig
   snapshot A/B; Match Reference apply/discard.
5. **Container smoke (new, CI-optional):** build the worker image, run a fixture
   bake job inside it, validate the artifacts — gated on runner availability.

## Success Criteria

V3 is done when: an artist on an M1 MacBook opens a saved scene, presses **Make Scene
Relightable** with zero configuration, watches one progress bar while the bake and
training run on a remote GPU (or locally for a draft), and gets a validated cache and
auto-loaded proxy without touching a terminal; a lighter then blocks in a rig from the
gaffer list in stops and kelvin, solos lights against the live preview, saves and A/Bs
two rig versions, and matches a reference with locked parameters — while every
artifact remains loadable by `nrp` and `ComfyUI-NeuralRenderProxy` unchanged, every
remote job is cancellable and reconcilable, and every remaining approximation is still
named in a report file, not discovered by the user.
