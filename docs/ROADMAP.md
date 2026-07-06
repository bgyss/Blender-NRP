# Blender-NRP Roadmap: V2 → Production-Quality Scenes

V2 (shipped, `v0.2.0-alpha`) reached parity with the `nrp` reference: real multi-bounce
capture, genuine torch training, gradient light solving, quad lights, packed caches, and
a live in-Blender preview. But it is honest about its own ceiling: sampling is
Python-driven over `scene.ray_cast`, the BSDF model is Lambertian-diffuse only, and the
practical envelope is ~64–256 px at 16–128 paths/pixel — minutes on an M1, and an M1 is
the *fast* case for the current code because nothing GPU-accelerates the bake at all.

Getting to **production-quality scenes** means closing three gaps, in this order of
leverage:

1. **Compute** — bakes and training must be able to run on a fast GPU that is not the
   artist's machine (cloud or LAN render node), asynchronously, with the add-on acting
   as a thin submit/monitor/fetch client.
2. **UX** — the default experience must collapse to one button ("Make it relightable")
   with progressive disclosure for the current granular controls.
3. **Lighting workflow** — the relight interface must feel like a lighter's tool
   (gaffer-style light list, soloing, exposure/white-balance, snapshots/versions), not
   like a research harness.

A fourth track — **capture fidelity** (glossy/transmissive BSDFs, textured lights,
higher resolutions) — matters for final quality but is deliberately sequenced *after*
remote compute, because fidelity multiplies cost and is pointless while the only
compute budget is a laptop CPU core running Python.

---

## Phase A — Remote & Accelerated Baking (the compute unlock)

**Problem.** `cycles_capture` traces every path in Python on the Blender main process.
A production shot (1080p+, 256–1k paths/pixel, real BSDFs) is hours-to-days on an M1
and would still be slow on a desktop GPU because nothing is vectorized onto a device.

**Strategy: separate the *job* from the *machine*.** The bake becomes a serializable
job description; where it executes is a backend choice.

### A1. Headless bake job format + local worker (foundation, no cloud yet)

- Define `bake_job.json`: scene reference (packed `.blend` or exported scene bundle),
  camera, resolution, budget, backend, seed, output manifest. Deterministic and
  versioned like every other artifact in this project.
- A `scripts/run_bake_job.py` worker that consumes a job in
  `blender --background` and produces `path_cache.npz` + `metadata.json` +
  `bake_report.json` — the exact artifacts the add-on already validates.
- The add-on gains a "bake out of process" mode: submit to a local subprocess, poll a
  status file, ingest results. This alone un-freezes Blender for long bakes and is the
  protocol every remote backend reuses.

### A2. GPU-vectorized tracing core

- Port the inner path loop to a batched, device-resident implementation (torch on
  MPS/CUDA — the dependency already exists for training): BVH intersection via a
  torch-side BVH or embree/optix binding on the worker, batched BSDF sampling,
  wavefront-style segment accumulation straight into the packed cache layout.
- The Python `scene.ray_cast` path stays as the zero-dependency fallback; the report
  records which engine traced the cache.
- Target: 512×512 @ 256 paths/pixel in minutes, not hours, on a single modern GPU.

### A3. Remote execution backends

- **Transport-agnostic `RemoteBakeBackend` Protocol**: `submit(job) -> job_id`,
  `status(job_id)`, `fetch(job_id) -> artifacts`, `cancel(job_id)`. Mirrors the
  existing `PathCacheBackend` discipline.
- **Tier 1 — SSH/LAN node**: rsync the job bundle to a machine you own (Linux box with
  a 4090, etc.), run the A1 worker there, pull artifacts back. No accounts, no billing,
  works today for studios with render nodes.
- **Tier 2 — cloud GPU**: an adapter for at least one commodity GPU-rental API
  (RunPod/Lambda/Modal-style: upload bundle → start container → poll → download).
  Container image published from this repo (Blender headless + torch + the worker).
  Credentials live in add-on preferences, never in the scene file.
- Training and solving ride the same rails: `train_job.json` / `solve_job.json` reuse
  the submit/poll/fetch protocol, because a 4090 trains the proxy 20–50× faster than
  an M1 and the artifacts (`model.pt`, reports) are already machine-portable.
- In-panel experience: submit → progress chip ("remote: 42%, ~3 min left, $0.xx est") →
  auto-ingest + auto-validate on completion, identical to a local bake finishing.

### A4. Job queue & resilience

- Multiple queued/parallel jobs (e.g. bake several cameras of one scene), persisted
  across Blender restarts (queue state on disk, not in `bpy` memory).
- Retry/resume semantics; a killed Blender never orphans a cloud instance silently —
  reconcile on next startup and surface anything still running/billing.

**Exit criteria:** an artist on an M1 MacBook submits a 1080p bake + training run to a
remote GPU from the panel, keeps working, and gets a validated cache + auto-loaded
proxy back without touching a terminal.

---

## Phase B — One-Button UX (progressive disclosure)

**Problem.** The panel today is three numbered stages with ~15 visible parameters.
Correct, but the default experience should be: pick a camera, press one button, get a
relightable scene.

### B1. The One Button

- A single **"Make Scene Relightable"** operator that chains the whole pipeline:
  resolve settings → bake → validate → train → auto-load proxy → create a starter
  light + open the preview. One progress bar with named sub-stages
  ("Baking 3/5…", "Training…"), one Esc/X to cancel the whole chain, one final toast.
- Runs locally or remotely depending on a single "Compute: This Machine / <node> /
  Cloud" dropdown — the only choice surfaced by default.
- Failure lands in plain language at the failed stage with a "Details…" expando into
  the honest report JSON; never a silent stall.

### B2. Smart defaults & presets

- Resolution defaults derived from the render settings (proportional preview size),
  budget presets **Draft / Standard / Final** replacing raw paths/bounces/iterations
  numbers. The raw numbers move behind an **Advanced** disclosure that preserves the
  full V2 surface for power users — nothing is removed, only re-layered.
- Scene ID auto-derived from the `.blend` name; output dir defaults sensibly. The
  ideal first-run flow has *zero* required fields.

### B3. Status you can trust

- Persistent per-stage state written into the `.blend` (what's baked, with which
  settings hash, whether the scene changed since) so the button can say "cache is
  stale — geometry changed" and offer *re-bake* vs *use existing*.
- Estimated time/cost preview before launching (from resolution × budget × backend
  benchmark table, refined by observed history).

**Exit criteria:** a new user with a saved scene reaches a live relight preview with
exactly one button press and zero pre-configuration; every V2 control remains
reachable under Advanced.

---

## Phase C — A Lighter's Interface

**Problem.** Lights are edited as scattered scene objects with a custom-prop panel;
fine for validation, wrong for the person whose job is lighting.

### C1. Light list / gaffer panel

- A dedicated **NRP Lights** list (3D Viewport N-panel and/or Scene panel): every NRP
  light with name, type icon, color swatch, intensity, and per-light **enable /
  solo / mute** toggles — the DCC-standard gaffer layout. Selection syncs both ways
  with the viewport.
- Add/duplicate/delete from the list; rename inline.

### C2. Lighting ergonomics

- Intensity in **stops** (±EV) alongside linear; color via **kelvin temperature** +
  tint as an alternative to raw RGB. Preview **exposure/gamma** controls styled like
  Blender's own color management.
- Interactive placement: click-in-viewport to place a light on/offset-from a surface
  (normal-aligned for quads), and gizmos for quad width/height/aim instead of numeric
  fields only.
- Because the proxy makes relighting near-free, add **A/B snapshots**: store named
  light-rig versions (JSON under the hood — the format already exists), flip between
  them live, diff two rigs side-by-side in the preview.

### C3. Solver as a lighting tool

- Reframe "Solve" as **"Match Reference"**: pick a reference image (or a paint-over of
  the current preview), choose which lights/parameters are free vs locked
  (per-light lock toggles in the gaffer list), run, review a before/after wipe, then
  *apply* or *discard*. Solve remains write-back-to-objects under the hood.

**Exit criteria:** a lighter can block in a rig, solo lights, dial stops/kelvin, save
and compare two versions, and match a reference — without ever opening the raw
custom-property editor or the Advanced disclosure.

---

## Phase D — Capture Fidelity for Final Quality (follows compute)

Sequenced after Phase A because every item multiplies bake cost:

- Glossy/transmissive BSDF sampling (Principled metallic/roughness/transmission), with
  per-lobe throughput handled per the paper; report which lobes were captured.
- Higher-resolution production caches (1080p+) — viable only on A2/A3 compute; packed
  layout becomes the default at these sizes.
- Textured and environment lights, tracking `nrp`'s extensions roadmap.
- Denoised training targets (port `nrp`'s pool) for cleaner proxies at low budgets.
- Multi-camera caches per scene (the `camera_id` plumbing already exists) for shot
  coverage.

---

## Sequencing & Dependencies

```
A1 headless job format ──► A3 remote backends ──► A4 queue
        │                        │
        ▼                        ▼
A2 GPU tracing core      B1 one button (needs async submit/poll)
                                 │
                                 ▼
                          B2 presets ──► B3 staleness/estimates
                                 │
                                 ▼
C1–C3 lighter UX (parallel to B after B1)          D fidelity (after A2/A3)
```

Recommended release slices:

- **v0.3** — Phase A1 + A2 (out-of-process + GPU-vectorized local bake), B1 minimal
  (chained one-button running locally).
- **v0.4** — A3 + A4 (SSH node, then cloud adapter; queue), B2/B3.
- **v0.5** — Phase C complete (gaffer list, ergonomics, Match Reference).
- **v0.6+** — Phase D fidelity items as compute allows.

## Invariants that do not change

- Wire compatibility with `nrp` / `ComfyUI-NeuralRenderProxy` (cache, metadata, light
  JSON, `model.pt`) — remote workers produce the same artifacts; the round-trip script
  gates every phase.
- `core/` importable without `bpy`; torch optional; every artifact gets an honest JSON
  report with `ok` + `limitations`.
- No secrets in scene files; cloud credentials in add-on preferences only.
- Four-tier validation grows with each phase (job-format round-trips, mocked remote
  backend tests, worker-in-container smoke, manual UI checks).
