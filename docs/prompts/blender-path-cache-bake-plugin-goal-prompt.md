# Goal Prompt: Build Blender-NRP as a Unified Bake and Relight Add-on

## Revision Note

This project was renamed from a baker-only concept to `Blender-NRP` because the target product is now a single Blender add-on that handles both sides of the Neural Render Proxy workflow:

1. Bake a fixed Blender shot into a light-agnostic path cache.
2. Train, load, preview, and interactively relight that shot inside Blender.

`ComfyUI-NeuralRenderProxy` remains an important compatibility target and optional graph UI, but the V1 success path must not require leaving Blender once the add-on is installed.

---

## Objective

Build an installable Blender add-on named `Blender-NRP` that lets a user open a Blender scene, bake an NRP-compatible scene cache, create or load a compact proxy, edit relightable sphere-light rigs, preview fixed-camera relighting, and export or import solved light rigs.

Prioritize a functional end-to-end system over performance. V1 may update previews on button press, on debounced light edits, or at modest resolution. It is more important that the cache, proxy, light rig, metadata, and Blender scene objects round trip correctly than that the first implementation is fast.

The add-on should work in stock Blender first. It should clearly separate the stock-Blender approximation backend from a future renderer-instrumented Cycles backend.

---

## Prior Art to Reuse

Use these local projects as behavioral and schema prior art, not as code that must be copied directly:

- Sancho et al., [*Neural Render Proxies for Interactive and Differentiable Lighting*](https://studios.disneyresearch.com/2026/07/01/neural-render-proxies-for-interactive-and-differentiable-lighting/): the original Disney Research paper defining the SAMPLEPATHS / GATHERLIGHT split, neural proxy relighting, and differentiable inverse-lighting workflow.
- [`bgyss/nrp`](https://github.com/bgyss/nrp): the sample reimplementation and reference for this add-on's behavior and file formats. It defines the path-cache vocabulary, sphere and quad light abstractions, PyTorch training, proxy relighting, and inverse-light optimization.
- `ComfyUI-NeuralRenderProxy`: the current node-based UI and interchange surface. It defines ComfyUI nodes such as `NRP Load Path Cache`, `NRP Train Proxy`, `NRP Sphere Light`, `NRP Combine Lights`, `NRP Relight`, `NRP Optimize Lights From Target`, `NRP Save Lights`, and `NRP Load Lights`.

The Blender add-on should mirror the compatible data contracts from those projects:

- `.npz` path-cache arrays.
- `metadata.json` fields.
- sphere-light JSON rigs.
- PyTorch `model.pt` proxy artifacts where practical.
- fixed scene, fixed camera, fixed geometry, fixed materials for V1.

---

## Core Product Hypothesis

Artists and technical directors should be able to stay in Blender while using Neural Render Proxies for lighting iteration:

1. Choose a camera, frame, resolution, and cache backend.
2. Bake a light-agnostic path cache for the current shot.
3. Train or load a small proxy for that cache.
4. Edit NRP sphere lights as native Blender objects.
5. Preview the relit fixed-camera image inside Blender.
6. Optionally solve light parameters from a target image or reference render.
7. Commit the solved light rig back into the Blender scene or export JSON for ComfyUI and other tools.

This turns NRP relighting into an artist-facing Blender workflow rather than a one-way cache export.

---

## Target Repository Shape

Implement in this repository as the `Blender-NRP` project:

```text
Blender-NRP/
  README.md
  pyproject.toml
  blender_nrp/
    __init__.py
    addon.py
    panels.py
    properties.py
    operators/
      bake_cache.py
      validate_cache.py
      train_proxy.py
      load_proxy.py
      relight_preview.py
      optimize_lights.py
      import_lights.py
      export_lights.py
    core/
      camera.py
      gbuffer.py
      path_cache.py
      metadata.py
      lights.py
      proxy.py
      gather.py
      validation.py
      reports.py
    backends/
      stock_blender_hemi.py
      interface.py
    ui/
      preview_image.py
      light_editor.py
      status.py
  scripts/
    package_addon.py
    bake_fixture.py
    relight_fixture.py
    validate_cache.py
    validate_light_json.py
  tests/
    test_cache_schema.py
    test_camera_rays.py
    test_light_json.py
    test_metadata.py
    test_fixture_workflow.py
    fixtures/
      minimal_scene.blend
  examples/
    scene_manifests/
      blender_studio_spring.json
      blender_studio_sprite_fright.json
```

The add-on zip should install through Blender's normal add-on preferences. Development should also support Blender background-mode commands for automated fixture bake and relight tests.

---

## Scene Targets

Use Blender-native scenes for validation before generic glTF scenes:

- **Primary production-style target:** Blender Studio's `Spring`, because it is a Blender open movie with visually rich forest shots and public project material.
- **Secondary production-style target:** Blender Studio's `Sprite Fright`, because it has production shot files, set dressing, stylized materials, and forest environments that exercise fixed-camera relighting.
- **Mandatory fixture target:** a small local `.blend` fixture created in the repo with a Cornell-style room, textured primitives, at least one camera, and at least one conventional Blender light. This fixture is required so CI and first-run tests do not depend on downloading large production files.

Do not commit production `.blend` files, trained weights, or generated caches. Store only scene manifests with upstream URLs, license notes, recommended camera/frame settings, expected artifact names, and manual setup notes.

Useful upstream references:

- Disney Research Neural Render Proxies paper page: `https://studios.disneyresearch.com/2026/07/01/neural-render-proxies-for-interactive-and-differentiable-lighting/`
- bgyss/nrp sample reimplementation: `https://github.com/bgyss/nrp`
- Blender Studio Spring project: `https://studio.blender.org/projects/spring/`
- Blender Studio Sprite Fright project: `https://studio.blender.org/projects/sprite-fright/`
- Khronos Sponza glTF fallback: `https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza`

---

## Data Contracts

### Path Cache

`path_cache.npz` must remain compatible with the NRP cache vocabulary:

```text
n_paths: (H*W,)
seg_pixel: (S,)
seg_origin: (S, 3)
seg_dir: (S, 3)
seg_tmax: (S,)
seg_throughput: (S, 3)
albedo: (H, W, 3)
normal: (H, W, 3)
depth: (H, W)
position: (H, W, 3)
```

If the add-on writes extra arrays such as masks, object IDs, roughness, or packed layouts, current readers must still be able to ignore them or load a strict compatibility cache.

### Metadata

Write `metadata.json` beside the cache and model artifacts. It must include the fields expected by `ComfyUI-NeuralRenderProxy` and may add Blender-specific fields only if current readers tolerate them:

- `scene_id`
- `camera_id`
- `resolution`
- `light_type`: `sphere`
- `aux_features`: `["albedo", "normal", "depth"]`
- `bbox_min`
- `bbox_max`
- `model_width`
- `model_depth`
- `coordinate_system`

Blender-specific details belong in `bake_report.json` unless they are part of the strict metadata contract:

- `blender_file_name`
- `frame`
- `backend`
- `backend_version`
- `cache_schema_version`
- dependency versions
- warnings and approximation limits

### Light Rig JSON

Implement import and export for this compatible sphere-light rig schema:

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

The add-on should preserve enough custom properties on Blender objects to round trip this JSON without losing semantic fields.

---

## V1 Bake Strategy

### Stock Blender Approximation Backend

Implement a backend that works in an unmodified Blender install:

1. Resolve the selected camera, frame, render resolution, dependency graph, and evaluated scene.
2. Generate one camera ray per pixel, with an optional deterministic subpixel pattern for `spp > 1`.
3. Use Blender scene ray casting to find the first visible surface.
4. Export:
   - world-space hit position,
   - world-space normal,
   - depth,
   - approximate diffuse albedo,
   - pixel-to-segment mapping.
5. Generate a deterministic set of outgoing candidate light-transport segments per hit point:
   - V1 default: normal-oriented hemisphere spokes.
   - Store each spoke as `seg_origin`, `seg_dir`, `seg_tmax`, and `seg_throughput`.
   - Use albedo, cosine weighting, configured sample count, and optional blocker ray casts for `seg_throughput`.
   - Store no-hit pixels explicitly through `n_paths` and zeroed auxiliary buffers or a documented miss convention.

This does not claim to be physically exact path tracing. It is a renderer-independent approximation that produces a valid cache so the full Blender workflow can be built and tested.

### Future Renderer Backend Interface

Define an interface for future backends that can capture true path vertices and throughput from Cycles or another renderer:

```python
class PathCacheBackend:
    id: str
    label: str

    def bake(self, context: BakeContext, settings: BakeSettings) -> PathCachePayload:
        ...
```

The V1 add-on ships with `stock_blender_hemi`. The interface should leave room for `cycles_instrumented` without requiring a custom Blender build on day one.

---

## Blender UX

Create a top-level `Blender-NRP` panel in the scene or render properties. It can be split into subpanels once the workflow grows.

### Bake Panel

Controls:

- Scene ID.
- Camera selector.
- Frame selector.
- Resolution override.
- Samples per pixel.
- Hemisphere segment count per hit.
- Max segment distance.
- Backend selector.
- Output directory.
- Random seed.

Buttons:

- `Bake Path Cache`
- `Validate Cache`
- `Open Output Folder`
- `Load Existing Cache`

Show status for selected camera, expected resolution, last bake path, segment count, warnings, and validation result.

### Proxy Panel

Controls:

- Cache path.
- Model path.
- Network width.
- Network depth.
- Training iterations.
- Batch pixels.
- Learning rate.
- Device selector where available.

Buttons:

- `Train Proxy`
- `Load Proxy`
- `Validate Proxy`
- `Save Proxy Metadata`

V1 may call a bundled lightweight trainer, an optional `nrp`-compatible module, or a subprocess. The Blender UI must report progress and leave a training report beside the model.

### Relighting Panel

Controls:

- Active proxy.
- Active NRP light collection.
- Preview mode:
  - cache gather preview,
  - proxy preview,
  - Blender scene light preview.
- Exposure.
- Tonemap mode.
- Preview scale.
- Update mode:
  - manual,
  - debounced on light edit.

Buttons:

- `Create NRP Sphere Light`
- `Preview Relight`
- `Apply NRP Lights To Scene`
- `Render Reference With Blender`
- `Compare Preview To Reference`

The preview can be an Image Editor image, a custom UI preview, or a camera-view overlay. It does not need to be real time in V1.

### Inverse Relighting Panel

Controls:

- Target image source:
  - external image,
  - current Blender render,
  - stored preview image.
- Initial light rig.
- Optimize position.
- Optimize radius.
- Optimize color and intensity.
- Iterations.
- Learning rate.
- Pixel fraction.
- Optional mask image.

Buttons:

- `Optimize Lights From Target`
- `Preview Solved Lights`
- `Apply Solved Lights`
- `Export Solved Lights JSON`

This should mirror the ComfyUI `NRP Optimize Lights From Target` behavior at a functional level.

### Interchange Panel

Buttons:

- `Import NRP Lights JSON`
- `Export Selected NRP Lights JSON`
- `Export Current Blender Lights As NRP JSON`
- `Export ComfyUI Compatibility Bundle`

The compatibility bundle should contain cache, metadata, model if available, preview images, and light JSON in a layout that can be loaded by `ComfyUI-NeuralRenderProxy`.

---

## Blender Light Representation

On import or creation, represent NRP sphere lights as native Blender scene objects:

- V1 default: UV sphere mesh emitters so radius is visible in the viewport and render.
- Optional: point lights with radius or soft-size, linked to the sphere object.
- Names should follow a stable pattern such as `NRP_Sphere_001`.
- Store custom properties:
  - `nrp_light_type`
  - `nrp_scene_id`
  - `nrp_camera_id`
  - `nrp_coordinate_system`
  - `nrp_radius`
  - `nrp_color`
  - `nrp_intensity`

The add-on should be able to discover existing NRP light objects, update them from JSON, serialize selected lights back to JSON, and apply solved values without recreating objects unnecessarily.

Warn when imported light rigs do not match the active scene ID, camera ID, or coordinate system.

---

## File Outputs

For a scene ID such as `fixture_room_001`, write:

```text
output/
  fixture_room_001/
    path_cache.npz
    metadata.json
    bake_report.json
    model.pt
    train_report.json
    relight_preview.png
    solved_lights.json
    preview_albedo.png
    preview_normal.png
    preview_depth.png
```

Generated artifacts must live under an ignored output directory. The repository should commit only source, tests, fixture scenes, manifests, and small hand-authored examples.

---

## Must Have

- Installable Blender add-on zip named `Blender-NRP`.
- Works in current stock Blender without custom renderer builds.
- Provides a tiny fixture scene and background-mode bake command for tests.
- Exports `.npz` cache arrays matching the NRP schema.
- Exports strict `metadata.json` that `ComfyUI-NeuralRenderProxy` can load.
- Exports preview images for albedo, normal, depth, and at least one relit preview.
- Creates, edits, imports, and exports NRP sphere-light rigs inside Blender.
- Trains or loads a basic proxy from the baked cache.
- Previews relighting inside Blender from a loaded proxy or from cache gather.
- Optimizes at least one sphere light from a target image through the proxy.
- Applies solved lights back to native Blender objects.
- Writes machine-readable bake, validation, training, and solve reports.
- Documents the approximation limits of `stock_blender_hemi`.
- Provides scene manifests for Spring and Sprite Fright without committing large assets.

---

## Should Have

- Optional launch or export path for ComfyUI workflows that point at the baked cache and model.
- Progress reporting and cancellation for large bakes and training runs.
- Tiled or chunked baking to avoid memory spikes.
- Deterministic outputs for fixed scene, frame, camera, resolution, backend settings, and seed.
- Validation report with array shapes, finite-value checks, bbox, segment counts, and estimated cache size.
- A simple side-by-side preview comparing proxy relight, cache-gather relight, and Blender-rendered reference.
- A clear dependency strategy for PyTorch in Blender, including graceful errors when training or proxy inference dependencies are unavailable.

---

## Nice To Have

- Cycles instrumented backend for true multi-bounce path transport.
- Quad or area light support after sphere-light parity is proven.
- USD light export/import.
- Cryptomatte or object-mask buffers for masked optimization.
- Material AOV export for roughness, metallic, and emission.
- Blender viewport overlay showing sampled cache points and hemisphere segments.
- Live preview mode after the manual preview path is reliable.
- Compatibility with packed cache layouts from the `nrp` reference implementation.

---

## Explicit Non-Goals for V1

- Do not require a custom Blender build.
- Do not claim the stock backend is physically exact path tracing.
- Do not optimize for high frame-rate preview before the functional loop works.
- Do not support animated geometry or moving cameras in a single proxy.
- Do not bake arbitrary material edits into one reusable proxy.
- Do not commit large production `.blend` files, trained weights, or generated caches.
- Do not require a running ComfyUI server for the basic Blender bake, train, preview, solve, import, or export path.
- Do not attempt broad light-type coverage before sphere lights round trip correctly.

---

## Acceptance Criteria

1. The add-on installs in Blender from a zip and exposes `Blender-NRP` panels.
2. A minimal fixture `.blend` can be baked in Blender background mode.
3. The bake writes `path_cache.npz`, `metadata.json`, preview images, and `bake_report.json`.
4. A validation command confirms all cache arrays have the required keys, shapes, dtypes, finite values, and non-empty segment data.
5. The exported cache loads through `NRP Load Path Cache` in `ComfyUI-NeuralRenderProxy`.
6. A basic proxy can be trained from the exported cache, saved as `model.pt`, and reloaded in Blender.
7. Blender can create at least one NRP sphere light, preview the fixed-camera relight, and write `relight_preview.png`.
8. A target-image solve can update one sphere light's position, radius, color, or intensity through the proxy.
9. Solved lights can be applied back to native Blender scene objects.
10. `NRP Save Lights` output from ComfyUI can be imported into Blender as editable NRP sphere lights.
11. Blender-exported NRP light JSON can be loaded through `NRP Load Lights` in ComfyUI.
12. The README includes a tested end-to-end fixture path and a documented manual path for Spring or Sprite Fright.

---

## Verification Commands

Run from the repository:

```bash
python -m pytest
blender --background tests/fixtures/minimal_scene.blend --python scripts/bake_fixture.py
python scripts/validate_cache.py build/nrp/fixture_room_001/path_cache.npz build/nrp/fixture_room_001/metadata.json
blender --background tests/fixtures/minimal_scene.blend --python scripts/relight_fixture.py
python scripts/validate_light_json.py build/nrp/fixture_room_001/solved_lights.json
```

Then install the add-on in Blender and manually verify:

1. Bake the fixture scene from the UI.
2. Validate the cache from the UI.
3. Train or load a small proxy.
4. Create an NRP sphere light in Blender.
5. Preview the relit fixed-camera image in Blender.
6. Render or load a target image.
7. Optimize the light against the target image.
8. Apply the solved light values to Blender objects.
9. Export light JSON.
10. Load the cache, model, and light JSON in ComfyUI as a compatibility check.

---

## Implementation Notes

- Prefer Blender Python APIs for V1: `bpy`, evaluated dependency graph, camera projection utilities, and `scene.ray_cast`.
- Keep cache serialization, metadata, light JSON, gather, proxy loading, and validation independent of Blender UI modules so they can be tested with normal Python where possible.
- Keep coordinate transforms explicit. Blender is Z-up; existing NRP examples may use right-handed Y-up conventions. Metadata must identify the convention and any conversion performed.
- Use NumPy `.npz` output for compatibility. Do not invent a new cache format unless a converter and strict compatibility mode are included.
- Treat exact cache gather as a useful debugging preview even if proxy preview is the main artist-facing mode.
- Make failures readable: missing camera, unsupported material, no surface hit, zero segments, missing dependencies, unwritable output directory, invalid model path, mismatched metadata, and mismatched light JSON should produce actionable errors.
- Keep the first implementation boring and functional. Once the complete Blender loop works, performance, live preview, packed caches, and renderer instrumentation can be optimized separately.
