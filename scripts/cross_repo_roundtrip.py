#!/usr/bin/env python
"""Cross-repo round-trip check (verification tier 2).

Bakes a fixture cache with the pure-Python cycles_capture path, then verifies the
artifacts against the *actual* sibling implementations:

1. `nrp` (reference repo): `PathCache.load` reads the cache (default and packed
   layouts), and `gather_light` agrees with `blender_nrp`'s gather to <= 1e-8 for
   sphere and quad lights.
2. `nrp` torch backend (if torch is importable): `TorchNRP.load` reads a model.pt
   trained here and evaluates identically to our copy of the architecture.
3. `ComfyUI-NeuralRenderProxy` (if torch is importable): `NRPLightRig.from_json`
   parses a sphere rig exported in right_handed_y_up, and its non-normalizing
   gather over our pre-divided ComfyUI export bundle matches the reference gather.

Sibling repo locations default to ../nrp and ../ComfyUI-NeuralRenderProxy next to
this repo; override with NRP_REPO / COMFY_REPO. Both siblings ship a top-level
package called `nrp`, so they are imported one at a time with a sys.modules purge
in between.

Exit code 0 = all applicable checks passed (torch-dependent checks report SKIP
without failing when torch is absent).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NRP_REPO = Path(os.environ.get("NRP_REPO", ROOT.parent / "nrp"))
COMFY_REPO = Path(os.environ.get("COMFY_REPO", ROOT.parent / "ComfyUI-NeuralRenderProxy"))

TOLERANCE = 1e-8


def _purge_nrp_modules() -> None:
    for name in [m for m in sys.modules if m == "nrp" or m.startswith("nrp.")]:
        del sys.modules[name]


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    from blender_nrp.backends import cycles_capture
    from blender_nrp.backends.interface import BakeSettings
    from blender_nrp.core.comfy_export import export_comfy_bundle
    from blender_nrp.core.coords import RIGHT_HANDED_Y_UP, convert_rig
    from blender_nrp.core.gather import gather_hdr
    from blender_nrp.core.lights import LightRig, QuadLight, SphereLight
    from blender_nrp.core.metadata import NRPMetadata
    from blender_nrp.core.path_cache import load_arrays, save_arrays

    out_dir = ROOT / "build" / "cross_repo"
    settings = BakeSettings(
        scene_id="roundtrip",
        output_dir=out_dir,
        width=24,
        height=20,
        segment_count=1,
        max_segment_distance=100.0,
        paths_per_pixel=16,
        max_bounces=4,
        seed=3,
    )
    result = cycles_capture.bake(None, settings)
    arrays = load_arrays(result.cache_path).arrays
    packed_path = result.output_dir / "path_cache_packed.npz"
    save_arrays(packed_path, arrays, width=settings.width, height=settings.height, packed=True)
    print(f"baked fixture cache: {result.cache_path}")

    sphere = SphereLight(
        position=(0.1, -0.2, 1.4), radius=0.45, color=(2.0, 1.0, 0.5), intensity=1.5
    )
    quad = QuadLight(
        position=(0.0, 0.4, 1.1),
        normal=(0.2, -0.6, 0.75),
        width=1.3,
        height=0.9,
        color=(1.0, 1.0, 1.0),
        intensity=2.0,
    )
    ours = {
        "sphere": gather_hdr(arrays, (sphere,)),
        "quad": gather_hdr(arrays, (quad,)),
    }

    # ---- Check 1: nrp reference loads both layouts and gathers identically.
    if not NRP_REPO.exists():
        _fail(f"nrp repo not found at {NRP_REPO} (set NRP_REPO)")
    _purge_nrp_modules()
    sys.path.insert(0, str(NRP_REPO))
    from nrp.gather_light import gather_light as nrp_gather
    from nrp.lights import QuadLight as NrpQuad
    from nrp.lights import SphereLight as NrpSphere
    from nrp.path_cache import PathCache as NrpPathCache

    for label, path in (("default", result.cache_path), ("packed", packed_path)):
        cache = NrpPathCache.load(str(path))
        if cache.segment_count != int(arrays["seg_pixel"].shape[0]):
            _fail(f"nrp loaded {label} cache with wrong segment count")
        if label == "packed":
            continue  # fp16/rgb9e5 quantization: only exact-layout gathers compared
        nrp_sphere_img = nrp_gather(
            cache,
            NrpSphere(
                center=list(sphere.position),
                radius=sphere.radius,
                rgb=[c * sphere.intensity for c in sphere.color],
            ),
        )
        nrp_quad_img = nrp_gather(
            cache,
            NrpQuad(
                center=list(quad.position),
                normal=list(quad.normal),
                width=quad.width,
                height=quad.height,
                rgb=[c * quad.intensity for c in quad.color],
            ),
        )
        for kind, reference in (("sphere", nrp_sphere_img), ("quad", nrp_quad_img)):
            diff = float(np.abs(reference - ours[kind]).max())
            if diff > TOLERANCE:
                _fail(f"nrp {kind} gather differs from blender_nrp by {diff} (> {TOLERANCE})")
            print(f"PASS: nrp {label} {kind} gather agreement (max diff {diff:.2e})")
    print("PASS: nrp reads the packed cache natively")

    # ---- Check 2: nrp's TorchNRP loads a model.pt trained here.
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False
    if torch_available:
        from blender_nrp.core.torch_proxy.model import TorchNRP as OurTorchNRP
        from blender_nrp.core.torch_proxy.train import train_proxy

        model_path = result.output_dir / "model.pt"
        train_proxy(
            arrays,
            model_path,
            iterations=50,
            batch_size=1024,
            pool_size=4,
            n_val_lights=2,
            device="cpu",
            checkpoint_every=0,
        )
        from nrp.torch_backend.model import TorchNRP as NrpTorchNRP

        theirs = NrpTorchNRP.load(str(model_path))
        mine = OurTorchNRP.load(str(model_path))
        gen = torch.Generator().manual_seed(0)
        xy = torch.rand(64, 2, generator=gen)
        aux = torch.rand(64, 7, generator=gen)
        params = torch.rand(64, 4, generator=gen)
        if not torch.equal(theirs(xy, aux, params), mine(xy, aux, params)):
            _fail("nrp TorchNRP evaluates our model.pt differently")
        print("PASS: nrp TorchNRP loads model.pt and evaluates identically")
    else:
        print("SKIP: torch unavailable, model.pt cross-load not checked")

    sys.path.remove(str(NRP_REPO))
    _purge_nrp_modules()

    # ---- Check 3: ComfyUI parses our rig JSON; its gather matches over the
    # pre-divided export bundle.
    if not COMFY_REPO.exists():
        _fail(f"ComfyUI repo not found at {COMFY_REPO} (set COMFY_REPO)")
    if not torch_available:
        print("SKIP: torch unavailable, ComfyUI (torch-based) round-trip not checked")
        print("cross-repo round-trip complete")
        return 0

    import torch

    sys.path.insert(0, str(COMFY_REPO))
    from nrp.gather import gather_sphere
    from nrp.lights import NRPLightRig

    rig = LightRig((sphere,), scene_id="roundtrip", camera_id="Camera")
    rig_yup = convert_rig(rig, RIGHT_HANDED_Y_UP)
    rig_path = result.output_dir / "lights_y_up.json"
    rig_yup.save(rig_path)
    comfy_rig = NRPLightRig.from_json(json.loads(rig_path.read_text()))
    if comfy_rig.coordinate_system != RIGHT_HANDED_Y_UP or len(comfy_rig.lights) != 1:
        _fail("ComfyUI NRPLightRig parse mismatch")
    print("PASS: ComfyUI NRPLightRig parses the exported y-up rig")

    metadata = NRPMetadata.load(result.metadata_path)
    comfy_cache_path = result.output_dir / "comfy_cache.npz"
    export_comfy_bundle(
        arrays, metadata, comfy_cache_path, result.output_dir / "comfy_metadata.json"
    )
    from nrp.cache import PathCache as ComfyPathCache

    comfy_cache = ComfyPathCache.from_npz(comfy_cache_path)
    comfy_image = gather_sphere(comfy_cache, comfy_rig).cpu().numpy().astype(np.float64)
    diff = float(np.abs(comfy_image - ours["sphere"]).max())
    # ComfyUI stores float32 tensors; agreement is float32-level, not 1e-8.
    if diff > 1e-4:
        _fail(f"ComfyUI gather over the export bundle differs by {diff}")
    print(f"PASS: ComfyUI gather matches reference gather (max diff {diff:.2e})")
    del torch
    print("cross-repo round-trip complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
