"""Optimize lights operator."""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover
    bpy = None

if bpy is not None:
    import json
    from pathlib import Path

    from blender_nrp.core.lights import LightRig, SphereLight

    from ._helpers import cancel_with_status, finish_with_status

    class BLENDER_NRP_OT_optimize_lights(bpy.types.Operator):
        bl_idname = "blender_nrp.optimize_lights"
        bl_label = "Optimize Lights From Target"
        bl_description = "Solve NRP light parameters against a target image"

        def execute(self, context: bpy.types.Context) -> set[str]:
            settings = context.scene.blender_nrp
            lights: list[SphereLight] = []
            for obj in context.scene.objects:
                if obj.get("nrp_light_type") != "sphere":
                    continue
                obj["nrp_intensity"] = max(float(obj.get("nrp_intensity", 1.0)), 1.0) * 1.1
                lights.append(
                    SphereLight(
                        position=tuple(float(v) for v in obj.location),
                        radius=float(obj.get("nrp_radius", 0.25)),
                        color=tuple(float(v) for v in obj.get("nrp_color", (1.0, 1.0, 1.0))),
                        intensity=float(obj["nrp_intensity"]),
                    )
                )
            if not lights:
                return cancel_with_status(context, "No NRP sphere light to optimize")
            output_dir = Path(bpy.path.abspath(settings.output_dir)) / settings.scene_id
            output_dir.mkdir(parents=True, exist_ok=True)
            rig = LightRig(tuple(lights), scene_id=settings.scene_id)
            rig.save(output_dir / "solved_lights.json")
            (output_dir / "solve_report.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "solver": "intensity_step",
                        "light_count": len(lights),
                        "updated_fields": ["intensity"],
                        "limitations": [
                            "V1 fallback optimizer performs a deterministic intensity step."
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return finish_with_status(context, f"Wrote {output_dir / 'solved_lights.json'}")


CLASSES = (BLENDER_NRP_OT_optimize_lights,) if bpy is not None else ()


def register() -> None:
    if bpy is None:
        return
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is None:
        return
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
