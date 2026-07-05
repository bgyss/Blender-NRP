"""Object-level panel for NRP light properties (intensity, color, radius, …).

Raw custom properties are awkward to find and edit in Blender's UI. This panel
sits in the Object Properties tab and appears only when the active object has an
``nrp_light_type`` custom property, giving users proper labelled sliders with
sensible min/max ranges. Changing any value through the panel tags the object as
updated so the ``depsgraph_update_post`` handler can trigger a live-preview
refresh.
"""

from __future__ import annotations

try:
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised only inside Blender.
    bpy = None


if bpy is not None:

    def _tag_update(obj: bpy.types.Object) -> None:
        """Notify the depsgraph that this object changed (drives live preview)."""
        obj.update_tag()

    def _get_intensity(obj: bpy.types.Object) -> float:
        return float(obj.get("nrp_intensity", 1.0))

    def _set_intensity(obj: bpy.types.Object, value: float) -> None:
        obj["nrp_intensity"] = float(value)
        _tag_update(obj)

    def _get_color(obj: bpy.types.Object) -> tuple[float, float, float]:
        c = obj.get("nrp_color", (1.0, 1.0, 1.0))
        return (float(c[0]), float(c[1]), float(c[2]))

    def _set_color(obj: bpy.types.Object, value) -> None:
        obj["nrp_color"] = [float(value[0]), float(value[1]), float(value[2])]
        _tag_update(obj)

    def _get_radius(obj: bpy.types.Object) -> float:
        return float(obj.get("nrp_radius", 0.25))

    def _set_radius(obj: bpy.types.Object, value: float) -> None:
        obj["nrp_radius"] = float(value)
        _tag_update(obj)

    def _get_width(obj: bpy.types.Object) -> float:
        return float(obj.get("nrp_width", 1.0))

    def _set_width(obj: bpy.types.Object, value: float) -> None:
        obj["nrp_width"] = float(value)
        _tag_update(obj)

    def _get_height(obj: bpy.types.Object) -> float:
        return float(obj.get("nrp_height", 1.0))

    def _set_height(obj: bpy.types.Object, value: float) -> None:
        obj["nrp_height"] = float(value)
        _tag_update(obj)

    # -- Annotation-driven properties on bpy.types.Object -----------------------
    # These act as typed wrappers around the raw custom properties so that
    # Blender can draw proper sliders / colour pickers in the panel.

    _PROPS: list[tuple[str, bpy.props._PropertyDeferred]] = [
        (
            "nrp_intensity_prop",
            bpy.props.FloatProperty(
                name="Intensity",
                description="NRP light intensity (linear multiplier on emission)",
                get=_get_intensity,
                set=_set_intensity,
                min=0.0,
                soft_max=100.0,
                default=1.0,
            ),
        ),
        (
            "nrp_color_prop",
            bpy.props.FloatVectorProperty(
                name="Color",
                description="NRP light emission color",
                subtype="COLOR",
                size=3,
                get=_get_color,
                set=_set_color,
                min=0.0,
                max=1.0,
                default=(1.0, 1.0, 1.0),
            ),
        ),
        (
            "nrp_radius_prop",
            bpy.props.FloatProperty(
                name="Radius",
                description="NRP sphere light radius",
                get=_get_radius,
                set=_set_radius,
                min=0.001,
                soft_max=10.0,
                default=0.25,
            ),
        ),
        (
            "nrp_width_prop",
            bpy.props.FloatProperty(
                name="Width",
                description="NRP quad light width",
                get=_get_width,
                set=_set_width,
                min=0.001,
                soft_max=50.0,
                default=1.0,
            ),
        ),
        (
            "nrp_height_prop",
            bpy.props.FloatProperty(
                name="Height",
                description="NRP quad light height",
                get=_get_height,
                set=_set_height,
                min=0.001,
                soft_max=50.0,
                default=1.0,
            ),
        ),
    ]

    class BLENDER_NRP_PT_light(bpy.types.Panel):
        bl_label = "NRP Light"
        bl_idname = "BLENDER_NRP_PT_light"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "object"

        @classmethod
        def poll(cls, context: bpy.types.Context) -> bool:
            obj = context.active_object
            return obj is not None and obj.get("nrp_light_type") in ("sphere", "quad")

        def draw(self, context: bpy.types.Context) -> None:
            layout = self.layout
            obj = context.active_object
            kind = obj.get("nrp_light_type")

            layout.label(
                text=f"Type: {kind.title()}",
                icon="LIGHT_POINT" if kind == "sphere" else "LIGHT_AREA",
            )
            layout.prop(obj, "nrp_intensity_prop")
            layout.prop(obj, "nrp_color_prop")
            if kind == "sphere":
                layout.prop(obj, "nrp_radius_prop")
            elif kind == "quad":
                layout.prop(obj, "nrp_width_prop")
                layout.prop(obj, "nrp_height_prop")

    CLASSES = (BLENDER_NRP_PT_light,)

else:
    _PROPS = []
    CLASSES = ()


def register() -> None:
    if bpy is None:
        return
    for attr, prop in _PROPS:
        setattr(bpy.types.Object, attr, prop)
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    if bpy is None:
        return
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    for attr, _prop in reversed(_PROPS):
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
