"""SLD handle dragging for the diagnostic plot controller.

The SLD companion pane draws a grabbable handle for every editable layer edge,
level and roughness whisker; dragging one asks the panel to commit the matching
structure edit.  That gesture layer lives here, apart from
:mod:`xrr_fitter.gui.plots.interactions`, so each module stays a single
responsibility the complexity gate can weigh on its own.

:class:`SldHandleDragMixin` is mixed into ``PlotInteractionController``: at run
time ``self`` is the controller, so the methods read its ``_panel``, ``_views``,
``_navigators`` and ``_sld_drag`` state directly.  The module-level ``_edit_*``
helpers rebuild an immutable structure from a dragged value and are pure, so
they carry no controller state and are tested through the panel's edit signal.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite, nextafter

import numpy as np
from matplotlib.backend_bases import MouseButton

from xrr_fitter.gui.plots.live import LiveReflectivityPlot
from xrr_fitter.gui.plots.sld import (
    MIN_DENSITY_SCALE,
    MIN_LAYER_THICKNESS_A,
    draggable_interfaces,
    draggable_layers,
    draggable_roughness,
)


def _edit_layer_thickness(structure: object, index: int, depth_nm: float) -> object | None:
    """Rebuild ``structure`` so layer ``index``'s backing edge sits at ``depth_nm``.

    A handle stands at a layer's backing edge, whose depth is the running sum of
    every thickness up to and including it, so moving it to ``depth_nm`` (in nm)
    sets this layer's thickness to that depth minus everything stacked before it.
    The layer is never driven below :data:`MIN_LAYER_THICKNESS_A`, so a gesture
    cannot invert the stack.  Returns ``None`` when the thickness is unchanged,
    sparing the caller a redundant structure commit and the refit it would force.
    """
    components = tuple(structure.components)
    prior_a = sum(float(components[i].thickness_a) for i in range(index))
    new_a = max(depth_nm * 10.0 - prior_a, MIN_LAYER_THICKNESS_A)
    if new_a == float(components[index].thickness_a):
        return None
    edited = replace(components[index], thickness_a=new_a)
    return replace(structure, components=(*components[:index], edited, *components[index + 1 :]))


def _edit_layer_density(
    structure: object,
    index: int,
    from_level: float,
    to_level: float,
) -> object | None:
    """Rebuild ``structure`` so layer ``index``'s drawn SLD level becomes ``to_level``.

    A layer's SLD is exactly linear in its ``density_scale`` - the override path
    multiplies the tabulated value by it, the formula path multiplies the bulk
    density by it - so the drag inverts in closed form: the new scale is the old
    one times the ratio of the levels, and no search over the profile is needed.
    The scale is held at :data:`MIN_DENSITY_SCALE` or above so a level dragged to
    or below the axis floor still commits a valid layer.  Returns ``None`` when
    the ratio is unusable or the scale is unchanged, sparing the caller a
    redundant commit and the refit it would force.
    """
    if not isfinite(from_level) or not isfinite(to_level) or from_level == 0.0:
        return None
    components = tuple(structure.components)
    old_scale = float(components[index].density_scale)
    new_scale = max(old_scale * (to_level / from_level), MIN_DENSITY_SCALE)
    if new_scale == old_scale:
        return None
    edited = replace(components[index], density_scale=new_scale)
    return replace(structure, components=(*components[:index], edited, *components[index + 1 :]))


# The physics validator rejects any interface roughness that reaches
# ``0.49 * min(neighbouring thickness)`` - equality is refused, not just excess -
# so a dragged width has to land strictly below that ceiling.  The neighbour
# search mirrors ``physics.stack._finite_neighbors`` on the padded thickness list
# ``[0, t0, ..., t_{N-1}, 0]`` a plain stack expands to; replicated here rather
# than imported because the GUI layer may reach only into ``api``, and kept
# byte-identical so a committed width is exactly one the validator would accept.
def _roughness_ceiling_a(components: tuple[object, ...], stack_index: int) -> float:
    thickness = [0.0, *(float(component.thickness_a) for component in components), 0.0]
    neighbors: list[float] = []
    if stack_index != 0:
        neighbors.append(thickness[stack_index])
    if stack_index + 1 != len(thickness) - 1:
        neighbors.append(thickness[stack_index + 1])
    return 0.49 * min(neighbors) if neighbors else 50.0


def _edit_layer_roughness(structure: object, index: int, sigma_a: float) -> object | None:
    """Rebuild ``structure`` so the interface below layer ``index`` has width ``sigma_a``.

    Handle ``index`` stands at layer ``index``'s backing edge, which is smeared by
    the roughness one step deeper in the expanded stack: an interior layer commits
    ``components[index + 1].roughness_a`` and the deepest layer commits the stack's
    ``backing_roughness_a``.  The width is held in ``[0, ceiling)`` - the largest
    float below the ceiling the validator enforces - so a commit never lands on or
    past the value the validator rejects.  A layer whose width comes from an
    interface transition is refused outright, because a transition and a plain
    roughness cannot both set it.  Returns ``None`` when the width is unchanged,
    sparing the caller a redundant commit and the refit it would force.
    """
    components = tuple(structure.components)
    stack_index = index + 1
    ceiling = _roughness_ceiling_a(components, stack_index)
    new_sigma = min(max(sigma_a, 0.0), nextafter(ceiling, 0.0))
    if stack_index == len(components):
        if new_sigma == float(getattr(structure, "backing_roughness_a", 0.0)):
            return None
        return replace(structure, backing_roughness_a=new_sigma)
    target = components[stack_index]
    if getattr(target, "transition", None) is not None:
        return None
    if new_sigma == float(target.roughness_a):
        return None
    edited = replace(target, roughness_a=new_sigma)
    return replace(structure, components=(*components[:stack_index], edited, *components[stack_index + 1 :]))


def _roughness_tip_hit(
    interface_nm: float,
    tip_x: float,
    tip_y: float,
    xdata: float,
    ydata: float,
    xspan: float,
    yspan: float,
) -> bool:
    """Return whether a press at ``(xdata, ydata)`` lands on a whisker's tip zone.

    The grab has to fall past the interface the whisker grows from
    (``interface_nm < xdata``) and close to the drawn tip in both depth and
    height, each judged against a small fraction of the visible span so the same
    gesture works whatever depth scale the pane shows.
    """
    return interface_nm < xdata and abs(xdata - tip_x) <= 0.05 * xspan and abs(ydata - tip_y) <= 0.08 * yspan


class SldHandleDragMixin:
    """SLD handle dragging mixed into :class:`PlotInteractionController`.

    Every method runs with ``self`` bound to the controller, so it reads the
    controller's ``_panel``/``_views``/``_navigators`` and the ``_sld_drag`` slot
    the controller initialises and its ``release`` clears.
    """

    def _sld_axes(self) -> object | None:
        """Return the companion SLD pane's axes, or ``None`` when it is gone."""
        view = self._views.get("sld")
        if view is None or isinstance(view, LiveReflectivityPlot):
            return None
        return view.axes

    def _sld_handle(self, kind: str, index: int) -> object | None:
        """Find the line that stands for handle ``index`` of the given kind."""
        axes = self._sld_axes()
        if axes is None:
            return None
        label = f"_{kind}_{index}"
        return next((line for line in axes.lines if line.get_label() == label), None)

    def _sld_grab_roughness(self, axes: object, xdata: float, ydata: float) -> tuple[str, int, object, float] | None:
        """Offer the roughness whisker whose outer tip the press landed on.

        The whisker runs outward from an interface, so a grab has to fall past
        the interface (``interface_nm < xdata``) and close to the drawn tip in
        both depth and height.  An interface press sits on the interface and a
        level press sits within the shallower layer body, so neither reaches
        here; only the narrow tip zone does.  The inner end - the interface the
        width is measured from - rides along as the origin the release measures
        the new width against.
        """
        low_x, high_x = (float(value) for value in axes.get_xlim())
        low_y, high_y = (float(value) for value in axes.get_ylim())
        xspan = abs(high_x - low_x)
        yspan = abs(high_y - low_y)
        if xspan <= 0.0 or yspan <= 0.0:
            return None
        for index, _interface_nm, _sigma_nm in draggable_roughness(getattr(self._panel, "_structure", None)):
            handle = self._sld_handle("roughness", index)
            if handle is None:
                continue
            xs = np.asarray(handle.get_xdata(), dtype=float)
            ys = np.asarray(handle.get_ydata(), dtype=float)
            interface_nm = float(xs[0])
            tip_x = float(xs[-1])
            tip_y = float(ys[0])
            if _roughness_tip_hit(interface_nm, tip_x, tip_y, xdata, ydata, xspan, yspan):
                return ("roughness", index, handle, interface_nm)
        return None

    def _sld_grab_interface(self, axes: object, xdata: float) -> tuple[str, int, object, float] | None:
        """Offer the interface handle whose depth the press landed close to.

        Closeness is judged against a small fraction of the visible depth span so
        the same gesture works whether the pane shows nanometres or tens of them.
        """
        interfaces = draggable_interfaces(getattr(self._panel, "_structure", None))
        if not interfaces:
            return None
        low, high = (float(value) for value in axes.get_xlim())
        span = abs(high - low)
        if span <= 0.0:
            return None
        index, depth_nm = min(interfaces, key=lambda item: abs(item[1] - xdata))
        if abs(depth_nm - xdata) > 0.05 * span:
            return None
        handle = self._sld_handle("interface", index)
        return None if handle is None else ("interface", index, handle, depth_nm)

    def _sld_grab_level(self, xdata: float) -> tuple[str, int, object, float] | None:
        """Offer the level handle of whichever layer the press landed inside.

        A level handle spans its whole layer, so the press only has to fall
        between that layer's two edges; there is no proximity radius to miss.
        The height the handle was drawn at rides along as the gesture's origin,
        because that is the level the layer's current density scale produces and
        so the "from" of the ratio the release inverts.
        """
        for index, front_nm, backing_nm in draggable_layers(getattr(self._panel, "_structure", None)):
            if front_nm <= xdata <= backing_nm:
                handle = self._sld_handle("level", index)
                if handle is not None:
                    drawn = float(np.asarray(handle.get_ydata(), dtype=float)[0])
                    return ("level", index, handle, drawn)
        return None

    def _sld_left_press_xy(self, event: object, axes: object) -> tuple[float, float | None] | None:
        """Return the ``(x, y)`` a usable left-press delivered on the SLD axes.

        ``y`` is ``None`` when the pointer carried no finite height; the grabs
        that need one check for it themselves.  ``None`` for the whole result
        means the event was not a finite-``x`` left-press sitting on the pane, so
        there is nothing to grab.
        """
        if getattr(event, "inaxes", None) is not axes:
            return None
        if getattr(event, "button", None) != MouseButton.LEFT:
            return None
        xdata = getattr(event, "xdata", None)
        if xdata is None or not isfinite(float(xdata)):
            return None
        ydata = getattr(event, "ydata", None)
        y = float(ydata) if ydata is not None and isfinite(float(ydata)) else None
        return float(xdata), y

    def _sld_press(self, event: object) -> None:
        """Grab whichever SLD handle a left-press landed on.

        The gesture stands down while a navigator owns the canvas (pan or box
        zoom) so hand-editing never fights a zoom drag.  A roughness whisker wins
        the first look: it is the narrowest target, a tip zone sitting past a
        layer's backing edge, so a press there reads as "widen this interface".
        Failing that an interface handle wins over a level one, since it is the
        narrower target, and a press on none starts nothing.
        """
        if self._panel is None:
            return
        navigator = self._navigators.get("sld")
        if navigator is not None and navigator.mode:
            return
        axes = self._sld_axes()
        if axes is None:
            return
        point = self._sld_left_press_xy(event, axes)
        if point is None:
            return
        xdata, ydata = point
        self._sld_drag = self._sld_pick_grab(axes, xdata, ydata)

    def _sld_pick_grab(self, axes: object, xdata: float, ydata: float | None) -> tuple[str, int, object, float] | None:
        """Pick the handle a left-press claims, narrowest target first.

        A roughness whisker wins the first look when the press carried a finite
        height, its tip zone being the smallest target; an interface handle then
        beats a level one for being narrower, and a press matching none grabs
        nothing.
        """
        grab = self._sld_grab_roughness(axes, xdata, ydata) if ydata is not None else None
        return grab or self._sld_grab_interface(axes, xdata) or self._sld_grab_level(xdata)

    def _sld_motion(self, event: object) -> None:
        """Track the grabbed handle under the pointer without committing yet.

        An interface handle follows the pointer sideways, a level handle follows
        it up and down, and a roughness whisker stretches its outer tip sideways
        from the fixed interface, each along the one axis it means something on,
        so a wobble in the other direction cannot smear the gesture.
        """
        drag = self._sld_drag
        if drag is None or self._panel is None:
            return
        kind, _index, handle, origin = drag
        coordinate = getattr(event, "ydata" if kind == "level" else "xdata", None)
        if coordinate is None or not isfinite(float(coordinate)):
            return
        self._sld_track_handle(kind, handle, origin, float(coordinate))
        axes = self._sld_axes()
        if axes is not None:
            axes.figure.canvas.draw_idle()

    def _sld_track_handle(self, kind: str, handle: object, origin: float, coordinate: float) -> None:
        """Slide ``handle`` to ``coordinate`` along the one axis its kind lives on.

        An interface handle moves sideways, a level handle up and down, and a
        roughness whisker stretches its outer tip sideways from the fixed
        ``origin`` interface, so a wobble off that axis cannot smear the gesture.
        """
        if kind == "interface":
            handle.set_xdata([coordinate, coordinate])
        elif kind == "roughness":
            handle.set_xdata([origin, max(coordinate, origin)])
        else:
            handle.set_ydata([coordinate, coordinate])

    def _sld_edit_from_release(
        self,
        event: object,
        kind: str,
        index: int,
        origin: float,
        structure: object,
    ) -> object | None:
        """Map a release position to the structure edit the grabbed handle names.

        A level handle reads its new height off ``ydata``; an interface or
        roughness handle reads its new depth off ``xdata``.  A release that
        carried no finite coordinate on the axis the handle lives on names no
        edit, and neither does one the ``_edit_*`` helper judges a no-op.
        """
        if kind == "level":
            ydata = getattr(event, "ydata", None)
            if ydata is None or not isfinite(float(ydata)):
                return None
            return _edit_layer_density(structure, index, origin, float(ydata))
        xdata = getattr(event, "xdata", None)
        if xdata is None or not isfinite(float(xdata)):
            return None
        if kind == "interface":
            return _edit_layer_thickness(structure, index, float(xdata))
        return _edit_layer_roughness(structure, index, (float(xdata) - origin) * 10.0)

    def _sld_release(self, event: object) -> None:
        """Commit the dragged edit once, on release, then drop the grab.

        The edit is requested through the panel signal so a hand edit travels the
        same commit path as a typed one.  A release that lands off the axes or on
        an unchanged value commits nothing; since no redraw follows to correct it,
        the handle the motion moved is snapped back to what it names.
        """
        drag = self._sld_drag
        self._sld_drag = None
        if drag is None or self._panel is None:
            return
        kind, index, handle, origin = drag
        structure = getattr(self._panel, "_structure", None)
        if structure is None:
            return
        edited = self._sld_edit_from_release(event, kind, index, origin, structure)
        if edited is None:
            self._sld_restore_handle(kind, index, handle, origin)
            return
        self._panel.structure_edit_requested.emit(edited)

    def _sld_restore_handle(self, kind: str, index: int, handle: object, origin: float) -> None:
        """Put a released handle back where it was grabbed when nothing changed.

        No redraw follows a commit that did not happen, so the artist the motion
        moved would otherwise sit at a value the structure never took.  A
        roughness whisker is restored to the width the current structure still
        carries, read back off the same source that drew it.
        """
        if kind == "interface":
            handle.set_xdata([origin, origin])
        elif kind == "roughness":
            handle.set_xdata([origin, origin + self._sld_current_roughness_nm(index)])
        else:
            handle.set_ydata([origin, origin])
        axes = self._sld_axes()
        if axes is not None:
            axes.figure.canvas.draw_idle()

    def _sld_current_roughness_nm(self, index: int) -> float:
        """Read back the width (nm) the structure still carries for whisker ``index``.

        Snaps a released roughness handle to the width the structure holds when a
        drag committed nothing, read off the same source that drew it; returns
        ``0.0`` when the interface is no longer draggable.
        """
        for candidate, _interface_nm, width_nm in draggable_roughness(getattr(self._panel, "_structure", None)):
            if candidate == index:
                return width_nm
        return 0.0
