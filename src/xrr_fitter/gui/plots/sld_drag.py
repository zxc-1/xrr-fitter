"""SLD handle dragging for the diagnostic plot controller.

The SLD companion pane draws a grabbable handle for every editable layer edge,
level and roughness whisker; dragging one asks the panel to commit the matching
structure edit.  That gesture layer lives here, apart from
:mod:`xrr_fitter.gui.plots.interactions`, so each module stays a single
responsibility the complexity gate can weigh on its own.

:class:`SldHandleDragMixin` is mixed into ``PlotInteractionController``: at run
time ``self`` is the controller, so the methods read its ``_panel``, ``_views``,
``_navigators``, ``_sld_drag`` and ``_sld_hover`` state directly.  The
module-level ``_edit_*`` helpers rebuild an immutable structure from a dragged
value and are pure, so they carry no controller state and are tested through the
panel's edit signal.

一次结构提交会清空拟合结果，所以「抓到了」与「改了」分开判：抓取按像素距离判（层把手要贴着
自己那条电平线，界面圆点优先于粗糙度须线），提交要先越过拖动阈值，单击只是单击。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import hypot, isfinite, nextafter

import numpy as np
from matplotlib.backend_bases import MouseButton, cursors

from xrr_fitter.gui.plots.live import LiveReflectivityPlot
from xrr_fitter.gui.plots.sld import (
    MIN_DENSITY_SCALE,
    MIN_LAYER_THICKNESS_A,
    draggable_roughness,
)

# 指针沿把手那一轴移过这么多逻辑像素才算拖动。手按下去本来就会抖一两个像素，阈值以内松开
# 是单击——单击不该改结构，更不该顺手清掉拟合结果。
DRAG_THRESHOLD_PX = 3.0

# 线形把手（界面竖线、层电平线、粗糙度须线）的抓取半宽，逻辑像素。按像素而不是按视野比例：
# 视野一宽，比例式的容差会覆盖半个图层，按在哪儿都像是按在把手上。
GRAB_TOLERANCE_PX = 6.0

# 圆点的抓取半径在画出来的半径外再放这么多：点只有 5px 来宽，正好按在点上要求太苛刻。
DOT_SLACK_PX = 2.0

# 粗糙度须线只有 σ 那么长，σ 小时几乎整根缩在圆点里。圆点右缘之外至少留这么宽的一段给它，
# 否则粗糙度永远抓不到；这一段从圆点外缘起算，所以按在点上仍然是界面。
WHISKER_MIN_GRAB_PX = 6.0

# 悬停时把手加粗这么多倍：读者在按下之前就该知道这一下会抓到谁。
HOVER_EMPHASIS = 2.0

HANDLE_LABEL = re.compile(r"_(interface|level|roughness)_(\d+)")
HOVER_CURSORS = {"level": cursors.RESIZE_VERTICAL, "interface": cursors.RESIZE_HORIZONTAL}


@dataclass(slots=True)
class SldDrag:
    """一次进行中的把手手势：抓到的是谁、从哪儿按下、有没有越过拖动阈值。

    ``origin`` 在按下时就记下，而不是松开时回读：拖动会移动那条线，松开时它已经不记得手势
    从哪儿开始。``press_px`` 是按下时的像素坐标，阈值按它量。
    """

    kind: str
    index: int
    handle: object
    origin: float
    press_px: tuple[float, float]
    armed: bool = False


def _handle_key(line: object) -> tuple[str, int] | None:
    match = HANDLE_LABEL.fullmatch(str(line.get_label()))
    return None if match is None else (match.group(1), int(match.group(2)))


def _segment_gap(point: tuple[float, float], start: np.ndarray, end: np.ndarray) -> float:
    """``point`` 到线段 ``start``–``end`` 的像素距离；线段退化成一点时就是到那一点的距离。"""
    delta = end - start
    length = float(delta @ delta)
    offset = np.asarray(point, dtype=float) - start
    fraction = 0.0 if length == 0.0 else min(max(float(offset @ delta) / length, 0.0), 1.0)
    nearest = start + fraction * delta
    return hypot(float(point[0] - nearest[0]), float(point[1] - nearest[1]))


def _pixels(axes: object, line: object) -> np.ndarray:
    data = np.column_stack((np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)))
    return np.asarray(axes.transData.transform(data), dtype=float)


def _dot_radius_px(axes: object, line: object) -> float:
    """界面圆点画在屏幕上的半径（像素），含描边。"""
    points = 0.5 * float(line.get_markersize()) + 0.5 * float(line.get_markeredgewidth())
    return points * float(axes.figure.dpi) / 72.0


@dataclass(frozen=True, slots=True)
class HandleProbe:
    """一次按像素判的抓取：指针在哪儿、各把手线落在屏幕上的哪儿、容差按设备像素比放大多少。"""

    axes: object
    point: tuple[float, float]
    scale: float

    def dot_gap(self, line: object) -> float | None:
        tip = _pixels(self.axes, line)[-1]
        gap = hypot(self.point[0] - tip[0], self.point[1] - tip[1])
        return gap if gap <= _dot_radius_px(self.axes, line) + DOT_SLACK_PX * self.scale else None

    def whisker_gap(self, line: object, dot: object | None) -> float | None:
        """须线的抓取区：从圆点外缘起、到须尖（至少 ``WHISKER_MIN_GRAB_PX``）为止的一条横带。"""
        start, tip = _pixels(self.axes, line)
        edge = start[0] + (0.0 if dot is None else _dot_radius_px(self.axes, dot))
        reach = max(tip[0], edge + WHISKER_MIN_GRAB_PX * self.scale) + GRAB_TOLERANCE_PX * self.scale
        gap = abs(self.point[1] - start[1])
        inside = edge < self.point[0] <= reach and gap <= GRAB_TOLERANCE_PX * self.scale
        return gap if inside else None

    def line_gap(self, line: object) -> float | None:
        start, end = _pixels(self.axes, line)[[0, -1]]
        gap = _segment_gap(self.point, start, end)
        return gap if gap <= GRAB_TOLERANCE_PX * self.scale else None


def _named_handles(axes: object) -> dict[tuple[str, int], object]:
    return {key: line for line in axes.lines if (key := _handle_key(line)) is not None}


def _closest(
    handles: dict[tuple[str, int], object],
    kind: str,
    gap: Callable[[int, object], float | None],
) -> tuple[str, int, object] | None:
    """同一类把手里离指针最近、且落在容差内的那一个。"""
    best: tuple[float, int, object] | None = None
    for (handle_kind, index), line in handles.items():
        if handle_kind != kind:
            continue
        distance = gap(index, line)
        if distance is not None and (best is None or distance < best[0]):
            best = (distance, index, line)
    return None if best is None else (kind, best[1], best[2])


def pick_sld_handle(axes: object, point: tuple[float, float], scale: float = 1.0) -> tuple[str, int, object] | None:
    """指针在 ``point``（像素）时会抓到哪个把手，按目标从窄到宽依次判。

    界面圆点最先：它是读者瞄准的那个点，按在点上就该是界面——此前粗糙度先判，点右半边归了
    须线。然后是圆点之外的那截须线、界面竖线，最后才是层电平线；电平线横跨整层，但只在线附近
    才算抓到它，按在层里别处什么都不抓。
    """
    handles = _named_handles(axes)
    probe = HandleProbe(axes, point, scale)
    tiers = (
        ("interface", lambda _index, line: probe.dot_gap(line)),
        ("roughness", lambda index, line: probe.whisker_gap(line, handles.get(("interface", index)))),
        ("interface", lambda _index, line: probe.line_gap(line)),
        ("level", lambda _index, line: probe.line_gap(line)),
    )
    for kind, gap in tiers:
        picked = _closest(handles, kind, gap)
        if picked is not None:
            return picked
    return None


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


class SldHandleDragMixin:
    """SLD handle dragging mixed into :class:`PlotInteractionController`.

    Every method runs with ``self`` bound to the controller, so it reads the
    controller's ``_panel``/``_views``/``_navigators`` and the ``_sld_drag`` /
    ``_sld_hover`` slots the controller initialises and its ``release`` clears.
    """

    def _sld_axes(self) -> object | None:
        """Return the companion SLD pane's axes, or ``None`` when it is gone."""
        view = self._views.get("sld")
        if view is None or isinstance(view, LiveReflectivityPlot):
            return None
        return view.axes

    def _sld_scale(self, axes: object) -> float:
        """逻辑像素到事件像素的倍数：高分屏上事件坐标是物理像素，容差要跟着放大。"""
        return float(getattr(axes.figure.canvas, "device_pixel_ratio", 1.0) or 1.0)

    def _sld_redraw(self) -> None:
        axes = self._sld_axes()
        if axes is not None:
            axes.figure.canvas.draw_idle()

    def _sld_gestures_live(self) -> bool:
        """把手此刻接不接手势：运行中锁定编辑、或平移/框选占着画布时都不接。"""
        if self._panel is None or not self._editing_enabled:
            return False
        navigator = self._navigators.get("sld")
        return navigator is None or not navigator.mode

    def _sld_point(self, event: object, axes: object) -> tuple[float, float] | None:
        """事件落在 SLD 轴内时的像素坐标；落在轴外或没有坐标时为 ``None``。"""
        if getattr(event, "inaxes", None) is not axes:
            return None
        x, y = getattr(event, "x", None), getattr(event, "y", None)
        if x is None or y is None:
            return None
        return float(x), float(y)

    def _sld_press(self, event: object) -> None:
        """Grab whichever SLD handle a left-press landed on, without editing yet.

        The gesture stands down while a navigator owns the canvas (pan or box
        zoom) or a run has locked editing.  Which handle the press claims is
        :func:`pick_sld_handle`'s call; a press on none starts nothing.
        """
        axes = self._sld_axes()
        if axes is None or not self._sld_gestures_live() or getattr(event, "button", None) != MouseButton.LEFT:
            return
        point = self._sld_point(event, axes)
        picked = None if point is None else pick_sld_handle(axes, point, self._sld_scale(axes))
        self._sld_cancel_drag()
        if picked is not None:
            kind, index, handle = picked
            self._sld_drag = SldDrag(kind, index, handle, self._sld_origin(kind, handle), point)

    @staticmethod
    def _sld_origin(kind: str, handle: object) -> float:
        """手势的起点：层把手记电平高度（比例的分母），其余记界面深度。"""
        values = handle.get_ydata() if kind == "level" else handle.get_xdata()
        return float(np.asarray(values, dtype=float)[0])

    def _sld_crossed(self, drag: SldDrag, event: object) -> bool:
        """指针沿把手那一轴是否已经移过拖动阈值。"""
        x, y = getattr(event, "x", None), getattr(event, "y", None)
        if x is None or y is None:
            return False
        axes = self._sld_axes()
        scale = 1.0 if axes is None else self._sld_scale(axes)
        moved = float(y) - drag.press_px[1] if drag.kind == "level" else float(x) - drag.press_px[0]
        return abs(moved) >= DRAG_THRESHOLD_PX * scale

    @staticmethod
    def _axis_value(event: object, kind: str) -> float | None:
        """把手那一轴上的数据坐标：层把手读高度，其余读深度；没有有限值时为 ``None``。"""
        value = getattr(event, "ydata" if kind == "level" else "xdata", None)
        return float(value) if value is not None and isfinite(float(value)) else None

    def _sld_motion(self, event: object) -> None:
        """拖动中只跟手不提交；没抓着什么时这一下是悬停。

        越过拖动阈值之前把手不动：手抖的那一两个像素不该让线先跳一下再弹回去。
        """
        drag = self._sld_drag
        if drag is None:
            self._sld_hover_at(event)
            return
        drag.armed = drag.armed or self._sld_crossed(drag, event)
        coordinate = self._axis_value(event, drag.kind)
        if not drag.armed or coordinate is None:
            return
        self._sld_track_handle(drag.kind, drag.handle, drag.origin, coordinate)
        self._sld_redraw()

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

    def _sld_edit_from_release(self, event: object, drag: SldDrag, structure: object) -> object | None:
        """Map a release position to the structure edit the grabbed handle names.

        A release that carried no finite coordinate on the axis the handle lives
        on names no edit, and neither does one the ``_edit_*`` helper judges a no-op.
        """
        coordinate = self._axis_value(event, drag.kind)
        if coordinate is None:
            return None
        if drag.kind == "level":
            return _edit_layer_density(structure, drag.index, drag.origin, coordinate)
        if drag.kind == "interface":
            return _edit_layer_thickness(structure, drag.index, coordinate)
        return _edit_layer_roughness(structure, drag.index, (coordinate - drag.origin) * 10.0)

    def _sld_release(self, event: object) -> None:
        """Commit the dragged edit once, on release, then drop the grab.

        只有越过拖动阈值的手势才提交：单击落在把手上也只是单击。The edit is
        requested through the panel signal so a hand edit travels the same commit
        path as a typed one; a release that names no edit snaps the moved handle
        back, since no redraw follows to correct it.
        """
        drag = self._sld_drag
        self._sld_drag = None
        if drag is None or self._panel is None:
            return
        structure = getattr(self._panel, "_structure", None)
        crossed = drag.armed or self._sld_crossed(drag, event)
        edited = None if structure is None or not crossed else self._sld_edit_from_release(event, drag, structure)
        if edited is None:
            self._sld_restore_handle(drag)
            return
        self._panel.structure_edit_requested.emit(edited)

    def _sld_restore_handle(self, drag: SldDrag) -> None:
        """Put a handle back where it was grabbed when nothing changed.

        A roughness whisker is restored to the width the current structure still
        carries, read back off the same source that drew it.
        """
        if drag.kind == "interface":
            drag.handle.set_xdata([drag.origin, drag.origin])
        elif drag.kind == "roughness":
            drag.handle.set_xdata([drag.origin, drag.origin + self._sld_current_roughness_nm(drag.index)])
        else:
            drag.handle.set_ydata([drag.origin, drag.origin])
        self._sld_redraw()

    def _sld_cancel_drag(self) -> bool:
        """放弃进行中的拖动并把把手放回原处；确实有拖动可放弃时返回 ``True``。

        Esc 与运行中锁定编辑都走这里：此后的松开不再提交任何东西。
        """
        drag = self._sld_drag
        self._sld_drag = None
        if drag is None:
            return False
        self._sld_restore_handle(drag)
        return True

    def _sld_hover_at(self, event: object) -> None:
        """悬停时把指针下会被抓到的那个把手加粗，并换成它能移动的方向的光标。"""
        axes = self._sld_axes()
        if axes is None:
            return
        point = self._sld_point(event, axes) if self._sld_gestures_live() else None
        picked = None if point is None else pick_sld_handle(axes, point, self._sld_scale(axes))
        self._sld_set_hover(axes, picked)

    def _sld_set_hover(self, axes: object, picked: tuple[str, int, object] | None) -> None:
        hovered = None if self._sld_hover is None else self._sld_hover[0]
        target = None if picked is None else picked[2]
        if hovered is target:
            return
        self._sld_unhover()
        if picked is None:
            # 平移/框选占着画布时光标归导航器，不替它改回箭头。
            if self._sld_gestures_live():
                axes.figure.canvas.set_cursor(cursors.POINTER)
        else:
            kind, _index, line = picked
            self._sld_hover = (line, line.get_linewidth(), line.get_markersize())
            line.set_linewidth(line.get_linewidth() * HOVER_EMPHASIS)
            line.set_markersize(line.get_markersize() * 1.3)
            axes.figure.canvas.set_cursor(HOVER_CURSORS.get(kind, cursors.RESIZE_HORIZONTAL))
        axes.figure.canvas.draw_idle()

    def _sld_unhover(self) -> None:
        """把上一次悬停加粗的把手还原；那条线已被重画替换掉时还原它也无妨。"""
        hover = self._sld_hover
        self._sld_hover = None
        if hover is not None:
            line, width, size = hover
            line.set_linewidth(width)
            line.set_markersize(size)

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
