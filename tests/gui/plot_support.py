"""Plot diagnostics preserve scientific identity across Qt projections.

The plot panel consumes immutable prepared data and fit results. Display floors,
nanometre conversions, profile normalization, and qz transforms therefore act
only on presentation copies. Candidate and uncertainty ownership remain
explicit so inspecting an archived or invalid row cannot relabel evidence.

Interaction tests use the same stored two-theta and prepared-point index spaces
as the data service. Invalid gestures and failed redraws must leave both the
visible figures and the immutable project snapshot unchanged.
"""

from __future__ import annotations

import gc
import warnings
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QToolButton, QTreeWidget, QVBoxLayout, QWidget
from shiboken6 import isValid
from tests.support.model_cases import (
    final_fit_result,
    fit_candidate,
    prepared_data,
)

import xrr_fitter.api as api
from xrr_fitter.gui.plots.live import LiveReflectivityPlot

# The SLD profile left the tab bar for a permanent companion pane, so these are
# the switchable diagnostic tabs only, with the log view leading.
TAB_TITLES = (
    "对数反射率",
    "原始数据与模型",
    "qz⁴R",
    "加权残差",
    "候选解比较",
    "残差热图",
    "参数热图",
    "相关性与区间",
    "批量趋势",
)


def _candidate(data, candidate_id="candidate-a", *, objective=0.2, **changes):
    size = data.two_theta_deg.size
    values = {
        "objective": objective,
        "qz_a_inv": np.linspace(0.015, 0.25, size),
        "model_normalized": np.geomspace(0.9, 2e-5, size),
        "log_residuals_decades": np.linspace(-0.2, 0.2, size),
        "weighted_residuals": np.linspace(-1.0, 1.0, size),
        "sld_depth_a": np.array([0.0, 20.0, 50.0]),
        "sld_profile_a2": np.array([0.0 + 0.0j, 2e-5 + 1e-7j, 4e-6 + 0.0j]),
    }
    values.update(changes)
    base_objective = objective if np.isfinite(objective) else 1.0
    return replace(fit_candidate(candidate_id, base_objective), **values)


def _uncertainty(candidate_id="candidate-a", *, profiles=()):
    return api.UncertaintyReport(
        correlation_names=("component.0.thickness_a", "instrument.scale"),
        correlation_matrix=np.array([[1.0, -0.65], [-0.65, 1.0]]),
        profiles=profiles,
        bootstrap_intervals=(("component.0.thickness_a", 35.0, 48.0),),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _result(data, *, uncertainty=True):
    first = _candidate(data)
    second = _candidate(data, "candidate-b", objective=0.3)
    value = final_fit_result(first, second)
    if uncertainty:
        value = replace(value, uncertainty=_uncertainty())
    return value


def _panel(qtbot, *, data=None, result=None, bands=None):
    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    qtbot.addWidget(panel)
    if data is not None:
        panel.set_dataset("curve", data)
    if bands is not None and result is None:
        result = replace(_result(data), uncertainty=replace(_uncertainty(), sld_bands=bands))
    if result is not None:
        panel.set_result(result, "candidate-a")
    return panel


def _is_live(view):
    """Report whether a view is a pyqtgraph reflectivity pane rather than mpl."""
    return isinstance(view, LiveReflectivityPlot)


# The four pg panes expose fixed managed items set once at construction, so a
# curve's contextual label (what the matplotlib draw_* functions passed to the
# legend) maps to the owning item rather than to a per-draw artist. The residual
# curve and the raw included points both ride the observed item, so several
# labels collapse onto one attribute.
PG_ITEM_BY_LABEL = {
    "归一化数据": "observed_item",
    "拟合点": "observed_item",
    "加权残差": "observed_item",
    "当前候选模型": "model_item",
    "排除点": "excluded_item",
    "零参考线": "reference_item",
    "搜索中模型": "preview_item",
}


def _pg_xy(item):
    """Coerce a managed item's data to finite arrays; an empty item reads empty."""
    if item is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = item.getData()
    x = np.asarray([] if x is None else x, dtype=float)
    y = np.asarray([] if y is None else y, dtype=float)
    return x, y


def _line_y(view, label):
    if _is_live(view):
        y = _pg_xy(getattr(view, PG_ITEM_BY_LABEL[label]))[1]
        # A log-mode viewbox delivers y in log10 display space; delinearize so the
        # assertion reads the physical value the reader sees on the axis.
        if view.plot_item.ctrl.logYCheck.isChecked():
            return np.power(10.0, y)
        return y
    return next(line.get_ydata() for line in view.axes.lines if line.get_label() == label)


def _line_x(view, label):
    if _is_live(view):
        return _pg_xy(getattr(view, PG_ITEM_BY_LABEL[label]))[0]
    return next(line.get_xdata() for line in view.axes.lines if line.get_label() == label)


def _marker(view, label):
    if _is_live(view):
        return getattr(view, PG_ITEM_BY_LABEL[label]).opts["symbol"]
    return next(line.get_marker() for line in view.axes.lines if line.get_label() == label)


def _marker_filled(view, label):
    if _is_live(view):
        return getattr(view, PG_ITEM_BY_LABEL[label]).opts["symbolBrush"] is not None
    facecolor = next(line.get_markerfacecolor() for line in view.axes.lines if line.get_label() == label)
    return facecolor != "none"


def _view_xrange(view):
    if _is_live(view):
        return tuple(view.plot_item.vb.viewRange()[0])
    return view.axes.get_xlim()


def _addressable(view):
    """Return the widget that carries the object name and takes focus.

    A pg pane is its own widget; an mpl view exposes it through ``.canvas``.
    """
    return view if _is_live(view) else view.canvas


def _mpl_view_keys(panel):
    """The still-matplotlib view keys, in panel order (the six static views)."""
    return tuple(key for key in panel.view_keys() if not _is_live(panel.view(key)))


def _colour_keys(view):
    """The colour-key axes on a figure, in figure order.

    Matplotlib's own label on a colorbar axes is the only handle on it, so that
    label is what tells a key apart from a plotted pane.
    """
    from xrr_fitter.gui.plots.diagnostics import COLORBAR_AXES_LABEL

    return tuple(axes for axes in view.figure.axes if axes.get_label() == COLORBAR_AXES_LABEL)


def _artist_snapshot(panel):
    snapshot = []
    for key in panel.view_keys():
        view = panel.view(key)
        if _is_live(view):
            # A pg pane has no artist list; its state is the length of each managed
            # curve plus the two annotations, which is what a redraw would change.
            lengths = tuple(
                _pg_xy(getattr(view, attr))[0].size
                for attr in ("observed_item", "model_item", "excluded_item", "reference_item", "preview_item")
            )
            snapshot.append((key, lengths, view.placeholder_text(), view.quality_caption_text()))
        else:
            snapshot.append(
                (
                    key,
                    len(view.axes.lines),
                    len(view.axes.collections),
                    tuple(text.get_text() for text in view.axes.texts),
                )
            )
    return tuple(snapshot)


def _write_curve(path: Path, *, offset: float = 0.0, invalid_index: int | None = None) -> Path:
    rows = []
    for index in range(32):
        intensity = 1000.0 / (index + 1)
        if index == invalid_index:
            intensity = -intensity
        rows.append(f"{0.05 + offset + index * 0.02:.6f} {intensity:.12g}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _project_with_curves(tmp_path: Path, count: int = 2):
    value = api.new_project()
    for index in range(count):
        value = api.add_dataset(
            value,
            _write_curve(tmp_path / f"curve-{index}.xy", offset=index * 0.01),
            api.InstrumentSpec(instrument_id=f"plot-{index}"),
        )
    return value


def _mouse_event(name: str, panel, xdata: float, *, button: int = 1) -> MouseEvent:
    view = panel.view("raw")
    view.canvas.draw()
    ydata = float(np.nanmedian(view.axes.lines[0].get_ydata()))
    x, y = view.axes.transData.transform((xdata, ydata))
    return MouseEvent(name, view.canvas, x, y, button=button)


def _drag_range(panel, lower: float, upper: float) -> None:
    raw = panel.view("raw")
    if _is_live(raw):
        # The pg range selector is a draggable region; a test drives it by setting
        # the region and firing the finished signal the widget listens on, which is
        # the twin of the matplotlib press/motion/release the mpl branch replays.
        region = raw.range_item
        if region is None:
            return
        # setRegion already emits sigRegionChangeFinished once; block it so the
        # explicit emit below fires exactly one commit, matching a real drag.
        region.blockSignals(True)
        region.setRegion((lower, upper))
        region.blockSignals(False)
        region.sigRegionChangeFinished.emit(region)
        return
    canvas = raw.canvas
    canvas.callbacks.process("button_press_event", _mouse_event("button_press_event", panel, lower))
    canvas.callbacks.process("motion_notify_event", _mouse_event("motion_notify_event", panel, upper))
    canvas.callbacks.process("button_release_event", _mouse_event("button_release_event", panel, upper))


def _drag_sld_interface(panel, from_nm: float, to_nm: float) -> None:
    """Drag an interface handle on the SLD companion pane, in depth (nm).

    The handle is a full-height vertical line, so the vertical position of the
    gesture carries no meaning; the middle of the current view is used so the
    press lands inside the axes whatever the SLD range happens to be.
    """
    view = panel.view("sld")
    view.canvas.draw()
    lower, upper = view.axes.get_ylim()
    ydata = 0.5 * (float(lower) + float(upper))
    names = ("button_press_event", "motion_notify_event", "button_release_event")
    for name, depth_nm in zip(names, (from_nm, to_nm, to_nm), strict=True):
        x, y = view.axes.transData.transform((depth_nm, ydata))
        # A motion event also has to say which buttons are still down, the same
        # way a pan gesture does; see _drag_on_view.
        held = {MouseButton.LEFT} if name == "motion_notify_event" else None
        event = MouseEvent(name, view.canvas, x, y, button=MouseButton.LEFT, buttons=held)
        view.canvas.callbacks.process(name, event)


def _drag_sld_level(panel, index: int, depth_nm: float, factor: float) -> tuple[float, float]:
    """Drag layer ``index``'s SLD level handle to ``factor`` times its height.

    The gesture is expressed as a ratio rather than an absolute SLD so a case
    stays readable without restating the material's tabulated scattering length;
    the handle's current height is read off the pane and returned so the caller
    can state what it expected the ratio to become.
    """
    view = panel.view("sld")
    view.canvas.draw()
    label = f"_level_{index}"
    handle = next(line for line in view.axes.lines if line.get_label() == label)
    from_level = float(np.asarray(handle.get_ydata())[0])
    to_level = from_level * float(factor)
    names = ("button_press_event", "motion_notify_event", "button_release_event")
    for name, level in zip(names, (from_level, to_level, to_level), strict=True):
        x, y = view.axes.transData.transform((depth_nm, level))
        # A motion event also has to say which buttons are still down; see _drag_on_view.
        held = {MouseButton.LEFT} if name == "motion_notify_event" else None
        event = MouseEvent(name, view.canvas, x, y, button=MouseButton.LEFT, buttons=held)
        view.canvas.callbacks.process(name, event)
    return from_level, to_level


def _drag_sld_roughness(panel, index: int, to_sigma_nm: float) -> tuple[float, float]:
    """Drag interface ``index``'s roughness whisker so its width becomes ``to_sigma_nm``.

    The whisker runs from the interface depth outward by the interface's current
    roughness; its outer tip is the grab point.  The gesture reads the tip off the
    pane and drags it to ``interface + to_sigma_nm`` along the tip's own height, so
    the press lands on the handle whatever the SLD range happens to be.  The width
    the whisker was drawn at and the width the drag asks for are returned so the
    caller can state what it expected the interface roughness to become.
    """
    view = panel.view("sld")
    view.canvas.draw()
    label = f"_roughness_{index}"
    handle = next(line for line in view.axes.lines if line.get_label() == label)
    xs = np.asarray(handle.get_xdata(), dtype=float)
    ys = np.asarray(handle.get_ydata(), dtype=float)
    interface_nm = float(xs[0])
    height = float(ys[0])
    from_sigma_nm = float(xs[1]) - interface_nm
    to_x = interface_nm + float(to_sigma_nm)
    names = ("button_press_event", "motion_notify_event", "button_release_event")
    for name, depth_nm in zip(names, (float(xs[1]), to_x, to_x), strict=True):
        x, y = view.axes.transData.transform((depth_nm, height))
        # A motion event also has to say which buttons are still down; see _drag_on_view.
        held = {MouseButton.LEFT} if name == "motion_notify_event" else None
        event = MouseEvent(name, view.canvas, x, y, button=MouseButton.LEFT, buttons=held)
        view.canvas.callbacks.process(name, event)
    return from_sigma_nm, float(to_sigma_nm)


def _zero_width_bands():
    """A degenerate band: five identical quantile faces on one depth grid.

    Mirrors the io-side export fixture so the on-screen caption and the exported
    caption are pinned to the same ``SldUncertaintyBands.caption()`` text.
    """
    depth = np.linspace(0.0, 40.0, 4)
    levels = (0.025, 0.16, 0.5, 0.84, 0.975)
    real = np.tile(np.arange(len(levels), dtype=float)[:, None], (1, depth.size))
    return api.SldUncertaintyBands(
        depth_a=depth,
        quantiles=levels,
        real=real,
        imaginary=real * 0.5,
        align_label="基底界面",
        sample_count=500,
        total_samples=2000,
        failure_rate=0.0,
    )


__all__ = (
    "gc",
    "replace",
    "Path",
    "SimpleNamespace",
    "warnings",
    "weakref",
    "np",
    "pytest",
    "MouseButton",
    "MouseEvent",
    "QCoreApplication",
    "QEvent",
    "Qt",
    "QApplication",
    "QLineEdit",
    "QToolButton",
    "QTreeWidget",
    "QVBoxLayout",
    "QWidget",
    "isValid",
    "api",
    "LiveReflectivityPlot",
    "final_fit_result",
    "fit_candidate",
    "prepared_data",
    "TAB_TITLES",
    "_candidate",
    "_uncertainty",
    "_result",
    "_panel",
    "_is_live",
    "PG_ITEM_BY_LABEL",
    "_pg_xy",
    "_line_y",
    "_line_x",
    "_marker",
    "_marker_filled",
    "_view_xrange",
    "_addressable",
    "_mpl_view_keys",
    "_colour_keys",
    "_artist_snapshot",
    "_write_curve",
    "_project_with_curves",
    "_mouse_event",
    "_drag_range",
    "_drag_sld_interface",
    "_drag_sld_level",
    "_drag_sld_roughness",
    "_zero_width_bands",
)
