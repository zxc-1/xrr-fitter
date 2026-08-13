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
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QToolButton, QTreeWidget, QVBoxLayout, QWidget
from shiboken6 import isValid
from tests.support.model_cases import (
    final_fit_result,
    fit_candidate,
    prepared_data,
)

import xrr_fitter.api as api

# The SLD profile left the tab bar for a permanent companion pane, so these are
# the switchable diagnostic tabs only, with the log view leading.
TAB_TITLES = (
    "对数反射率",
    "原始数据与模型",
    "qz⁴R",
    "加权残差",
    "候选解比较",
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


def _line_y(view, label):
    return next(line.get_ydata() for line in view.axes.lines if line.get_label() == label)


def _artist_snapshot(panel):
    return tuple(
        (
            key,
            len(panel.view(key).axes.lines),
            len(panel.view(key).axes.collections),
            tuple(text.get_text() for text in panel.view(key).axes.texts),
        )
        for key in panel.view_keys()
    )


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
    canvas = panel.view("raw").canvas
    canvas.callbacks.process("button_press_event", _mouse_event("button_press_event", panel, lower))
    canvas.callbacks.process("motion_notify_event", _mouse_event("motion_notify_event", panel, upper))
    canvas.callbacks.process("button_release_event", _mouse_event("button_release_event", panel, upper))


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
    "final_fit_result",
    "fit_candidate",
    "prepared_data",
    "TAB_TITLES",
    "_candidate",
    "_uncertainty",
    "_result",
    "_panel",
    "_line_y",
    "_artist_snapshot",
    "_write_curve",
    "_project_with_curves",
    "_mouse_event",
    "_drag_range",
    "_zero_width_bands",
)
