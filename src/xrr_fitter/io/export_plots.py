"""Deterministic headless PNG serialization for exported fit results."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from io import BytesIO
from typing import ParamSpec

from matplotlib import rc_context, rcParamsDefault
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import numpy as np

from xrr_fitter.io.export_tables import DatasetExportData, _contexts


PNG_METADATA = {
    "Software": "xrr-fitter",
    "Creation Time": None,
}
TREND_LABEL_FONT = FontProperties(
    family=("PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans")
)
P = ParamSpec("P")


def _default_matplotlib_style(render: Callable[P, bytes]) -> Callable[P, bytes]:
    @wraps(render)
    def isolated(*args: P.args, **kwargs: P.kwargs) -> bytes:
        with rc_context(rc=rcParamsDefault):
            return render(*args, **kwargs)

    return isolated


def _context(value: DatasetExportData) -> DatasetExportData:
    if not isinstance(value, DatasetExportData):
        raise TypeError("context must be DatasetExportData")
    return value


def _png(figure: Figure) -> bytes:
    buffer = BytesIO()
    try:
        FigureCanvasAgg(figure)
        figure.savefig(
            buffer,
            format="png",
            dpi=120,
            metadata=PNG_METADATA,
            facecolor="white",
        )
        return buffer.getvalue()
    finally:
        figure.clear()


def _base_figure(height: float) -> Figure:
    figure = Figure(figsize=(7.2, height), dpi=120, facecolor="white")
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.94, hspace=0.24)
    return figure


@_default_matplotlib_style
def fit_overview_png(context: DatasetExportData) -> bytes:
    """Render measured/model reflectivity and log residual evidence."""
    value = _context(context)
    data = value.data
    selected = value.selected
    figure = _base_figure(5.8)
    reflectivity, residual = figure.subplots(2, 1, sharex=True)
    measured = np.maximum(data.intensity_normalized, data.r_floor)
    model = np.maximum(selected.model_normalized, data.r_floor)
    reflectivity.semilogy(selected.qz_a_inv, measured, color="#1f2937", linewidth=1.0, label="data")
    reflectivity.semilogy(selected.qz_a_inv, model, color="#c2410c", linewidth=1.2, label="model")
    reflectivity.set_ylabel("Reflectivity")
    reflectivity.grid(True, alpha=0.20)
    reflectivity.legend(loc="best", frameon=False)
    residual.axhline(0.0, color="#6b7280", linewidth=0.8)
    residual.plot(
        selected.qz_a_inv,
        selected.log_residuals_decades,
        color="#0f766e",
        linewidth=1.0,
    )
    residual.set_xlabel("qz (1/A)")
    residual.set_ylabel("Log residual")
    residual.grid(True, alpha=0.20)
    return _png(figure)


@_default_matplotlib_style
def sld_profile_png(context: DatasetExportData) -> bytes:
    """Render real and imaginary selected SLD profiles."""
    selected = _context(context).selected
    figure = _base_figure(4.2)
    axis = figure.subplots(1, 1)
    profile = np.asarray(selected.sld_profile_a2, dtype=complex)
    axis.plot(
        selected.sld_depth_a,
        profile.real,
        color="#1d4ed8",
        linewidth=1.4,
        label="real",
    )
    axis.plot(
        selected.sld_depth_a,
        profile.imag,
        color="#be123c",
        linewidth=1.1,
        label="imaginary",
    )
    axis.axhline(0.0, color="#6b7280", linewidth=0.8)
    axis.set_xlabel("Depth (A)")
    axis.set_ylabel("SLD (1/A^2)")
    axis.grid(True, alpha=0.20)
    axis.legend(loc="best", frameon=False)
    return _png(figure)


def _excluded_intervals(
    qz_a_inv: np.ndarray,
    fit_mask: tuple[bool, ...],
) -> tuple[tuple[float, float], ...]:
    excluded = np.flatnonzero(~np.asarray(fit_mask, dtype=bool))
    if excluded.size == 0:
        return ()
    groups = np.split(excluded, np.flatnonzero(np.diff(excluded) > 1) + 1)
    return tuple(
        (float(qz_a_inv[group[0]]), float(qz_a_inv[group[-1]]))
        for group in groups
    )


@_default_matplotlib_style
def residuals_png(context: DatasetExportData) -> bytes:
    """Render both residual definitions with disjoint exclusion spans."""
    value = _context(context)
    selected = value.selected
    figure = _base_figure(5.6)
    log_axis, weighted_axis = figure.subplots(2, 1, sharex=True)
    intervals = _excluded_intervals(selected.qz_a_inv, value.dataset.fit_mask)
    for axis in (log_axis, weighted_axis):
        for lower, upper in intervals:
            axis.axvspan(lower, upper, color="#d1d5db", alpha=0.45, linewidth=0.0)
        axis.axhline(0.0, color="#6b7280", linewidth=0.8)
        axis.grid(True, alpha=0.20)
    log_axis.plot(
        selected.qz_a_inv,
        selected.log_residuals_decades,
        color="#0f766e",
        linewidth=1.0,
    )
    weighted_axis.plot(
        selected.qz_a_inv,
        selected.weighted_residuals,
        color="#7e22ce",
        linewidth=1.0,
    )
    log_axis.set_ylabel("Log residual")
    weighted_axis.set_ylabel("Weighted residual")
    weighted_axis.set_xlabel("qz (1/A)")
    return _png(figure)


def _trend_contexts(contexts: object) -> tuple[DatasetExportData, ...]:
    values = _contexts(contexts)
    if len(values) < 2:
        raise ValueError("parameter trends require at least two dataset export values")
    return values


def _common_parameter_names(values: tuple[DatasetExportData, ...]) -> tuple[str, ...]:
    parameter_sets = tuple(
        {parameter.name for parameter in value.selected.parameters}
        for value in values
    )
    return tuple(sorted(set.intersection(*parameter_sets)))


def _parameter_sample(context: DatasetExportData, name: str) -> float:
    return next(
        parameter.value
        for parameter in context.selected.parameters
        if parameter.name == name
    )


def _plot_parameter_lines(
    axis: object,
    values: tuple[DatasetExportData, ...],
    names: tuple[str, ...],
    positions: np.ndarray,
) -> None:
    for index, name in enumerate(names):
        samples = [_parameter_sample(value, name) for value in values]
        axis.plot(
            positions,
            samples,
            marker="o",
            linewidth=1.0,
            label=name,
            color=f"C{index % 10}",
        )


@_default_matplotlib_style
def parameter_trends_png(contexts: object) -> bytes:
    """Render selected parameter values in project dataset order."""
    values = _trend_contexts(contexts)
    names = _common_parameter_names(values)
    figure = _base_figure(4.6)
    axis = figure.subplots(1, 1)
    positions = np.arange(len(values), dtype=float)
    _plot_parameter_lines(axis, values, names, positions)
    axis.set_xticks(
        positions,
        [value.dataset.dataset_id for value in values],
        rotation=30,
        ha="right",
        fontproperties=TREND_LABEL_FONT,
    )
    axis.set_ylabel("Selected value")
    axis.grid(True, alpha=0.20)
    if names:
        axis.legend(loc="best", frameon=False, fontsize=7)
    return _png(figure)
