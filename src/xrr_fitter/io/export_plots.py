"""Deterministic headless PNG serialization for exported fit results."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from io import BytesIO
from typing import ParamSpec

from matplotlib import rc_context, rcParamsDefault
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from xrr_fitter.io.export_tables import DatasetExportData, _contexts


PNG_SOFTWARE = "Matplotlib version3.11.0, https://matplotlib.org/"
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
        canvas = FigureCanvasAgg(figure)
        canvas.print_png(buffer, metadata={"Software": PNG_SOFTWARE})
        return buffer.getvalue()
    finally:
        figure.clear()


@_default_matplotlib_style
def fit_overview_png(context: DatasetExportData) -> bytes:
    """Render normalized curves and distinguish excluded source rows."""
    value = _context(context)
    data = value.data
    selected = value.selected
    figure = Figure(figsize=(6.4, 4.0), layout="constrained")
    axis = figure.subplots()
    axis.set_yscale("log")
    included = np.asarray(value.dataset.fit_mask, dtype=bool)
    axis.plot(data.qz_a_inv, data.intensity_normalized, label="observed")
    axis.plot(selected.qz_a_inv, selected.model_normalized, label="model")
    excluded = ~included
    if np.any(excluded):
        axis.scatter(
            data.qz_a_inv[excluded],
            data.intensity_normalized[excluded],
            label="excluded",
            marker="x",
        )
    axis.set_xlabel("qz (1/Angstrom)")
    axis.set_ylabel("Normalized intensity")
    axis.legend()
    return _png(figure)


@_default_matplotlib_style
def sld_profile_png(context: DatasetExportData) -> bytes:
    """Render real and imaginary selected SLD profiles."""
    selected = _context(context).selected
    figure = Figure(figsize=(6.4, 4.0), layout="constrained")
    axis = figure.subplots()
    profile = np.asarray(selected.sld_profile_a2, dtype=complex)
    axis.plot(selected.sld_depth_a, profile.real, label="real")
    axis.plot(selected.sld_depth_a, profile.imag, label="imaginary")
    axis.set_xlabel("Depth (Angstrom)")
    axis.set_ylabel("SLD (1/Angstrom^2)")
    axis.legend()
    return _png(figure)


def _excluded_intervals(
    qz_a_inv: np.ndarray,
    fit_mask: tuple[bool, ...],
) -> tuple[tuple[float, float], ...]:
    excluded = np.flatnonzero(~np.asarray(fit_mask, dtype=bool))
    if excluded.size == 0:
        return ()
    groups = np.split(excluded, np.flatnonzero(np.diff(excluded) > 1) + 1)
    intervals = []
    for group in groups:
        values = qz_a_inv[group]
        finite = values[np.isfinite(values)]
        if finite.size:
            intervals.append((float(np.min(finite)), float(np.max(finite))))
    return tuple(intervals)


@_default_matplotlib_style
def residuals_png(context: DatasetExportData) -> bytes:
    """Render both residual definitions with disjoint exclusion spans."""
    value = _context(context)
    selected = value.selected
    figure = Figure(figsize=(6.4, 4.0), layout="constrained")
    log_axis, weighted_axis = figure.subplots(2, 1, squeeze=False).ravel()
    intervals = _excluded_intervals(selected.qz_a_inv, value.dataset.fit_mask)
    log_axis.plot(selected.qz_a_inv, selected.log_residuals_decades)
    weighted_axis.plot(selected.qz_a_inv, selected.weighted_residuals)
    log_axis.set_ylabel("Log residual (decades)")
    weighted_axis.set_ylabel("Weighted residual")
    for axis in (log_axis, weighted_axis):
        axis.set_xlabel("qz (1/Angstrom)")
        for lower, upper in intervals:
            axis.axvspan(lower, upper, alpha=0.08)
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
    for name in names:
        samples = [_parameter_sample(value, name) for value in values]
        axis.plot(
            positions,
            samples,
            marker="o",
            label=name,
        )


@_default_matplotlib_style
def parameter_trends_png(contexts: object) -> bytes:
    """Render selected parameter values in project dataset order."""
    values = _trend_contexts(contexts)
    names = _common_parameter_names(values)
    figure = Figure(figsize=(7.2, 4.2), layout="constrained")
    axis = figure.subplots()
    positions = np.arange(len(values), dtype=int)
    _plot_parameter_lines(axis, values, names, positions)
    axis.set_xticks(
        positions,
        tuple(str(index + 1) for index in positions),
        rotation=20,
    )
    axis.set_xlabel("Dataset order")
    axis.set_ylabel("Selected value")
    if names:
        axis.legend(fontsize="small")
    return _png(figure)
