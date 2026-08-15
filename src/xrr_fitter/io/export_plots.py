"""Deterministic headless PNG serialization for exported fit results."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from io import BytesIO
from typing import ParamSpec

import numpy as np
from matplotlib import font_manager, rc_context, rcParamsDefault
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

from xrr_fitter.io.export_tables import DatasetExportData, _contexts

PNG_SOFTWARE = "Matplotlib version3.11.1, https://matplotlib.org/"
P = ParamSpec("P")

# io cannot import the GUI font helper (the dependency contract forbids io->gui
# and there is no shared layer between them), so the export path resolves a CJK
# family independently. The list mirrors gui.plots.diagnostics.CJK_FONT_FAMILIES;
# without it every Chinese caption/title renders as tofu under DejaVu Sans.
CJK_FONT_FAMILIES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "Arial Unicode MS",
)


def _cjk_font_families() -> tuple[str, ...]:
    # Return the first installed CJK family ahead of DejaVu Sans so Latin axis
    # text still falls back cleanly; if none is installed, sans-serif keeps the
    # historical behaviour rather than forcing a missing family.
    for family in CJK_FONT_FAMILIES:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        return (family, "DejaVu Sans")
    return ("sans-serif",)


def _apply_cjk_font(figure: Figure) -> None:
    # Swap only the family on every text artist so each artist keeps its own size
    # (e.g. the fontsize=8 band caption); the explicit family overrides the
    # rcParamsDefault DejaVu font that the style isolation forces.
    families = list(_cjk_font_families())
    for artist in figure.findobj(match=Text):
        artist.set_fontfamily(families)


def _default_matplotlib_style(render: Callable[P, bytes]) -> Callable[P, bytes]:  # noqa: UP047
    @wraps(render)
    def isolated(*args: P.args, **kwargs: P.kwargs) -> bytes:
        with rc_context(rc=rcParamsDefault):
            return render(*args, **kwargs)

    return isolated


def _context(value: DatasetExportData) -> DatasetExportData:
    if not isinstance(value, DatasetExportData):
        raise TypeError("context must be DatasetExportData")
    return value


def _png(figure: Figure, *, cjk_text: bool = False) -> bytes:
    buffer = BytesIO()
    try:
        if cjk_text:
            _apply_cjk_font(figure)
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


# Published credible bands: (quantile pair, fill alpha, legend label). The inner
# 16-84% band is drawn more opaque than the outer 2.5-97.5% band so overlap reads
# as nested intervals. The order is fixed to keep PNG output byte-deterministic.
BAND_PAIRS = (
    ((0.16, 0.84), 0.28, "16–84%"),
    ((0.025, 0.975), 0.14, "2.5–97.5%"),
)


def _band_index(quantiles: tuple[float, ...], level: float) -> int | None:
    # Exact match only: quantile faces are stored verbatim, so an absent level
    # means the report never sampled it and the whole pair must be skipped.
    return next((i for i, value in enumerate(quantiles) if value == level), None)


def _draw_band_pair(
    axis: object,
    bands: object,
    pair: tuple[float, float],
    alpha: float,
    label: str,
) -> None:
    lower = _band_index(bands.quantiles, pair[0])
    upper = _band_index(bands.quantiles, pair[1])
    if lower is None or upper is None:
        return
    axis.fill_between(bands.depth_a, bands.real[lower], bands.real[upper], alpha=alpha, label=label)
    axis.fill_between(bands.depth_a, bands.imaginary[lower], bands.imaginary[upper], alpha=alpha)


def _draw_bands(axis: object, bands: object) -> None:
    for pair, alpha, label in BAND_PAIRS:
        _draw_band_pair(axis, bands, pair, alpha, label)


def _selected_bands(context: DatasetExportData) -> object | None:
    # Bands hang off the persisted result rather than the selected candidate:
    # MCMC replay attaches them to the dataset's last valid uncertainty report.
    result = context.dataset.last_valid_result
    report = None if result is None else result.uncertainty
    return None if report is None else report.sld_bands


@_default_matplotlib_style
def sld_profile_png(context: DatasetExportData) -> bytes:
    """Render real and imaginary selected SLD profiles with credible bands."""
    value = _context(context)
    selected = value.selected
    figure = Figure(figsize=(6.4, 4.0), layout="constrained")
    axis = figure.subplots()
    profile = np.asarray(selected.sld_profile_a2, dtype=complex)
    axis.plot(selected.sld_depth_a, profile.real, label="real")
    axis.plot(selected.sld_depth_a, profile.imag, label="imaginary")
    bands = _selected_bands(value)
    if bands is not None:
        _draw_bands(axis, bands)
        axis.set_title(bands.caption(), fontsize=8, loc="left")
    axis.set_xlabel("Depth (Angstrom)")
    axis.set_ylabel("SLD (1/Angstrom^2)")
    axis.legend()
    return _png(figure, cjk_text=bands is not None)


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
    parameter_sets = tuple({parameter.name for parameter in value.selected.parameters} for value in values)
    return tuple(sorted(set.intersection(*parameter_sets)))


def _parameter_sample(context: DatasetExportData, name: str) -> float:
    return next(parameter.value for parameter in context.selected.parameters if parameter.name == name)


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
