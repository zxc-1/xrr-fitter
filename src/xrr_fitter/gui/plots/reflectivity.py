"""Prepared-data and candidate reflectivity diagnostic rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.ticker import LogFormatter

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.diagnostics import (
    DiagnosticView,
    current_plot_palette,
    draw_empty,
    finish_view,
)


@dataclass(frozen=True, slots=True)
class PreparedProjectPlots:
    datasets: dict[str, api.PreparedData]
    masks: dict[str, np.ndarray]
    dataset_id: str | None
    data: api.PreparedData | None
    mask: np.ndarray | None
    result: object | None
    candidate_id: str | None


def validate_plot_data(data: api.PreparedData, mask: object) -> np.ndarray:
    arrays = (
        np.asarray(data.two_theta_deg),
        np.asarray(data.intensity_raw),
        np.asarray(data.intensity_normalized),
    )
    sizes = {array.size for array in arrays}
    converted = np.asarray(mask, dtype=bool)
    if len(sizes) != 1 or converted.ndim != 1 or converted.size != arrays[0].size:
        raise ValueError("plot data arrays and mask must have equal lengths")
    return converted.copy()


def validate_candidate(data: api.PreparedData, candidate: object) -> None:
    size = data.two_theta_deg.size
    arrays = (
        candidate.qz_a_inv,
        candidate.model_normalized,
        candidate.log_residuals_decades,
        candidate.weighted_residuals,
    )
    if any(np.asarray(values).size != size for values in arrays):
        raise ValueError("candidate arrays must match prepared point count")
    if np.asarray(candidate.sld_depth_a).size != np.asarray(candidate.sld_profile_a2).size:
        raise ValueError("candidate SLD arrays must have equal lengths")


def validate_result(data: api.PreparedData, result: object) -> None:
    for candidate in result.candidates:
        validate_candidate(data, candidate)


def _source_path(project: api.XrrProject, dataset: api.DatasetProject) -> Path:
    source = Path(dataset.source_path)
    if source.is_absolute() or project.base_directory is None:
        return source
    return Path(project.base_directory) / source


def _prepared_dataset(
    project: api.XrrProject,
    dataset: api.DatasetProject,
) -> tuple[api.PreparedData, np.ndarray] | None:
    try:
        data = api.import_data(
            _source_path(project, dataset),
            dataset.beam,
            dataset.import_angle_offset_deg,
            dataset.column_mapping,
        )
    except (OSError, ValueError):
        return None
    return data, validate_plot_data(data, dataset.fit_mask)


def _prepared_datasets(
    project: api.XrrProject,
) -> tuple[dict[str, api.PreparedData], dict[str, np.ndarray]]:
    datasets: dict[str, api.PreparedData] = {}
    masks: dict[str, np.ndarray] = {}
    for dataset in project.datasets:
        prepared = _prepared_dataset(project, dataset)
        if prepared is not None:
            datasets[dataset.dataset_id], masks[dataset.dataset_id] = prepared
    return datasets, masks


def _project_dataset(
    project: api.XrrProject,
    dataset_id: str,
) -> api.DatasetProject:
    matches = tuple(dataset for dataset in project.datasets if dataset.dataset_id == dataset_id)
    if len(matches) != 1:
        raise KeyError(f"unknown project dataset: {dataset_id}")
    return matches[0]


def _project_candidate_id(
    project: api.XrrProject,
    dataset: api.DatasetProject | None,
) -> str | None:
    if dataset is None or dataset.last_valid_result is None:
        return None
    selected = dict(project.ui_state.selected_candidate_ids).get(dataset.dataset_id)
    best = dataset.last_valid_result.best_candidate
    return selected if selected is not None else (None if best is None else best.candidate_id)


def prepare_project_plots(project: api.XrrProject) -> PreparedProjectPlots:
    if not isinstance(project, api.XrrProject):
        raise TypeError("project must be an XrrProject")
    datasets, masks = _prepared_datasets(project)
    dataset_id = project.ui_state.active_dataset_id
    data = None if dataset_id is None else datasets.get(dataset_id)
    mask = None if dataset_id is None else masks.get(dataset_id)
    active = None if dataset_id is None else _project_dataset(project, dataset_id)
    result = None if active is None or data is None else active.last_valid_result
    candidate_id = None if data is None else _project_candidate_id(project, active)
    if data is not None and result is not None:
        validate_result(data, result)
    return PreparedProjectPlots(
        datasets,
        masks,
        dataset_id,
        data,
        mask,
        result,
        candidate_id,
    )


def _quality_caption(axes: object, candidate: object | None) -> None:
    """Annotate how well the drawn candidate agrees with the data.

    Two overlaid curves on a log axis look close whatever their disagreement, so
    the axes state the objective the search minimised together with the mean
    absolute log-decade miss, which reads as "off by this many decades on
    average" and is the same quantity the exported residual plot labels.
    """
    if candidate is None:
        return
    residuals = np.asarray(candidate.log_residuals_decades, dtype=float)
    finite = residuals[np.isfinite(residuals)]
    parts = [f"J={candidate.objective:.4g}"]
    if finite.size:
        parts.append(f"平均残差 {np.mean(np.abs(finite)):.3g} decade")
    axes.text(
        0.99,
        0.02,
        " · ".join(parts),
        ha="right",
        va="bottom",
        transform=axes.transAxes,
        color=current_plot_palette().muted,
        fontsize=theme.FONT_PT_SM,
    )


def draw_raw(
    view: DiagnosticView,
    data: api.PreparedData,
    mask: np.ndarray,
    candidate: object | None,
) -> None:
    axes = view.axes
    axes.clear()
    angles = np.asarray(data.two_theta_deg, dtype=float)
    raw = np.asarray(data.intensity_raw, dtype=float)
    finite = np.isfinite(angles) & np.isfinite(raw)
    included = finite & mask
    excluded = finite & ~mask
    axes.plot(
        angles[included],
        raw[included],
        linestyle="none",
        marker="o",
        color=theme.DATA_OBSERVED,
        markerfacecolor=theme.DATA_OBSERVED,
        label="拟合点",
    )
    axes.plot(
        angles[excluded],
        raw[excluded],
        linestyle="none",
        marker="x",
        # Structural rather than a data series: excluded points read as "struck
        # out", so they follow the foreground and would vanish on a dark canvas
        # if left at a fixed near-black.
        color=current_plot_palette().foreground,
        markerfacecolor="none",
        label="排除点",
    )
    if candidate is not None:
        axes.plot(
            angles,
            np.asarray(candidate.model_normalized) * data.normalization,
            "--",
            color=theme.DATA_CANDIDATE,
            label="当前候选模型",
        )
    axes.set(title="原始数据与模型", xlabel="2θ (deg)", ylabel="原始强度")
    axes.legend()
    _quality_caption(axes, candidate)
    finish_view(view)


def draw_log(
    view: DiagnosticView,
    data: api.PreparedData,
    candidate: object | None,
) -> None:
    axes = view.axes
    axes.clear()
    angles = np.asarray(data.two_theta_deg, dtype=float)
    observed = np.maximum(np.asarray(data.intensity_normalized, dtype=float), data.r_floor)
    axes.plot(angles, observed, "o", color=theme.DATA_OBSERVED, label="归一化数据")
    if candidate is not None:
        model = np.maximum(np.asarray(candidate.model_normalized, dtype=float), data.r_floor)
        axes.plot(angles, model, "--", color=theme.DATA_CANDIDATE, label="当前候选模型")
    axes.set(
        title="对数反射率",
        xlabel="2θ (deg)",
        ylabel=f"归一化 R (display floor={data.r_floor:g})",
        yscale="log",
    )
    axes.yaxis.set_major_formatter(LogFormatter())
    axes.legend()
    _quality_caption(axes, candidate)
    finish_view(view)


def preview_display_values(
    data: api.PreparedData,
    qz_a_inv: np.ndarray,
    model_normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a live preview curve onto the angle axis used by the log view.

    A preview is published on the searching context's q grid, which is a subset
    of the prepared q grid, so interpolating against the prepared pairs recovers
    the display angle exactly at those points. The display floor matches the
    published curve so a preview and a committed candidate stay comparable.
    """
    qz = np.asarray(qz_a_inv, dtype=float)
    model = np.asarray(model_normalized, dtype=float)
    if qz.ndim != 1 or model.ndim != 1 or qz.size != model.size or qz.size == 0:
        raise ValueError("preview axes must be nonempty one-dimensional pairs")
    reference_qz = np.asarray(data.qz_a_inv, dtype=float)
    reference_angle = np.asarray(data.two_theta_deg, dtype=float)
    order = np.argsort(reference_qz, kind="stable")
    angles = np.interp(qz, reference_qz[order], reference_angle[order])
    return angles, np.maximum(model, data.r_floor)


def _qz4_display_values(
    qz: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return finite qz⁴ diagnostics, scaling only when float64 cannot hold them.

    The unscaled transform remains the default so ordinary plots retain their
    historical physical units.  An extreme candidate can make ``qz**4``
    overflow even though the curve shape is meaningful; in that case the
    q-axis is normalized by its largest finite magnitude and the label records
    the reference explicitly.
    """
    qz = np.asarray(qz, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(qz) & np.isfinite(values)
    if not np.any(finite):
        return qz[finite], values[finite], "qz⁴R（无有限数据）"
    qz_finite = qz[finite]
    values_finite = values[finite]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        unscaled = qz_finite**4 * values_finite
    if np.all(np.isfinite(unscaled)):
        return qz_finite, unscaled, "qz⁴R"

    q_reference = float(np.max(np.abs(qz_finite)))
    if q_reference == 0.0:
        q_reference = 1.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        normalized_qz4 = (qz_finite / q_reference) ** 4
        scaled = normalized_qz4 * values_finite
    if not np.all(np.isfinite(scaled)):
        value_reference = float(np.max(np.abs(values_finite)))
        if value_reference == 0.0 or not np.isfinite(value_reference):
            value_reference = 1.0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scaled = normalized_qz4 * (values_finite / value_reference)
        label = f"归一化 (qz/{q_reference:g} Å⁻¹)⁴(R/{value_reference:g})"
    else:
        label = f"归一化 (qz/{q_reference:g} Å⁻¹)⁴R"
    return qz_finite, scaled, label


def draw_qz4(
    view: DiagnosticView,
    data: api.PreparedData,
    candidate: object | None,
) -> None:
    if candidate is None:
        draw_empty(view, "qz⁴R", "暂无当前候选")
        return
    qz = np.asarray(candidate.qz_a_inv, dtype=float)
    data_qz, data_transform, ylabel = _qz4_display_values(
        qz,
        np.asarray(data.intensity_normalized, dtype=float),
    )
    model_qz, model_transform, model_label = _qz4_display_values(
        qz,
        np.asarray(candidate.model_normalized, dtype=float),
    )
    axes = view.axes
    axes.clear()
    axes.plot(data_qz, data_transform, "o", label="归一化数据")
    axes.plot(model_qz, model_transform, "--", label="当前候选模型")
    if ylabel == model_label:
        axis_label = ylabel
    else:
        axis_label = f"数据: {ylabel}; 模型: {model_label}"
    axes.set(title="qz⁴R 诊断变换（非拟合数据）", xlabel="qz (Å⁻¹)", ylabel=axis_label)
    axes.legend()
    finish_view(view)


def draw_residual(view: DiagnosticView, candidate: object | None) -> None:
    if candidate is None:
        draw_empty(view, "加权残差", "暂无当前候选")
        return
    qz = np.asarray(candidate.qz_a_inv, dtype=float)
    axes = view.axes
    axes.clear()
    axes.plot(qz, candidate.weighted_residuals, "-o", label="加权残差")
    axes.plot(qz, np.zeros_like(qz), ":", label="零参考线")
    axes.set(title="加权残差", xlabel="qz (Å⁻¹)", ylabel="加权残差")
    axes.legend()
    finish_view(view)
