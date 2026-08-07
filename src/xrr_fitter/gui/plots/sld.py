"""Candidate-owned SLD and uncertainty diagnostic rendering."""

from __future__ import annotations

import numpy as np

from xrr_fitter.gui.plots.diagnostics import (
    DiagnosticView,
    apply_figure_font,
    draw_empty,
)


def _finish(view: DiagnosticView) -> None:
    apply_figure_font(view.figure)
    view.canvas.draw_idle()


def draw_sld(view: DiagnosticView, candidate: object | None) -> None:
    if candidate is None:
        draw_empty(view, "SLD 深度剖面", "暂无当前候选")
        return
    depth_nm = np.asarray(candidate.sld_depth_a, dtype=float) / 10.0
    profile = np.asarray(candidate.sld_profile_a2, dtype=complex)
    axes = view.axes
    axes.clear()
    axes.plot(depth_nm, profile.real, label="SLD 实部")
    axes.plot(depth_nm, profile.imag, "--", label="SLD 虚部")
    axes.set(title="SLD 深度剖面", xlabel="深度 (nm)", ylabel="SLD (Å⁻²)")
    axes.legend()
    _finish(view)


def _reset_uncertainty_axes(view: DiagnosticView) -> tuple[object, object]:
    primary = tuple(view.figure.axes[:2])
    if len(primary) != 2:
        view.figure.clear()
        primary = tuple(view.figure.subplots(1, 2))
        view.axes = primary[0]
    for axes in tuple(view.figure.axes[2:]):
        axes.remove()
    for axes in primary:
        axes.clear()
    return primary


def _unavailable(
    view: DiagnosticView,
    message: str,
) -> None:
    correlation, profile = _reset_uncertainty_axes(view)
    correlation.set_title("相关矩阵")
    correlation.text(0.5, 0.5, message, ha="center", va="center", transform=correlation.transAxes)
    profile.set_title("profile likelihood 与区间")
    profile.text(0.5, 0.5, "区间证据不可用", ha="center", va="center", transform=profile.transAxes)
    _finish(view)


def _draw_profiles(axes: object, report: object) -> None:
    profiles = tuple(report.profiles)
    for profile in profiles:
        values = np.asarray(profile.values, dtype=float)
        span = float(np.ptp(values))
        normalized = np.zeros_like(values) if span == 0.0 else (values - np.min(values)) / span
        axes.plot(normalized, profile.objectives, "-o", label=profile.name)
    for name, lower, upper in report.bootstrap_intervals:
        axes.text(
            0.02,
            0.98 - 0.08 * len(axes.texts),
            f"{name}: [{lower:g}, {upper:g}]",
            ha="left",
            va="top",
            transform=axes.transAxes,
        )
    if profiles:
        axes.legend(fontsize="small")
    elif not report.bootstrap_intervals:
        axes.text(0.5, 0.5, "profile 与区间证据不可用", ha="center", va="center", transform=axes.transAxes)
    axes.set(
        title="profile likelihood 与区间",
        xlabel="参数坐标（各参数独立归一化至 [0, 1]）",
        ylabel="目标函数",
    )


def draw_uncertainty(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    report = None if result is None else result.uncertainty
    if report is None:
        _unavailable(view, "不确定性报告不可用")
        return
    owner = report.candidate_id
    if candidate_id is None or owner != candidate_id:
        _unavailable(
            view,
            f"不确定性证据属于 {owner or '未标识候选'}，当前查看 {candidate_id or '未选择候选'}",
        )
        return
    correlation, profile = _reset_uncertainty_axes(view)
    matrix = np.asarray(report.correlation_matrix, dtype=float)
    image = correlation.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    image.set_clim(-1.0, 1.0)
    positions = tuple(range(len(report.correlation_names)))
    correlation.set_xticks(positions, report.correlation_names, rotation=45, ha="right")
    correlation.set_yticks(positions, report.correlation_names)
    correlation.set_title("相关矩阵（固定范围 [-1, 1]）")
    _draw_profiles(profile, report)
    _finish(view)
