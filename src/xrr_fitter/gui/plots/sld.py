"""Candidate-owned SLD and uncertainty diagnostic rendering."""

from __future__ import annotations

import numpy as np

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.diagnostics import (
    DiagnosticView,
    apply_figure_palette,
    candidate_is_inspection_only,
    draw_empty,
    finish_view,
)


def draw_sld(
    view: DiagnosticView,
    candidate: object | None,
    others: tuple[object, ...] = (),
    bands: object | None = None,
) -> None:
    """Draw the selected candidate's SLD with comparison overlays from others.

    The reflectivity comparison tab already overlays all candidates so users can
    judge which model best matches the data. This extends that comparison to the
    SLD depth profiles: the selected candidate's real and imaginary parts stay
    prominent at full opacity, while other valid candidates' real parts appear as
    faint overlays behind them. This lets users compare structural interpretations
    side-by-side without switching between rows.
    """
    if candidate is None:
        draw_empty(view, "SLD 深度剖面", "暂无当前候选")
        return
    depth_nm = np.asarray(candidate.sld_depth_a, dtype=float) / 10.0
    profile = np.asarray(candidate.sld_profile_a2, dtype=complex)
    axes = view.axes
    axes.clear()
    # Draw comparison overlays first so the selected candidate's curves sit on top
    selected_id = candidate.candidate_id
    overlays: list[object] = []
    for other in others:
        if other.candidate_id == selected_id or candidate_is_inspection_only(other):
            continue
        other_depth = np.asarray(other.sld_depth_a, dtype=float) / 10.0
        other_profile = np.asarray(other.sld_profile_a2, dtype=complex)
        overlays.extend(
            axes.plot(
                other_depth,
                other_profile.real,
                linewidth=0.8,
                alpha=0.35,
                label=f"{other.candidate_id} 实部",
            )
        )
    # Draw the selected candidate's real/imag at full opacity on top
    axes.plot(depth_nm, profile.real, label="SLD 实部")
    axes.plot(depth_nm, profile.imag, "--", label="SLD 虚部")
    axes.set(title="SLD 深度剖面", xlabel="深度 (nm)", ylabel="SLD (Å⁻²)")
    if bands is not None:
        _draw_bands(axes, bands)
        axes.set_title(f"SLD 深度剖面 — {bands.caption()}", fontsize=theme.FONT_PT_SM)
    _limit_depth_axis(axes, depth_nm)
    _draw_legend(axes, overlays)
    finish_view(view)


def _limit_depth_axis(axes: object, depth_nm: np.ndarray) -> None:
    """Scale the depth axis to the selected candidate, not to the overlays.

    A run that keeps a badly fitted candidate can carry a stack hundreds of nm
    thick.  Letting the shared autoscale see that overlay stretched the axis to
    its depth, and the selected profile - the one the user is reading - collapsed
    into a sliver at the left edge.  The overlays stay drawn for comparison; they
    just no longer decide the scale, so the part of them that falls outside the
    selected range is simply out of frame.
    """
    finite = depth_nm[np.isfinite(depth_nm)]
    if finite.size == 0:
        return
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    if span <= 0.0:
        return
    margin = 0.05 * span
    axes.set_xlim(lower - margin, upper + margin)


def _draw_legend(axes: object, overlays: list[object]) -> None:
    """Key the selected profile and bands, folding comparison curves into one row.

    A run that keeps several candidates gave the key one row per comparison
    curve, and the key then took most of the pane's width and squeezed the
    profiles into a strip.  Every curve keeps its own label so selection and
    inspection still name it; only the key collapses them into a counted entry.
    """
    handles, labels = axes.get_legend_handles_labels()
    entries = [(handle, label) for handle, label in zip(handles, labels, strict=True) if handle not in overlays]
    if overlays:
        entries.append((overlays[0], f"其他候选 实部 ×{len(overlays)}"))
    axes.legend(
        [handle for handle, _ in entries],
        [label for _, label in entries],
        fontsize=theme.FONT_PT_SM,
        loc="best",
        framealpha=0.75,
    )


# Credible bands mirror the exported PNG: (quantile pair, fill alpha, legend
# label). The inner 16-84% band is more opaque than the outer 2.5-97.5% band, so
# nesting reads correctly. Depth is converted Å→nm to match the profile curves.
BAND_PAIRS = (
    ((0.16, 0.84), 0.28, "16–84%"),
    ((0.025, 0.975), 0.14, "2.5–97.5%"),
)


def _band_index(quantiles: tuple[float, ...], level: float) -> int | None:
    return next((i for i, value in enumerate(quantiles) if value == level), None)


def _draw_band_pair(
    axes: object,
    bands: object,
    pair: tuple[float, float],
    alpha: float,
    label: str,
) -> None:
    lower = _band_index(bands.quantiles, pair[0])
    upper = _band_index(bands.quantiles, pair[1])
    if lower is None or upper is None:
        return
    depth_nm = np.asarray(bands.depth_a, dtype=float) / 10.0
    axes.fill_between(depth_nm, bands.real[lower], bands.real[upper], alpha=alpha, label=label)
    axes.fill_between(depth_nm, bands.imaginary[lower], bands.imaginary[upper], alpha=alpha)


def _draw_bands(axes: object, bands: object) -> None:
    for pair, alpha, label in BAND_PAIRS:
        _draw_band_pair(axes, bands, pair, alpha, label)


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
    # Both halves get their ticks blanked and their placeholder drawn in the
    # muted colour.  Leaving the default 0-1 ticks in place made an empty pane
    # read as a real plot whose curve had failed to draw.
    correlation, profile = _reset_uncertainty_axes(view)
    for axes in (correlation, profile):
        axes.set_xticks(())
        axes.set_yticks(())
    correlation.set_title("相关矩阵")
    profile.set_title("profile likelihood 与区间")
    palette = apply_figure_palette(view.figure)
    for axes, text in ((correlation, message), (profile, "区间证据不可用")):
        axes.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=axes.transAxes,
            color=palette.muted,
        )
    finish_view(view)


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
        axes.legend(fontsize=theme.FONT_PT_SM)
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
    # Without a colour key the matrix is a field of colours with no stated
    # scale, so a cell only reads as "reddish" rather than as a correlation near
    # +1.  The key is rebuilt each draw because the reset above removes it along
    # with every other companion axes.
    view.figure.colorbar(image, ax=correlation)
    positions = tuple(range(len(report.correlation_names)))
    correlation.set_xticks(positions, report.correlation_names, rotation=45, ha="right")
    correlation.set_yticks(positions, report.correlation_names)
    correlation.set_title("相关矩阵（固定范围 [-1, 1]）")
    _draw_profiles(profile, report)
    finish_view(view)
