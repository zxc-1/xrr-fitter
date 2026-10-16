"""Candidate-by-column heatmap rendering for the diagnostic tabs.

Both views answer a question the line plots cannot. The residual and candidate
comparison tabs overlay every candidate on one pair of axes, so ten candidates
become ten curves that hide each other. Laying the same evidence out as one row
per candidate keeps every candidate readable at once, and makes a q range that
fits badly across all of them read as a vertical stripe rather than as ten
separate wobbles.
"""

from __future__ import annotations

import numpy as np

from xrr_fitter.gui.noise import residual_label
from xrr_fitter.gui.plots.diagnostics import (
    DiagnosticView,
    apply_figure_palette,
    candidate_is_inspection_only,
    diverging_colormap,
    draw_empty,
    finish_view,
    listed_candidates,
)
from xrr_fitter.gui.plots.parameter_labels import result_definitions, short_labels
from xrr_fitter.gui.results.candidate_row import candidate_title

RESIDUAL_TITLE = "残差热图"
PARAMETER_TITLE = "参数热图"

# Six labelled columns read without crowding; more overlap once each label
# carries three significant figures of qz.
QZ_TICK_COUNT = 6


def reset_single_axes(view: DiagnosticView) -> object:
    """Drop the previous draw's colour key and hand back the cleared axes.

    A colorbar arrives as an extra axes on the figure and is rebuilt on every
    draw, so without removing the old one each redraw would surrender another
    strip of width to a stack of stale colour keys. The view's own axes is kept
    rather than replaced: it is the one the navigation toolbar and the tab's
    accessibility wiring already point at.
    """
    for axes in tuple(view.figure.axes):
        if axes is not view.axes:
            axes.remove()
    view.axes.clear()
    return view.axes


def _comparable(result: object | None) -> tuple[object, ...]:
    """Candidates that deserve a row, best objective first."""
    candidates = () if result is None else tuple(result.candidates)
    kept = [candidate for candidate in candidates if not candidate_is_inspection_only(candidate)]
    return tuple(sorted(kept, key=lambda candidate: float(candidate.objective)))


def _unavailable(view: DiagnosticView, title: str, message: str) -> None:
    # The reset runs first: draw_empty blanks the axes it is given but knows
    # nothing of a colour key left behind by an earlier successful draw.
    reset_single_axes(view)
    draw_empty(view, title, message)


def _row_labels(result: object, candidates: tuple[object, ...], candidate_id: str | None) -> tuple[str, ...]:
    # The candidate on screen elsewhere is marked, so a row can be tied back to
    # the curves and the SLD profile the other panes are showing.  行名用候选在结果
    # 列表里的序号（候选 A / B …），和列表、候选比较图同一套叫法；行序仍按目标值。
    ordinals = {candidate.candidate_id: ordinal for ordinal, candidate in enumerate(listed_candidates(result))}
    labels = []
    for candidate in candidates:
        title = candidate_title(ordinals[candidate.candidate_id])
        labels.append(f"▶ {title}" if candidate.candidate_id == candidate_id else title)
    return tuple(labels)


def _label_rows(axes: object, labels: tuple[str, ...]) -> None:
    axes.set_yticks(tuple(range(len(labels))), labels)


def _symmetric_limit(matrix: np.ndarray) -> float:
    """A diverging scale centred on zero, so residual sign reads as colour.

    An over- and an under-shoot of the same size have to read as equally far
    from the centre, which an autoscaled range straddling zero unevenly would
    not deliver.
    """
    finite = matrix[np.isfinite(matrix)]
    limit = float(np.max(np.abs(finite))) if finite.size else 0.0
    return limit if limit > 0.0 else 1.0


def _label_columns_with_qz(axes: object, candidate: object, columns: int) -> str:
    """Name the qz each labelled column carries, and report the axis label.

    The columns are point indices rather than a linear qz axis: qz is
    4π·sin(θ)/λ, which is not uniform in the stored angle, so stretching the
    image across a qz extent would place most cells at a q they do not hold.
    Each tick names the qz of the exact index it sits on instead.
    """
    qz = np.asarray(candidate.qz_a_inv, dtype=float)
    if qz.size != columns:
        return "数据点序号"
    positions = np.unique(np.linspace(0, columns - 1, num=min(QZ_TICK_COUNT, columns), dtype=int))
    axes.set_xticks(positions, tuple(f"{qz[index]:.3g}" for index in positions))
    return "qz (Å⁻¹)"


def draw_residual_heatmap(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    """Draw one row of weighted residuals per candidate, best fit at the top."""
    candidates = _comparable(result)
    if not candidates:
        _unavailable(view, RESIDUAL_TITLE, "暂无可比较候选")
        return
    # Every candidate is validated against the prepared point count before it
    # reaches a draw, so the rows are known to be equal in length here.
    matrix = np.vstack([np.asarray(candidate.weighted_residuals, dtype=float) for candidate in candidates])
    limit = _symmetric_limit(matrix)
    axes = reset_single_axes(view)
    image = axes.imshow(matrix, aspect="auto", cmap=diverging_colormap(), vmin=-limit, vmax=limit)
    view.figure.colorbar(image, ax=axes, label=residual_label(candidates[0]))
    _label_rows(axes, _row_labels(result, candidates, candidate_id))
    xlabel = _label_columns_with_qz(axes, candidates[0], matrix.shape[1])
    axes.set(title=RESIDUAL_TITLE, xlabel=xlabel, ylabel="候选")
    _caption(view, f"固定范围 ±{limit:.3g}")
    finish_view(view)


def _caption(view: DiagnosticView, text: str) -> None:
    """标题右上那行小字，与标题同色。

    颜色从调色板取，不从标题身上抄：``clear()`` 之后、``finish_view`` 套调色板之前，标题还是
    matplotlib 的默认黑，深色画布上这行字就跟着成了黑字。
    """
    view.axes.text(
        1.0,
        1.02,
        text,
        transform=view.axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=apply_figure_palette(view.figure).foreground,
        alpha=0.7,
    )


def _normalized_parameters(parameters: tuple[object, ...]) -> np.ndarray:
    """Place each parameter on [0, 1] within its own bounds.

    A thickness in ångström and a unitless scale factor share no range, so raw
    values on one colour scale would leave every column but the widest-ranged
    one reading as a flat zero. A parameter pinned to a single value holds no
    position within its bounds, so it is drawn mid-scale rather than at an end
    it does not occupy.
    """
    values = []
    for parameter in parameters:
        span = float(parameter.upper) - float(parameter.lower)
        offset = float(parameter.value) - float(parameter.lower)
        values.append(0.5 if span <= 0.0 else offset / span)
    return np.asarray(values, dtype=float)


def draw_parameter_heatmap(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    """Draw one row of normalized parameter values per candidate.

    Candidates that agree on the data can still disagree on the structure that
    produced it, and that shows up here as a column where the rows differ while
    the residual heatmap stays uniform.
    """
    candidates = _comparable(result)
    if not candidates:
        _unavailable(view, PARAMETER_TITLE, "暂无可比较候选")
        return
    names = tuple(parameter.name for parameter in candidates[0].parameters)
    columns_differ = any(
        tuple(parameter.name for parameter in candidate.parameters) != names for candidate in candidates
    )
    if not names or columns_differ:
        _unavailable(view, PARAMETER_TITLE, "候选参数集合不一致，无法逐列并排比较")
        return
    matrix = np.vstack([_normalized_parameters(tuple(candidate.parameters)) for candidate in candidates])
    axes = reset_single_axes(view)
    image = axes.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    view.figure.colorbar(image, ax=axes)
    _label_rows(axes, _row_labels(result, candidates, candidate_id))
    # 列名与相关矩阵的刻度同一套短名（``d·ox``）：两张图是拿来互相对照的，一边写机器路径就多一次翻译。
    labels = short_labels(names, result_definitions(result))
    axes.set_xticks(tuple(range(len(names))), labels, rotation=45, ha="right")
    axes.set(
        title=PARAMETER_TITLE,
        xlabel="参数",
        ylabel="候选",
    )
    _caption(view, "各参数独立归一化至 [0, 1]")
    finish_view(view)
