"""Stable Matplotlib diagnostic views and non-scientific summaries.

Figures are constructed directly, without pyplot or global rcParams changes.
Each Qt canvas owns its queued redraw timer and every text artist receives a
local CJK font before drawing. Candidate comparison and batch-trend rendering
live here because they combine result evidence rather than one scientific
curve family.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import LogFormatter
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QTabWidget


# The switchable diagnostic tabs, in display order. The log view leads because
# reflectivity spans several decades: on a linear axis everything below the
# critical angle collapses onto the baseline, so the raw view answers far fewer
# questions than its former first position implied.
TAB_SPECS = (
    ("log", "对数反射率", "查看仅应用显示下限的归一化反射率"),
    ("raw", "原始数据与模型", "查看存储角度、原始强度、拟合点和排除点"),
    ("qz4", "qz⁴R", "查看当前候选 qz 网格上的 qz 四次方诊断"),
    ("residual", "加权残差", "查看当前候选发布的全长加权残差"),
    ("candidates", "候选解比较", "比较全部保留候选及其审计状态"),
    ("uncertainty", "相关性与区间", "查看当前候选拥有的相关和区间证据"),
    ("trend", "批量趋势", "查看多个数据集的厚度和周期趋势"),
)

# The SLD depth profile is not a tab. Judging a fit means reading curve
# agreement and the resulting structure together, so it occupies a companion
# pane that stays on screen whichever diagnostic tab is selected.
COMPANION_SPEC = ("sld", "SLD 深度剖面", "查看当前候选的实部和虚部 SLD")

VIEW_SPECS = (*TAB_SPECS, COMPANION_SPEC)

DIAGNOSTIC_LABELS = {
    "gauss_hermite_unconverged": "Gauss-Hermite 积分未收敛",
    "ideal_reflectivity_above_one": "理想反射率超过 1",
    "nevot_croce_applicability_exceeded": "Nevot-Croce 适用范围超限",
    "suspected_diffuse_background": "疑似漫散射背景",
    "suspected_unmodeled_footprint": "疑似未建模的足迹效应",
}

CJK_FONT_FAMILIES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "Arial Unicode MS",
)


@dataclass(frozen=True, slots=True)
class DiagnosticView:
    """One fixed figure, canvas, and primary axes for a diagnostic tab."""

    figure: Figure
    canvas: object
    axes: object


class DiagnosticCanvas(FigureCanvasQTAgg):
    """Coalesce redraws on a QObject-owned timer that can be cancelled."""

    def __init__(self, figure: Figure) -> None:
        super().__init__(figure)
        self._draw_pending = False
        self._released = False
        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.timeout.connect(self._draw_queued)

    def draw_idle(self) -> None:
        if self.figure is not None and not self._released and not self._draw_pending:
            self._draw_pending = True
            if self.isVisible():
                self._draw_timer.start(0)

    def cancel_pending_draw(self) -> None:
        self._draw_timer.stop()
        self._draw_pending = False

    def release(self) -> None:
        self.cancel_pending_draw()
        self._released = True
        if hasattr(self, "renderer"):
            del self.renderer
        self._lastKey = None

    def _draw_queued(self) -> None:
        if not self._draw_pending:
            return
        if self.figure is None or self._released:
            self._draw_pending = False
            return
        if not self.isVisible():
            return
        self._draw_pending = False
        self.renderer = self.get_renderer()
        self.renderer.clear()
        self.figure.draw(self.renderer)
        try:
            self.update()
        except RuntimeError as error:
            deleted = "Internal C++ object (DiagnosticCanvas) already deleted"
            if not self._released and deleted not in str(error):
                raise
            self._released = True

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
        if self._draw_pending and not self._released:
            self._draw_timer.start(0)

    def closeEvent(self, event: object) -> None:
        self.release()
        super().closeEvent(event)


SCRATCH_VIEWS: dict[str, DiagnosticView] | None = None


def _cjk_font() -> font_manager.FontProperties:
    for family in CJK_FONT_FAMILIES:
        properties = font_manager.FontProperties(family=family)
        try:
            font_manager.findfont(properties, fallback_to_default=False)
        except ValueError:
            continue
        return font_manager.FontProperties(family=(family, "DejaVu Sans"))
    return font_manager.FontProperties(family="sans-serif")


def apply_figure_font(figure: Figure) -> None:
    """Apply a local font to current artists without mutating global style."""
    properties = _cjk_font()
    for artist in figure.findobj(match=Text):
        artist.set_fontproperties(properties)


def draw_empty(view: DiagnosticView, title: str, message: str = "暂无可用数据") -> None:
    # Spine visibility survives Axes.clear(), so only ticks are blanked here;
    # hiding spines would leave later real draws without an axes frame.
    for axes in view.figure.axes:
        axes.clear()
        axes.set_xticks(())
        axes.set_yticks(())
    view.axes.set_title(title)
    view.axes.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=view.axes.transAxes,
        color="#8A8A8E",
    )
    apply_figure_font(view.figure)
    view.canvas.draw_idle()


def _axes(figure: Figure, key: str) -> object:
    if key == "uncertainty":
        correlation, profile = figure.subplots(1, 2)
        profile.set_title("profile likelihood 与区间")
        return correlation
    return figure.subplots()


def _view(key: str, *, qt: bool) -> DiagnosticView:
    title = next(title for name, title, _description in VIEW_SPECS if name == key)
    figure = Figure(layout="constrained") if qt else Figure(figsize=(0.8, 0.6), dpi=25)
    canvas = DiagnosticCanvas(figure) if qt else FigureCanvasAgg(figure)
    axes = _axes(figure, key)
    view = DiagnosticView(figure, canvas, axes)
    if qt:
        draw_empty(view, title)
    return view


def build_tabs() -> tuple[QTabWidget, dict[str, DiagnosticView]]:
    tabs = QTabWidget()
    tabs.setObjectName("diagnosticTabs")
    tabs.setAccessibleName("拟合诊断图标签")
    tabs.setToolTip("切换原始曲线、残差、候选解和专家诊断视图")
    views: dict[str, DiagnosticView] = {}
    for key, title, description in VIEW_SPECS:
        view = _view(key, qt=True)
        view.canvas.setObjectName(f"diagnosticCanvas:{key}")
        view.canvas.setAccessibleName(title)
        view.canvas.setAccessibleDescription(description)
        view.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        views[key] = view
        if key != COMPANION_SPEC[0]:
            tabs.addTab(view.canvas, title)
    return tabs, views


def build_scratch_views() -> dict[str, DiagnosticView]:
    global SCRATCH_VIEWS
    if SCRATCH_VIEWS is None:
        SCRATCH_VIEWS = {
            key: _view(key, qt=False) for key, _title, _description in VIEW_SPECS
        }
    return dict(SCRATCH_VIEWS)


def release_scratch_views(views: dict[str, DiagnosticView]) -> None:
    for view in views.values():
        if hasattr(view.canvas, "renderer"):
            del view.canvas.renderer
        view.canvas._lastKey = None
    views.clear()


def diagnostic_text(diagnostic: object) -> str:
    code = str(diagnostic.code)
    message = str(diagnostic.message)
    label = DIAGNOSTIC_LABELS.get(code)
    if label is None:
        return f"{code}: {message}"
    return f"{label}（{code}: {message}）"


def _candidate_is_inspection_only(candidate: object) -> bool:
    ranking = getattr(candidate, "ranking_objective", None)
    return bool(
        not getattr(candidate, "valid", False)
        or getattr(candidate, "stop_reason", "") == "early_eliminated"
        or not isfinite(float(getattr(candidate, "objective", float("inf"))))
        or (ranking is not None and not isfinite(float(ranking)))
    )


def candidate_label(candidate: object, *, selected: bool) -> str:
    parts = [str(candidate.candidate_id), f"J={candidate.objective:g}"]
    if _candidate_is_inspection_only(candidate):
        parts.append("仅供检查")
    else:
        parts.append("有效")
    if getattr(candidate, "stop_reason", "") == "early_eliminated":
        parts.append("早期淘汰")
    if selected:
        parts.append("查看中")
    return " · ".join(parts)


def draw_candidate_comparison(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    axes = view.axes
    axes.clear()
    candidates = () if result is None else result.candidates
    for candidate in candidates:
        qz = np.asarray(candidate.qz_a_inv, dtype=float)
        model = np.asarray(candidate.model_normalized, dtype=float)
        display = np.where(np.isfinite(model) & (model > 0.0), model, np.nan)
        selected = candidate.candidate_id == candidate_id
        axes.plot(
            qz,
            display,
            linewidth=2.0 if selected else 1.0,
            alpha=1.0 if selected else 0.65,
            label=candidate_label(candidate, selected=selected),
        )
    if not candidates:
        axes.text(0.5, 0.5, "暂无候选解", ha="center", va="center", transform=axes.transAxes)
    elif candidate_id is None:
        axes.text(0.02, 0.02, "尚未选择候选解", transform=axes.transAxes)
    axes.set(title="候选解比较", xlabel="qz (Å⁻¹)", ylabel="归一化 R", yscale="log")
    axes.yaxis.set_major_formatter(LogFormatter())
    if candidates:
        axes.legend(fontsize="small")
    apply_figure_font(view.figure)
    view.canvas.draw_idle()


def validate_batch_trends(
    dataset_ids: tuple[str, ...],
    thickness_a: tuple[float, ...],
    period_a: tuple[float, ...],
) -> None:
    if len(dataset_ids) < 2:
        raise ValueError("batch trend requires at least two datasets")
    if len({len(dataset_ids), len(thickness_a), len(period_a)}) != 1:
        raise ValueError("batch trend columns must have equal lengths")
    if any(not value.strip() for value in dataset_ids) or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("batch trend dataset ids must be nonempty and unique")
    if any(not isfinite(float(value)) for value in thickness_a + period_a):
        raise ValueError("batch trend values must be finite")


def draw_batch_trends(
    view: DiagnosticView,
    trends: tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]] | None,
) -> None:
    if trends is None:
        draw_empty(view, "批量趋势", "暂无批量趋势；请提供项目级趋势")
        return
    dataset_ids, thickness_a, period_a = trends
    axes = view.axes
    axes.clear()
    positions = tuple(range(len(dataset_ids)))
    axes.plot(positions, np.asarray(thickness_a) / 10.0, "o-", label="厚度")
    axes.plot(positions, np.asarray(period_a) / 10.0, "s--", label="周期")
    axes.set_xticks(positions, dataset_ids)
    axes.set(title="批量趋势", xlabel="数据集", ylabel="长度 (nm)")
    axes.legend()
    apply_figure_font(view.figure)
    view.canvas.draw_idle()
