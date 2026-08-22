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
from matplotlib.ticker import AutoMinorLocator, FixedLocator, LogFormatter
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QTabWidget

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.live import LiveReflectivityPlot

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
    ("residual_map", "残差热图", "按候选逐行比较加权残差，定位共同失配的 q 区间"),
    ("parameter_map", "参数热图", "按候选逐行比较归一化参数值，查看结构解的分歧"),
    ("uncertainty", "相关性与区间", "查看当前候选拥有的相关和区间证据"),
    ("trend", "批量趋势", "查看多个数据集的厚度和周期趋势"),
)

# The SLD depth profile is not a tab. Judging a fit means reading curve
# agreement and the resulting structure together, so it occupies a companion
# pane that stays on screen whichever diagnostic tab is selected.
COMPANION_SPEC = ("sld", "SLD 深度剖面", "查看当前候选的实部和虚部 SLD")

VIEW_SPECS = (*TAB_SPECS, COMPANION_SPEC)

# The four interactive reflectivity panes render live through pyqtgraph
# (LiveReflectivityPlot); every other view stays matplotlib. build_tabs keys on
# this set to build heterogeneous tabs, and panel dispatch keys on the widget
# type (isinstance) so the all-matplotlib scratch preflight still takes the
# draw_* path while the live dict takes the show_* path.
LIVE_PANE_KEYS = ("log", "raw", "qz4", "residual")

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
    """Apply the local font and theme palette to the figure's current artists.

    Every draw ends here, and ``Axes.clear()`` resets facecolor and tick colours
    to the Matplotlib defaults, so the palette has to be reapplied at this point
    rather than only at construction.

    Only the family is overridden.  Assigning a whole ``FontProperties`` used to
    replace every other property along with it, which reset each artist's size
    back to the rcParam default and made a caller's explicit ``fontsize`` dead on
    arrival: a caption asking to be smaller than its title came out the same size.
    """
    families = _cjk_font().get_family()
    for artist in figure.findobj(match=Text):
        artist.set_fontfamily(families)
    _left_align_titles(figure)
    apply_figure_palette(figure)


def _left_align_titles(figure: Figure) -> None:
    """Send each axes title to its left edge, clear of the floating mode bar.

    That bar hovers over the plot's top-right corner, which is the band a centred
    title occupies: at the width the plot gets once both docks are open, the title
    runs under the bar's leftmost glyph.  Moving it to the opposite corner lets the
    two share one band.

    The title Matplotlib already owns is repositioned rather than swapped for the
    ``loc="left"`` one, because that alternative is a *different* artist: the text
    would move to ``_left_title`` and the default ``get_title()`` would start
    answering with an empty string, which is how the drawn titles are asserted.
    This runs on every draw because ``Axes.clear()`` rebuilds the title centred,
    and each draw_* clears before it paints.
    """
    for axes in figure.axes:
        title = axes.title
        title.set_horizontalalignment("left")
        # _update_title_position rewrites y on every render but preserves x, so
        # this survives resizes and redraws without being reapplied there.
        title.set_position((0.0, title.get_position()[1]))


def draw_empty(view: DiagnosticView, title: str, message: str = "暂无可用数据") -> None:
    # Spine visibility survives Axes.clear(), so only ticks are blanked here;
    # hiding spines would leave later real draws without an axes frame.
    #
    # Every pane gets its own caption.  On the uncertainty figure, which is split
    # in two, only the primary pane used to be titled and captioned, and
    # ``clear()`` also drops the title its companion was given at construction:
    # that half came up blank on startup, reading as a real plot whose curve had
    # failed to draw.  A title an axes already carries is put back, so each pane
    # keeps saying what it will show once evidence arrives.
    kept = tuple((axes, axes.get_title()) for axes in view.figure.axes)
    for axes in view.figure.axes:
        axes.clear()
        axes.set_xticks(())
        axes.set_yticks(())
    palette = apply_figure_palette(view.figure)
    for axes, previous_title in kept:
        axes.set_title(title if axes is view.axes else previous_title)
        axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axes.transAxes,
            color=palette.muted,
        )
    apply_figure_font(view.figure)
    view.canvas.draw_idle()


def _axes(figure: Figure, key: str) -> object:
    if key == "uncertainty":
        correlation, profile = figure.subplots(1, 2)
        profile.set_title("参数剖面似然与区间")
        return correlation
    return figure.subplots()


def current_plot_palette() -> theme.PlotPalette:
    """Resolve the figure palette from the running application's palette.

    Delegates to :func:`theme.current_plot_palette` so the matplotlib figures
    and the pyqtgraph panes share one resolver.  Kept as a module-global name
    here because ``apply_figure_palette`` and the ``draw_*`` functions call it
    late-bound, and tests monkeypatch ``diagnostics.current_plot_palette`` to
    force a palette onto that matplotlib path.
    """
    return theme.current_plot_palette()


def apply_figure_palette(figure: Figure) -> theme.PlotPalette:
    """Paint one figure's structural colours and return the palette used."""
    palette = current_plot_palette()
    figure.set_facecolor(palette.background)
    for axes in figure.axes:
        axes.set_facecolor(palette.background)
        axes.tick_params(colors=palette.foreground, which="both")
        for spine in axes.spines.values():
            spine.set_color(palette.muted)
        for text in (axes.title, axes.xaxis.label, axes.yaxis.label):
            text.set_color(palette.foreground)
    return palette


# Matplotlib labels the axes it creates for a colorbar, and that label is the
# only handle on it: a colorbar axes reports data of its own and holds no image,
# so nothing else tells it apart from a plotted pane.
COLORBAR_AXES_LABEL = "<colorbar>"


def apply_axes_grid(figure: Figure, palette: theme.PlotPalette) -> None:
    """Draw a reference grid behind the data on every plotted axes.

    Reading a value off a reflectivity curve means tracing it back to both
    axes, which bare tick marks make into guesswork across four decades.  The
    palette already carried a ``grid`` colour that nothing used; this is what
    consumes it.  Axes holding an image are skipped: a grid over the
    correlation matrix would sit on top of the cells it is meant to help read.
    """
    for axes in figure.axes:
        # An image axes would have its cells overdrawn, and a placeholder axes
        # holds only centred text: gridding either one adds lines that carry no
        # reading.  ``has_data`` is false for a text-only placeholder.  A
        # colorbar reports data of its own without holding an image, so it needs
        # its own exclusion: it is a legend for another axes, not a plot.
        if axes.images or axes.get_label() == COLORBAR_AXES_LABEL or not axes.has_data():
            continue
        axes.set_axisbelow(True)
        axes.grid(True, which="major", color=palette.grid, linewidth=0.6)
        # A log axis already places decade subdivisions itself, and a fixed
        # locator means the ticks are categories (one per dataset) rather than a
        # measurable scale: subdividing either one invents readings that are not
        # there.
        for axis, scale in ((axes.xaxis, axes.get_xscale()), (axes.yaxis, axes.get_yscale())):
            if scale == "linear" and not isinstance(axis.get_major_locator(), FixedLocator):
                axis.set_minor_locator(AutoMinorLocator())
        axes.grid(True, which="minor", color=palette.grid, linewidth=0.3, alpha=0.6)
        axes.tick_params(which="minor", length=2)


def _view(key: str, *, qt: bool) -> DiagnosticView:
    title = next(title for name, title, _description in VIEW_SPECS if name == key)
    figure = Figure(layout="constrained") if qt else Figure(figsize=(0.8, 0.6), dpi=25)
    canvas = DiagnosticCanvas(figure) if qt else FigureCanvasAgg(figure)
    axes = _axes(figure, key)
    view = DiagnosticView(figure, canvas, axes)
    apply_figure_palette(figure)
    if qt:
        draw_empty(view, title)
    return view


def build_tabs() -> tuple[QTabWidget, dict[str, DiagnosticView | LiveReflectivityPlot]]:
    tabs = QTabWidget()
    tabs.setObjectName("diagnosticTabs")
    tabs.setAccessibleName("拟合诊断图标签")
    tabs.setToolTip("切换原始曲线、残差、候选解和专家诊断视图")
    # The nine labels want 784px; with both docks open the stack gets 568px, so
    # Qt's default response is to hide the last three behind scroll arrows --
    # three diagnostics a user cannot see are three they will not know exist.
    # Eliding instead keeps every tab on screen and readable: the labels share
    # the shortfall as trimmed characters rather than one of them absorbing it
    # as total absence.  tabText() still returns the full label, so lookups and
    # the documented titles are unaffected.
    tab_bar = tabs.tabBar()
    tab_bar.setUsesScrollButtons(False)
    tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
    views: dict[str, DiagnosticView | LiveReflectivityPlot] = {}
    for key, title, description in VIEW_SPECS:
        # The live reflectivity panes ARE their own QWidget; the matplotlib views
        # expose their addable/queryable widget through .canvas. Both accept the
        # same object-name and accessibility contract so lookups keep working.
        if key in LIVE_PANE_KEYS:
            view: DiagnosticView | LiveReflectivityPlot = LiveReflectivityPlot()
            # A pg pane is not styled by the Qt stylesheet or by
            # apply_figure_palette, so it needs the resolved palette handed to it
            # explicitly -- otherwise it keeps pyqtgraph's default black canvas
            # regardless of the desktop appearance.
            view.apply_palette(current_plot_palette())
            canvas: object = view
        else:
            view = _view(key, qt=True)
            canvas = view.canvas
        canvas.setObjectName(f"diagnosticCanvas:{key}")
        canvas.setAccessibleName(title)
        canvas.setAccessibleDescription(description)
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        views[key] = view
        if key != COMPANION_SPEC[0]:
            index = tabs.addTab(canvas, title)
            # Eliding trims characters off the label, so the full name has to be
            # reachable some other way: the per-tab tip carries it plus what the
            # view answers, which the bar-wide tip cannot do because it says the
            # same sentence over all nine.
            tabs.setTabToolTip(index, f"{title} — {description}")
    return tabs, views


def build_scratch_views() -> dict[str, DiagnosticView]:
    global SCRATCH_VIEWS
    if SCRATCH_VIEWS is None:
        SCRATCH_VIEWS = {key: _view(key, qt=False) for key, _title, _description in VIEW_SPECS}
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


def candidate_is_inspection_only(candidate: object) -> bool:
    """Report whether a candidate may only be inspected, never compared.

    Three views ask this same question, so it is answered in one place: the
    comparison overlay, the candidate label, and the heatmaps, which would
    otherwise devote a row to a candidate whose objective is not even finite.
    """
    ranking = getattr(candidate, "ranking_objective", None)
    return bool(
        not getattr(candidate, "valid", False)
        or getattr(candidate, "stop_reason", "") == "early_eliminated"
        or not isfinite(float(getattr(candidate, "objective", float("inf"))))
        or (ranking is not None and not isfinite(float(ranking)))
    )


def finish_view(view: DiagnosticView) -> None:
    """Apply the grid, then the font, then queue the redraw.

    Grid first: it can install a minor locator, and the font pass afterwards
    then reaches every tick label that exists by the time it runs.
    """
    apply_axes_grid(view.figure, current_plot_palette())
    apply_figure_font(view.figure)
    view.canvas.draw_idle()


def candidate_label(candidate: object, *, selected: bool) -> str:
    parts = [str(candidate.candidate_id), f"J={candidate.objective:g}"]
    if candidate_is_inspection_only(candidate):
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
        axes.legend(fontsize=theme.FONT_PT_SM)
    apply_axes_grid(view.figure, current_plot_palette())
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
    apply_axes_grid(view.figure, current_plot_palette())
    apply_figure_font(view.figure)
    view.canvas.draw_idle()
