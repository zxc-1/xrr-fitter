"""Posterior histograms and the four-page uncertainty evidence container.

The evidence a reader needs to answer "is this thickness actually determined?"
arrives in four unrelated shapes -- a correlation matrix, profile likelihood
curves, SLD credible bands, and marginal posteriors -- and only the last of them
had no rendering at all.  All four live here as pages of one tab widget so the
judgement can be made without leaving the panel; the first three delegate to the
drawing functions that already own those figures.

Matplotlib and NumPy are confined to ``gui.plots`` by the dependency gate, so
``gui.results.uncertainty`` assembles this widget rather than plotting itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.diagnostics import (
    DiagnosticCanvas,
    DiagnosticView,
    apply_figure_palette,
    finish_view,
)
from xrr_fitter.gui.plots.sld import (
    _owned_report,
    _page_placeholder,
    draw_band_page,
    draw_correlation_page,
    draw_profile_page,
)

# The three marginal quantiles drawn on every parameter panel.  A median alone
# makes "determined to ±1 Å" and "the prior width handed straight back" look
# identical on screen; the 16th and 84th percentiles are the width itself, and
# they are the same pair the evidence text reports, so the picture and the text
# cannot disagree.
POSTERIOR_QUANTILES = (0.16, 0.5, 0.84)

# Freedman-Diaconis on four retained samples asks for one bin, which turns a
# marginal into a single block. A fixed count keeps a short chain readable while
# staying coarse enough not to invent structure in a long one.
POSTERIOR_BINS = 24

# 没跑过采样时后验页只说这一句。写「未运行」而不是「无数据」：这两件事的下一步完全
# 不同——一个是去跑采样，一个是去查为什么算出来的样本是空的。
POSTERIOR_UNAVAILABLE_TEXT = "未运行 MCMC 采样，无后验样本"

# 每格标题下面那行分位读数保留的有效位。参数跨厚度（数十 Å）和比例因子（约 1）两个
# 量级，定点小数会把其中一个显示成 0.00 或者 40.00000。
QUANTILE_DIGITS = 4

# The page titles, in tab order.  Read left to right they are the order the
# evidence is meant to be judged in: what is entangled, how far each parameter
# can move on its own, what that does to the structure, and finally the full
# posterior when sampling was actually run. The generic profile title does not
# relabel exploratory loss-support evidence as calibrated likelihood inference.
UNCERTAINTY_PAGE_TITLES = ("相关矩阵", "参数剖面", "SLD 可信带", "MCMC 后验")

# Matplotlib's default figure is 6.4x4.8in, which a canvas turns into a 480px
# height request -- taller than the verdict and the fitted values put together,
# so the pages would push the readings the evidence talks about out of the
# inspector.  This is a page that still reads as a plot at the width the
# inspector column actually gets.
PAGE_FIGURE_INCHES = (4.0, 2.6)

# The shortest evidence page still worth drawing: two axes with their labels and
# enough room to tell a peaked marginal from a flat one.  The floor is this plus
# whatever the tab bar asks for, so it is derived rather than counted; see
# ``UncertaintyPages.__init__``.
MIN_PAGE_PLOT_H = 200

# 右栏分位段那条紧凑直方图的画布尺寸（英寸）。它不是页面版后验的等比缩小：那一版每格带标题、
# 两轴刻度和一行分位读数，缩到右栏这点宽度上字会互相叠住。这一版只留形状，数字全部交给它下面
# 那张分位表念。
QUANTILE_STRIP_INCHES = (2.6, 0.95)

# 比 ``POSTERIOR_BINS`` 略多一点。页面版一列里要塞下每个参数一格，这条只画一个参数，横向能用
# 的像素多些；分箱跟着少的话，一个偏斜的后验会被抹成对称的方块。
QUANTILE_STRIP_BINS = 26

# 这条图的高度地板。再矮下去柱与柱的高低差就读不出来了，而读形状是它唯一的用处。宽度不设地板：
# matplotlib 的画布自己不要宽度，右栏窄下去时它跟着窄，形状还在。
QUANTILE_STRIP_MIN_H = 76


def _page_view() -> DiagnosticView:
    """One matplotlib page: its own figure, canvas, and primary axes."""
    figure = Figure(figsize=PAGE_FIGURE_INCHES, layout="constrained")
    canvas = DiagnosticCanvas(figure)
    view = DiagnosticView(figure, canvas, figure.subplots())
    apply_figure_palette(figure)
    return view


def _strip_view() -> DiagnosticView:
    """右栏那条紧凑直方图：同样是一图一画布一轴，只是比页面版小一号。"""
    figure = Figure(figsize=QUANTILE_STRIP_INCHES, layout="constrained")
    canvas = DiagnosticCanvas(figure)
    view = DiagnosticView(figure, canvas, figure.subplots())
    apply_figure_palette(figure)
    return view


def _quantile_readout(values: np.ndarray) -> str:
    """The three quantiles as one line, so a panel states its own numbers."""
    low, mid, high = (float(value) for value in np.quantile(values, POSTERIOR_QUANTILES))
    return f"P16 {low:.{QUANTILE_DIGITS}g} · P50 {mid:.{QUANTILE_DIGITS}g} · P84 {high:.{QUANTILE_DIGITS}g}"


def _draw_marginal(axes: object, name: str, values: np.ndarray, palette: object) -> None:
    """One parameter's marginal posterior with its three quantile markers."""
    axes.hist(
        values,
        bins=POSTERIOR_BINS,
        color=theme.DATA_OBSERVED,
        alpha=0.55,
    )
    for probability in POSTERIOR_QUANTILES:
        position = float(np.quantile(values, probability))
        # The median is solid and the pair around it dashed: the three would
        # otherwise be three identical lines, and which one is the centre is the
        # first thing a reader needs off this panel.
        axes.axvline(
            position,
            color=theme.current_accent() if probability == 0.5 else palette.muted,
            linewidth=1.0 if probability == 0.5 else 0.8,
            linestyle="-" if probability == 0.5 else "--",
        )
    axes.set_title(name, fontsize=theme.FONT_PT_SM)
    axes.set_xlabel(_quantile_readout(values), fontsize=theme.FONT_PT_SM)
    axes.tick_params(labelsize=theme.FONT_PT_SM)


class McmcHistogramPlot(QWidget):
    """Marginal posterior histograms, one panel per sampled parameter.

    The panel count follows the report, so the grid is rebuilt on every draw.
    ``DiagnosticView`` is frozen and its primary axes cannot be replaced, so that
    axes is kept and re-slotted into the new grid while the rest are added and
    removed around it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mcmcHistogramPlot")
        self.setAccessibleName("MCMC 边缘后验分布")
        self._view = _page_view()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view.canvas)
        self._blank(POSTERIOR_UNAVAILABLE_TEXT)

    @property
    def figure(self) -> Figure:
        return self._view.figure

    @property
    def canvas(self) -> object:
        return self._view.canvas

    def _reset(self) -> object:
        for axes in tuple(self._view.figure.axes):
            if axes is not self._view.axes:
                axes.remove()
        self._view.axes.clear()
        return self._view.axes

    def _blank(self, message: str) -> None:
        axes = self._reset()
        axes.set_xticks(())
        axes.set_yticks(())
        palette = apply_figure_palette(self._view.figure)
        axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axes.transAxes,
            color=palette.muted,
        )
        finish_view(self._view)

    def _panel_axes(self, count: int) -> tuple[object, ...]:
        """A one-column grid of ``count`` panels, first slot reusing the kept axes.

        Stacked rather than tiled: the panels share nothing on the vertical axis
        (each marginal has its own count scale) but the reader compares widths,
        and a single column gives every parameter the same horizontal extent.
        """
        grid = self._view.figure.add_gridspec(count, 1)
        self._view.axes.set_subplotspec(grid[0])
        rest = tuple(self._view.figure.add_subplot(grid[index]) for index in range(1, count))
        return (self._view.axes, *rest)

    def show_report(self, result: object | None, candidate_id: str | None) -> None:
        """Draw the marginals the inspected candidate owns, or say why not.

        Ownership is checked on the enclosing uncertainty report, not on the MCMC
        report alone: a chain is evidence about the candidate whose report carries
        it, and showing it beside a different candidate would relabel it.
        """
        report, message = _owned_report(result, candidate_id)
        if report is None:
            self._blank(message)
            return
        chain = report.mcmc
        if chain is None:
            self._blank(POSTERIOR_UNAVAILABLE_TEXT)
            return
        samples = np.asarray(chain.samples_physical, dtype=float)
        names = tuple(chain.parameter_names)
        if samples.ndim != 2 or samples.shape[0] == 0 or not names:
            self._blank(POSTERIOR_UNAVAILABLE_TEXT)
            return
        self._reset()
        palette = apply_figure_palette(self._view.figure)
        for index, (axes, name) in enumerate(zip(self._panel_axes(len(names)), names, strict=True)):
            _draw_marginal(axes, name, samples[:, index], palette)
        finish_view(self._view)


class UncertaintyPages(QTabWidget):
    """The four kinds of uncertainty evidence, one page each.

    Every page redraws on ``set_result`` rather than on tab change: a page that
    only drew when selected would hand a reader who switched tabs the previous
    candidate's evidence for as long as it took to repaint.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("uncertaintyPages")
        self.setAccessibleName("不确定度证据页")
        tab_bar = self.tabBar()
        tab_bar.setUsesScrollButtons(False)
        self.correlation = _page_view()
        self.profile = _page_view()
        self.bands = _page_view()
        self.posterior = McmcHistogramPlot()
        for widget, title in zip(self._widgets(), UNCERTAINTY_PAGE_TITLES, strict=True):
            self.addTab(widget, title)
        # The floor is derived, not counted: ask the assembled tab widget what its
        # chrome needs and add a page worth drawing.  A hand-written total goes
        # stale the moment the tab bar grows a row, and the widget then honours the
        # number by collapsing the axes instead of by growing.
        canvas_floor = self.correlation.canvas.minimumSizeHint().height()
        chrome = max(self.minimumSizeHint().height() - canvas_floor, 0)
        self.setMinimumHeight(chrome + MIN_PAGE_PLOT_H)
        self.clear_pages("尚无拟合结果")

    def _widgets(self) -> tuple[QWidget, ...]:
        return (
            self.correlation.canvas,
            self.profile.canvas,
            self.bands.canvas,
            self.posterior,
        )

    def figures(self) -> tuple[Figure, ...]:
        """Each page's figure, in tab order, for inspection and export."""
        return (
            self.correlation.figure,
            self.profile.figure,
            self.bands.figure,
            self.posterior.figure,
        )

    def set_result(self, result: object | None, candidate_id: str | None) -> None:
        draw_correlation_page(self.correlation, result, candidate_id)
        draw_profile_page(self.profile, result, candidate_id)
        draw_band_page(self.bands, result, candidate_id)
        self.posterior.show_report(result, candidate_id)

    def clear_pages(self, message: str) -> None:
        """Blank all four pages with one message: no page keeps stale evidence."""
        for view in (self.correlation, self.profile, self.bands):
            _page_placeholder(view, message)
        self.posterior._blank(message)


class PosteriorQuantilePlot(QWidget):
    """一个参数的后验形状，画在右栏分位表的上方（帧⑤ 设计稿把两者叠在同一段里）。

    表里那三行数字说的是分布上的三个点，形状说的是别的事：偏斜、双峰、贴着边界堆起来。三个
    分位一模一样的两个后验可以长得完全不同，所以这一段既要数字也要形状。

    图上不写任何数字。同一段里把三个分位念两遍，读者会去找两处的差别；念的那一遍归表，画的
    这一遍归图。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("posteriorQuantilePlot")
        self.setAccessibleName("后验分布形状")
        self._view = _strip_view()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view.canvas)
        self.setMinimumHeight(QUANTILE_STRIP_MIN_H)
        self.show_unavailable(POSTERIOR_UNAVAILABLE_TEXT)

    @property
    def view(self) -> DiagnosticView:
        return self._view

    @property
    def figure(self) -> Figure:
        return self._view.figure

    @property
    def canvas(self) -> object:
        return self._view.canvas

    def _bare_axes(self) -> object:
        """清空后的那根轴：两轴刻度与四面边框都撤掉，只留画的东西。

        ``clear()`` 会把 spine 的可见性一并还原成默认，所以撤边框这一步必须排在它后面。
        """
        axes = self._view.axes
        axes.clear()
        axes.set_xticks(())
        axes.set_yticks(())
        for spine in axes.spines.values():
            spine.set_visible(False)
        return axes

    def show_values(self, values: Sequence[float], quantiles: Sequence[float]) -> None:
        """画这一列采样点的形状，并在三个分位处各落一条线。

        分位值由调用方算好传进来，这里不重算：同一段里的表念的必须和这三条线站的是同一组
        数，各算一次的话图与表会在小数位上分叉。
        """
        axes = self._bare_axes()
        palette = apply_figure_palette(self._view.figure)
        axes.hist(tuple(values), bins=QUANTILE_STRIP_BINS, color=theme.DATA_OBSERVED, alpha=0.55)
        for probability, position in zip(POSTERIOR_QUANTILES, quantiles, strict=True):
            # 中位实线、两侧虚线，与页面版同一套画法：三条一样的线里认不出哪条是中心。
            axes.axvline(
                position,
                color=theme.current_accent() if probability == 0.5 else palette.muted,
                linewidth=1.0 if probability == 0.5 else 0.8,
                linestyle="-" if probability == 0.5 else "--",
            )
        finish_view(self._view)

    def show_unavailable(self, message: str) -> None:
        """没有样本时只留一句话——空着的坐标框读起来像「这个后验是平的」。"""
        axes = self._bare_axes()
        palette = apply_figure_palette(self._view.figure)
        axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axes.transAxes,
            color=palette.muted,
            fontsize=theme.FONT_PT_SM,
        )
        finish_view(self._view)
