"""pyqtgraph-backed live reflectivity view for real-time preview and fluid gestures.

matplotlib stays the renderer for the static diagnostics (correlation matrix,
credible-band profile, residual and parameter heatmaps, batch trends) and for the
byte-identity export in ``io/export_plots.py``; both are decoupled from what draws on
screen, so this module can own the interactive reflectivity family without touching
them.  The win here is a search publishing a new model many times per second into a
single mutating curve, plus a draggable region and a click that reads back a q, in
place of the matplotlib navigation toolbar.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.series_key import PEN_STYLES, overlay_style

# 上标字符表。对数轴的刻度值本身就是指数，写成 10ⁿ 只差一次逐字符转写。
SUPERSCRIPT_DIGITS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
# 设计稿量出来的测量点：反射率三张 r=2.6（直径 5.2），残差那张 r=2.2（直径 4.4）。残差
# 点上还压着一条连线，点子跟着小一号，密集处才不糊成一条带；反射率只有点，反而要大一点。
# 截断点那个 ▽ 是 6px 宽的小三角，比测量点略大但不抢眼。
REFLECTIVITY_MARKER_PX = 5.2
RESIDUAL_MARKER_PX = 4.4
FLOOR_MARKER_PX = 6.0
# 拟合窗口说明要报出当前横轴的单位，而单位就写在轴标题末尾的括号里。度号贴着数字写，
# 其余单位隔一个空格——两种都照排版惯例，不是同一个规则的例外。
FINAL_UNIT = re.compile(r"\(([^()]*)\)\s*$")
TIGHT_UNITS = ("°",)
# 图里那两段小字照设计稿的字号走（10.5px、9.5px），都比 ``FONT_PX_SM`` 再小一档：它们贴着曲线画，
# 用正文字号会盖住数据。按像素给，取 96 dpi 下原先点数落到的整像素（11、10）：点数在 Mac 上会
# 再缩 25%，``setPixelSize`` 又只收整数。
FIT_RANGE_CAPTION_PX = 11
SIGMA_LABEL_PX = 10
# 反射率两张与残差那张共用的横轴。2θ 是散射角；入射角是 θ——叫它入射角，等于说轴上的数是实际值的
# 两倍（导入那一步正是把入射角乘二归一到散射角）。
ANGLE_AXIS_LABEL = "散射角 2θ (°)"
# 残差画到 ±5σ 为止，越界的点钉在边上、换成朝外的空心三角。一个 40σ 的离群点会让自动缩放把
# ±1σ 带压成一条线，整张图就只剩「有个点很远」这一句；钉住之后带子仍占得住画面的一成多，越界
# 也照样看得见。
RESIDUAL_CLAMP_SIGMA = 5.0
# 掩膜点按的是指针底下那个点：离它不到这么多像素才算点中。此前只比横坐标、挑最近的一列，曲线正上方
# 很远处点一下也会排掉一个点。6px 比测量点的半径大一圈，手抖一点仍点得中，相邻两点又不会抢同一下。
MASK_PICK_RADIUS_PX = 6.0
# 悬停时套在那个点外的圈：比测量点大一号、空心、用前景色——深浅两种画布上都看得出。
MASK_HOVER_RING_PX = 11.0


def _without_decades(values: list[float], decades: list[float]) -> list[float]:
    """次级网格保留原顺序，只去掉与整数量级重合的位置。"""
    return [value for value in values if all(abs(value - decade) > 1e-9 for decade in decades)]


class DecadeAxis(pg.AxisItem):
    """对数刻度一律写成 10ⁿ，并保证整数量级本身在刻度里。

    pyqtgraph 的 ``logTickStrings`` 先把 10ⁿ 按 ``"%0.1g"`` 写成十进制，只有当结果里
    出现 "e" 时才改写成幂次。于是同一条轴上混着四种写法：0 写成 1，-2 写成 0.01，
    -4 写成 0.0001，-6 才写成 10⁻⁶。读者要判断的是「差了几个数量级」，而混排的写法把
    这件事变成了数零个数——设计稿写的是齐整的一列 10ⁿ。

    只写法不够：``logTickStrings`` 只能改写它收到的那些值，而值由 ``tickValues`` 挑。量程
    跨不到两个频程时（帧① 的曲线从 1 掉到 0.03 就是这样）基类按 0.05 个频程一档挑刻度，
    交上来的全是非整数指数，写成 10ⁿ 就等于把 10^-1.45 谎报成 10⁻¹——只能交回基类写十进制，
    于是纵轴退回一列 ``0.04 0.05 … 0.9 1``。所以两处都要覆写：``tickValues`` 保证整数频程
    在场，``logTickStrings`` 负责把它们写成 10ⁿ。

    两个覆写都只管对数刻度：线性刻度（原始强度、qz⁴R、加权残差）走 ``tickStrings`` 那条
    分支，且 ``tickValues`` 在非对数模式下原样交回基类。
    """

    def tickValues(self, min_val: float, max_val: float, size: float) -> list[tuple[float | None, list[float]]]:
        # 形参改成 snake_case（基类写的是 ``minVal`` / ``maxVal``）：R23 命名门禁不给 Qt 覆写的
        # 形参开例外，而 pyqtgraph 是按位置调这个方法的（``AxisItem`` 的 ``generateDrawSpecs``），
        # 改名不影响派发。
        levels = super().tickValues(min_val, max_val, size)
        if not self.logMode:
            return levels
        low, high = sorted((float(min_val), float(max_val)))
        decades = [float(exponent) for exponent in range(math.ceil(low), math.floor(high) + 1)]
        # 量程整个落在一个频程内部时一个整数指数都不在场，凑不出 10ⁿ 的刻度——交回基类，
        # 宁可写十进制，也不要把 10^-1.45 写成 10⁻¹。
        if not decades:
            return levels
        # 基类第一级已经落在整数指数上（跨好几个频程时就是这样）就别动：它挑的疏密是按可用
        # 高度算的，插一级更密的进去会把纵轴挤成一列连续数字。
        if levels and all(abs(value - round(value)) <= 1e-9 for value in levels[0][1]):
            return levels
        # 整数频程排第一级——``maxTextLevel`` 是 0，只有第一级写字。后续级留着画网格线，但要
        # 剔掉和第一级重合的值，否则同一个位置会叠两条线。
        thinned = [(spacing, _without_decades(values, decades)) for spacing, values in levels]
        return [(1.0, decades), *thinned]

    def logTickStrings(self, values: object, scale: float, spacing: float) -> list[str]:
        exponents = [round(float(value)) for value in values]
        drifted = any(abs(float(value) - exponent) > 1e-9 for value, exponent in zip(values, exponents, strict=True))
        # 非整数指数（放大到一个十倍频程以内时会出现）和被缩放过的刻度都不是「整数量级」，
        # 写成 10ⁿ 会把 10^-4.5 谎报成 10⁻⁴。这两种情况交回基类，宁可写十进制。
        if drifted or scale != 1.0:
            return super().logTickStrings(values, scale, spacing)
        return [f"10{str(exponent).translate(SUPERSCRIPT_DIGITS)}" for exponent in exponents]


def _release_menus_on_destroy(plot: pg.PlotWidget) -> None:
    # pyqtgraph 的弹出菜单没有 QWidget 父级，关闭 PlotItem 也只会解除引用。
    # 绑定拥有者的销毁信号，确保父窗口销毁和单独关闭绘图两条路径都释放菜单。
    for menu in (plot.plotItem.ctrlMenu, plot.plotItem.vb.menu):
        plot.destroyed.connect(menu.deleteLater)


class ObjectiveTracePlot(pg.PlotWidget):
    """设计稿帧④ 画布第三段：逐帧攒出来的目标值轨迹。

    横轴用整次运行的完成比例而不是帧序号——帧的疏密取决于轮询间隔，用序号画出来的
    斜率会随着轮询退避而变形；也不用本阶段的比例，那个每换一个阶段就回卷到 0。纵轴是
    对数轴：一次运行里 J 跨好几个数量级（全局搜索的预览帧能冲到 10⁹），线性轴会把下降
    压成贴底的一条。住在这一层是因为架构门禁只允许绘图模块直接依赖
    pyqtgraph；拟合面板拿到的是这个已配好的部件。
    """

    Y_LABEL = "目标值 J"
    X_LABEL = "总完成比例"

    def __init__(self, parent: object | None = None) -> None:
        super().__init__(parent)
        _release_menus_on_destroy(self)
        self.setObjectName("fitObjectiveTrace")
        # 先换纵轴再写轴名：标签记在轴对象上，换轴会把已写的那行一起换掉。刻度写法同反射率页（见 DecadeAxis）。
        self.setAxisItems({"left": DecadeAxis(orientation="left")})
        self.setLogMode(y=True)
        self.setLabel("left", self.Y_LABEL)
        self.setLabel("bottom", self.X_LABEL)
        self.trace_item = self.plot([], [], pen=pg.mkPen(theme.DATA_CANDIDATE, width=1.6))
        self._x: list[float] = []
        self._y: list[float] = []

    def append(self, fraction: float, objective: float) -> None:
        # 对数轴放不下非有限或不为正的值：J 恒不为负，这里拦下的是还没有候选时那个 +inf。
        if not math.isfinite(objective) or objective <= 0.0:
            return
        self._x.append(float(fraction))
        self._y.append(float(objective))
        self.trace_item.setData(self._x, self._y)

    def reset(self) -> None:
        self._x.clear()
        self._y.clear()
        self.trace_item.setData([], [])

    def _apply_theme_palette(self) -> None:
        """Refresh the progress chart's structural colours after a theme switch."""
        palette = theme.current_plot_palette()
        self.setBackground(QColor.fromRgbF(*palette.background))
        foreground = QColor.fromRgbF(*palette.foreground)
        muted = QColor.fromRgbF(*palette.muted)
        for side in ("left", "bottom"):
            axis = self.plotItem.getAxis(side)
            axis.setTextPen(foreground)
            axis.setPen(muted)
        self.showGrid(x=True, y=True, alpha=palette.grid[3])


class LiveReflectivityPlot(pg.PlotWidget):
    """One reflectivity pane with an owned preview curve and native gestures.

    The observed and model artists are created once and updated in place; the
    preview artist is created lazily on its first publish so a pane that never
    hosts a search carries no spare line.  Selection and masking are opt-in so a
    read-only pane does not intercept the drags a reader uses to pan and zoom.
    """

    fit_range_selected = Signal(float, float)
    # 报的是点中那个点在面板收到的数组里的下标；面板收到的是整份预处理数据，所以它就是预处理点的下标。
    point_mask_requested = Signal(int)
    cursor_moved = Signal(float, float)
    cursor_left = Signal()

    def __init__(self, parent: object | None = None) -> None:
        super().__init__(parent)
        _release_menus_on_destroy(self)
        self.plot_item = self.getPlotItem()
        # 换掉纵轴：对数刻度的写法要齐整（见 DecadeAxis）。趁面板还空着换，之后的
        # 网格、配色、量程设置才都落在这个轴上。
        self.plot_item.setAxisItems({"left": DecadeAxis(orientation="left")})
        self._released = False
        self._masking = False
        self._range_accepted_mouse_buttons: Qt.MouseButton | None = None
        # 掩膜点按得中的那些点：（预处理下标, 横坐标, 画出来的纵坐标），只有原始与对数两页会填。
        self._mask_targets: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.empty(0, dtype=int),
            np.empty(0),
            np.empty(0),
        )
        self._hover_target: int | None = None
        self.preview_item: pg.PlotDataItem | None = None
        self.range_item: pg.LinearRegionItem | None = None
        self._overlay_items: list[pg.PlotDataItem] = []
        self._overlay_keys: tuple[tuple[str, str, str], ...] = ()
        self.observed_item = self.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=REFLECTIVITY_MARKER_PX,
            symbolPen=None,
            symbolBrush=pg.mkColor(theme.DATA_OBSERVED),
            name="观测数据",
        )
        self.model_item = self.plot(
            [],
            [],
            pen=pg.mkPen(theme.DATA_CANDIDATE, width=1.4, style=Qt.PenStyle.DashLine),
            name="当前拟合模型",
        )
        # The raw view struck-out points and the residual zero baseline are always
        # present but empty; a pane that never renders those projections simply
        # leaves them blank rather than paying a lazy-construction branch per draw.
        self.excluded_item = self.plot(
            [],
            [],
            pen=None,
            symbol="x",
            symbolSize=7,
            symbolPen=pg.mkPen(theme.DATA_NEUTRAL),
            symbolBrush=None,
            name="排除点",
        )
        self.reference_item = self.plot(
            [],
            [],
            # 实线，而且是这张图上唯一的实线基准：零是残差唯一的绝对参考，虚线会把它读成
            # 「一条提示」而跟 ±2σ 那两条提示线混在一起。
            pen=pg.mkPen(theme.DATA_NEUTRAL, width=1.0, style=Qt.PenStyle.SolidLine),
            name="零参考线",
        )
        # The log projection clamps a below-floor point up to the floor, where it
        # then reads as a real measurement lying on a flat baseline -- the one
        # misreading the display floor introduces.  A hollow downward triangle at
        # the clamped position says "at most this" instead, and it does so by shape
        # rather than by a new hue, so the key stays readable in greyscale.
        self.clipped_item = self.plot(
            [],
            [],
            pen=None,
            symbol="t",
            symbolSize=FLOOR_MARKER_PX,
            symbolPen=pg.mkPen(theme.DATA_NEUTRAL),
            symbolBrush=None,
            name="截断点",
        )
        # 残差越过 ±RESIDUAL_CLAMP_SIGMA 的点：钉在边上，三角朝越界的那一侧。同截断点一样靠形状
        # 而不是新颜色说话，颜色仍是它所属的观测色。
        self.residual_edge_item = self.plot(
            [],
            [],
            pen=None,
            symbol="t1",
            symbolSize=FLOOR_MARKER_PX,
            symbolPen=pg.mkPen(theme.DATA_OBSERVED),
            symbolBrush=None,
            name="越界残差",
        )
        # 掩膜模式下指针停在一个点上时，那个点外面套一圈：点下去会切换的就是它。不参与自动量程。
        self.mask_hover_item = pg.PlotDataItem(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=MASK_HOVER_RING_PX,
            symbolBrush=None,
        )
        self.mask_hover_item.setZValue(15)
        self.addItem(self.mask_hover_item, ignoreBounds=True)
        self._x_peers: tuple[LiveReflectivityPlot, ...] = ()
        self.fit_range_item: pg.LinearRegionItem | None = None
        self.sigma_band_item: pg.LinearRegionItem | None = None
        self.sigma_guide_items: tuple[pg.InfiniteLine, ...] = ()
        self._xlabel = ""
        self._ylabel = ""
        self._fit_range: tuple[float, float] | None = None
        self._placeholder_text: str | None = None
        self._nav_mode = "pan"
        # A reference grid lets a reader read values off the curve, matching the
        # grid the matplotlib draw_* functions drew.
        self.plot_item.showGrid(x=True, y=True, alpha=theme.LIGHT_PLOT_PALETTE.grid[3])
        # 图内的注解都挂在场景上、都不参与 autoRange（ignoreBounds），由一个 range-change
        # 回调钉到各自该在的位置，于是平移缩放时跟着走——这和当年 matplotlib 那套
        # axes-fraction 文本是一个意思。颜色一律取自下面套上的画板调色板而不是写死的灰：
        # 浅色桌面上一段浅字读不出来，正是这张面板的 matplotlib 前身修过的毛病。
        #
        # 设计稿把两段说明画在图里而不是图例或状态栏里：拟合窗口那行压在色带顶端居中，
        # ±1σ 贴在残差那条浅带的左上角。挨着自己说明的东西，读者不用再去别处找对应。
        window_font = self.font()
        window_font.setPixelSize(FIT_RANGE_CAPTION_PX)
        self.fit_range_label_item = pg.TextItem(anchor=(0.5, 0))
        self.fit_range_label_item.setZValue(20)
        self.fit_range_label_item.setFont(window_font)
        self.fit_range_label_item.hide()
        self.addItem(self.fit_range_label_item, ignoreBounds=True)
        sigma_font = self.font()
        sigma_font.setPixelSize(SIGMA_LABEL_PX)
        self.sigma_label_item = pg.TextItem("±1σ", anchor=(0, 1))
        self.sigma_label_item.setZValue(20)
        self.sigma_label_item.setFont(sigma_font)
        self.sigma_label_item.hide()
        self.addItem(self.sigma_label_item, ignoreBounds=True)
        self._placeholder_item = pg.TextItem(anchor=(0.5, 0.5))
        self._placeholder_item.setZValue(20)
        self._placeholder_item.hide()
        self.addItem(self._placeholder_item, ignoreBounds=True)
        self._palette = theme.LIGHT_PLOT_PALETTE
        # pyqtgraph offers to divide the tick values by a power of ten and print
        # the factor beside the axis label.  Every axis here is already labelled
        # with its unit, so that rescaling silently contradicts the label: a qz
        # axis spanning 0..0.4 Å⁻¹ comes out ticked 0..400 captioned "(x0.001)",
        # and the number a reader lifts off the axis is a thousand times the
        # quantity the label names.  Ticks stay in the labelled unit.
        for side in ("left", "bottom"):
            axis = self.plot_item.getAxis(side)
            axis.enableAutoSIPrefix(False)
            # 设计稿每一格都有网格线，但隔一格才写数字。pyqtgraph 默认给次级刻度也配
            # 标签（maxTextLevel 是 2，只有挤不下时才让位），那会把奇数量级、半整数角度
            # 一并写上，轴上挤成一列连续数字，反倒看不出刻度的节奏。
            axis.setStyle(maxTextLevel=0)
        # 纵轴钉死宽度，叠放的两块面板才共用一条左边界（见 theme.PLOT_LEFT_GUTTER_PX）。
        self.plot_item.getAxis("left").setWidth(theme.PLOT_LEFT_GUTTER_PX)
        # Crosshair: two thin lines + coordinate readout, hidden until hover.
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self._crosshair_label = pg.TextItem(anchor=(0, 1))
        self._crosshair_label.setZValue(30)
        for line in (self._crosshair_v, self._crosshair_h):
            line.setZValue(30)
            self.addItem(line, ignoreBounds=True)
        self.addItem(self._crosshair_label, ignoreBounds=True)
        self._set_crosshair_visible(False)
        self.apply_palette(theme.current_plot_palette())
        self.plot_item.vb.sigRangeChanged.connect(self._reposition_annotations)
        self.plot_item.vb.sigResized.connect(self._reposition_annotations)
        self.plot_item.vb.sigRangeChangedManually.connect(self._mirror_x_range)
        self.scene().sigMouseMoved.connect(self._on_scene_moved)
        self.scene().sigMouseClicked.connect(self._on_scene_clicked)

    def _set_observed_style(self, *, diameter: float, connected: bool) -> None:
        """观测点的直径，以及要不要给它连线。

        反射率三张只画点，残差那张点上还压一条细线。线是 ``observed_item`` 自己的画笔，
        所以每张面板都得把自己要哪一种说全——只在残差里打开、别处不关掉，切回反射率就会
        留着上一张的连线。
        """
        self.observed_item.setSymbolSize(diameter)
        if not connected:
            self.observed_item.setPen(None)
            return
        # 设计稿那条残差折线是 ``stroke-width:1.3`` 配 ``opacity:0.55``：线只负责把相邻点
        # 串起来看趋势，读数还是落在点上，所以线要压得比点淡。
        line = pg.mkColor(theme.DATA_OBSERVED)
        line.setAlpha(140)
        self.observed_item.setPen(pg.mkPen(line, width=1.3))

    def _apply_labels(self, xlabel: str, ylabel: str) -> None:
        # No in-plot title: the pane is framed by a titled plot card, and
        # pyqtgraph draws its title inside the axes -- indented by the y-axis
        # width, so it aligns with neither the card header above nor the plot
        # below.  Leaving it unset also gives the picture back the 30px band
        # PlotItem.setTitle reserves for the label.
        self._xlabel, self._ylabel = xlabel, ylabel
        self.plot_item.setLabel("bottom", xlabel)
        self.plot_item.setLabel("left", ylabel)
        # 换轴就得重排说明里的单位：同一条带子留在图上、面板改画另一个量，
        # 说明却还报着上一根轴的单位，那是在替读者读错。
        self._refresh_fit_range_caption()

    def axis_labels(self) -> tuple[str, str]:
        """Return the drawn (x-label, y-label) for parity assertions."""
        return (self._xlabel, self._ylabel)

    def apply_palette(self, palette: theme.PlotPalette) -> None:
        """Paint the pane's structural colours from a resolved theme palette.

        A pg pane is drawn outside the Qt stylesheet and is not reached by
        ``diagnostics.apply_figure_palette``, so — like the matplotlib figures —
        it must be handed the palette explicitly and re-read it on every draw;
        otherwise it keeps pyqtgraph's default black canvas when the desktop
        switches appearance.  The Okabe-Ito data-series hues are left fixed on
        purpose: a curve that changed colour with the desktop would invalidate
        the reference a reader built ("the blue curve is my data").
        """
        self._palette = palette
        self.setBackground(QColor.fromRgbF(*palette.background))
        foreground = QColor.fromRgbF(*palette.foreground)
        muted = QColor.fromRgbF(*palette.muted)
        for side in ("left", "bottom"):
            axis = self.plot_item.getAxis(side)
            # Tick-value text and the axis label take the foreground; the axis
            # line and ticks take the quieter muted, matching the matplotlib
            # spines/tick_params split in apply_figure_palette.
            axis.setTextPen(foreground)
            axis.setPen(muted)
        # 拟合窗口那行字读的底其实是画布：色带只有一成不透明，压不动读者眼里的亮度。
        # 所以它的颜色跟着画布两端各挑一支，而 ±1σ 是注解，取和占位文字同一档的 muted。
        self.fit_range_label_item.setColor(theme.range_label_colour(QColor.fromRgbF(*palette.background).name()))
        self.sigma_label_item.setColor(muted)
        self._placeholder_item.setColor(muted)
        self.mask_hover_item.setSymbolPen(pg.mkPen(foreground, width=1.6))
        self._apply_crosshair_pen()
        self.plot_item.showGrid(x=True, y=True, alpha=palette.grid[3])

    def background_color(self) -> tuple[float, float, float, float]:
        """Read back the painted canvas background as an RGBA 0..1 tuple.

        QColor stores channels at reduced precision, so this round-trips to only
        ~1e-6 of the palette tuple; callers compare with a tolerance, not ==.
        """
        return self.backgroundBrush().color().getRgbF()

    def set_observed(self, angles: object, values: object) -> None:
        self.observed_item.setData(np.asarray(angles, dtype=float), np.asarray(values, dtype=float))

    def set_model(self, angles: object | None, values: object | None) -> None:
        if angles is None or values is None:
            self.model_item.setData([], [])
            return
        self.model_item.setData(np.asarray(angles, dtype=float), np.asarray(values, dtype=float))

    def show_log_reflectivity(
        self,
        angles: object,
        observed: object,
        model: object | None,
        *,
        r_floor: float,
        mask: object | None = None,
    ) -> None:
        """Project the reflectivity family onto the log axis with draw_log's floor.

        ``r_floor`` is a presentation clamp, not a fit input: a measured or a
        modelled point below it reads at the floor rather than diverging down the
        log axis, so this pane and the matplotlib ``draw_log`` render the same data
        identically.  Every point the clamp moved is also marked, so the floor stays
        legible as a floor.  A missing candidate leaves the model curve empty rather
        than drawing a flat line at the floor.

        ``mask`` 是拟合掩膜：掩掉的点改画成灰色 ✕（和原始页同一种记号），不再和拟合点混成同一种蓝点。
        省略时视作全部参与拟合。
        """
        self.apply_palette(theme.current_plot_palette())
        floor = float(r_floor)
        x = np.asarray(angles, dtype=float)
        values = np.asarray(observed, dtype=float)
        shown = np.maximum(values, floor)
        keep = np.ones(x.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        excluded = ~keep & np.isfinite(x) & np.isfinite(shown)
        self.set_observed(x[keep], shown[keep])
        self.excluded_item.setData(x[excluded], shown[excluded])
        self._set_mask_targets(x, shown, log_axis=True)
        clipped = np.isfinite(values) & (values < floor)
        self.clipped_item.setData(x[clipped], np.full_like(x[clipped], floor))
        if model is None:
            self.set_model(None, None)
        else:
            self.set_model(x, np.maximum(np.asarray(model, dtype=float), floor))
        # 显示下限不写进轴标题：被夹到下限的点已经由 ▽ 空心标记就地标出（clipped_item），
        # 轴标题只回答「这条轴是什么量」。
        self._apply_labels(ANGLE_AXIS_LABEL, "反射率 R（归一化 · 对数）")
        self._set_observed_style(diameter=REFLECTIVITY_MARKER_PX, connected=False)
        self._set_residual_edges([], [])
        self._set_sigma_reference(False)
        self._hide_placeholder()
        self.set_log_mode(True)

    def show_raw_reflectivity(
        self,
        angles: object,
        raw: object,
        mask: object,
        model: object | None,
    ) -> None:
        """Render stored angle / raw intensity with fit and excluded points split.

        The raw family reads on a linear axis: raw intensity spans well under a
        decade, so the log projection the ``show_log_reflectivity`` pane applies
        would only crush it against the baseline.  Included and excluded points
        live on separate artists so the mask a reader clicked reads back visually.
        """
        self.apply_palette(theme.current_plot_palette())
        x = np.asarray(angles, dtype=float)
        y = np.asarray(raw, dtype=float)
        keep = np.asarray(mask, dtype=bool)
        finite = np.isfinite(x) & np.isfinite(y)
        included = finite & keep
        excluded = finite & ~keep
        self.observed_item.setData(x[included], y[included])
        self.excluded_item.setData(x[excluded], y[excluded])
        self._set_mask_targets(x, y, log_axis=False)
        # The clipped glyph belongs to the log floor alone; a linear axis shows a
        # small intensity as a small intensity, with nothing clamped to mark.
        self.clipped_item.setData([], [])
        if model is None:
            self.set_model(None, None)
        else:
            self.set_model(x, np.asarray(model, dtype=float))
        self._apply_labels(ANGLE_AXIS_LABEL, "原始强度")
        self._set_observed_style(diameter=REFLECTIVITY_MARKER_PX, connected=False)
        self._set_residual_edges([], [])
        self._set_sigma_reference(False)
        self._hide_placeholder()
        self.set_log_mode(False)

    def show_qz4(
        self,
        data_qz: object,
        data_values: object,
        model_qz: object | None,
        model_values: object | None,
        *,
        ylabel: str = "qz⁴R",
    ) -> None:
        """Render the qz⁴R diagnostic transform on a linear axis.

        The qz⁴ weighting is computed upstream (it needs overflow-aware scaling),
        so this pane only plots the ready pairs and shows the ``ylabel`` that
        projection chose, which records the scaling reference when it had to
        normalize.  A missing candidate leaves the model curve empty rather than
        drawing a bare data series as a fit.
        """
        self.apply_palette(theme.current_plot_palette())
        self.observed_item.setData(np.asarray(data_qz, dtype=float), np.asarray(data_values, dtype=float))
        self.excluded_item.setData([], [])
        self.clipped_item.setData([], [])
        if model_qz is None or model_values is None:
            self.set_model(None, None)
        else:
            self.set_model(np.asarray(model_qz, dtype=float), np.asarray(model_values, dtype=float))
        self._apply_labels("散射矢量 qz (Å⁻¹)", ylabel)
        self._set_observed_style(diameter=REFLECTIVITY_MARKER_PX, connected=False)
        self._set_residual_edges([], [])
        self._set_sigma_reference(False)
        self._hide_placeholder()
        self.set_log_mode(False)

    def show_residual(
        self, angles: object, weighted: object, *, ylabel: str = "加权残差 (σ)", sigma_reference: bool = True
    ) -> None:
        """Render the weighted residual as a marked line above a zero baseline.

        横轴与它上面那张反射率卡同为散射角，同一个点在两张卡上落在同一条竖线上。有 σ 带时点钉在
        ±``RESIDUAL_CLAMP_SIGMA`` 以内，越界的换成朝外的空心三角；没有 σ 带（非高斯噪声）时不钉——
        那时纵轴上没有一把「几个 σ」的尺子可保护。
        """
        self.apply_palette(theme.current_plot_palette())
        x = np.asarray(angles, dtype=float)
        y = np.asarray(weighted, dtype=float)
        limit = RESIDUAL_CLAMP_SIGMA if sigma_reference else np.inf
        beyond = np.isfinite(y) & (np.abs(y) > limit)
        self.observed_item.setData(x, np.clip(y, -limit, limit))
        self._set_residual_edges(x[beyond], np.sign(y[beyond]) * limit)
        self.reference_item.setData(x, np.zeros_like(x))
        self.set_model(None, None)
        self.excluded_item.setData([], [])
        self.clipped_item.setData([], [])
        self._apply_labels(ANGLE_AXIS_LABEL, ylabel)
        self._set_observed_style(diameter=RESIDUAL_MARKER_PX, connected=True)
        self._set_sigma_reference(sigma_reference)
        self._hide_placeholder()
        self.set_log_mode(False)

    def _set_residual_edges(self, angles: object, values: object) -> None:
        # 每个点的三角朝向随正负变，所以记号是逐点给的一列；清空时也得把记号列一起换成空的，
        # 否则 pyqtgraph 拿上一次的那列去配这一次的点。
        y = np.asarray(values, dtype=float)
        symbols = ["t1" if value > 0.0 else "t" for value in y]
        self.residual_edge_item.setData(np.asarray(angles, dtype=float), y, symbol=symbols)

    def clear_series(self) -> None:
        """Empty every managed curve for the panel's no-data state."""
        self.observed_item.setData([], [])
        self.model_item.setData([], [])
        self.excluded_item.setData([], [])
        self.reference_item.setData([], [])
        self.clipped_item.setData([], [])
        self._set_residual_edges([], [])
        self._set_mask_targets([], [], log_axis=False)

    def _fit_range_caption_text(self, span: tuple[float, float]) -> str:
        """ "拟合窗口 <低>–<高><单位>"，单位取自当前横轴标题末尾的括号。"""
        low, high = span
        match = FINAL_UNIT.search(self._xlabel)
        unit = match.group(1) if match else ""
        suffix = unit if unit in TIGHT_UNITS else (f" {unit}" if unit else "")
        return f"拟合窗口 {low:g}–{high:g}{suffix}"

    def _refresh_fit_range_caption(self) -> None:
        if self._released:
            return
        if self._fit_range is None:
            self.fit_range_label_item.hide()
            return
        self.fit_range_label_item.setText(self._fit_range_caption_text(self._fit_range))
        self.fit_range_label_item.show()
        self._reposition_annotations()

    def fit_range_caption(self) -> str | None:
        """图上那行拟合窗口说明，没有带子时是 None。"""
        if self._fit_range is None:
            return None
        return self.fit_range_label_item.toPlainText()

    def show_placeholder(self, text: str) -> None:
        """Empty the pane and centre a placeholder, matching draw_empty.

        A pane with no data or no candidate reads as an explicit "暂无…" note
        rather than a blank canvas, and the stale fit-range band goes with it: a
        window drawn over nothing still claims a range was fitted here.
        """
        self.apply_palette(theme.current_plot_palette())
        self.clear_series()
        self.clear_fit_range()
        self._placeholder_text = text
        muted = QColor.fromRgbF(*self._palette.muted)
        html = (
            f'<div style="text-align:center; color:{muted.name()};"'
            f'><p style="font-size:24px; margin:0;">📈</p>'
            f'<p style="font-size:11px; margin:4px 0 0 0;">{text}</p></div>'
        )
        self._placeholder_item.setHtml(html)
        self._placeholder_item.show()
        self._reposition_annotations()

    def placeholder_text(self) -> str | None:
        return self._placeholder_text

    def _hide_placeholder(self) -> None:
        self._placeholder_text = None
        self._placeholder_item.hide()

    def _reposition_annotations(self, *_args: object) -> None:
        view_box = self.plot_item.vb
        # 藏在别的 tab 里的面板是零尺寸，那时的视图变换是奇异的：量出来的像素既没有意义，pyqtgraph
        # 求逆时还会落进 NumPy 2.5 弃用的那条路。等它有了尺寸（``sigResized``）再落位。
        size = view_box.size()
        if size.width() <= 0.0 or size.height() <= 0.0:
            return
        (x0, x1), (y0, y1) = view_box.viewRange()
        px_x, px_y = view_box.viewPixelSize()
        self._placeholder_item.setPos((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        # 拟合窗口那行字跟着色带露在视野里的那一段居中。带子被平移出去一半时，说明仍压在
        # 看得见的那半段上；写死在区间中点的话，缩到一角就跟着飘出视野了。
        if self._fit_range is not None:
            low, high = self._fit_range
            centre = (max(low, x0) + min(high, x1)) / 2.0
            self.fit_range_label_item.setPos(centre, y1 - theme.SPACE_XS * px_y)
        # ±1σ 落在带子顶边（也就是 +1σ）之上、贴着左边界，和设计稿一致。带子本身是常数
        # ±1，所以这个纵坐标不随缩放变。
        self.sigma_label_item.setPos(x0 + theme.SPACE_XS * px_x, 1.0)

    def show_fit_range(self, low: float, high: float) -> None:
        """Shade the committed fit range as a read-only band.

        This is the presentation twin of the matplotlib ``axvspan`` mirror, not
        the drag selector: the band is not user-movable and does not arm the
        interactive handles, so a read-only pane can show the range it fitted.
        """
        lo, hi = sorted((float(low), float(high)))
        if self.fit_range_item is None:
            fill = pg.mkColor(theme.DATA_RANGE)
            fill.setAlpha(41)
            region = pg.LinearRegionItem(
                values=(lo, hi),
                brush=pg.mkBrush(fill),
                pen=pg.mkPen(theme.DATA_RANGE),
                movable=False,
            )
            region.setZValue(-10)
            self.addItem(region)
            self.fit_range_item = region
        else:
            self.fit_range_item.setRegion((lo, hi))
        self._fit_range = (lo, hi)
        # 选框开着时也跟上：另一页拖完、或项目刚提交了范围，这一页的选框得落在同一段上。屏蔽信号，
        # 免得这次跟随被当成读者拖动、又提交一遍。
        if self.range_item is not None:
            self.range_item.blockSignals(True)
            self.range_item.setRegion((lo, hi))
            self.range_item.blockSignals(False)
        self._refresh_fit_range_caption()

    def clear_fit_range(self) -> None:
        item = self.fit_range_item
        self.fit_range_item = None
        self._fit_range = None
        self._refresh_fit_range_caption()
        if item is None or self._released:
            return
        self.removeItem(item)

    def _set_sigma_reference(self, visible: bool) -> None:
        """±1σ 带与 ±2σ 虚线：残差图的绝对刻度，装一次然后只开关可见性。

        残差已经除过 σ，所以「大不大」有绝对答案，可纵轴是随这一轮残差自动缩放的——同一个
        形状在量级差十倍时画出来一模一样。常数 ±1 的浅带把那把尺子画回图上：带内是噪声量级，
        越出 ±2 就该解释。一秒里残差要重画很多次，所以带和参考线只在第一次要用时创建，
        之后靠可见性切换，免得每帧长出一份新的图元。
        """
        if self._released:
            return
        if visible and self.sigma_band_item is None:
            # 设计稿那块浅底是 ``--d-observed`` 走 ``opacity:0.08``，也就是 8% 的观测色；
            # 它是读数背景而不是选区，所以不可拖动，并压在数据下面。
            fill = pg.mkColor(theme.DATA_OBSERVED)
            fill.setAlpha(20)
            band = pg.LinearRegionItem(
                values=(-1.0, 1.0),
                orientation="horizontal",
                brush=pg.mkBrush(fill),
                pen=pg.mkPen(None),
                movable=False,
            )
            band.setZValue(-20)
            self.addItem(band)
            self.sigma_band_item = band
            guide_colour = pg.mkColor(theme.DATA_NEUTRAL)
            guide_colour.setAlpha(64)
            guides = []
            for value in (-2.0, 2.0):
                # 虚线，因为这是提示线：实线只留给零那条基准。
                line = pg.InfiniteLine(
                    pos=value,
                    angle=0,
                    pen=pg.mkPen(guide_colour, width=1.0, style=Qt.PenStyle.DashLine),
                    movable=False,
                )
                line.setZValue(-15)
                self.addItem(line)
                guides.append(line)
            self.sigma_guide_items = tuple(guides)
        self.sigma_label_item.setVisible(visible)
        if self.sigma_band_item is not None:
            self.sigma_band_item.setVisible(visible)
            for guide in self.sigma_guide_items:
                guide.setVisible(visible)

    def set_view_xrange(self, low: float, high: float) -> None:
        self.plot_item.vb.setXRange(float(low), float(high), padding=0)

    def autoscale_view(self) -> None:
        self.plot_item.vb.autoRange()

    def set_navigation_mode(self, mode: str) -> None:
        """Select pan (drag translates) or zoom (drag rubber-bands a rectangle).

        These are the pg twins of the matplotlib pan and zoom toolbar buttons, so
        the same toolbar can drive whichever backend renders the visible tab.
        """
        vb = self.plot_item.vb
        if mode == "zoom":
            self._nav_mode = "zoom"
            vb.setMouseMode(pg.ViewBox.RectMode)
        else:
            self._nav_mode = "pan"
            vb.setMouseMode(pg.ViewBox.PanMode)

    def navigation_mode(self) -> str:
        return self._nav_mode

    def go_home(self) -> None:
        """Restore the latest draw's data bounds, the pg twin of the home button."""
        self.plot_item.vb.autoRange()
        self._mirror_x_range([True, False])

    def share_x_range(self, peers: tuple[LiveReflectivityPlot, ...]) -> None:
        """读者在这张上缩放、平移横轴时，``peers`` 跟着落到同一段角度。

        不用 ``setXLink``：它按两只 ViewBox 在屏上的像素位置对齐，而这几张卡有一张总在另一个 tab 里，
        几何是陈旧的或者干脆是零，对出来的量程是错的；它还会让被联动的那张永远停掉自动量程，换数据集
        时一直停在上一个数据集的窗口上。这里只转发读者的手势：程序画数据各自自动量程——横轴是同一组
        角度，量出来本来就一样。
        """
        self._x_peers = tuple(peer for peer in peers if peer is not self)

    def _mirror_x_range(self, mask: object) -> None:
        if self._released or not mask[0]:
            return
        low, high = self.plot_item.vb.viewRange()[0]
        for peer in self._x_peers:
            # ``setXRange`` 是程序改量程，不会再发 ``sigRangeChangedManually``，所以转发不会来回弹。
            if not peer._released:
                peer.set_view_xrange(low, high)

    def _set_crosshair_visible(self, visible: bool) -> None:
        for item in (self._crosshair_v, self._crosshair_h, self._crosshair_label):
            item.setVisible(visible)

    def _apply_crosshair_pen(self) -> None:
        muted = QColor.fromRgbF(*self._palette.muted)
        muted.setAlpha(120)
        pen = pg.mkPen(muted, width=1.0, style=Qt.PenStyle.DashLine)
        self._crosshair_v.setPen(pen)
        self._crosshair_h.setPen(pen)
        self._crosshair_label.setColor(QColor.fromRgbF(*self._palette.muted))

    def _emit_cursor(self, x_view: float, y_view: float) -> None:
        if self._released:
            return
        # A log-mode viewbox carries y in log10 space; report the physical value so
        # the readout matches what the reader reads off the axis.
        y = 10.0**y_view if self.plot_item.ctrl.logYCheck.isChecked() else y_view
        self.cursor_moved.emit(float(x_view), float(y))

    def _emit_cursor_left(self) -> None:
        if self._released:
            return
        self.cursor_left.emit()

    def _on_scene_moved(self, scene_pos: object) -> None:
        if self._released:
            return
        if self.plot_item.sceneBoundingRect().contains(scene_pos):
            point = self.plot_item.vb.mapSceneToView(scene_pos)
            self._crosshair_v.setPos(point.x())
            self._crosshair_h.setPos(point.y())
            y_display = 10.0 ** point.y() if self.plot_item.ctrl.logYCheck.isChecked() else point.y()
            self._crosshair_label.setText(f"{point.x():.4g}, {y_display:.4g}")
            self._crosshair_label.setPos(point.x(), point.y())
            self._set_crosshair_visible(True)
            self._emit_cursor(point.x(), point.y())
        else:
            self._set_crosshair_visible(False)
            self._emit_cursor_left()
        self._set_mask_hover(self._target_at(scene_pos) if self._masking else None)

    def set_preview(self, angles: object, values: object) -> bool:
        """Publish the searching model into one owned curve; False once released."""
        if self._released:
            return False
        x = np.asarray(angles, dtype=float)
        y = np.asarray(values, dtype=float)
        if self.preview_item is None:
            self.preview_item = self.plot(x, y, pen=pg.mkPen(theme.DATA_PREVIEW, width=2.5), name="搜索中模型")
        else:
            self.preview_item.setData(x, y)
        return True

    def clear_preview(self) -> None:
        item = self.preview_item
        self.preview_item = None
        if item is None or self._released:
            return
        self.removeItem(item)

    def set_overlay_datasets(
        self,
        datasets: tuple[tuple[str, object, object], ...],
    ) -> None:
        """Draw other datasets' curves as an overlay comparison.

        Each entry is ``(label, angles, values)``.  The n-th curve is drawn at
        ×10⁻ⁿ so the stack separates by decades on the log axis, and the factor
        is written into its key caption -- an unlabelled offset would read as a
        real intensity difference.  Colour and dash come from ``overlay_style``,
        which keeps clear of the hues this pane already spends on its own marks.
        """
        self.clear_overlay()
        if self._released:
            return
        keys: list[tuple[str, str, str]] = []
        for index, (label, angles, values) in enumerate(datasets):
            colour, dash = overlay_style(index)
            caption = label if index == 0 else f"{label} ×10{str(-index).translate(SUPERSCRIPT_DIGITS)}"
            item = self.plot(
                np.asarray(angles, dtype=float),
                np.asarray(values, dtype=float) * 10.0**-index,
                pen=pg.mkPen(colour, width=1.2, style=PEN_STYLES[dash]),
                name=caption,
            )
            self._overlay_items.append(item)
            keys.append((caption, colour, dash))
        self._overlay_keys = tuple(keys)

    def overlay_keys(self) -> tuple[tuple[str, str, str], ...]:
        """叠加曲线在色标里的 ``(caption, colour, dash)``，与画的先后一致。"""
        return self._overlay_keys

    def clear_overlay(self) -> None:
        """Remove all overlay curves."""
        for item in self._overlay_items:
            if not self._released:
                self.removeItem(item)
        self._overlay_items.clear()
        self._overlay_keys = ()

    def set_log_mode(self, enabled: bool) -> None:
        self.plot_item.setLogMode(y=bool(enabled))

    def enable_range_selection(self, enabled: bool) -> None:
        if enabled:
            if self.range_item is None:
                brush = pg.mkColor(theme.DATA_RANGE)
                brush.setAlpha(50)
                # 选框从当前范围起手：此前每次都落在 0–1，读者得先把它从角落里拖回来。没有范围就取此刻的
                # 视野，不缩不扩，坐标系不在指针底下变。构造时就给定区间，信号这时还没接，不会凭空提交一次。
                region = pg.LinearRegionItem(
                    values=self._fit_range or tuple(self.plot_item.vb.viewRange()[0]),
                    brush=pg.mkBrush(brush),
                    pen=pg.mkPen(theme.DATA_RANGE),
                )
                region.sigRegionChangeFinished.connect(self._emit_range)
                vb = self.plot_item.vb
                self._range_accepted_mouse_buttons = vb.acceptedMouseButtons()
                # The region handles must own the drag gesture: ViewBox accepts the
                # press at the scene level, so its mouseDragEvent would otherwise
                # swallow a range-handle drag.  Remove that competing target for the
                # duration of range mode and restore the exact button set later.
                # ``mouseEnabled`` stays as the reader left it: the wheel reads only
                # that flag, and zooming in to place a handle is what range mode is for.
                vb.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                region.setZValue(10)
                # A selector is an overlay: including it in autoRange moves the
                # coordinate system while its handles are being dragged.
                self.addItem(region, ignoreBounds=True)
                self.range_item = region
            return
        if self.range_item is not None:
            self.removeItem(self.range_item)
            self.range_item = None
        accepted_mouse_buttons = self._range_accepted_mouse_buttons
        self._range_accepted_mouse_buttons = None
        if accepted_mouse_buttons is not None:
            self.plot_item.vb.setAcceptedMouseButtons(accepted_mouse_buttons)

    def _apply_theme_palette(self) -> None:
        """Refresh an existing pyqtgraph pane after the application appearance flips."""
        if not self._released:
            self.apply_palette(theme.current_plot_palette())

    def enable_masking(self, enabled: bool) -> None:
        self._masking = bool(enabled)
        if not self._masking:
            self._set_mask_hover(None)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._x_peers = ()
        for signal, slot in (
            (self.scene().sigMouseClicked, self._on_scene_clicked),
            (self.scene().sigMouseMoved, self._on_scene_moved),
            (self.plot_item.vb.sigRangeChangedManually, self._mirror_x_range),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                # Idempotent teardown: the scene C++ object may already be gone, or
                # the slot may have been disconnected by an earlier release.
                pass

    def _emit_range(self, *_args: object) -> None:
        region = self.range_item
        if region is None:
            return
        low, high = sorted(float(value) for value in region.getRegion())
        self.fit_range_selected.emit(low, high)

    def _set_mask_targets(self, angles: object, values: object, *, log_axis: bool) -> None:
        """记下掩膜点按得中的点：画在屏上的那些（有限值；对数轴上还得是正值），下标对齐面板收到的数组。"""
        x = np.asarray(angles, dtype=float)
        y = np.asarray(values, dtype=float)
        drawable = np.isfinite(x) & np.isfinite(y)
        if log_axis:
            drawable &= y > 0.0
        self._mask_targets = (np.flatnonzero(drawable), x[drawable], y[drawable])
        self._set_mask_hover(None)

    def _target_at(self, scene_pos: object) -> int | None:
        """指针底下那个点在 ``_mask_targets`` 里的位置；``MASK_PICK_RADIUS_PX`` 以内没有点时为 None。

        按屏幕像素量距离而不是按数据单位：两条轴的单位不同，对数轴上更是一个频程一个样。
        """
        indices, x, y = self._mask_targets
        view_box = self.plot_item.vb
        size = view_box.size()
        if not indices.size or size.width() <= 0.0 or size.height() <= 0.0:
            return None
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return None
        point = view_box.mapSceneToView(scene_pos)
        y_view = np.log10(y) if self.plot_item.ctrl.logYCheck.isChecked() else y
        px_x, px_y = view_box.viewPixelSize()
        distance = np.hypot((x - point.x()) / px_x, (y_view - point.y()) / px_y)
        nearest = int(np.argmin(distance))
        return nearest if distance[nearest] <= MASK_PICK_RADIUS_PX else None

    def point_index_at(self, scene_pos: object) -> int | None:
        """指针底下那个测量点的下标（对齐面板收到的数组）；没有点中任何点时为 None。"""
        target = self._target_at(scene_pos)
        return None if target is None else int(self._mask_targets[0][target])

    def _set_mask_hover(self, target: int | None) -> None:
        """掩膜模式下给指针底下那个点套圈、换成手形光标；离开所有点就撤掉。"""
        if target == self._hover_target:
            return
        self._hover_target = target
        if target is None:
            self.mask_hover_item.setData([], [])
            self.unsetCursor()
            return
        _indices, x, y = self._mask_targets
        self.mask_hover_item.setData(x[target : target + 1], y[target : target + 1])
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _on_scene_clicked(self, event: object) -> None:
        if self._released or not self._masking or event.button() != Qt.MouseButton.LeftButton:
            return
        index = self.point_index_at(event.scenePos())
        if index is not None:
            self.point_mask_requested.emit(index)
