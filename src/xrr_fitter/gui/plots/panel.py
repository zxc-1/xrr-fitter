"""Transactional Qt projection of prepared data and fit diagnostics.

Every mutation validates a complete scratch projection before changing live
artists. A failed live draw restores the committed projection and all SLD view
state. SLD bands are persisted evidence; alternate alignments are view-only
cache entries owned by one dataset and one MCMC report. Dataset or report
changes invalidate that cache, while ordinary redraws reuse it.

The panel publishes Python state only after a successful draw. Project and
legacy dataset transitions therefore restore their dictionaries, structure,
selector, toggle, and cache together. Preview artists remain outside this
transaction and are discarded by the next full projection; teardown owns the
figures and callbacks idempotently.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.canvas_top import canvas_tab_group
from xrr_fitter.gui.plots.diagnostics import (
    ANALYSIS_KEYS,
    REFLECTIVITY_KEYS,
    RESIDUAL_KEY,
    TAB_SPECS,
    VIEW_SPECS,
    DiagnosticView,
    apply_plot_card_captions,
    build_residual_companion_pane,
    build_scratch_views,
    build_tabs,
    draw_batch_trends,
    draw_candidate_comparison,
    draw_empty,
    release_scratch_views,
    residual_tab_host,
    validate_batch_trends,
)
from xrr_fitter.gui.plots.heatmaps import draw_parameter_heatmap, draw_residual_heatmap
from xrr_fitter.gui.plots.interactions import (
    PlotInteractionController,
    PlotInteractionToolbar,
    ordered_finite_range,
)
from xrr_fitter.gui.plots.live import LiveReflectivityPlot
from xrr_fitter.gui.plots.reflectivity import (
    draw_log,
    draw_qz4,
    draw_raw,
    draw_residual,
    prepare_project_plots,
    preview_display_values,
    reflectivity_pane_arrays,
    validate_plot_data,
    validate_result,
)
from xrr_fitter.gui.plots.sld import draw_sld, draw_uncertainty
from xrr_fitter.gui.plots.sld_state import (
    ALIGN_KEYS,
    BatchTrends,
    Projection,
    SldBandReplay,
    SldViewState,
    alignment_index_from_cache,
    build_sld_companion_pane,
    cache_matches,
    candidate_for_result,
    capture_sld_view_state,
    committed_projection,
    comparison_candidates,
    current_projection,
    project_structure,
    projection_bands,
    projection_mcmc,
    reset_band_view,
    restore_sld_view_state,
    set_alignment_index,
    sync_band_controls,
    visible_bands,
)

# 画布在每一步露哪几段。设计稿六帧里的画布一律是两张卡：帧①「反射率 + 加权残差」、
# 帧③「层堆叠 + SLD 深度剖面」、帧④「总进度 + 实时反射率」、帧⑤「参数相关矩阵 +
# Profile 似然」。绘图栈的四段此前全都常驻，按 3:2:2:2 分同一个 ~800px 的画布，每段落到
# 180px 上下——装不下一张带坐标轴和图例的图：帧⑤ 的两张子图叠在一起连标签都读不出来，
# 帧③ 的 SLD 剖面干脆被挤到折叠线以下。右栏早就按步收窄了（``STEP_INSPECTOR_SECTIONS``，
# 那儿的注释记着「五段内容高约 1900px、视口 815px，2.3 倍」），这里是同一个毛病的另一半。
#
# 键是 ``PIPELINE_STEPS`` 的下标：0 数据 / 1 结构 / 2 参数 / 3 拟合 / 4 结果 / 5 导出。
# 0-2 这三步画布上半是层堆叠（``STEPS_WITH_LAYER_STACK``），所以下半只留一段。
#
STEP_PLOT_PANES: dict[int, tuple[str, ...]] = {
    # 数据：还没有结构可画，这一步要看的就是刚导进来的那条曲线。
    0: ("reflectivity",),
    # 结构（帧③）：画布是「层堆叠 + SLD 深度剖面」两张卡，层堆叠由结构面板出，这里只出剖面。
    # 反射率与残差在这一步画的都是上一轮的旧曲线，摆在层堆叠下面会被读成「改完之后的样子」。
    1: ("sld",),
    # 参数：仍在编辑同一个模型，剖面留着——边界是照着它设的。
    2: ("sld",),
    # 拟合中（帧④）：只有正在动的那条实时曲线。残差和候选解画的都是上一轮的收敛结果，
    # 和进度摆在一起会被读成本轮读数——这和右栏在运行中只剩控制那一段是同一个理由。
    3: ("reflectivity",),
    # 结果（帧①）：反射率与它的加权残差。残差紧跟它判读的那张图。
    4: ("reflectivity", "residual"),
    5: ("reflectivity", "residual"),
}

# 剖面画「正在编辑的这套结构」而不是「上一轮拟合出来的候选」的那两步，键同样是
# ``PIPELINE_STEPS`` 的下标。取 ``STEP_PLOT_PANES`` 里只有 ``sld`` 的两步：那时画布上半是
# 层堆叠，改一层下半当场跟着动，而跟着动的只可能是标称结构那条线。
STRUCTURE_EDIT_STEPS = (1, 2)

# 结果态选中分析页时画布整块换成分析页（帧⑤ 那一屏：相关矩阵 + Profile 似然，两张卡都在
# 分析页自己那张图里）。分析页是另一种看法而不是第三张图：反射率与残差一起让位给它。
#
# 让位也带走模式条——它是反射率页的角落控件。设计稿帧③ 与帧⑤ 的 ``.canvas-top`` 里确实都
# 没有模式条：帧③ 那一行是三个结构按钮，帧⑤ 只有标签页。所以这不是牺牲，是照着做。剩下的
# 那几张 matplotlib 静态图仍能滚轮缩放（``scroll_event`` 直接连在画布上），而返回反射率那
# 一组走菜单「视图」——它不经过标签页，收起标签页不会把人困住。
ANALYSIS_STEPS = (4, 5)

# 分析页至少要有的高度。不确定度那一页自己再上下切成两张子图，坐标轴、刻度、图例都得读得
# 出来；低于这个数就回到叠字那种状态，所以这是下限而不是某一次量到的像素值。
ANALYSIS_MIN_H = 300


#
# Empty state widget
#
def _empty_state_widget(panel: PlotPanel) -> QWidget:
    widget = QWidget(panel)
    widget.setObjectName("plotEmptyState")
    title = QLabel("尚未导入数据")
    title.setObjectName("emptyStateTitle")
    title.setProperty("emptyTitle", True)
    title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    body = QLabel("导入 .xy / .dat / .txt 反射率数据后，这里将显示曲线、SLD 剖面与拟合诊断。")
    body.setProperty("mutedText", True)
    body.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    body.setWordWrap(True)
    button = QPushButton("导入数据文件…")
    button.setObjectName("emptyStateImportButton")
    button.setProperty("primary", True)
    button.clicked.connect(panel.import_requested.emit)
    hint = QLabel("也可以使用菜单「文件 ▸ 导入数据文件…」或快捷键 Ctrl+I")
    hint.setProperty("mutedText", True)
    hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(button)
    row.addStretch(1)
    layout = QVBoxLayout(widget)
    layout.setSpacing(theme.SPACE_MD)
    layout.addStretch(2)
    layout.addWidget(title)
    layout.addWidget(body)
    layout.addLayout(row)
    layout.addWidget(hint)
    layout.addStretch(3)
    return widget


class PlotPanel(QWidget):
    """Render one selected dataset only after a complete drawing preflight."""

    fit_range_requested = Signal(float, float)
    point_mask_requested = Signal(int)
    view_changed = Signal(int)
    import_requested = Signal()
    # Emitted when an interface handle on the SLD pane is dragged: the payload is
    # the edited StructureSpec, committed by the window through the structure
    # editor so a hand edit and a typed edit take exactly the same path.
    structure_edit_requested = Signal(object)

    # 指针不在曲线上时这条读数带什么都不写。设计稿在残差卡和状态栏之间没有常驻提示：
    # 一句永远挂着的操作说明占的是数据的位置，而读数带空着本身就说明「现在没有读数」。
    CURSOR_IDLE_HINT = ""

    #
    # Init
    #
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotPanel")
        self.setAccessibleName("反射率、SLD 与拟合诊断")
        self._datasets: dict[str, api.PreparedData] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._dataset_id: str | None = None
        self._result: object | None = None
        self._candidate_id: str | None = None
        self._structure: object | None = None
        self._sld_band_cache: SldBandReplay | None = None
        self._trends: BatchTrends | None = None
        self._visible_range: tuple[float, float] | None = None
        self._released = False
        # 画布当前按哪一步收窄。None 表示还没有人告诉过它（独立构造的面板、以及旧测试），
        # 这时四段按老样子全露——收窄是窗口接线加上去的，不是面板自己的默认姿势。
        self._step: int | None = None
        # 专家模式开没开。同样用 None 表示「还没有人说过」：此前 SLD 剖面的可见性只由
        # ``interactions._apply_tabs`` 写，没跑过它的独立面板剖面就是露着的，None 保住这个行为。
        self._expert_mode: bool | None = None
        self.toolbar = PlotInteractionToolbar(self)
        self.reflectivity_tabs, self.analysis_tabs, self._views = build_tabs()
        # Backward-compatible alias: external code that references panel.tabs
        # (workspace findChild, some tests) gets the reflectivity group.
        self.tabs = self.reflectivity_tabs
        # 设计稿 帧①/⑤ 的画布也以 ``.canvas-top`` 开头（HTML 148-152）：一条扁 tab 条，
        # 紧跟着这一页的动作。``QTabWidget`` 自带的 tab 是带边框的小盒子，而它唯一的行内
        # 附加位 ``setCornerWidget`` 只有「贴右边框」这一种摆法——模式条会离开它作用的那四
        # 个 tab 六百多像素。所以两组都收进 ``canvas_tab_group``：容器退成页面壳，露在外面
        # 的是那一行；模式条排在 tab 后面的动作位上。
        self.reflectivity_group, self.reflectivity_canvas_tabs, reflectivity_actions = canvas_tab_group(
            self.reflectivity_tabs,
            parent=self,
            name="reflectivityCanvas",
            row_name="reflectivityCanvasTop",
            tabs_name="reflectivityCanvasTabs",
        )
        reflectivity_actions.addWidget(self.toolbar)
        self.analysis_group, self.analysis_canvas_tabs, _analysis_actions = canvas_tab_group(
            self.analysis_tabs,
            parent=self,
            name="analysisCanvas",
            row_name="analysisCanvasTop",
            tabs_name="analysisCanvasTabs",
        )
        self.residual_pane = build_residual_companion_pane(self._views)
        # 「加权残差」那一页的空壳，和它在 splitter 里的那一格。卡只有一份，在这两处之间搬。
        self._residual_host = residual_tab_host(self.reflectivity_tabs)
        self._residual_slot = 1
        self.sld_pane, self.sld_bands_toggle, self.sld_align_selector = build_sld_companion_pane(
            self,
            self._views,
            self._on_bands_toggled,
            self._on_align_changed,
        )
        # 四段竖排，顺序固定：反射率页 / 加权残差 / 分析页 / SLD 深度剖面。残差紧挨着它
        # 判读的那张图，隔两段的复核是不会有人看的复核。哪几段露由 ``set_step_scope``
        # 按步骤定（``STEP_PLOT_PANES``）——四段同时在场时每段只有 180px，那是叠字，不是图。
        # 顺序在这里排一次，换步时就不必把控件从 splitter 里摘下来重插。
        # 例外只有残差这一格：选中「加权残差」那个 tab 时，那张卡搬去当那一页的正文
        # （见 ``_sync_residual_home``），离开时再插回 ``_residual_slot``。
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.plot_splitter.setObjectName("plotSplitter")
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot_splitter.addWidget(self.reflectivity_group)
        self.plot_splitter.addWidget(self.residual_pane)
        self.plot_splitter.addWidget(self.analysis_group)
        self.plot_splitter.addWidget(self.sld_pane)
        # 设计稿 帧① 的两张卡是 345 : 199.8 高（≈ 7:4）。3:2 把残差压到 199 以下，那条
        # 曲线在 ±3σ 的参考线之间只剩几个像素可摆。帧③/④/⑤ 的画布列各只剩 24px 富余，
        # 帧① 有 274px——那段白是它右栏太高撑出来的 CSS grid 拉伸，不是给某一张卡的高度，
        # 所以这里对的是比例而不是绝对值。
        self.plot_splitter.setStretchFactor(0, 7)
        self.plot_splitter.setStretchFactor(1, 4)
        self.plot_splitter.setStretchFactor(2, 4)
        self.plot_splitter.setStretchFactor(3, 4)
        self.analysis_tabs.setMinimumHeight(ANALYSIS_MIN_H)
        content = QWidget(self)
        content.setObjectName("plotContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(theme.SPACE_SM)
        content_layout.addWidget(self.plot_splitter, 1)
        self.cursor_readout = QLabel(self.CURSOR_IDLE_HINT, content)
        self.cursor_readout.setObjectName("plotCursorReadout")
        self.cursor_readout.setProperty("mutedText", True)
        content_layout.addWidget(self.cursor_readout)
        self._pages = QStackedLayout(self)
        self._pages.addWidget(_empty_state_widget(self))
        self._pages.addWidget(content)
        self._sync_pages()
        self._interactions = PlotInteractionController(self, self.toolbar)
        self._install_view_shortcuts()
        self.toolbar.overlay_toggled.connect(self._on_overlay_toggled)
        # 换视图要重算画布范围：结果态选到分析页时残差让位给分析页（见 ``ANALYSIS_STEPS``）。
        # 挂在 ``view_changed`` 上而不是两个 tab 的 currentChanged 上，因为菜单「视图」也走
        # ``select_view``，那条路不经过 tab 的点击。
        self.view_changed.connect(self._on_view_changed_scope)
        # 控制器就位后才问得出「此刻选中的是哪一段」，所以残差卡的落脚点在这里对一次。
        self._sync_residual_home()

    #
    # Install view shortcuts
    #
    def _install_view_shortcuts(self) -> None:
        """Bind one Alt+N per diagnostic tab, addressed by visible position.

        Users switch among the diagnostic plots constantly; clicking or cycling
        with Ctrl+Tab is slow.  The keys count the tabs, not the views: the SLD
        profile is a companion pane that never leaves the screen, so it is not
        addressable and consumes no number.  The weighted residual counts,
        because it is both -- a tab of its own *and* the card pinned under the
        strip; the one card moves between the two homes.
        Numbering by visible position rather than by fixed view key means a
        hidden tab shifts the keys up instead of leaving a key that lands on
        nothing and a gap in the sequence.
        """
        self.view_shortcuts: list[QShortcut] = []
        for position in range(len(self.tab_keys())):
            shortcut = QShortcut(QKeySequence(f"Alt+{position + 1}"), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setProperty("viewPosition", position)
            # Bind the position through the sender rather than a lambda closing
            # over self; a self-capturing closure held by the shortcut's signal
            # forms a cycle PySide cannot break, leaking the panel on teardown.
            shortcut.activated.connect(self._view_shortcut_activated)
            self.view_shortcuts.append(shortcut)

    #
    # View shortcut activated
    #
    def _view_shortcut_activated(self) -> None:
        shortcut = self.sender()
        if shortcut is not None:
            self.select_visible_view(int(shortcut.property("viewPosition")))

    #
    # Select visible view
    #
    def select_visible_view(self, position: int) -> bool:
        """Select the Nth (0-based) currently-visible diagnostic view."""
        visible: list[str] = []
        for i in range(self.reflectivity_tabs.count()):
            if self.reflectivity_tabs.isTabVisible(i):
                visible.append(REFLECTIVITY_KEYS[i])
        for i in range(self.analysis_tabs.count()):
            if self.analysis_tabs.isTabVisible(i):
                visible.append(ANALYSIS_KEYS[i])
        if not 0 <= position < len(visible):
            return False
        self.select_view(visible[position])
        return True

    #
    # Sync pages
    #
    def _sync_pages(self) -> None:
        self._pages.setCurrentIndex(0 if self._dataset_id is None else 1)
        self._sync_analysis_visibility()
        # Every path that commits a new active dataset or a new result ends here, so
        # this is the one place the plot-card captions have to be rewritten -- doing
        # it per caller would leave whichever path was added last showing a stale
        # name, or a stale χ²ᵥ on the residual card.
        apply_plot_card_captions(
            self,
            self._dataset_id,
            self._result,
            candidate_for_result(self._result, self._candidate_id),
        )

    #
    # Set step scope
    #
    def set_step_scope(self, step: int | None) -> None:
        """按流程步收窄画布，只留这一步该露的那两段（``STEP_PLOT_PANES``）。

        ``None`` 交回四段全露的老姿势，给独立构造的面板用。窗口那边由
        ``window_layout.apply_step_scope`` 在同一处调用，和右栏收窄同进同退。

        换了步还要重画一次：剖面在结构那两步画的是正在编辑的结构、别的步画的是拟合候选
        （``_sld_follows_edits``），只改可见性的话，从「结果」回到「结构」看到的还是结果态
        那张图，而它不会跟着层堆叠动。
        """
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be int or None")
        changed = step != self._step
        self._step = step
        self._sync_pane_scope()
        if changed and not self._released and self._dataset_id is not None:
            self._transact(self._current_projection())

    #
    # Canvas pane keys
    #
    def canvas_pane_keys(self) -> tuple[str, ...]:
        """当前这一步画布该露的段，按 splitter 里的固定顺序。

        报的是「这一步的画布是什么」，不是「此刻屏幕上有什么」：没有数据集时整块面板停在
        空态页上，那时每个子控件都读作不可见，拿可见性反推步骤范围会得到一个和步骤无关的答
        案。空态与「这一步该露哪几段」是两件事，所以分开报。
        """
        if self._step is None:
            return ("reflectivity", "residual", "analysis", "sld")
        if self._step in ANALYSIS_STEPS and self._analysis_group_is_active():
            return ("analysis",)
        return STEP_PLOT_PANES.get(self._step, STEP_PLOT_PANES[0])

    #
    # Analysis group is active
    #
    def _analysis_group_is_active(self) -> bool:
        """分析页是否是当前选中的那一组视图。

        交互控制器是在 ``__init__`` 末尾才建起来的，而第一次 ``_sync_pages`` 在它之前就跑
        了，所以这里不能假定它已经在场。
        """
        if getattr(self, "_interactions", None) is None:
            return False
        return self.current_view_key() in ANALYSIS_KEYS

    #
    # Active analysis view
    #
    def active_analysis_view(self) -> str | None:
        """此刻选中的分析页是哪一页；不在分析组里（或控制器还没建起来）时为 ``None``。

        右栏按视图挑段落时要问的正是这一句。``current_view_key`` 单独用不了：控制器缺席时它
        直接抛，而反射率那几页选中时它报的键也不该拿去和分析页的名字比。
        """
        if not self._analysis_group_is_active():
            return None
        return self.current_view_key()

    #
    # On view changed scope
    #
    def _on_view_changed_scope(self, _index: int) -> None:
        self._sync_residual_home()
        self._sync_pane_scope()

    #
    # Residual is docked
    #
    def _residual_is_docked(self) -> bool:
        """残差卡此刻是不是正当「加权残差」这一页的正文。"""
        return self._residual_host.layout().indexOf(self.residual_pane) != -1

    #
    # Sync residual home
    #
    def _sync_residual_home(self) -> None:
        """把那张唯一的残差卡放到此刻该在的位置。

        设计稿 帧① 两处都要它：tab 条上有「加权残差」，画布下半段又钉着这张卡。一个控件
        只有一个父件，真做两份就是画两遍、各记一套缩放状态——读者在 tab 里放大完回到下半
        段会看到另一个视野，而两边本该是同一条曲线。所以选中这一段时卡搬进那一页当正文，
        其余时候搬回 splitter 里紧跟 tab 组的那一格。
        """
        if getattr(self, "_interactions", None) is None:
            return
        if self.current_view_key() == RESIDUAL_KEY:
            if not self._residual_is_docked():
                self._residual_host.layout().addWidget(self.residual_pane)
            # 钉在下半段时可能被步骤范围关掉过；搬进来当正文就得亮着，之后由 tab 决定这
            # 一页露不露。
            self.residual_pane.show()
            return
        if self.plot_splitter.indexOf(self.residual_pane) == -1:
            self.plot_splitter.insertWidget(self._residual_slot, self.residual_pane)
            self.plot_splitter.setStretchFactor(self._residual_slot, 4)

    #
    # Sync pane scope
    #
    def _sync_pane_scope(self) -> None:
        """把 ``canvas_pane_keys`` 落到四段的可见性上。

        分析页还有一道自己的闸，和步骤是「与」而不是覆盖：没有结果时它是四张空图，哪一步
        都不该露。SLD 剖面那一道闸交给 ``_sld_is_wanted`` 判。
        """
        wanted = self.canvas_pane_keys()
        # 收的是整组（行头 + 页面），不是单独那个页面壳：只藏页面会留下一条孤零零的 tab
        # 条，读者点得动它却看不到任何结果。
        self.reflectivity_group.setVisible("reflectivity" in wanted)
        # 残差卡当着「加权残差」那一页的正文时，露不露由 tab 说了算：读者刚点开这一段，
        # 步骤范围没把它算进来就把它关掉，点开的会是一页空白。
        if not self._residual_is_docked():
            self.residual_pane.setVisible("residual" in wanted)
        self.sld_pane.setVisible(self._sld_is_wanted(wanted))
        self.analysis_group.setVisible("analysis" in wanted and self._result is not None)

    def _sld_is_wanted(self, wanted: tuple[str, ...]) -> bool:
        """SLD 剖面此刻该不该在画布上。

        没有步骤作用域时（裸面板、以及步骤化之前那套四段常驻布局）沿用专家模式那道闸：那时
        四段同屏，剖面是可以省掉的一段，省掉了画布上还剩三张图。

        有步骤作用域时由步骤说了算。设计稿帧③ 的画布就是层堆叠加剖面这两张卡，对应
        ``STEP_PLOT_PANES`` 里步骤 1/2 只有 ``sld`` 这一项——再压一层专家模式，默认项目
        （``expert_mode`` 出厂是关的）走到结构那一步会拿到一整块空白画布，而这一步的全部
        看点就是「改一层，剖面当场跟着动」。
        """
        if "sld" not in wanted:
            return False
        return self._step is not None or self._expert_mode is not False

    def _sld_follows_edits(self, projection: Projection) -> bool:
        """剖面此刻画的是「读者正在编辑的结构」还是「上一轮拟合出来的候选」。

        结构与参数这两步（``STRUCTURE_EDIT_STEPS``）画布上半是层堆叠，改一层下半当场跟着
        动——跟着动的只可能是标称结构那条线，拟合候选不会因为改了一层而变。设计稿帧③ 的图区
        因此只有一条剖面、它的带和几个手柄：候选（连它的虚部、以及折进图例的「其他候选 实部
        ×N」那一行）画在这儿会被读成「改完之后的样子」，和反射率、残差在这一步退场是同一个
        理由。

        没有步骤作用域（裸面板、以及步骤化之前那套四段常驻布局）时照旧画候选。
        """
        return self._step in STRUCTURE_EDIT_STEPS and projection.structure is not None

    def set_expert_pane_scope(self, enabled: bool) -> None:
        """记下专家模式，然后重算两段的可见性。

        由 ``interactions._apply_tabs`` 调，取代它此前直接 ``sld_pane.setVisible(expert_mode)``：
        可见性只在 ``_sync_pane_scope`` 一处落地，两道闸才谈得上「与」。
        """
        self._expert_mode = bool(enabled)
        self._sync_pane_scope()

    #
    # Sync analysis visibility
    #
    def _sync_analysis_visibility(self) -> None:
        """Hide the analysis pane when no fit result exists for the active dataset."""
        self._sync_pane_scope()

    #
    # Tab titles
    #
    def tab_titles(self) -> tuple[str, ...]:
        ref = tuple(self.reflectivity_tabs.tabText(i) for i in range(self.reflectivity_tabs.count()))
        ana = tuple(self.analysis_tabs.tabText(i) for i in range(self.analysis_tabs.count()))
        return ref + ana

    #
    # Tab keys
    #
    def tab_keys(self) -> tuple[str, ...]:
        """The switchable diagnostic tabs, in tab-bar order (both groups)."""
        return tuple(key for key, _title, _description in TAB_SPECS)

    #
    # Reflectivity tab keys
    #
    def reflectivity_tab_keys(self) -> tuple[str, ...]:
        """Tab keys belonging to the reflectivity group."""
        return REFLECTIVITY_KEYS

    #
    # Analysis tab keys
    #
    def analysis_tab_keys(self) -> tuple[str, ...]:
        """Tab keys belonging to the analysis/diagnostic group."""
        return ANALYSIS_KEYS

    #
    # View keys
    #
    def view_keys(self) -> tuple[str, ...]:
        """Every owned view, including the companion pane outside the tab bar."""
        return tuple(key for key, _title, _description in VIEW_SPECS)

    #
    # View
    #
    def view(self, key: str) -> DiagnosticView | LiveReflectivityPlot:
        try:
            return self._views[key]
        except KeyError as error:
            raise KeyError(f"unknown diagnostic view: {key}") from error

    #
    # Selected dataset id
    #
    def selected_dataset_id(self) -> str | None:
        return self._dataset_id

    #
    # Selected candidate id
    #
    def selected_candidate_id(self) -> str | None:
        return self._candidate_id

    #
    # Current view key
    #
    def current_view_key(self) -> str:
        return self._interactions.current_view_key()

    #
    # Select view
    #
    def select_view(self, key: str) -> None:
        self._interactions.select_view(key)

    #
    # Set expert mode
    #
    def set_expert_mode(self, enabled: bool) -> None:
        self._interactions.set_expert_mode(enabled)

    #
    # Apply workspace
    #
    def apply_workspace(self, *, expert_mode: bool, tab_index: int, analysis_tab_index: int = 0) -> None:
        self._interactions.apply_workspace(expert_mode, tab_index, analysis_tab_index)

    #
    # Mode buttons
    #
    def mode_buttons(self) -> dict[str, object]:
        return self.toolbar.buttons()

    #
    # Mode actions
    #
    def mode_actions(self) -> dict[str, object]:
        """三档交互模式的 ``QAction``——状态住在这里，不在按钮上。

        设计稿的 ``.modebar`` 只给「范围」留了字形，查看与掩膜从条的右键菜单进，所以问
        「模式现在是哪一档」得问 action，``mode_buttons()`` 只剩留在条上的那一枚。
        """
        return self.toolbar.mode_actions()

    #
    # Navigation buttons
    #
    def navigation_buttons(self) -> dict[str, object]:
        return self.toolbar.navigation_buttons()

    #
    # Navigation mode
    #
    def navigation_mode(self) -> str:
        return self._interactions.navigation_mode()

    #
    # Navigators
    #
    def navigators(self) -> dict[str, object]:
        return self._interactions.navigators()

    #
    # Interaction mode
    #
    def interaction_mode(self) -> str:
        return self.toolbar.mode()

    #
    # Set interaction mode
    #
    def set_interaction_mode(self, mode: str) -> None:
        self.toolbar.set_mode(mode)

    #
    # Set dataset
    #
    def set_dataset(self, dataset_id: str, data: api.PreparedData) -> None:
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset id must be nonempty")
        if not isinstance(data, api.PreparedData):
            raise TypeError("data must be PreparedData")
        mask = validate_plot_data(data, data.fit_mask)
        projection = Projection(
            data,
            mask,
            None,
            None,
            self._trends,
            self._visible_range,
            dataset_id,
            None,
        )
        previous_context = (
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._structure,
            self._sld_band_cache,
            self.sld_align_selector.currentIndex(),
        )
        self._reset_sld_band_view(projection)
        try:
            self._transact(projection)
        except Exception:
            (
                self._datasets,
                self._masks,
                self._dataset_id,
                self._result,
                self._candidate_id,
                self._structure,
                self._sld_band_cache,
                alignment_index,
            ) = previous_context
            set_alignment_index(self.sld_align_selector, alignment_index)
            raise
        self._datasets = {**self._datasets, dataset_id: data}
        self._masks = {**self._masks, dataset_id: mask}
        self._dataset_id = dataset_id
        self._result = None
        self._candidate_id = None
        self._structure = None
        self._sync_pages()

    #
    # Select dataset
    #
    def select_dataset(self, dataset_id: str) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"unknown dataset: {dataset_id}")
        projection = Projection(
            self._datasets[dataset_id],
            self._masks[dataset_id],
            None,
            None,
            self._trends,
            self._visible_range,
            dataset_id,
            None,
        )
        previous = (
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._structure,
            self._sld_band_cache,
            self.sld_align_selector.currentIndex(),
        )
        self._reset_sld_band_view(projection)
        try:
            self._transact(projection)
        except Exception:
            (
                self._dataset_id,
                self._result,
                self._candidate_id,
                self._structure,
                self._sld_band_cache,
                alignment_index,
            ) = previous
            set_alignment_index(self.sld_align_selector, alignment_index)
            raise
        self._dataset_id = dataset_id
        self._result = None
        self._candidate_id = None
        self._structure = None
        self._sync_pages()

    #
    # Update mask
    #
    def update_mask(self, dataset_id: str, mask: object) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"unknown dataset: {dataset_id}")
        converted = validate_plot_data(self._datasets[dataset_id], mask)
        if dataset_id != self._dataset_id:
            self._masks[dataset_id] = converted
            return
        projection = Projection(
            self._datasets[dataset_id],
            converted,
            self._result,
            self._candidate_id,
            self._trends,
            self._visible_range,
            dataset_id,
            self._structure,
        )
        self._transact(projection)
        self._masks[dataset_id] = converted

    #
    # Set result
    #
    def set_result(self, result: object, candidate_id: str | None) -> None:
        data = self._active_data()
        validate_result(data, result)
        candidate_for_result(result, candidate_id)
        previous_projection = self._current_projection()
        projection = Projection(
            data,
            self._active_mask(),
            result,
            candidate_id,
            self._trends,
            self._visible_range,
            self._dataset_id,
            self._structure,
        )
        current_report = projection_mcmc(self._result)
        report_changed = projection_mcmc(projection.result) is not current_report
        previous_sld_state = capture_sld_view_state(
            self._sld_band_cache,
            self.sld_bands_toggle,
            self.sld_align_selector,
        )
        if report_changed:
            self._reset_sld_band_view(projection)
        self._transact(
            projection,
            rollback_projection=previous_projection,
            rollback_sld_state=previous_sld_state,
        )
        self._result = result
        self._candidate_id = candidate_id
        # 残差卡的抬头写这次拟合的 χ²ᵥ，所以换了结果就得重写一遍抬头；``_sync_pages``
        # 里已经含 ``_sync_analysis_visibility``。
        self._sync_pages()

    #
    # Set batch trends
    #
    def set_batch_trends(
        self,
        dataset_ids: tuple[str, ...],
        thickness_a: tuple[float, ...],
        period_a: tuple[float, ...],
    ) -> None:
        trends = (tuple(dataset_ids), tuple(thickness_a), tuple(period_a))
        validate_batch_trends(*trends)
        projection = self._current_projection(trends=trends)
        self._transact(projection)
        self._trends = trends

    #
    # Select fit range
    #
    def select_fit_range(self, first: float, second: float) -> bool:
        return self._interactions.select_fit_range(first, second)

    #
    # Request point mask
    #
    def request_point_mask(self, index: int) -> bool:
        return self._interactions.request_point_mask(index)

    #
    # Show range
    #
    def show_range(self, lower: float, upper: float) -> None:
        visible = ordered_finite_range(lower, upper)
        projection = self._current_projection(visible_range=visible)
        self._transact(projection)
        self._visible_range = visible

    #
    # Visible range
    #
    def visible_range(self) -> tuple[float, float] | None:
        return self._visible_range

    #
    # Zoom to range
    #
    def zoom_to_range(self) -> bool:
        """Focus the angle-domain views on the active fit range.

        The fit range is often a small window of a wide scan, so keeping the
        full sweep on screen buries the region the user is actually judging.
        This clamps the raw and log x-axes to the highlighted range; it is a
        pure view operation that leaves the committed projection untouched, so
        the next redraw restores the full sweep on its own.
        """
        visible = self._visible_range
        if visible is None or self._released or self._dataset_id is None:
            return False
        for key in ("raw", "log"):
            view = self._views[key]
            if isinstance(view, LiveReflectivityPlot):
                view.set_view_xrange(*visible)
        return True

    #
    # Reset zoom
    #
    def reset_zoom(self) -> bool:
        """Return the angle-domain views to their data-driven autoscale."""
        if self._released or self._dataset_id is None:
            return False
        for key in ("raw", "log"):
            view = self._views[key]
            if isinstance(view, LiveReflectivityPlot):
                view.autoscale_view()
        return True

    #
    # Cancel interaction
    #
    def cancel_interaction(self) -> None:
        self._interactions.cancel()

    #
    # Clear visible range
    #
    def _clear_visible_range(self) -> None:
        projection = self._current_projection(visible_range=None)
        self._transact(projection)
        self._visible_range = None

    #
    # Displayed prepared indices
    #
    def displayed_prepared_indices(self) -> tuple[int, ...]:
        data = self._active_data()
        finite = np.isfinite(data.two_theta_deg) & np.isfinite(data.intensity_raw)
        return tuple(int(index) for index in np.flatnonzero(finite))

    #
    # Callback counts
    #
    def callback_counts(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return self._interactions.callback_counts()

    #
    # Resources released
    #
    def resources_released(self) -> bool:
        return self._released

    #
    # Set cursor readout
    #
    def set_cursor_readout(self, message: str) -> None:
        """Show the pointer's coordinates, or the standing hint when it leaves.

        Matplotlib clears the read-out with an empty string as the pointer
        leaves the axes, so an empty message restores the idle hint rather than
        letting a stale reading linger as if the pointer were still on the curve.
        """
        if self._released:
            return
        self.cursor_readout.setText(message if message else self.CURSOR_IDLE_HINT)

    #
    # Project project
    #
    def project_project(self, project: api.XrrProject) -> None:
        prepared = prepare_project_plots(project)
        candidate_for_result(prepared.result, prepared.candidate_id)
        projection = Projection(
            prepared.data,
            prepared.mask,
            prepared.result,
            prepared.candidate_id,
            self._trends,
            self._visible_range,
            prepared.dataset_id,
            project_structure(project, prepared.dataset_id),
        )
        previous_dataset_id = self._dataset_id
        previous_report = projection_mcmc(self._result)
        previous_context = (
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._structure,
            self._sld_band_cache,
            self.sld_align_selector.currentIndex(),
        )
        previous_projection = self._committed_projection()
        previous_sld_state = capture_sld_view_state(
            self._sld_band_cache,
            self.sld_bands_toggle,
            self.sld_align_selector,
        )
        self._datasets, self._masks = prepared.datasets, prepared.masks
        self._dataset_id = prepared.dataset_id
        self._structure = projection.structure
        if prepared.dataset_id != previous_dataset_id or projection_mcmc(projection.result) is not previous_report:
            self._reset_sld_band_view(projection)
        try:
            self._transact(
                projection,
                rollback_projection=previous_projection,
                rollback_sld_state=previous_sld_state,
            )
        except Exception:
            (
                self._datasets,
                self._masks,
                self._dataset_id,
                self._result,
                self._candidate_id,
                self._structure,
                self._sld_band_cache,
                alignment_index,
            ) = previous_context
            set_alignment_index(self.sld_align_selector, alignment_index)
            raise
        self._result = prepared.result
        self._candidate_id = prepared.candidate_id
        self._sync_pages()
        self.apply_workspace(
            expert_mode=project.ui_state.expert_mode,
            tab_index=project.ui_state.plot_tab_index,
            # 分析组的序号也要跟着投，否则每次投射都替读者按一次分析组的第一个 tab：换视图
            # 自己就会绕回这里（``view_changed`` → ``_capture_workspace`` → 换项目 → 投射），
            # 画布停在读者选的那一张、屏上那条 tab 条却退回开头。
            analysis_tab_index=project.ui_state.analysis_tab_index,
        )

    #
    # Active data
    #
    def _active_data(self) -> api.PreparedData:
        if self._dataset_id is None:
            raise RuntimeError("no active plot dataset")
        return self._datasets[self._dataset_id]

    #
    # Active mask
    #
    def _active_mask(self) -> np.ndarray:
        if self._dataset_id is None:
            raise RuntimeError("no active plot dataset")
        return self._masks[self._dataset_id]

    #
    # Reset sld band view
    #
    def _reset_sld_band_view(self, projection: Projection) -> None:
        """Drop a view-only replay when its dataset or MCMC owner changes."""
        self._sld_band_cache = None
        reset_band_view(projection_bands(projection.result), self.sld_align_selector)

    #
    # On bands toggled
    #
    def _on_bands_toggled(self) -> None:
        if self._released or self._dataset_id is None:
            return
        if self.sld_bands_toggle.isChecked():
            self._on_align_changed(self.sld_align_selector.currentIndex())
            return
        self._transact(self._current_projection())

    #
    # On align changed
    #
    def _on_align_changed(self, index: int) -> None:
        """Recompute the bands for the picked alignment as a view-only overlay."""
        if self._released or self._dataset_id is None or self._structure is None or not 0 <= index < len(ALIGN_KEYS):
            return
        projection = self._current_projection()
        report = projection_mcmc(projection.result)
        if report is None or not self.sld_bands_toggle.isChecked():
            return
        alignment = ALIGN_KEYS[index]
        if cache_matches(self._sld_band_cache, self._dataset_id, report, alignment):
            self._transact(projection)
            return
        previous_index = alignment_index_from_cache(
            self._sld_band_cache,
            projection_bands(projection.result),
            self.sld_align_selector.itemText(1),
        )
        previous_sld_state = capture_sld_view_state(
            self._sld_band_cache,
            self.sld_bands_toggle,
            self.sld_align_selector,
        )
        rollback_sld_state = replace(previous_sld_state, alignment_index=previous_index)
        try:
            bands = api.sld_uncertainty_bands(
                self._structure,
                report,
                wavelength_a=self._active_data().beam.effective_wavelength_a,
                align=alignment,
            )
            self._sld_band_cache = SldBandReplay(self._dataset_id, report, alignment, bands)
            self._transact(
                projection,
                rollback_projection=self._committed_projection(),
                rollback_sld_state=rollback_sld_state,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError):
            self._sld_band_cache = restore_sld_view_state(
                rollback_sld_state,
                self.sld_bands_toggle,
                self.sld_align_selector,
            )

    #
    # On overlay toggled
    #
    def _on_overlay_toggled(self, checked: bool) -> None:
        """Show or hide all non-active dataset curves on the log pane."""
        view = self._views["log"]
        if not isinstance(view, LiveReflectivityPlot):
            return
        if not checked:
            view.clear_overlay()
            return
        if self._dataset_id is None or len(self._datasets) < 2:
            view.clear_overlay()
            return
        entries: list[tuple[str, object, object]] = []
        for did, data in self._datasets.items():
            if did == self._dataset_id:
                continue
            label = data.source_path.stem if data.source_path else did
            entries.append((label, data.two_theta_deg, data.intensity_normalized))
        view.set_overlay_datasets(tuple(entries))

    #
    # Current projection
    #
    def _current_projection(self, **changes: object) -> Projection:
        return current_projection(
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._trends,
            self._visible_range,
            self._structure,
            changes,
        )

    #
    # Set preview curve
    #
    def set_preview_curve(
        self,
        qz_a_inv: object,
        model_normalized: object,
    ) -> bool:
        """Overlay the searching model on the log view without a projection.

        A live preview updates many times per search, so it pushes into one
        owned pyqtgraph curve instead of running the transactional redraw. It
        carries no committed evidence and is discarded whenever a real
        projection lands.
        """
        if self._dataset_id is None or self._released:
            return False
        angles, values = preview_display_values(
            self._active_data(),
            qz_a_inv,
            model_normalized,
        )
        view = self._views["log"]
        if not isinstance(view, LiveReflectivityPlot):
            return False
        return view.set_preview(angles, values)

    #
    # Clear preview curve
    #
    def clear_preview_curve(self) -> None:
        """Drop the live overlay so committed evidence renders on its own."""
        if self._released:
            return
        view = self._views["log"]
        if isinstance(view, LiveReflectivityPlot):
            view.clear_preview()

    #
    # Transact
    #
    def _transact(
        self,
        projection: Projection,
        *,
        rollback_projection: Projection | None = None,
        rollback_sld_state: SldViewState | None = None,
    ) -> None:
        if self._released:
            raise RuntimeError("plot panel resources have been released")
        # A full projection supersedes the live preview overlay. The pyqtgraph
        # log pane keeps its preview curve across show_log_reflectivity (unlike
        # the old axes.clear() path), so the overlay is dropped explicitly here.
        self.clear_preview_curve()
        scratch = build_scratch_views()
        try:
            self._draw(scratch, projection)
        finally:
            release_scratch_views(scratch)
        previous = self._committed_projection() if rollback_projection is None else rollback_projection
        try:
            self._draw(self._views, projection)
        except Exception:
            if rollback_sld_state is not None:
                self._sld_band_cache = restore_sld_view_state(
                    rollback_sld_state,
                    self.sld_bands_toggle,
                    self.sld_align_selector,
                )
            self._draw(self._views, previous)
            raise
        finally:
            # Whatever ended up on screen, the new projection or the rolled back
            # one, is the view the reset button has to return to.
            self._interactions.refresh_navigation_baselines()

    #
    # Committed projection
    #
    def _committed_projection(self) -> Projection:
        return committed_projection(
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._trends,
            self._visible_range,
            self._structure,
        )

    #
    # Draw
    #
    def _draw(
        self,
        views: dict[str, DiagnosticView | LiveReflectivityPlot],
        projection: Projection,
    ) -> None:
        data = projection.data
        candidate = candidate_for_result(projection.result, projection.candidate_id)
        bands = projection_bands(projection.result)
        sync_band_controls(
            self.sld_bands_toggle,
            self.sld_align_selector,
            bands=bands,
            has_structure=projection.structure is not None,
        )
        if data is None or projection.mask is None:
            for key in ("raw", "log", "qz4", "residual", "sld"):
                view = views[key]
                title = next(title for name, title, _description in VIEW_SPECS if name == key)
                if isinstance(view, LiveReflectivityPlot):
                    view.show_placeholder("暂无可用数据")
                    view.clear_fit_range()
                else:
                    draw_empty(view, title)
        else:
            self._draw_reflectivity_panes(views, data, projection.mask, candidate)
            shown = visible_bands(
                checked=self.sld_bands_toggle.isChecked(),
                cache=self._sld_band_cache,
                persisted=projection_bands(projection.result),
                dataset_id=projection.dataset_id,
                report=projection_mcmc(projection.result),
                alignment=ALIGN_KEYS[self.sld_align_selector.currentIndex()],
                surface_label=self.sld_align_selector.itemText(1),
            )
            # 结构那两步画的是读者手上这套结构，候选连同它折进图例的那一行都不进画面。
            edited = self._sld_follows_edits(projection)
            draw_sld(
                views["sld"],
                None if edited else candidate,
                () if edited else comparison_candidates(projection.result, projection.candidate_id),
                shown,
                structure=projection.structure,
                wavelength_a=data.beam.effective_wavelength_a,
            )
            self._draw_range(views, projection.visible_range)
        draw_candidate_comparison(views["candidates"], projection.result, projection.candidate_id)
        draw_residual_heatmap(views["residual_map"], projection.result, projection.candidate_id)
        draw_parameter_heatmap(views["parameter_map"], projection.result, projection.candidate_id)
        draw_uncertainty(views["uncertainty"], projection.result, projection.candidate_id)
        draw_batch_trends(views["trend"], projection.trends)

    #
    # Draw reflectivity panes
    #
    def _draw_reflectivity_panes(
        self,
        views: dict[str, DiagnosticView | LiveReflectivityPlot],
        data: api.PreparedData,
        mask: np.ndarray,
        candidate: object | None,
    ) -> None:
        """Draw the four reflectivity panes through whichever backend holds them.

        The scratch preflight dict is all-matplotlib, while the live dict holds
        pyqtgraph widgets for these four keys, so one projection reaches two
        rendering paths. The four panes always share a backend within a dict, so
        the log pane's type decides for all of them. The matplotlib path keeps
        the byte-identical draw_* functions; the live path computes the pane
        arrays once and pushes them into the pyqtgraph curves.
        """
        log = views["log"]
        if not isinstance(log, LiveReflectivityPlot):
            draw_raw(views["raw"], data, mask, candidate)
            draw_log(views["log"], data, candidate)
            draw_qz4(views["qz4"], data, candidate)
            draw_residual(views["residual"], candidate)
            return
        arrays = reflectivity_pane_arrays(data, mask, candidate)
        log.show_log_reflectivity(
            arrays.log_angles,
            arrays.log_observed,
            arrays.log_model,
            r_floor=arrays.r_floor,
        )
        raw = views["raw"]
        raw.show_raw_reflectivity(arrays.raw_angles, arrays.raw_intensity, arrays.raw_mask, arrays.raw_model)
        qz4 = views["qz4"]
        if arrays.qz4 is None:
            qz4.show_placeholder("暂无当前候选")
        else:
            qz4.show_qz4(*arrays.qz4, ylabel=arrays.qz4_ylabel or "qz⁴R")
        residual = views["residual"]
        if arrays.residual is None:
            residual.show_placeholder("暂无当前候选")
        else:
            residual.show_residual(*arrays.residual)

    #
    # Draw range
    #
    def _draw_range(
        self,
        views: dict[str, DiagnosticView | LiveReflectivityPlot],
        visible_range: tuple[float, float] | None,
    ) -> None:
        for key in ("raw", "log"):
            view = views[key]
            if isinstance(view, LiveReflectivityPlot):
                # pyqtgraph range items persist across show_*, so an absent
                # range must be cleared explicitly rather than relying on a
                # cleared axes.
                if visible_range is None:
                    view.clear_fit_range()
                else:
                    view.show_fit_range(*visible_range)
            elif visible_range is not None:
                view.axes.axvspan(*visible_range, color=theme.DATA_RANGE, alpha=0.16, label="拟合范围")
                view.canvas.draw_idle()

    #
    # Release resources
    #
    def release_resources(self) -> None:
        if self._released:
            return
        self._released = True
        self._interactions.release()
        for view in self._views.values():
            if isinstance(view, LiveReflectivityPlot):
                view.release()
            else:
                view.canvas.release()
                view.figure.clear()
                view.figure.set_canvas(None)
        self._datasets.clear()
        self._masks.clear()
        self._result = None
        self._candidate_id = None
        self._structure = None
        self._sld_band_cache = None

    #
    # Closeevent
    #
    def closeEvent(self, event: object) -> None:
        self.release_resources()
        super().closeEvent(event)

    #
    # Event
    #
    def event(self, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.DeferredDelete, QEvent.Type.Destroy):
            self.release_resources()
        elif event.type() == QEvent.Type.ParentChange:
            controller = getattr(self, "_interactions", None)
            if controller is not None:
                controller.watch_parent()
        return super().event(event)
