"""Transactional Qt projection of prepared data and fit diagnostics.

Every mutation computes the projection's fallible arrays before changing live
artists; nothing is rendered off screen to find out. A failed live draw restores
the committed projection and all SLD view state. Under a step scope a matplotlib
view that is not on screen is not redrawn: it is marked stale with the projection
it owes and drawn the moment it is shown or read. SLD bands are persisted
evidence; alternate alignments are view-only cache entries owned by one dataset
and one MCMC report. Dataset or report changes invalidate that cache, while
ordinary redraws reuse it.

The panel publishes Python state only after a successful draw. Project and
legacy dataset transitions therefore restore their dictionaries, structure,
selector, toggle, and cache together. Preview artists remain outside this
transaction and are discarded by the next full projection; teardown owns the
figures and callbacks idempotently.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
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
    REFLECTIVITY_KEYS,
    DiagnosticView,
    apply_plot_card_captions,
    build_residual_companion_pane,
    build_tabs,
    draw_batch_trends,
    draw_candidate_comparison,
    draw_empty,
    residual_tab_host,
    validate_batch_trends,
)
from xrr_fitter.gui.plots.heatmaps import draw_parameter_heatmap, draw_residual_heatmap
from xrr_fitter.gui.plots.interactions import PlotInteractionController, ordered_finite_range
from xrr_fitter.gui.plots.live import LiveReflectivityPlot
from xrr_fitter.gui.plots.pane_scope import PaneScopeMixin
from xrr_fitter.gui.plots.reflectivity import (
    PreparedDataCache,
    prepare_project_plots,
    reflectivity_pane_arrays,
    validate_plot_data,
    validate_result,
)
from xrr_fitter.gui.plots.sld import draw_sld, draw_uncertainty
from xrr_fitter.gui.plots.sld_controls import SldBandControlsMixin
from xrr_fitter.gui.plots.sld_state import (
    ALIGN_KEYS,
    BatchTrends,
    Projection,
    SldBandReplay,
    SldViewState,
    build_sld_companion_pane,
    candidate_for_result,
    capture_sld_view_state,
    check_new_bands,
    committed_projection,
    comparison_candidates,
    current_projection,
    project_structure,
    projection_bands,
    projection_mcmc,
    restore_sld_view_state,
    set_alignment_index,
    sync_band_controls,
    visible_bands,
)
from xrr_fitter.gui.plots.toolbar import PlotInteractionToolbar
from xrr_fitter.gui.plots.transient_series import TransientSeriesMixin
from xrr_fitter.gui.plots.view_controls import ViewControlsMixin

# 分析页至少要有的高度。不确定度那一页自己再上下切成两张子图，坐标轴、刻度、图例都得读得
# 出来；低于这个数就回到叠字那种状态，所以这是下限而不是某一次量到的像素值。
ANALYSIS_MIN_H = 300

# 可以欠着重画的视图：matplotlib 那几张。一次重画要清轴、重建刻度与文字，不在屏上时这些全是白做；
# 四张 pyqtgraph 卡只是换一份数组，当场就画。顺序即重画顺序：剖面在前，与此前一致。
DEFERRABLE_KEYS = ("sld", "candidates", "residual_map", "parameter_map", "uncertainty", "trend")

# 横轴都是散射角的三张卡：缩放、平移、复位、缩放到拟合范围都一起走，残差点才一直对得上它上面那条曲线。
ANGLE_PANE_KEYS = ("log", "raw", "residual")


# 空状态最后那句提示。按键不写死在这里：它由窗口按平台给出（``set_import_shortcut_text``），
# macOS 上是 ⌘I——写死的「Ctrl+I」照着按什么也不会发生。
IMPORT_HINT_TEXT = "也可以使用菜单「文件 ▸ 导入数据文件…」"


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
    folder_button = QPushButton("导入文件夹…")
    folder_button.setObjectName("emptyStateImportFolderButton")
    folder_button.setAccessibleName("导入文件夹")
    folder_button.setToolTip("选择文件夹并预览批量导入的数据")
    folder_button.clicked.connect(panel.folder_import_requested.emit)
    hint = QLabel(IMPORT_HINT_TEXT)
    hint.setObjectName("emptyStateHint")
    hint.setProperty("mutedText", True)
    hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(button)
    row.addWidget(folder_button)
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


class PlotPanel(PaneScopeMixin, ViewControlsMixin, SldBandControlsMixin, TransientSeriesMixin, QWidget):
    """Render one selected dataset only after a complete drawing preflight."""

    fit_range_requested = Signal(float, float)
    point_mask_requested = Signal(int)
    view_changed = Signal(int)
    import_requested = Signal()
    folder_import_requested = Signal()
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
        # 拟合范围色带按数据集取：项目里每个数据集存着自己的 ``fit_range_two_theta_deg``，投影时按
        # 当前数据集那一个画。面板只记一份的话，切到别的数据集还画着上一个的范围。
        # ``_pending_range`` 是读者刚拖出、还没回到项目里的那一个，只属于当前数据集；下一次项目
        # 投影（提交或撤销之后都会有）或换数据集就作废。
        self._fit_ranges: dict[str, tuple[float, float]] = {}
        self._pending_range: tuple[float, float] | None = None
        # 当前数据集读不出来时那句说明（``reflectivity.SOURCE_ISSUE_TEXT``），只在投影提交后更新。
        self._source_issue: str | None = None
        # 未变的源文件不在每次投影时重读一遍。
        self._prepared_cache = PreparedDataCache()
        # 不在屏上、还欠一次重画的 matplotlib 视图，以及它欠的是哪一次投影。
        self._stale: dict[str, Projection] = {}
        # 换步时可见性先变、紧接着整次重投影；这期间不补画，免得刚露出来的视图被画两遍。
        self._holding_flush = False
        self._released = False
        # 画布当前按哪一步收窄。None 表示还没有人告诉过它（独立构造的面板、以及旧测试），
        # 这时四段按老样子全露——收窄是窗口接线加上去的，不是面板自己的默认姿势。
        self._step: int | None = None
        # 专家模式开没开。同样用 None 表示「还没有人说过」：此前 SLD 剖面的可见性只由
        # ``interactions._apply_tabs`` 写，没跑过它的独立面板剖面就是露着的，None 保住这个行为。
        self._expert_mode: bool | None = None
        # 叠加对比与动态色标里写的数据集名：项目里的 display_name，没有就用 dataset_id——和左栏同一个名字。
        self._dataset_names: dict[str, str] = {}
        # 两张反射率卡上动态色标此刻写着的条目；没变就不重建那一行（预览一秒要刷很多次）。
        self._series_keys: dict[str, tuple[tuple[str, str, str], ...]] = {}
        # 剖面上强调的结构组件：（序号, 点下时的数据集, 点下时的结构）。见 ``highlight_component``。
        self._highlight: tuple[int, str | None, object | None] | None = None
        self.toolbar = PlotInteractionToolbar(self)
        self.reflectivity_tabs, self.analysis_tabs, self._views = build_tabs()
        angle_panes = tuple(self._views[key] for key in ANGLE_PANE_KEYS)
        for pane in angle_panes:
            pane.share_x_range(angle_panes)
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
        self.toolbar.sld_profile_toggled.connect(self._on_sld_profile_toggled)
        # 换视图要重算画布范围：结果态选到分析页时残差让位给分析页（见 ``ANALYSIS_STEPS``）。
        # 挂在 ``view_changed`` 上而不是两个 tab 的 currentChanged 上，因为菜单「视图」也走
        # ``select_view``，那条路不经过 tab 的点击。
        self.view_changed.connect(self._on_view_changed_scope)
        # 分析组换页也可能是投影自己按项目恢复的（那时 ``view_changed`` 不发），欠着的那一页照样要补画。
        self.analysis_tabs.currentChanged.connect(self._flush_stale)
        # 控制器就位后才问得出「此刻选中的是哪一段」，所以残差卡的落脚点在这里对一次。
        self._sync_residual_home()

    #
    # Sync pages
    #
    def _sync_pages(self) -> None:
        self._pages.setCurrentIndex(0 if self._dataset_id is None else 1)
        controller = getattr(self, "_interactions", None)
        if controller is not None:
            # 选着一个数据集、却没有它的数据（源文件缺失或读不出）时，范围与掩膜手势一律不接。
            controller.set_data_available(self._dataset_id is None or self.active_data_available())
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
        # 叠加对比画的是当前这条之外的数据集，高亮认的是此刻的结构：两样都得跟着每一次提交走。
        self._drop_stale_highlight()
        self._sync_overlay()

    #
    # Set import shortcut text
    #
    def set_import_shortcut_text(self, text: str) -> None:
        """空状态提示里补上导入的按键，原样用窗口给的平台写法（macOS 上是 ⌘I）；空串就只提菜单。"""
        if not isinstance(text, str):
            raise TypeError("shortcut text must be str")
        hint = self.findChild(QLabel, "emptyStateHint")
        hint.setText(f"{IMPORT_HINT_TEXT}或快捷键 {text}" if text else IMPORT_HINT_TEXT)

    #
    # View
    #
    def view(self, key: str) -> DiagnosticView | LiveReflectivityPlot:
        """一张视图，交出去之前先把它欠的那次重画补上——拿到的永远是当前投影的样子。"""
        try:
            view = self._views[key]
        except KeyError as error:
            raise KeyError(f"unknown diagnostic view: {key}") from error
        owed = self._stale.pop(key, None)
        if owed is not None and not self._released:
            self._draw_view(key, owed)
        return view

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
    # Set dataset
    #
    def set_dataset(self, dataset_id: str, data: api.PreparedData) -> None:
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset id must be nonempty")
        if not isinstance(data, api.PreparedData):
            raise TypeError("data must be PreparedData")
        mask = validate_plot_data(data, data.fit_mask)
        switched = dataset_id != self._dataset_id
        projection = Projection(
            data,
            mask,
            None,
            None,
            self._trends,
            self._range_for(dataset_id),
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
        if switched:
            self._pending_range = None
        self._source_issue = None
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
        switched = dataset_id != self._dataset_id
        projection = Projection(
            self._datasets[dataset_id],
            self._masks[dataset_id],
            None,
            None,
            self._trends,
            self._range_for(dataset_id),
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
        if switched:
            self._pending_range = None
        self._source_issue = None
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
            self.visible_range(),
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
            self.visible_range(),
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
        check_new_bands(self.sld_bands_toggle, projection_bands(self._result), projection_bands(result))
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
    # Show range
    #
    def show_range(self, lower: float, upper: float) -> None:
        visible = ordered_finite_range(lower, upper)
        projection = self._current_projection(visible_range=visible)
        self._transact(projection)
        self._pending_range = visible

    #
    # Visible range
    #
    def visible_range(self) -> tuple[float, float] | None:
        """当前数据集画着的拟合范围：刚拖出未提交的那个优先，否则是项目里存下的那个。"""
        return self._range_for(self._dataset_id)

    def _range_for(self, dataset_id: str | None) -> tuple[float, float] | None:
        """投影 ``dataset_id`` 时该画的范围；待提交的拖动只属于当前数据集。"""
        if dataset_id is None:
            return None
        if dataset_id == self._dataset_id and self._pending_range is not None:
            return self._pending_range
        return self._fit_ranges.get(dataset_id)

    #
    # Zoom to range
    #
    def zoom_to_range(self) -> bool:
        """Focus the angle-domain views on the active fit range.

        The fit range is often a small window of a wide scan, so keeping the
        full sweep on screen buries the region the user is actually judging.
        This clamps the three angle-axis panes (``ANGLE_PANE_KEYS``) to the
        highlighted range; it is a pure view operation that leaves the committed
        projection untouched, so the next redraw restores the full sweep on its own.
        """
        visible = self.visible_range()
        if visible is None or self._released or self._dataset_id is None:
            return False
        for key in ANGLE_PANE_KEYS:
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
        for key in ANGLE_PANE_KEYS:
            view = self._views[key]
            if isinstance(view, LiveReflectivityPlot):
                view.autoscale_view()
        return True

    #
    # Drop pending range
    #
    def _drop_pending_range(self) -> None:
        """Esc 撤掉刚拖出、还没提交的范围；色带退回项目里存下的那一个（没有就不画）。"""
        if self._dataset_id is None:
            self._pending_range = None
            return
        projection = self._current_projection(visible_range=self._fit_ranges.get(self._dataset_id))
        self._transact(projection)
        self._pending_range = None

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
        prepared = prepare_project_plots(project, self._prepared_cache)
        candidate_for_result(prepared.result, prepared.candidate_id)
        # 每次投影都按项目里存下的范围画——提交、撤销、换数据集之后，色带说的都是此刻的项目。
        fit_ranges = {dataset.dataset_id: dataset.fit_range_two_theta_deg for dataset in project.datasets}
        projection = Projection(
            prepared.data,
            prepared.mask,
            prepared.result,
            prepared.candidate_id,
            self._trends,
            fit_ranges.get(prepared.dataset_id),
            prepared.dataset_id,
            project_structure(project, prepared.dataset_id),
            prepared.issues.get(prepared.dataset_id),
        )
        previous_dataset_id = self._dataset_id
        previous_report = projection_mcmc(self._result)
        previous_bands = projection_bands(self._result)
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
        check_new_bands(self.sld_bands_toggle, previous_bands, projection_bands(projection.result))
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
        self._fit_ranges = fit_ranges
        self._dataset_names = {
            dataset.dataset_id: dataset.display_name or dataset.dataset_id for dataset in project.datasets
        }
        self._pending_range = None
        self._source_issue = projection.source_issue
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
        if self._dataset_id not in self._datasets:
            raise RuntimeError(f"active plot dataset has no readable source: {self._dataset_id}")
        return self._datasets[self._dataset_id]

    def active_data_available(self) -> bool:
        """当前数据集的预处理数据是否在手：源文件缺失或读不出时为 ``False``。"""
        return self._dataset_id is not None and self._dataset_id in self._datasets

    #
    # Active mask
    #
    def _active_mask(self) -> np.ndarray:
        if self._dataset_id is None:
            raise RuntimeError("no active plot dataset")
        return self._masks[self._dataset_id]

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
            self.visible_range(),
            self._structure,
            changes,
            self._source_issue,
        )

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
        self._preflight(projection)
        previous = self._committed_projection() if rollback_projection is None else rollback_projection
        try:
            self._draw(projection)
        except Exception:
            if rollback_sld_state is not None:
                self._sld_band_cache = restore_sld_view_state(
                    rollback_sld_state,
                    self.sld_bands_toggle,
                    self.sld_align_selector,
                )
            self._draw(previous)
            raise
        finally:
            # Whatever ended up on screen, the new projection or the rolled back
            # one, is the view the reset button has to return to.
            self._interactions.refresh_navigation_baselines()

    def _preflight(self, projection: Projection) -> None:
        """在动任何活图元之前算出这次投影里会失败的那几样，失败就原样抛出、屏上不动。

        只算数组，不渲染：此前为了「先确认画得出来」把十张视图在离屏画布上整套画一遍，一次投影
        里一半的时间花在一张不会给任何人看的图上。
        """
        candidate_for_result(projection.result, projection.candidate_id)
        if projection.data is not None and projection.mask is not None:
            reflectivity_pane_arrays(
                projection.data,
                projection.mask,
                candidate_for_result(projection.result, projection.candidate_id),
            )
        if projection.trends is not None:
            validate_batch_trends(*projection.trends)

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
            self.visible_range(),
            self._structure,
            self._source_issue,
        )

    #
    # Draw
    #
    def _draw(self, projection: Projection) -> None:
        """把一次投影落到屏上：四张 pyqtgraph 卡当场画，matplotlib 视图不在屏上的先记欠账。"""
        bands = projection_bands(projection.result)
        sync_band_controls(
            self.sld_bands_toggle,
            self.sld_align_selector,
            bands=bands,
            has_structure=projection.structure is not None,
            drawable=not self._sld_follows_edits(projection),
        )
        self._draw_live_panes(projection)
        self._draw_figures(projection)

    def _draw_figures(self, projection: Projection) -> None:
        for key in DEFERRABLE_KEYS:
            self._draw_or_defer(key, projection)

    def _draw_or_defer(self, key: str, projection: Projection) -> None:
        """画这张 matplotlib 视图；它此刻不在屏上就先记下欠的是哪一次投影。"""
        if self._defers(key):
            self._stale[key] = projection
        else:
            self._stale.pop(key, None)
            self._draw_view(key, projection)

    def _apply_theme_palette(self) -> None:
        """外观切换后把 matplotlib 视图按已提交的投影重画一遍。

        画布的主题钩子只换得了结构色（底、轴、图例）；色标、相关矩阵里的读数、热图抬头那行小字、
        剖面上的手柄都是画的那一刻按调色板取的色，不重画就一直是切换前那一套——深色画布上的浅色
        色标、黑字。pyqtgraph 那几张卡的数据色不随外观变，结构色由它们自己的钩子换。
        """
        if self._released:
            return
        self._draw_figures(self._committed_projection())
        self._interactions.refresh_navigation_baselines()

    def _defers(self, key: str) -> bool:
        """这张 matplotlib 视图此刻能不能先欠着。

        只有按步骤收窄过的画布才欠：独立构造的面板（以及步骤化之前那套四段常驻布局）每次提交都整套
        重画，调用方一直按这个约定读图。欠着的视图在露出来或被 ``view()`` 读到时补画。
        """
        if self._step is None:
            return False
        return not self._views[key].canvas.isVisibleTo(self)

    def _flush_stale(self, *_args: object) -> None:
        """补画刚露出来、还欠着重画的视图。"""
        if self._released or self._holding_flush:
            return
        flushed = [key for key in tuple(self._stale) if not self._defers(key)]
        for key in flushed:
            self._draw_view(key, self._stale.pop(key))
        if flushed:
            self._interactions.refresh_navigation_baselines()

    def _draw_view(self, key: str, projection: Projection) -> None:
        if key == "sld":
            self._draw_sld(projection)
            return
        view = self._views[key]
        result, candidate_id = projection.result, projection.candidate_id
        if key == "candidates":
            draw_candidate_comparison(view, result, candidate_id)
        elif key == "residual_map":
            draw_residual_heatmap(view, result, candidate_id)
        elif key == "parameter_map":
            draw_parameter_heatmap(view, result, candidate_id)
        elif key == "uncertainty":
            draw_uncertainty(view, result, candidate_id)
        else:
            draw_batch_trends(view, projection.trends)

    def _draw_sld(self, projection: Projection) -> None:
        view = self._views["sld"]
        data = projection.data
        if data is None or projection.mask is None:
            draw_empty(view, "SLD 深度剖面", projection.source_issue or "暂无可用数据")
            return
        # 结构那两步画的是读者手上这套结构：候选、折进图例的对比候选、以拟合结果为中心的带都不进画面。
        edited = self._sld_follows_edits(projection)
        shown = visible_bands(
            checked=self.sld_bands_toggle.isChecked() and not edited,
            cache=self._sld_band_cache,
            persisted=projection_bands(projection.result),
            dataset_id=projection.dataset_id,
            report=projection_mcmc(projection.result),
            alignment=ALIGN_KEYS[self.sld_align_selector.currentIndex()],
            surface_label=self.sld_align_selector.itemText(1),
        )
        candidate = candidate_for_result(projection.result, projection.candidate_id)
        draw_sld(
            view,
            None if edited else candidate,
            () if edited else comparison_candidates(projection.result, projection.candidate_id),
            shown,
            structure=projection.structure,
            wavelength_a=data.beam.effective_wavelength_a,
            highlight=self._highlight_for(projection),
        )

    #
    # Draw live panes
    #
    def _draw_live_panes(self, projection: Projection) -> None:
        """四张 pyqtgraph 卡：换一份数组就够，便宜到不值得欠着。"""
        data = projection.data
        if data is None or projection.mask is None:
            for key in REFLECTIVITY_KEYS:
                self._views[key].show_placeholder(projection.source_issue or "暂无可用数据")
            return
        candidate = candidate_for_result(projection.result, projection.candidate_id)
        arrays = reflectivity_pane_arrays(data, projection.mask, candidate)
        log = self._views["log"]
        log.show_log_reflectivity(
            arrays.log_angles,
            arrays.log_observed,
            arrays.log_model,
            r_floor=arrays.r_floor,
            # 拟合掩膜对两页是同一份：对数页也要把掩掉的点画成 ✕，而不是混在拟合点里。
            mask=arrays.raw_mask,
        )
        raw = self._views["raw"]
        raw.show_raw_reflectivity(arrays.raw_angles, arrays.raw_intensity, arrays.raw_mask, arrays.raw_model)
        qz4 = self._views["qz4"]
        if arrays.qz4 is None:
            qz4.show_placeholder("暂无当前候选")
        else:
            qz4.show_qz4(*arrays.qz4, ylabel=arrays.qz4_ylabel or "qz⁴R")
        residual = self._views["residual"]
        if arrays.residual is None:
            residual.show_placeholder("暂无当前候选")
        else:
            residual.show_residual(
                *arrays.residual, ylabel=arrays.residual_ylabel, sigma_reference=arrays.residual_sigma_reference
            )
        self._draw_range(projection.visible_range)

    #
    # Draw range
    #
    def _draw_range(self, visible_range: tuple[float, float] | None) -> None:
        for key in ("raw", "log"):
            view = self._views[key]
            # pyqtgraph range items persist across show_*, so an absent range must
            # be cleared explicitly rather than relying on a cleared axes.
            if visible_range is None:
                view.clear_fit_range()
            else:
                view.show_fit_range(*visible_range)

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
        self._fit_ranges = {}
        self._pending_range = None
        self._stale.clear()
        self._prepared_cache = PreparedDataCache()
        self._result = None
        self._candidate_id = None
        self._structure = None
        self._sld_band_cache = None
        self._dataset_names = {}
        self._highlight = None

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
