"""Step scope: which plot panes the canvas shows at each pipeline step.

画布按流程步收窄成设计稿里那两张卡：哪几段 splitter 露出来、「加权残差」那张唯一的卡此刻住在哪、
SLD 剖面画的是正在编辑的结构还是拟合候选，都在这里判。这一块从 :mod:`xrr_fitter.gui.plots.panel`
分出来，面板模块只留投影与绘制事务，复杂度门禁才能各自衡量。

:class:`PaneScopeMixin` 混入 ``PlotPanel``：运行时 ``self`` 就是面板，所以方法直接读它的
``_step``、``_expert_mode``、各段控件与交互控制器。
"""

from __future__ import annotations

from xrr_fitter.gui.plots.diagnostics import ANALYSIS_KEYS, RESIDUAL_KEY
from xrr_fitter.gui.plots.sld_state import Projection

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


class PaneScopeMixin:
    """Narrow the canvas to the panes the current pipeline step reads."""

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
        self.toolbar.sld_profile_action.setEnabled(step in ANALYSIS_STEPS)
        reproject = changed and not self._released and self._dataset_id is not None
        self._holding_flush = reproject
        try:
            self._sync_pane_scope()
        finally:
            self._holding_flush = False
        if reproject:
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
        panes = STEP_PLOT_PANES.get(self._step, STEP_PLOT_PANES[0])
        # 结果步默认是设计稿那两张卡；读者要看拟合出来的剖面时，由「显示拟合 SLD 剖面」加在最后。
        if self._step in ANALYSIS_STEPS and self.toolbar.sld_profile_action.isChecked():
            return (*panes, "sld")
        return panes

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
        # 刚露出来的视图若欠着重画，现在补上：切过去时看到的必须是最新的。
        self._flush_stale()

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
    # On sld profile toggled
    #
    def _on_sld_profile_toggled(self, _checked: bool) -> None:
        """结果页加上或撤下拟合 SLD 剖面那张卡；剖面每次投影都画着，所以只需重算可见性。"""
        if not self._released:
            self._sync_pane_scope()
