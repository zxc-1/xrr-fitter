"""Fixed three-column MainWindow widget and action assembly.

The design's shell is a grid -- ``grid-template-columns:264px 1fr 340px`` -- with
a navigation rail, an adaptive canvas, and a context inspector.  Spending that
layout on QDockWidgets gave every column a title bar and a close button, showed
one inspector section at a time behind tabs, and let the arrangement be dragged
apart; none of which the design draws.  The columns are therefore a QSplitter
whose side widths are budgets and whose middle absorbs the rest.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.command_icons import command_icon
from xrr_fitter.gui.data.panel import DataPanel
from xrr_fitter.gui.fitting.panel import FitPanel
from xrr_fitter.gui.guidance.panel import GuidancePanel
from xrr_fitter.gui.navigation import panel as nav_panel
from xrr_fitter.gui.navigation.panel import PipelineNav
from xrr_fitter.gui.parameters.panel import ParametersPanel
from xrr_fitter.gui.plots.panel import PlotPanel
from xrr_fitter.gui.project.actions import ProjectActions
from xrr_fitter.gui.results.panel import ResultsPanel
from xrr_fitter.gui.sizing import AnchorSizedScroll, ContentSizedScroll, ContentSizedSplitter
from xrr_fitter.gui.structure import naming
from xrr_fitter.gui.structure.panel import StructurePanel
from xrr_fitter.gui.structure.selection import SelectedLayerCard, layer_bounds, layer_locks

WORKFLOW_ACTION_SPECS = (
    ("startFitAction", "一键拟合", "Ctrl+Return", "start_fit"),
    ("cancelFitAction", "取消拟合", "Esc", "cancel_fit"),
    ("exportResultsAction", "导出结果", "Ctrl+Shift+E", "export_results_dialog"),
)

# The side column widths, taken from the design's own grid rule:
# ``grid-template-columns:264px 1fr 340px``.  They are read off the persisted
# default rather than written here because a restored width always wins over a
# computed one -- ``showEvent`` applies these budgets and then lets the project's
# own state overwrite them -- so a second copy of the numbers would quietly
# decide the layout the moment the two drifted.
#
# They are a budget rather than a preference: ``_scrolled`` turns horizontal
# scrolling off, so a panel whose minimum width exceeds its column is clipped
# outright, with no scrollbar to reach the rest.  Panels are measured against
# these numbers, not the other way around -- widening a column to fit a panel
# would leave the canvas narrower than the design gives it.
LEFT_COLUMN_WIDTH = api.ProjectUiState().workspace_splitter_sizes[0]
RIGHT_COLUMN_WIDTH = api.ProjectUiState().workspace_splitter_sizes[2]

# 窗口能被拖到的最窄宽度（``MainWindow.setMinimumSize`` 的第一个数）。画布列在那一刻
# 拿到的就是这个数减去两侧固定列——住在画布里的面板得能在这个宽度下排完，否则它会在
# 列边缘被裁掉，而三条列都关掉了横向滚动。
#
# 这个数就是设计稿 ``.appwin`` 的宽度：1270 的内容（nav 264 + canvas 666 + inspector 340）
# 加左右各 1px 边框。比它大一像素，窗口就到不了设计稿那个尺寸，画布列会被多撑开那几像素。
MINIMUM_SHELL_WIDTH = 1272
CANVAS_COLUMN_FLOOR = MINIMUM_SHELL_WIDTH - LEFT_COLUMN_WIDTH - RIGHT_COLUMN_WIDTH

# 窗口能被拖到的最矮高度。设计稿最矮的一帧是帧②（引导模式）：``.appwin`` 731 高，减去
# 38px 的模拟标题栏（真窗口的标题栏归系统画）= 693。
MINIMUM_SHELL_HEIGHT = 693

# 层堆叠在画布里的高度地板。``QSplitter`` 用 stretch 只分“多余”高度，能压到多低由子
# 控件的最小高度说话，而滚动区的天然最小高度只有几十像素——不写这个数，下面绘图栈
# 624px 的最小高度会把层堆叠压到只剩一行。够装命令条（32px）、卡片抬头和四五行。
STRUCTURE_PANE_FLOOR_PX = 240

# 层堆叠长高时留给下面 SLD 剖面的余量。设计稿的 canvas-body 是「卡按内容占高、剖面
# 吃掉剩下的」，所以撑开上面那张卡时得先说清下面至少留多少——剖面自己报的最小高度是
# 228px，再多一点让坐标轴标题也在。
SLD_PANE_RESERVE_PX = 240

COLUMN_NAMES = ("navigationColumn", "canvasColumn", "inspectorColumn")

# 检视器里所有可能出现的段。构造顺序是固定的，露哪几段由 ``STEP_INSPECTOR_SECTIONS``
# 按项目走到的那一步决定——顺序在这里排一次，就不必在换步时把卡片从布局里摘下来重插。
# 每一段自己带抬头，因为没有停靠标题栏的列得给自己的段落做路标。
#
# 抬头右端那句话只在设计稿写了 ``<span class="faint">`` 的段落上有。``.insp-sec .h`` 是
# 一行 ``space-between``，右端空着时抬头就是一句话；替每一段补一句「怎么用」读起来匀，
# 却把设计稿区分过的东西抹平了——右端有字的那一段（实时指标 · 联合）是在报此刻的拟合
# 形态，不是在讲操作方法，而六段都挂一句之后它就混进同一列灰字里了。
INSPECTOR_SECTIONS = (
    # 帧① 右栏那三段全在这张卡里：拟合判定 / 参数 · 结果值 / 候选解，各自带抬头与计数，
    # 由 ``ResultsPanel`` 自己画。所以这一层不带抬头——设计稿数得出三段，套一句「拟合结果」
    # 就是第四句抬头加第二重边框，读者得先猜它和「拟合判定」是不是同一件事。抬头是 ``None``
    # 的段落走 ``_plain_section``：只占位、不画框。
    ("inspectorResults", None, None, "result_panel"),
    # 帧③ 的右栏第一张卡：画布里选中的那一层，就地改材料与几何。没有选中普通层时
    # 它说「未选择」而不是消失——一张会来回出现的卡会让下面几张跟着上下跳。抬头右端
    # 留给层名（见 ``_name_card_after_selection``），设计稿写的就是「选中层 · a-Si 非晶硅」。
    ("inspectorSelectedLayer", "选中层", "", "selected_layer_panel"),
    # 帧③ 右栏第二段：设计稿把它整段画完了（HTML 711-720），里面只有三档（自由 / 固定 /
    # 仅范围）和两枚徽标（先验、共享），说的都是选中那一个量一个人的事。所以这一段装的是
    # ``ParameterDisposition`` 而不是整张参数表——表在画布列那张「参数总览」卡里，也就是
    # ``.canvas-top`` 第三个 tab 指的地方。抬头右端留给量名（见 ``_name_card_after_parameter``），
    # 设计稿写的就是「参数化 · 厚度 d」。
    ("inspectorParameters", "参数化", "", "parameter_disposition"),
    # 紧跟参数，因为它诊断的就是刚改的那几个数；设计稿也把它排在结构编辑那几段之
    # 后、而不是整栏末尾。
    ("inspectorStructureDiagnostics", "结构诊断", "", "structure_diagnostics_panel"),
    # 设计稿帧④ 的右栏第一段：此刻跑到哪一阶段、目标值收敛到多少、每个数据集各自
    # 是什么走向。它只在运行那一步露面——不在跑的时候它报的是上一轮留下的残值。抬头
    # 右端那个「联合」是设计稿唯一写了 faint 的一处，而它报的是那一帧此刻的批量模式，
    # 不是这张卡的名号：留空由 ``_name_card_after_batch_mode`` 按项目填。
    ("inspectorLiveMetrics", "实时指标", "", "live_metrics"),
    # 设计稿里「拟合判定」只指判读那张卡（帧① 右栏第一节），运行控件那一节叫「控制」
    # （帧② 右栏末节）。这一节装的是批量模式与开始/取消/强制停止，所以用后者——两张
    # 卡同名同时挂在一栏里的话，抬头就不再是路标。
    ("inspectorFit", "控制", "", "fit_panel"),
    # 帧⑤ 的右栏三段。帧⑤ 与帧① 在实现里是同一个流程步（结果步），换的是画布列那一页：
    # 读者对着相关矩阵要问的是「这条链能不能当证据用」，而帧① 那三段一句也答不上。所以
    # 这三段不按步骤入场，按画布列此刻显示哪一页（见 ``UNCERTAINTY_INSPECTOR_SECTIONS``）。
    ("inspectorMcmcConvergence", "MCMC 收敛诊断", "", "convergence_panel"),
    # 抬头右端留给参数短名（见 ``_name_card_after_posterior``），设计稿写的就是
    # 「后验分位 · d·aSi」——那一位与中栏矩阵旁「最强相关」念的是同一对里的同一个。
    ("inspectorPosteriorQuantile", "后验分位", "", "quantile_panel"),
    ("inspectorBootstrap", "自助抽样（Bootstrap）", "", "bootstrap_panel"),
)

# 哪一步露哪几段。设计稿的每一帧右栏都恰好三段，而不是同一栏的六种滚动位置：帧③ 是
# 选中层 / 参数化 / 结构诊断，帧① 是判定 / 参数·结果值 / 候选解，帧④ 只剩运行那一段。
#
# 常驻五段解不掉的是溢出：1400×900 下五段内容高约 1900px、视口 815px，2.3 倍。哪一段
# 落在折叠线下就成了配置问题，而调顺序只是换一段藏起来——判定提到第一位就把选中层压
# 下去，点层编辑同样要它当场可见。按步骤只露相关的那几段，溢出本身就没了。
#
# 键是 ``PipelineNav`` 的步骤索引（``_determine_step``：0 数据 / 1 结构 / 2 参数 /
# 4 结果），值按 ``INSPECTOR_SECTIONS`` 的构造顺序显示。
STEP_INSPECTOR_SECTIONS = {
    # 数据与结构：还没拟合过，候选解是一张空卡，占的是「改完这一层去哪儿复核」那句
    # 诊断的位置。
    0: ("inspectorSelectedLayer", "inspectorParameters", "inspectorStructureDiagnostics"),
    1: ("inspectorSelectedLayer", "inspectorParameters", "inspectorStructureDiagnostics"),
    # 参数：结构立起来了，这一步之后就该按「开始拟合」，所以运行那一段在这里入场。
    # （批量模式此前只有这张卡能选，所以原注释说命令条上选不了；分段控件搬到命令条后
    # 那句已经不成立，两处都能选。）
    2: (
        "inspectorSelectedLayer",
        "inspectorParameters",
        "inspectorStructureDiagnostics",
        "inspectorFit",
    ),
    # 拟合中：参数表和候选解都还是上一轮的旧值，摆在进度旁边会被读成本轮读数。
    3: ("inspectorLiveMetrics", "inspectorFit"),
    # 结果（帧①）：只有判定 / 参数 · 结果值 / 候选解那三段，它们全在 ``inspectorResults``
    # 这一张卡里。参数化不在这一帧——它写的是交给求解器的初值，跟刚读完的结果值同一个量级、
    # 不同的含义，两张表叠在一栏里读者得先判断哪张是结果。控制也不在：重跑一次走命令栏的
    # ``⚡ 一键拟合``，取消与强制停止在拟合菜单里，收起这张卡不会让谁够不着。
    4: ("inspectorResults",),
    # 导出（帧⑥）按设计稿也不改右栏；``_determine_step`` 其实停在 4，这一条是备用。
    5: ("inspectorResults",),
}

# 结果步上切到「不确定度」那一页时，右栏换成帧⑤ 画的这三段。
#
# 是替换而不是并集：设计稿的每一帧右栏都恰好三段，并上帧① 那三段会变成六段，而六段在
# 1400×900 下又要溢出——正是 ``STEP_INSPECTOR_SECTIONS`` 那段注释算过的那笔账。
#
# 只有这一页换栏。别的分析页（候选解、残差图、参数图、趋势）看的是同一份结果的不同画法，
# 右栏那三段照样答得上；不确定度页问的是另一件事，右栏也得跟着换一套证据。
UNCERTAINTY_INSPECTOR_SECTIONS = (
    "inspectorMcmcConvergence",
    "inspectorPosteriorQuantile",
    "inspectorBootstrap",
)

# 层堆叠留在画布里的那几步。帧③ 的画布上半就是它，改一层看着下面的剖面动；帧① 的画布
# 只有反射率和残差两张图，画面里没有层堆叠——它是 ``central_stack`` 的兄弟，换页换不掉，
# 不显式收起就会在结果态压着两张图各扣掉一半高度。
STEPS_WITH_LAYER_STACK = (0, 1, 2)

# 结构、拟合中与结果这三步的索引。``_determine_step`` 永远不会报 3——它只看项目状态，看不见
# 「此刻正在跑」——所以运行那一步由拟合卡的 running 边压过来；而结果这一步在项目提交完
# 之前就得到位，因为 ``result_published`` 比项目状态先到一拍。
#
# 定义在 ``navigation.panel`` 里，和被它们索引的 ``PIPELINE_STEPS`` 放一起；这里保留名字，
# 因为跟着流程步走的收窄逻辑都写在本模块。
STRUCTURE_STEP_INDEX = nav_panel.STRUCTURE_STEP_INDEX
RUNNING_STEP_INDEX = nav_panel.RUNNING_STEP_INDEX
RESULT_STEP_INDEX = nav_panel.RESULT_STEP_INDEX


# 设计稿帧③ 的结构诊断原文。薄层上厚度与密度换着走能拟出几乎相同的曲线，所以措辞
# 停在「可能」，并把核实交给不确定度页——这里没有当前这组参数的相关矩阵，能说的只是
# 这类相关的存在。放大镜字形内联在文案里，和引导页的 💡 提示同一种写法。
STRUCTURE_CORRELATION_HINT_TEXT = (
    "🔎 <b>厚度 d 与密度 ρ 可能相关。</b>这类相关在薄层上常见；拟合后请到「不确定度」页用相关矩阵与 Profile 似然核查。"
)


def _structure_diagnostics_hint() -> QLabel:
    """The inspector's 结构诊断 body: one wrapped advisory line."""
    hint = QLabel(STRUCTURE_CORRELATION_HINT_TEXT)
    hint.setObjectName("structureCorrelationHint")
    theme.set_status_kind(hint, "info")
    theme.mark_hint(hint)
    hint.setWordWrap(True)
    return hint


def _scrolled(name: str, inner: QWidget, *, content_height: bool = False, anchor: QWidget | None = None) -> QScrollArea:
    """Wrap `inner` in a vertical-only scroll area.

    Horizontal scrolling is off by design: a column is a fixed budget, so a
    panel is expected to fit it rather than be reachable by scrolling sideways.
    The area takes no focus of its own so Tab order runs through the panels.

    ``content_height`` picks the variant that asks for no more height than it
    holds.  The rail needs it for both of its bands, because the design gives
    the spare height to neither list; a column that fills its own height keeps
    the default.

    ``anchor`` picks the variant that asks for the height of one child inside
    ``inner`` rather than all of it —— 上半段的高度预算归层堆叠，同一个滚动区里排在它
    下面的「参数总览」靠滚动去够。
    """
    if anchor is not None:
        scroll: QScrollArea = AnchorSizedScroll(anchor)
    else:
        scroll = ContentSizedScroll() if content_height else QScrollArea()
    scroll.setObjectName(name)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(inner)
    return scroll


def _column(name: str, accessible_name: str) -> tuple[QWidget, QVBoxLayout]:
    """One grid column: a plain widget whose layout stacks its sections.

    The accessible name is not decoration.  A QDockWidget announced itself, so
    dropping the title bars would otherwise hand a screen reader three
    unlabelled containers where it used to be told which panel it had entered.
    """
    column = QWidget()
    column.setObjectName(name)
    column.setAccessibleName(accessible_name)
    layout = QVBoxLayout(column)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    return column, layout


def _detached_summary(panel: DataPanel) -> QWidget:
    """Lift the dataset summary out of the panel and onto the rail's bottom edge.

    The design puts it there -- ``.ds-summary{margin-top:auto;border-top:1px
    solid var(--border)}`` -- and the panel already computes exactly the string
    it draws ("共 4 个数据集 · 可拟合 3 · 需注意 1") and refreshes it on every
    project change.  Moving the widget rather than mirroring its text keeps one
    fact in one place: a second label would need a refresh path of its own and
    would drift the first time one of the two was missed.
    """
    summary = panel.summary_label
    layout = panel.layout()
    if layout is not None:
        layout.removeWidget(summary)
    return summary


def _navigation_column(window) -> QWidget:
    """The design's left rail: what to work on, where you are, and how much.

    One column, not three stacked docks.  The rail carries the dataset list and
    the pipeline only -- 帧③ draws no structure list here at all, it edits the
    stack in the canvas beside the SLD profile that answers it.  Both lists take
    their contents' height and are scrolled, because the rail's own height is
    fixed by the window and either list can outgrow it.

    The rail's spare height belongs to neither list.  ``.nav`` is a flex column
    whose footer carries ``margin-top:auto``, so the slack collects in one band
    above the count -- not between the dataset rows and 分析管线, which the design
    draws 6px apart however few datasets there are.  Both bands are therefore
    capped to their contents and the leftover is claimed by a stretch below them.

    The splitter's handle is set to nothing: the design draws the rail as one
    continuous ``--panel-2`` surface, and a default 4px handle cuts a
    window-coloured band across it.  The split itself stays -- it is what keeps
    the pipeline visible when the dataset list grows -- and its sizes still
    round-trip through the project.
    """
    column, layout = _column("navigationColumn", "导航栏")
    lists = ContentSizedSplitter(Qt.Orientation.Vertical, column)
    lists.setObjectName("leftSplitter")
    lists.setHandleWidth(0)
    lists.addWidget(_scrolled("dataPanelScroll", window.data_panel, content_height=True))
    pipeline_scroll = _scrolled("pipelineNavScroll", window.pipeline_nav, content_height=True)
    # 管线是六步定长的地图，不是随高度伸缩的列表。滚动区按设计不把被滚控件的最小高度报给
    # 外面，于是分割器按 1:1 只分给它一半：六步里最后一步被裁在视口外，页脚也被顶到管线
    # 下边缘之上。这里把地图的自然高度显式报成下限——它跟样式表无关（量过：有无 QSS 都是
    # 同一个数），所以在建栏时定下来是稳的。
    pipeline_scroll.setMinimumHeight(window.pipeline_nav.minimumSizeHint().height())
    lists.addWidget(pipeline_scroll)
    layout.addWidget(lists, 1)
    layout.addStretch(0)
    layout.addWidget(window.dataset_summary)
    return column


def _follow_canvas_tabs(window, canvas: QSplitter, stack_scroll: QScrollArea, overview: QWidget) -> None:
    """画布行头的三个 tab 是这一列的目录，点一个就把那一段滚出来。

    三段内容本来就都在画布列里（层堆叠、SLD 深度剖面、参数总览），tab 不藏东西——它解的
    是「这一列往下还有什么」这件事在一屏放不下时读者看不见。所以 tab 做的是「带我去」而
    不是「换一页」：换页会让刚编辑的层堆叠从眼前消失，而读者点 SLD 往往正是要拿它和层
    对着看。
    """

    def follow(key: str) -> None:
        if key == "parameters":
            stack_scroll.ensureWidgetVisible(overview, 0, 0)
            return
        if key == "sld":
            pane = window.central_stack.findChild(QWidget, "sldPane")
            if pane is not None:
                # 画布是纵向 splitter：SLD 在下半段，要露出来得先让下半段有高度。
                # 段数不是固定的两段——进度视图搬进来之后中间还夹着一段，所以按
                # ``central_stack`` 自己的下标发尺寸，而不是假定它排在第二位。
                sizes = [0] * canvas.count()
                sizes[canvas.indexOf(stack_scroll)] = stack_scroll.minimumHeight()
                sizes[canvas.indexOf(window.central_stack)] = max(canvas.height(), 1)
                canvas.setSizes(sizes)
            return
        stack_scroll.ensureWidgetVisible(window.structure_panel, 0, 0)

    window.structure_panel.editor.canvas_view_changed.connect(follow)


def _overview_card(window) -> QFrame:
    """帧③ ``.canvas-top`` 第三个 tab「参数总览」指的那张卡（HTML 645）。

    设计稿只在那个 tab 上写过这四个字，没画这张卡的样子——``.canvas-body`` 里只有层堆叠与
    SLD 剖面两张 ``.plotcard``。而右栏「参数化」那一段是画完的（711-720）：三档加两枚徽标，
    没有表。表得有个去处，而 tab 是这一列的目录，于是它落在这一列。

    因此这张卡有抬头、没有副题：副题若写了，就是设计稿里没有的一句说明，而它和抬头一样醒目。

    卡里装的是整个 ``ParametersPanel``——勾选框、三页（参数 / 共享 / 约束）和底下那行状态字是
    同一个面板的零件。拆开它们等于把「哪一行在哪一页」这件事分到两栏去读。
    """
    card, body = theme.titled_card(None, "canvasParameterOverview", "参数总览", "")
    body.addWidget(window.parameters_panel, 1)
    return card


def _canvas_column(window) -> QWidget:
    """The adaptive middle: 帧③'s layer stack over guidance or the plots.

    ``min-width:0`` in the design's ``.canvas`` rule is what lets the middle be
    squeezed; the splitter carries the same permission by being the only
    stretching child of a zero-margin layout.

    The stack editor sits above the page stack rather than inside the plot
    panel: the plot panel swaps itself for an empty state until data arrives,
    and a structure declared before the first import would have gone with it --
    which is exactly when it is being built.

    「参数总览」那张卡跟着层堆叠共一个滚动区，而不是另开一段 splitter：
    ``setChildrenCollapsible(False)`` 之下每一段只要可见就长期占着高度，而 900px 高的窗口上
    那段高度正是 SLD 剖面在用的——同一张设计稿画着它。装在滚动区里，它只在读者滚到时才占地方。
    代价是它排在 SLD 上面，与 tab 的先后（样品结构 → SLD → 参数总览）相反。
    """
    column, layout = _column("canvasColumn", "画布")
    canvas = QSplitter(Qt.Orientation.Vertical, column)
    canvas.setObjectName("canvasSplitter")
    canvas.setChildrenCollapsible(False)
    overview = _overview_card(window)
    scrolled = QWidget()
    scrolled.setObjectName("structurePaneContents")
    scrolled_layout = QVBoxLayout(scrolled)
    scrolled_layout.setContentsMargins(0, 0, 0, 0)
    scrolled_layout.setSpacing(theme.SPACE_SM)
    scrolled_layout.addWidget(window.structure_panel)
    scrolled_layout.addWidget(overview)
    stack_scroll = _scrolled("structurePanelScroll", scrolled, anchor=window.structure_panel)
    stack_scroll.setMinimumHeight(STRUCTURE_PANE_FLOOR_PX)
    # 这一段的高度请求照层堆叠自己算（``anchor``），不照整段滚动内容：默认那个封死的 360px
    # 比层堆叠要的还矮，而照内容算又会把「参数总览」那三百来像素也讨过来——那是 SLD 的地方。
    # 单靠它不够：splitter 只在建栈时问一次高度，那时结构还是空的；真正把高度补齐的是下面
    # 的 ``_refit_on_structure_change``。
    canvas.addWidget(stack_scroll)
    # 设计稿帧④ 的画布列是「总进度 + 九阶段」压着实时反射率，右栏只剩控制与实时指标。
    # 进度视图因此从拟合面板里搬到这里——它在 340px 的右栏里九个阶段名会被压成两行一条，
    # 而这一帧里它就是主角。搬走不改归属：``fit_panel.progress_view`` 仍是同一个对象，
    # 起停仍由 ``_project_running_state`` 控制它的可见性，只是画在中间那一列。不另外包
    # 滚动区：包了的话没在运行时留下的是一个空框，而 splitter 认的是控件自己的可见性。
    canvas.addWidget(window.fit_panel.progress_view)
    canvas.addWidget(window.central_stack)
    canvas.setStretchFactor(0, 0)
    canvas.setStretchFactor(1, 0)
    canvas.setStretchFactor(2, 1)
    # 量高度问的是层堆叠自己，不是整个滚动内容：上半段的下限该跟着层数走，而参数总览
    # 是滚出来看的一张卡，把它算进下限等于永久多占三百来像素，而那是 SLD 的地方。
    _refit_on_structure_change(canvas, stack_scroll, window.structure_panel)
    _follow_canvas_tabs(window, canvas, stack_scroll, overview)
    layout.addWidget(canvas, 1)
    return column


def _name_card_after_selection(window, card: QWidget, name: str, title: str) -> None:
    """Write the selected layer's name into the card's own caption.

    设计稿的抬头是一整行「选中层 · a-Si 非晶硅」。抬头同时是这张卡的无障碍名称，所以
    层名写在这里比卡内再占一行更省高度，读屏进入这张卡时也直接听见改的是哪一层。

    层名走 ``naming.inline_name``：专家列表里这一层叫「a-Si · 非晶硅薄膜」，原样接在
    「选中层 · 」后面就是三段两个同样的符号，读者分不出哪一个是抬头与层名的界。
    """
    heading = card.findChild(QLabel, f"{name}Title")
    if heading is None:  # pragma: no cover - titled_card always names its heading
        return

    def rename(_index, component) -> None:
        layer = component if isinstance(component, api.LayerSpec) else None
        caption = title if layer is None else f"{title} · {naming.inline_name(layer.name)}"
        heading.setText(caption)
        card.setAccessibleName(caption)

    window.structure_panel.editor.component_selected.connect(rename)


def _name_card_after_parameter(window, card: QWidget, name: str, title: str) -> None:
    """Write the parameter row the reader is standing on into the card's caption.

    设计稿帧③ 这一段的抬头是「参数化 · 厚度 d」。卡里那三档（自由 / 固定 / 仅范围）和两枚
    徽标说的都是某一个量一个人的事，抬头不点名，读者就得回到表里去找哪一行是选中的——而那
    张表在同一张卡里往下滚，选中行常常已经不在眼前。抬头同时是无障碍名称，读屏进这张卡时
    也就直接听见在改哪个量。
    """
    heading = card.findChild(QLabel, f"{name}Title")
    if heading is None:  # pragma: no cover - titled_card always names its heading
        return
    table = window.parameters_panel.parameter_table

    def rename(*_args) -> None:
        quantity = table.current_quantity()
        caption = title if quantity is None else f"{title} · {quantity}"
        heading.setText(caption)
        card.setAccessibleName(caption)

    # ``currentCellChanged`` 而不是 ``itemSelectionChanged``：方向键走到标题行上也要改
    # 抬头，而标题行的格子未必进得了选区。
    table.currentCellChanged.connect(rename)
    # 换数据集、换结构都会重填整张表，选中行随之作废——那次作废发生在 ``QSignalBlocker``
    # 里，``currentCellChanged`` 一声不响，所以由表自己补一声。
    #
    # 两个发信方都必须是表自己。挂到 ``table.model()`` 上的连接会在表的 Python 包装失效
    # 之后、C++ 模型析构之前再响一次，那时 ``currentRow()`` 抛 shiboken 的「对象已删除」；
    # 这个异常落在 Qt 事件循环里，pytest-qt 把它记到*下一条*用例头上，于是一处接线错误在
    # 整个 GUI 套件里炸出上百条无关失败。
    table.rows_reloaded.connect(rename)


def _name_card_after_batch_mode(window, card: QWidget, name: str) -> None:
    """Write the batch mode in force into this card's right-hand caption.

    设计稿帧④ 在实时指标抬头右端写了「联合」，而那一格读的是项目的 ``batch_mode``，
    不是卡的名号：同一屏顶栏那枚模式高亮读的就是同一个字段。写成字面文案的话，独立
    批量的项目也会在右栏报「联合」，两处同屏说的是同一件事却互相打脸，读者只能二选
    一地信——而这一句正好在实时读数上头，被信错的那一半会把整段数字读成另一回事。

    短名取自 ``chrome.BATCH_MODE_SPECS``，与顶栏那两枚分段用的是同一份字面，免得两处
    各写一遍之后慢慢分岔。chrome 走函数内导入，因为它反过来要这个模块。
    """
    from xrr_fitter.gui.chrome import BATCH_MODE_SPECS

    caption = card.findChild(QLabel, f"{name}Subtitle")
    if caption is None:  # pragma: no cover - titled_card always names its caption
        return
    labels = {mode: label for mode, _object_name, label, _tooltip in BATCH_MODE_SPECS}

    def restate(project) -> None:
        text = labels.get(project.batch_mode, "")
        caption.setText(text)
        # ``titled_card`` 拿副标题当 tooltip，而这一句是 ``ElidingLabel``：换了可见文字
        # 不换 tooltip，被省略时读者悬停读到的就是上一个模式。
        caption.setToolTip(text)

    restate(window.document.project)
    window.document.project_changed.connect(restate)


def _name_card_after_posterior(window, card: QWidget, name: str, title: str) -> None:
    """Write the parameter this section is reading into its caption.

    设计稿帧⑤ 这一段的抬头是「后验分位 · d·aSi」，而那一位不是随手挑的：它是中栏矩阵旁
    「最强相关 d·aSi ↔ ρ·aSi」那一对里的第一位。三行数字不点名的话，同屏两处就会出现
    「矩阵说最强是这一对、分位表却在报另一个参数」，而读者没有别的线索能判断哪一处对。

    先打底再连接，照 ``_name_card_after_batch_mode``：投影发生在 ``ResultsPanel`` 的构造期，
    早于这张卡搭起来，只连信号的话第一次投影时抬头还没人接，那行字会永远停在「后验分位」。
    """
    heading = card.findChild(QLabel, f"{name}Title")
    if heading is None:  # pragma: no cover - titled_card always names its heading
        return
    panel = window.quantile_panel

    def rename(short_name: str) -> None:
        caption = title if not short_name else f"{title} · {short_name}"
        heading.setText(caption)
        card.setAccessibleName(caption)

    rename(panel.current_short_name())
    panel.parameter_changed.connect(rename)


def _refit_on_structure_change(canvas: QSplitter, scroll: QScrollArea, panel: QWidget) -> None:
    """Re-ask the stack card how tall it is whenever its contents change.

    A splitter asks once, while the stack is still empty, and remembers the
    answer.  Adding layers makes the card taller but nobody asks again, so the
    layer list below the header stays rows short however tall the window is.

    Re-dealing alone is not enough: a splitter satisfies its children's minimum
    heights before it shares out what is left, and the plot stack asks for more
    than the whole column on a 900px window -- so there is never any "left", and
    ``setSizes`` gets clamped straight back.  The lever that actually moves is
    the card's own floor, so the floor follows the content: as tall as the card
    now needs, minus a slab kept back for the SLD profile underneath it.
    """

    def refit() -> None:
        wanted = panel.sizeHint().height()
        total = sum(canvas.sizes())
        if total <= 0:
            return
        scroll.setMinimumHeight(min(wanted, max(total - SLD_PANE_RESERVE_PX, STRUCTURE_PANE_FLOOR_PX)))
        if canvas.sizes()[0] >= wanted:
            return
        canvas.setSizes([wanted, max(total - wanted, 0)])

    panel.editor.contents_resized.connect(refit)


def _reveal_when_relevant(window, scroll: QScrollArea, cards: dict[str, QWidget]) -> None:
    """Scroll whichever inspector section the user just made relevant into view.

    ``STEP_INSPECTOR_SECTIONS`` 已经把常驻五段那 2.3 倍溢出解掉了：一步只露三段，判定在
    结果态本来就在第一位且可见。这一层留着兜住窗口被拖矮、参数表长到十几层的情形——那时
    三段也会溢出，而谁现在相关谁就该滚出来这条规则跟步骤化不冲突：拟合出了结果就把判定
    滚出来，选中了一层就把编辑那一层的卡滚出来，落点都是用户刚要的东西。
    """
    verdict = cards.get("inspectorResults")
    selected_layer = cards.get("inspectorSelectedLayer")

    def reveal(card: QWidget | None) -> None:
        if card is None:  # pragma: no cover - both sections are always built
            return
        # 判定本体是这张卡里的一段，不是整张卡；整张卡装着候选解列表和不确定度，比视口还
        # 高，滚到它的顶边等于把判定推到上边界之外。所以滚判定那一段自己。
        target = card.findChild(QWidget, "resultConfidenceCard") or card
        scroll.ensureWidgetVisible(target, 0, 0)

    window.fit_panel.result_published.connect(lambda _result: reveal(verdict))
    window.structure_panel.editor.component_selected.connect(
        lambda _index, component: reveal(selected_layer) if isinstance(component, api.LayerSpec) else None
    )


def _plain_section(parent: QWidget, name: str) -> tuple[QFrame, QVBoxLayout]:
    """A section whose panel already captions itself, so the shell adds nothing.

    ``titled_card`` 给的是「一句抬头 + 一圈边框」，那是一段内容还没有自己抬头时该有的样子。
    结果那一段不是：它里面本来就是三张各自带抬头和边框的卡（判定 / 参数 · 结果值 / 候选解），
    再套一层就是双线边框加一句设计稿没有的第四抬头。零边距是因为里面那三张卡自己带内边距，
    外面再垫一圈会把三段一起往里挤，跟同一栏其他段落的左边界对不齐。
    """
    card = QFrame(parent)
    card.setObjectName(name)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_SM)
    return card, layout


def _inspector_column(window) -> QWidget:
    """The right column: the sections this step needs, scrolling as one body.

    Most panels go into a ``titled_card`` because a column with no dock title
    bars has to caption its own sections -- without the caption the bordered
    boxes read as one undifferentiated stack.  A panel that already draws its own
    captioned cards gets ``_plain_section`` instead, so the shell does not caption
    it a second time.  The trailing stretch keeps the cards at their natural
    heights instead of letting the last one absorb the column.

    每一段都建出来并留在布局里，露哪几段由 ``apply_step_scope`` 按步骤切可见性。建全套是
    因为抬头、候选计数、选中层改名这些接线都挂在构造期，按步骤增删卡片等于让同一条接线
    随步骤反复断连；而 ``findChild`` 不过滤可见性，靠名字找卡的那些契约照旧成立。
    """
    column, layout = _column("inspectorColumn", "上下文检查器")
    body = QWidget()
    body.setObjectName("inspectorBody")
    body_layout = QVBoxLayout(body)
    # 边距与段间距都归零：段与段之间只有 ``.insp-sec`` 那道横线，横线两侧留白时它离两边都
    # 远，读起来是一条装饰性的分隔线而不是「这一节到此为止」。内缩由每段自己的 12px 内边距
    # 提供，所以横线是通栏的。
    body_layout.setContentsMargins(0, 0, 0, 0)
    body_layout.setSpacing(0)
    cards: dict[str, QWidget] = {}
    for name, title, subtitle, attribute in INSPECTOR_SECTIONS:
        if title is None:
            card, card_layout = _plain_section(body, name)
        else:
            card, card_layout = theme.titled_card(body, name, title, subtitle, flat=True)
        card_layout.addWidget(getattr(window, attribute))
        body_layout.addWidget(card)
        cards[name] = card
        if attribute == "selected_layer_panel":
            _name_card_after_selection(window, card, name, title)
        if attribute == "parameter_disposition":
            _name_card_after_parameter(window, card, name, title)
        if attribute == "live_metrics":
            _name_card_after_batch_mode(window, card, name)
        if attribute == "quantile_panel":
            _name_card_after_posterior(window, card, name, title)
    body_layout.addStretch(1)
    # 结果那一段照设计稿只画三张卡；面板自己带的清除按钮、证据散文、状态行留给单独立起来
    # 的面板，命令改从拟合菜单进。
    window.result_panel.set_secondary_visible(False)
    scroll = _scrolled("inspectorScroll", body)
    layout.addWidget(scroll)
    window.inspector_cards = cards
    _reveal_when_relevant(window, scroll, cards)
    return column


def _uncertainty_evidence_is_ready(window) -> bool:
    """右栏此刻该不该换成帧⑤ 那三段。

    两件事都要成立：画布列停在「不确定度」那一页，且屏上这个候选解真有一条属于自己的链。

    第二个条件不是防御性检查。没有链时三段读出来是三张写着「不可用」的卡，而三张空卡不是
    证据——整段的答案是「去跑一次采样」，那个入口在拟合菜单里，不在右栏。换过去只会把判定
    与结果值一起换走，读者手里连原来那点东西都没了。

    问的是收敛那一段：三段读的是同一条链，它有没有东西可读就是三段有没有东西可读。
    """
    if window.plot_panel.active_analysis_view() != "uncertainty":
        return False
    return window.result_panel.convergence_panel.has_evidence()


def refresh_sampling_footer(window) -> None:
    """左栏页脚在帧⑤ 换成那句 walkers 下界，别的屏交还给数据集清点。

    判据复用 ``_uncertainty_evidence_is_ready``：页脚这一句与右栏那三段属于同一帧，两处各判一
    次「算不算在帧⑤ 上」的话，会出现右栏已经换成收敛诊断、页脚还在清点数据集。

    句子本身归右栏算（``ResultsPanel.sampling_rule_summary``）——两个数一个在候选解里、一个在
    那个 walkers spin box 里，都不在左栏手上。除了随 ``apply_step_scope`` 走一遍，主窗口还把
    spin box 的 ``valueChanged`` 连到这里：那一句预告的是「下一次采样会不会被拦」，读者改了
    walkers 它就得当即改口。
    """
    ready = _uncertainty_evidence_is_ready(window)
    window.data_panel.set_sampling_summary(window.result_panel.sampling_rule_summary() if ready else None)


def apply_step_scope(window, step: int | None = None) -> None:
    """Show the canvas and inspector content the project's current step calls for.

    设计稿的六帧是六屏，不是一屏的六种滚动位置。帧③ 的画布是层堆叠加剖面、右栏是选中层
    那三段；帧① 的画布只有两张图、右栏换成判定那三段；帧④ 右栏只剩运行那一段。此前三栏
    是「全都常驻」，于是不管项目走到哪一步屏幕都是同一张合页。

    步骤取自 ``PipelineNav``，仓库里唯一算过 furthest-reached-step 的地方；运行中额外压过
    一层，因为 ``_determine_step`` 只看项目状态，看不见「此刻正在跑」。``step`` 显式给值时
    压过这两者，留给「结果已经填进卡里、项目还没提交」那一拍。引导模式自己会把整个检视器
    收起来，这里不去碰它的可见性，只决定它内部露哪几段。
    """
    cards = getattr(window, "inspector_cards", None)
    if not cards:  # pragma: no cover - the column always builds its cards
        return
    if step is None:
        step = window.pipeline_nav.current_step_index()
        if window.fit_panel.is_running:
            step = RUNNING_STEP_INDEX
    visible = STEP_INSPECTOR_SECTIONS.get(step, STEP_INSPECTOR_SECTIONS[0])
    # 帧⑤ 与帧① 是同一个流程步，换的是画布那一页——所以这一层按步骤挑完之后，还要再问一次
    # 画布此刻在看什么。只在右栏摆着结果那张卡时替换：拟合中（第 3 步）右栏是进度，那一帧
    # 不该被分析页的选择改掉。
    if "inspectorResults" in visible and _uncertainty_evidence_is_ready(window):
        visible = UNCERTAINTY_INSPECTOR_SECTIONS
    for name, card in cards.items():
        card.setVisible(name in visible)
    # 末段的横线要摘掉，而哪一段是末段刚刚才定下来：``cards`` 按 ``INSPECTOR_SECTIONS`` 的
    # 次序遍历，就是它们在栏里自上而下的次序。
    theme.mark_last_section(cards.values())
    stack_pane = window.canvas_column.findChild(QWidget, "structurePanelScroll")
    if stack_pane is not None and not window.guidance_is_visible():
        stack_pane.setVisible(step in STEPS_WITH_LAYER_STACK)
    # 画布跟着同一个步骤收窄。绘图栈的四段此前全都常驻，和检视器五段常驻是同一个毛病的
    # 另一半：每段分到 ~180px，帧⑤ 的两张子图叠在一起、SLD 剖面掉到折叠线以下。
    window.plot_panel.set_step_scope(step)
    # 左栏也跟着这一步换措辞：卡片小字在结构这一步停在点数，页脚改说结构管着哪几条曲线
    # （设计稿帧③），不再清点数据集。
    window.data_panel.set_step(step)
    # 页脚再往下压一层：帧⑤ 与帧① 同属一个流程步，上面那句 ``set_step`` 分不开它们，而这一句
    # 与右栏换段用的是同一个判据。
    refresh_sampling_footer(window)
    # 状态栏也跟着这一步换内容：结构这一步报层数与选中层，别的步把这两段收起来。
    from xrr_fitter.gui.chrome import refresh_command_bar, refresh_structure_status

    refresh_structure_status(window, active=step == STRUCTURE_STEP_INDEX)
    # 命令栏的 批量 那一组跟着画布走，画布刚换完，所以在这里重算一次。
    refresh_command_bar(window)


def follow_pipeline_step(window) -> None:
    """Keep the scoped columns in step with the project, the run, and a finished fit.

    三个边：项目状态（导入、建结构、拟合完）由导航播报；运行起停由拟合卡播报，因为
    ``_determine_step`` 看不见「正在跑」；而 ``result_published`` 是结果落到项目之前就到的
    那条边——判定卡此刻已经填好，等项目提交完再换步会让它闪一下才出现。
    """
    window.pipeline_nav.step_changed.connect(lambda _index: apply_step_scope(window))
    # 联合批量那行「N 共享 + M 独立」要数自由参数，而 ``describe_parameters`` 会读源文件——
    # 参数面板已经为参数表编译过一次，左栏订阅那个结果而不是自己再读一遍。
    window.parameters_panel.definitions_changed.connect(window.pipeline_nav.set_parameter_definitions)
    window.pipeline_nav.set_parameter_definitions(window.parameters_panel.definitions)
    # 左栏也要跟着运行状态走（设计稿帧④：拟合 current、小字「进行中 62%」）。放在收窄那条边
    # 之前接，因为 ``set_running`` 自己会播报 ``step_changed``——收窄届时读到的已经是 3。
    window.fit_panel.running_changed.connect(window.pipeline_nav.set_running)
    # 百分数从进度条本身取（``overall_percent``），所以左栏不会报出一个和条形图不一样的位置。
    # 没在运行时进度条也会动（复位、初始化），``set_running_percent`` 那时只记数不改界面。
    progress_view = window.fit_panel.progress_view
    progress_view.bar.valueChanged.connect(
        lambda _value: window.pipeline_nav.set_running_percent(progress_view.overall_percent())
    )
    # 运行边用信号的载荷，不回读 ``is_running``：那个属性问的是控制器此刻的状态，而这条边
    # 关心的是「刚刚起/停」——两者在同一拍里不保证已经一致，和 ``result_published`` 比项目
    # 提交先到一拍是同一类先后问题。停下来时交回 ``None``，让步骤重新由项目状态说话。
    window.fit_panel.running_changed.connect(
        lambda running: apply_step_scope(window, RUNNING_STEP_INDEX if running else None)
    )
    window.fit_panel.result_published.connect(lambda _result: apply_step_scope(window, RESULT_STEP_INDEX))


def _pin_columns(splitter: QSplitter) -> None:
    """Take the grip off the seams: the design's side columns are values, not drags.

    ``grid-template-columns:264px 1fr 340px`` gives only the canvas a ``1fr``.  A
    splitter hands every seam a draggable handle, so opening at 264/340 proves the
    starting numbers and nothing about their staying there -- the columns could
    still be pulled apart, which is the arrangement the design does not have.

    Disabling the handles pins the user's path alone.  ``setSizes`` keeps working,
    which is what the defaults, ``reset_layout`` and a restore from a project all
    go through; the arrow cursor keeps the 1px seam from advertising a drag that
    no longer happens.
    """
    for index in range(splitter.count()):
        handle = splitter.handle(index)
        if handle is not None:
            handle.setEnabled(False)
            handle.setCursor(Qt.CursorShape.ArrowCursor)


def apply_default_columns(splitter: QSplitter) -> None:
    """Give the side columns their budgets and the canvas whatever is left.

    Written as a function because three callers need the same arithmetic: the
    initial assembly, the first ``showEvent`` (before which the splitter has no
    width to divide), and ``reset_layout``.  Qt raises a requested size to the
    widget's own minimum, so a column asking for more than its budget silently
    takes it -- which is what the ``fits_its_budget`` contracts measure.
    """
    total = sum(splitter.sizes()) or splitter.width()
    canvas = max(total - LEFT_COLUMN_WIDTH - RIGHT_COLUMN_WIDTH, 1)
    splitter.setSizes([LEFT_COLUMN_WIDTH, canvas, RIGHT_COLUMN_WIDTH])


def _guidance_actions(window) -> dict:
    """Bind each guided step's action to the panel that already performs it.

    Guidance never reimplements a workflow; it routes to the same buttons and
    methods the expert surface uses, so both paths share one code path.
    """
    return {
        "import_files": window.data_panel.import_files_button.click,
        "initialize_structure": window.structure_panel.initialize_button.click,
        "start_fit": window.start_fit,
    }


def _layer_bounds_lookup(window):
    """选中层那张卡上的界限从哪儿来：参数声明，和参数表读的是同一份。

    卡片自己不生成界限。它要是自己编一套，同一层的厚度就会在右栏说 40–60、在参数表
    里说 1.75–441——两处都在讲拟合器能把这个数推到哪儿，讲得不一样就有一处是假的。

    周期块和梯度层没有单一的材料，返回空表；卡片那时本来也是禁用的。
    """

    def lookup(index: int) -> dict[str, tuple[float, float]]:
        structure = window.structure_panel.structure
        if structure is None or not 0 <= index < len(structure.components):
            return {}
        material = getattr(structure.components[index], "material", None)
        if material is None:
            return {}
        return layer_bounds(window.parameters_panel.definitions, index, material)

    return lookup


def _layer_locks_lookup(window):
    """密度那枚方框说的话从哪儿来：和界限条同一批声明。

    锁没锁不是这张卡自己的状态——按住这个参数的是参数表里那一行，卡片只是把它画在标签
    旁边。两处读同一份声明，就不可能一处写「未锁定」而求解器那边根本不动它。

    声明是按名字查的，所以越界的下标自然什么也查不到，卡片跟着把方框收起来。
    """

    def lookup(index: int) -> dict[str, bool]:
        return layer_locks(window.parameters_panel.definitions, index)

    return lookup


def _build_panels(window, document) -> None:
    """Construct every panel the three columns are assembled from."""
    window.project_actions = ProjectActions(window, document)
    window.data_panel = DataPanel(document)
    window.structure_panel = StructurePanel(document)
    window.parameters_panel = ParametersPanel(document)
    # 三档与两枚徽标归参数面板所有（锁定回写、跟随选中行都在那里接的线），但它们画在右栏
    # 「参数化」那一段里，而面板自己带着的那张表在画布列。检视器按属性名取段，所以这里把
    # 这一件零件抬到窗口上，和 ``live_metrics`` 一样，而不是复制一份。
    window.parameter_disposition = window.parameters_panel.disposition
    window.fit_panel = FitPanel(document)
    # 指标视图归拟合面板所有——喂它的是同一条 ``progress_changed``，起停也随同一次运行。
    # 检视器按属性名取段，所以这里把它抬到窗口上，而不是复制一份。
    window.live_metrics = window.fit_panel.live_metrics
    window.result_panel = ResultsPanel(document)
    # 帧⑤ 右栏那三段同样归结果面板所有（投影跟着同一次候选解选中走），画在检视器的三张卡
    # 里。抬到窗口上的理由与 ``live_metrics`` 一条一样：检视器按属性名取段。
    window.convergence_panel = window.result_panel.convergence_panel
    window.quantile_panel = window.result_panel.quantile_panel
    window.bootstrap_panel = window.result_panel.bootstrap_panel
    window.plot_panel = PlotPanel()
    editor = window.structure_panel.editor
    window.selected_layer_panel = SelectedLayerCard(
        editor.replace_component,
        _layer_bounds_lookup(window),
        _layer_locks_lookup(window),
    )
    editor.component_selected.connect(window.selected_layer_panel.show_component)
    # 状态栏那一段读的是同一次选中：设计稿帧③ 屏幕最下面写「选中：a-Si」。
    from xrr_fitter.gui import chrome

    editor.component_selected.connect(lambda _index, component: chrome.set_selected_layer_status(window, component))
    # 换数据集、换结构都会重发一批声明。重发和「堆叠重新选中这一层」谁先谁后没有保证，
    # 所以两条路都通到卡片：后到的那一次刷新用的一定是新声明。
    window.parameters_panel.parameter_table.rows_reloaded.connect(window.selected_layer_panel.refresh_bounds)
    window.structure_diagnostics_panel = _structure_diagnostics_hint()
    window.pipeline_nav = PipelineNav(document)
    window.dataset_summary = _detached_summary(window.data_panel)
    # 设计稿栏上写的是「导出…」：省略号表示按下去先开对话框问导出什么、导到哪，不是当场
    # 就往磁盘写。要念的完整名字由 ``accessibleName`` 承担。
    window.export_button = QPushButton("导出…")
    window.export_button.setObjectName("exportResultsButton")
    window.export_button.setAccessibleName("导出拟合结果")
    window.export_button.setToolTip("将当前项目的拟合结果导出到所选目录")
    # 字形只挂在 文件 菜单那条 QAction 上：设计稿命令栏这枚是 ``.btn`` 纯文字格。
    window.export_button.clicked.connect(window.export_results_dialog)
    # Guidance and the plot share the canvas: they are two surfaces onto one
    # project, so a stack swaps them without either being rebuilt.
    window.guidance = GuidancePanel(document, _guidance_actions(window))
    window.central_stack = QStackedWidget()
    window.central_stack.setObjectName("centralStack")
    window.central_stack.addWidget(window.guidance)
    window.central_stack.addWidget(window.plot_panel)


def install_workspace(window, document) -> None:
    """Build the design's three fixed columns as the window's central widget.

    The columns are a splitter and not QDockWidgets: docks put a title bar and a
    close button on each column, showed one inspector section at a time behind
    tabs, and let the whole arrangement be dragged apart.  A splitter drops the
    title bars and the tabs, but it would keep the dragging -- so the seams are
    pinned here (:func:`_pin_columns`) and the columns are as fixed as the design's
    grid says, with none collapsible away.  The divider is narrowed to the design's
    ``1px solid var(--border)``; left at its default it is a 4px band of window
    colour, which reads as a pale gap between the columns rather than as a line.
    """
    _build_panels(window, document)
    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setObjectName("workspaceSplitter")
    splitter.setHandleWidth(1)
    window.nav_column = _navigation_column(window)
    window.canvas_column = _canvas_column(window)
    window.inspector_column = _inspector_column(window)
    for column in (window.nav_column, window.canvas_column, window.inspector_column):
        splitter.addWidget(column)
    _pin_columns(splitter)
    window.workspace_splitter = splitter
    window.setCentralWidget(splitter)
    apply_default_columns(splitter)
    follow_pipeline_step(window)
    apply_step_scope(window)


def install_workflow_actions(window) -> None:
    window._workflow_actions = {}
    for object_name, text, shortcut, callback_name in WORKFLOW_ACTION_SPECS:
        action = QAction(text, window)
        action.setObjectName(object_name)
        action.setShortcut(QKeySequence(shortcut))
        action.setToolTip(text)
        action.setStatusTip(text)
        action.setIcon(command_icon(callback_name))
        callback = getattr(window, callback_name)
        action.triggered.connect(lambda _checked=False, operation=callback: operation())
        window.addAction(action)
        window._workflow_actions[object_name] = action
