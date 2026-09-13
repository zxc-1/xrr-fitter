"""Pipeline step navigator for the left column.

Displays the five canonical pipeline stages (import → structure → parameters →
fit → results) as one connected run: every step carries a state-coloured
marker, its own name, a muted line saying what the step is for, and a rail down
to the next step.  The state is double-encoded exactly as the guided header
does it -- a reached step swaps its ordinal for a ✓ and the current step is
bolded -- so progress survives a monochrome display and being read aloud.
Only imports from ``xrr_fitter.api`` and ``xrr_fitter.gui.*``.
"""

from __future__ import annotations

from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.navigation.methods import UncertaintyMethodNav
from xrr_fitter.gui.navigation.steps import StepRow, build_step
from xrr_fitter.gui.structure.naming import NATIVE_OXIDE_SUFFIX

PIPELINE_STEPS = (
    # 第二项是这一步的用途。它常驻 tooltip（见 ``steps.build_step``），正文里只在一种情况下露面：
    # 项目正好走到这一步、而这一步又还没有数可报。项目还没走到的那几步照设计稿写「—」，见
    # ``_apply_state``。
    ("数据", "导入并选择数据集"),
    ("结构", "定义样品结构"),
    ("参数", "设定参数边界与先验"),
    ("拟合", "运行自动拟合"),
    ("结果", "查看候选解与不确定度"),
    # The design's sixth step.  It stays pending for good: nothing on the project
    # records that a file was written, and the mockup draws it pending even beside
    # a converged 可信 result, so the step states what is left rather than what is
    # finished.  ``_apply_index`` needs no special case for it -- the furthest
    # index the project can report is 结果, which leaves this one after the
    # current step in every state.
    ("导出", "导出 ORSO、Excel 与图"),
)

# ``PIPELINE_STEPS`` 里那几个别处也要指名道姓的下标。放在被索引的那个列表旁边，改列表时
# 一眼看得到它们；此前它们待在 ``window_layout`` 里，而那边并不持有这个列表。
#
# 结构这一步有两处读它：状态栏在这一步才报层数与选中层，左栏的数据集卡片与页脚也在这一步
# 换说法（设计稿帧③ 的小字停在点数，页脚写结构管着哪几条曲线）。数据面板不能从
# ``window_layout`` 取——那个模块反过来要导入 ``DataPanel``。
#
# 拟合这一步算不出来：``_determine_step`` 只读项目字段，而「正在跑」不是字段。所以运行中的高亮
# 靠 ``set_running`` 从外面盖上来（帧④），落在这个下标上。
STRUCTURE_STEP_INDEX = 1
RUNNING_STEP_INDEX = 3
RESULT_STEP_INDEX = 4

# 运行中那行小字。设计稿帧④ 写的是「进行中 62%」，百分数取自进度条本身。
RUNNING_STATE_TEMPLATE = "进行中 {percent}%"

# 结构立起来了、边界一个都没动过时，「参数」那行读什么。设计稿帧③ 写的就是这两个字；空项目
# 上同一行写的是用途说明「设定参数边界与先验」。两者得分得开，读者才知道现在是「还没轮到这
# 一步」还是「轮到了，等你看」。
PARAMETERS_PENDING_STATE = "待复核"

# 项目还没走到的那几步读什么。设计稿帧③ 与帧④ 的「结果 / 导出」两行都是这一个字符：这一步
# 现在没有可报的，而破折号说的正是这件事。用途那句话退到 tooltip 里，需要的人仍然找得到。
UNREACHED_STATE = "—"

# 「拟合」那行的两句话。设计稿帧① 是「全自动 · 已收敛」，帧③ 是「未开始」——都不带批量模式，
# 因为联合／独立已经在「数据」（「3 联合」）与「结构」（「4 层共享」）两行里各说过一次。
#
# 项目里没有 ``converged`` 字段，「已收敛」不能凭空造一个数：自动拟合只在收敛后才回写
# ``last_valid_result``，所以那份结果的存在本身就是这句话的依据。
FIT_CONVERGED_STATE = "全自动 · 已收敛"
FIT_PENDING_STATE = "未开始"

# 「导出」那行。设计稿帧① 写的是三个格式名，不是一句「导出 ORSO、Excel 与图」：这一行报的是
# 导出能给你什么，读者一眼要数得出有几种。
EXPORT_FORMATS_STATE = "ORSO / Excel / 图"

# 站在某一步上时，那行小字末尾多出来的一截。设计稿只有「结构」有：帧③ 是「4 层 · 编辑中」，
# 而同一个结构在别的帧里写的是「4 层」。所以「编辑中」不是项目里的另一个字段，它就是「你在
# 这儿」——层堆叠此刻正摆在画布上，改的就是这几层。没有状态可报（结构还没建）时不加，否则
# 会写出「定义样品结构 · 编辑中」这种把用途说明当成状态的句子。
CURRENT_STEP_SUFFIX = {"结构": "编辑中"}

# 这叠层里最上面那层是氧化层时，层数后面多出来的一截：设计稿帧① 写的是「5 层 · 含表面氧化」。
# 氧化层和别的层不是一回事——它由软件按氧化表建议、读者点过「接受」才进来，而它一进来就多吃
# 三个自由参数。回头问「12 个自由参数」里的 12 是怎么来的，答案有一部分在这一行里。
OXIDE_TRAIT = "含表面氧化"

# ``<div class="nav-sec" style="margin-top:6px">分析管线</div>``：管线抬头与上面那张数据集
# 列表之间的间隔。设计稿在这一处写的是行内的 6px，不是 ``--sm``，所以照抄它而不是就近取整
# 到 8——两句抬头之间那点差别正是左栏「一段结束了」的读法。
SECTION_GAP_PX = 6
PIPELINE_HEADING_TEXT = "分析管线"


def _active_dataset(project: api.XrrProject) -> api.DatasetProject | None:
    """The dataset every step reads its state from, or ``None`` before the first import.

    Falls back to the first dataset when nothing is selected: the steps report on
    *a* dataset rather than going blank, and an unselected project still has a
    structure and a result worth counting.
    """
    if not project.datasets:
        return None
    active_id = project.ui_state.active_dataset_id
    return next(
        (dataset for dataset in project.datasets if dataset.dataset_id == active_id),
        project.datasets[0],
    )


def _determine_step(project: api.XrrProject) -> int:
    """Return the 0-based index of the furthest reached pipeline step."""
    dataset = _active_dataset(project)
    if dataset is None:
        return 0
    if dataset.last_valid_result is not None:
        return 4
    if dataset.structure is not None:
        return 2
    return 1


# 「够 30 点才算可拟合」这条判定是数据面板定的，这里引用同一个数而不是各写一遍：同一个项目
# 在左栏报「3 可拟合」、在数据列表里标两行「数据点不足」的话，读者不知道该信哪个。
FITTABLE_POINT_FLOOR = 30

# 层堆叠列表两端那两行：入射介质与基底。它们不在 ``components`` 里，却各占一行也各自带
# 参数进拟合，设计稿的层数（帧③「4 层」= 空气 + SiO₂ + a-Si + c-Si）把它们算在内。
STACK_ENDS = 2


def _data_state(project: api.XrrProject) -> str:
    """有几条数据，以及它们此刻算是几次拟合。

    联合批量下「N 可拟合」数的是「有几条能各自拟合」，而那正是联合模式取消掉的事——
    三条曲线共用一个目标函数，一次收敛。设计稿帧④ 因此把这一行收成「3 联合」。

    全部都能拟合时只报一个数：设计稿帧③ 是「3 可拟合」，帧① 才写成「4 导入 · 3 可拟合」。
    第二个数只有在它和第一个不一样时才带信息量——差出来的那一条正是要读者去看的东西。
    """
    if not project.datasets:
        return ""
    if project.batch_mode == "joint":
        return f"{len(project.datasets)} 联合"
    fittable = sum(1 for dataset in project.datasets if sum(dataset.fit_mask) >= FITTABLE_POINT_FLOOR)
    if fittable == len(project.datasets):
        return f"{fittable} 可拟合"
    return f"{len(project.datasets)} 导入 · {fittable} 可拟合"


def _structure_state(project: api.XrrProject) -> str:
    dataset = _active_dataset(project)
    if dataset is None or dataset.structure is None:
        return ""
    # 联合批量下这一叠层是几条曲线共用的那一叠，不只是活动数据集自己的——设计稿帧④
    # 写「4 层共享」，说的就是这个区别。
    shared = "共享" if project.batch_mode == "joint" else ""
    # 数的是层堆叠里那几行：入射介质与基底也各占一行，各自带着粗糙度与密度进拟合。设计稿
    # 帧③ 的样品只有 SiO₂ + a-Si 两层，列表画四行，左栏与状态栏都写「4 层」。
    rows = len(dataset.structure.components) + STACK_ENDS
    return f"{rows} 层{shared}"


def _structure_trait(project: api.XrrProject) -> str:
    """这叠层除了层数还值得说的一件事：最上面那层是氧化层。

    判的是第一层而不是「这叠层里有没有」：``services.structures`` 按 ``location`` 把氧化层插在
    表面（``components`` 最前）或基底那一头，两头插进来的层名一模一样。基底侧那一层不在表面
    上，照着「含表面氧化」去表面找会找到一层不存在的东西。
    """
    dataset = _active_dataset(project)
    if dataset is None or dataset.structure is None or not dataset.structure.components:
        return ""
    return OXIDE_TRAIT if dataset.structure.components[0].name.endswith(NATIVE_OXIDE_SUFFIX) else ""


def _parameters_state(project: api.XrrProject, free_count: int | None = None, *, passed: bool = False) -> str:
    """这一步到哪儿了：走过了报自由参数的个数，还没走过报「待复核」。

    设计稿里这一行只有两种形状。走过之后是个数——帧① 的「12 个自由参数」，联合时帧④ 拆成
    「9 共享 + 6 独立」；还没走过是帧③ 的「待复核」：结构已经立起来了，边界一个都没动过也
    有话可说，说的是「轮到你了」。

    锁了几个不进这一行。锁定数在参数表里逐行看得见，而左栏这一行的活是报进度；把它写成
    「1 锁定」会让随手锁一个边界就顶掉「待复核」，那个形状设计稿六帧里一次都没有。
    """
    dataset = _active_dataset(project)
    if dataset is None or dataset.structure is None:
        return ""
    joint = _joint_parameters_state(project, free_count)
    if joint is not None:
        return joint
    if passed and free_count is not None:
        return f"{free_count} 个自由参数"
    return PARAMETERS_PENDING_STATE


def _joint_parameters_state(project: api.XrrProject, free_count: int | None) -> str | None:
    """帧④ 的「9 共享 + 6 独立」：联合拟合把参数分成两半，这一行报的是这个分界。

    共享的那几个是显式声明的绑定规则；剩下的自由参数每条曲线各有一份（标度、本底这类），
    所以独立数是「每条曲线剩下几个」乘曲线条数。只报共享数会让读者以为剩下的也绑在一起。

    自由数由参数面板算好递进来（``definitions_changed``）：``describe_parameters`` 会读源
    文件，这一行自己再读一遍等于把那次编译做两遍，源文件缺失时还会抛。数还没到手（面板尚
    未刷新、或读取失败）时交回 ``None``，让调用方退回持久化字段那套读数。
    """
    if project.batch_mode != "joint" or free_count is None:
        return None
    shared = len(project.sharing_rules)
    independent = max(free_count - shared, 0) * len(project.datasets)
    return f"{shared} 共享 + {independent} 独立"


def _fit_state(project: api.XrrProject) -> str:
    """跑没跑过，以及跑的是哪条路。

    结构还没立起来时交回空串：这一步连「未开始」都还谈不上，``_apply_state`` 会照设计稿
    写「—」。
    """
    dataset = _active_dataset(project)
    if dataset is None or dataset.structure is None:
        return ""
    finished = any(value.last_valid_result is not None for value in project.datasets)
    return FIT_CONVERGED_STATE if finished else FIT_PENDING_STATE


def _result_state(project: api.XrrProject) -> str:
    dataset = _active_dataset(project)
    result = None if dataset is None else dataset.last_valid_result
    if result is None:
        return ""
    best = result.best_candidate
    if best is None:
        return str(result.confidence.value)
    return f"{result.confidence.value} · J={best.objective:.2f}"


def _export_state(project: api.XrrProject) -> str:
    """有结果可导时列出三个格式名，没有就交回空串。

    项目里没有「导出过没有」这个字段，这一行报的因此不是进度而是能力：有一份结果在手，
    ORSO、Excel 与图三条路就都通了（设计稿帧①）。
    """
    dataset = _active_dataset(project)
    if dataset is None or dataset.last_valid_result is None:
        return ""
    return EXPORT_FORMATS_STATE


# 每一步那行小字读什么。设计稿在这个位置写的是这个项目此刻的数字，不是这一步的通用说明：
# 帧① 是「4 导入 · 3 可拟合 / 5 层 / 12 个自由参数 / 全自动 · 已收敛 / 可信 · J=1.83」，帧③
# 同一个位置换成「3 可拟合 / 4 层 · 编辑中 / 待复核 / 未开始」。常量文案会让左栏在整个流程里
# 一个字都不变，读者要知道「现在几层」得去右栏或画布里找。
#
# 交回空串的意思是「这一步还没有状态」，那行就退回 ``PIPELINE_STEPS`` 里的用途说明。导出没有
# 读取器：项目上没有任何字段记下文件写没写过，设计稿也把它画成待办。
STEP_STATE_READERS = {
    "数据": _data_state,
    "结构": _structure_state,
    "参数": _parameters_state,
    "拟合": _fit_state,
    "结果": _result_state,
    "导出": _export_state,
}


class PipelineNav(QWidget):
    """Vertical step indicator for the XRR analysis pipeline."""

    # 项目走到了哪一步，说给要跟着换内容的那些列听。设计稿的每一帧都是「一步一屏」：
    # 帧③ 的画布是层堆叠加剖面、右栏是选中层那三段，帧① 的画布只有两张图、右栏换成判定
    # 那三段。这个导航是仓库里唯一算过“furthest reached step”的地方，所以由它播报，而不是
    # 让画布和检视器各自再算一遍同一个判断。
    step_changed = Signal(int)

    def __init__(self, document=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pipelineNav")
        # The accessible name used to come from the enclosing QDockWidget's title.
        # On the design's left rail this is a plain child widget with no title bar
        # above it, so the name has to be its own or a screen reader reaches the
        # steps through an unnamed container.
        self.setAccessibleName("管线导航面板")
        self._document = document
        self._current_index = 0
        # 项目最远走到哪（``_determine_step``），以及读者手点停在哪。两件事分开存：走过的步
        # 还能回去看，所以「在看第几步」不等于「走到第几步」。手点的那个是 ``None`` 时就跟着
        # 项目走，这也是没人点过时的样子。
        self._reached_index = 0
        self._selected_index: int | None = None
        # 拟合是否正在跑，以及跑到哪了。这两个字段是覆盖，不是赋值：``_project_index`` 仍然
        # 照项目状态算，运行结束撤掉覆盖就原样回去，不必反推「运行前是第几步」。
        self._running = False
        self._running_percent = 0
        # 参数面板编译出来的自由参数个数。联合批量那行「N 共享 + M 独立」要用它，而它只有
        # 编译过参数的一方知道；没人报过时是 ``None``，那一行就退回持久化字段那套读数。
        self._free_parameters: int | None = None
        # 此刻在采的那条链有多少 walkers，没在采时是 ``None``。同样是覆盖而不是赋值：报告里那
        # 四样在不在照旧自己算，采样结束撤掉覆盖就原样交还给报告（``operation_state`` 两头都推）。
        self._sampling_walkers: int | None = None
        layout = QVBoxLayout(self)
        # ``.pipe{padding:var(--sm) 0}``: the band's own padding is vertical only --
        # each ``.pstep`` carries the horizontal inset, so a margin here would indent
        # the rows twice and pull the dots off the column the rail is drawn in.
        # 上边距是 ``.nav-sec`` 的 ``margin-top:6px``：抬头归这个控件（见下），所以栏顶到
        # 抬头之间那点空隙也归它；``.pipe`` 自己的上内边距改由抬头下面那段间隔给出。
        layout.setContentsMargins(0, SECTION_GAP_PX, 0, theme.SPACE_SM)
        # No spacing between steps: the connector is inside the row it belongs to, so
        # any gap here would break the line the six markers are meant to read as.
        layout.setSpacing(0)

        # ``.nav-sec 分析管线``：抬头是这条管线的一部分，不是栏的一部分——它随着它命名的
        # 那六行一起滚动、一起随步进重排，挂到栏上就会跟自己的列表脱开。
        self.section_heading = QLabel(PIPELINE_HEADING_TEXT, self)
        self.section_heading.setObjectName("pipelineNavHeader")
        theme.apply_section_heading(self.section_heading, tracking_px=theme.RAIL_SECTION_TRACKING_PX)
        self.section_heading.setContentsMargins(theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_SM)
        layout.addWidget(self.section_heading)
        # ``.pipe`` 的上内边距：抬头与第一步之间的 8px。栏的上边距已经让给了抬头，所以这一段
        # 单独加，否则第一步会贴着抬头。
        layout.addSpacing(theme.SPACE_SM)

        self.step_labels: list[QLabel] = []
        self._dots: list[QLabel] = []
        self._step_rows: list[StepRow] = []
        # 那行小字要按项目状态改写，所以留住它们；此前只留了标题和圆点，于是文案只能是常量。
        self._descriptions: list[QLabel] = []

        last = len(PIPELINE_STEPS) - 1
        for index, (title, description) in enumerate(PIPELINE_STEPS):
            parts = build_step(self, title, description, connected=index < last)
            layout.addWidget(parts.row)
            parts.row.clicked.connect(partial(self.select_step, index))
            self.step_labels.append(parts.label)
            self._dots.append(parts.marker)
            self._step_rows.append(parts.row)
            self._descriptions.append(parts.description)

        # 帧⑤ 的第二段。挂在六步下面，而不是取代它们：那一帧的左栏确实只画了这四行，可它是一张
        # 静态图，而实物里那六行是导航——从「结果」点回「结构」才改得了层。所以照方案的说法「在
        # 结果这一步展开」，六行留在原处，这一段跟着「结果」进出（见 ``_apply_index``）。
        self._methods = UncertaintyMethodNav(self)
        layout.addWidget(self._methods)
        # 抬头与那四个标题是只读入口：文字归 ``UncertaintyMethodNav`` 写，这里只把它们转出去，
        # 免得读者为了看一句抬头去摸私有字段。
        self.method_heading = self._methods.heading
        self.method_labels = tuple(self._methods.labels)

        layout.addStretch(1)
        self._apply_index(0)

        if self._document is not None:
            self._document.project_changed.connect(self._on_project_changed)
            self._sync_state(self._document.project)

    def set_document(self, document) -> None:
        """Late-bind to a document after construction."""
        if self._document is not None:
            self._document.project_changed.disconnect(self._on_project_changed)
        self._document = document
        if document is not None:
            document.project_changed.connect(self._on_project_changed)
            self._sync_state(document.project)

    def set_parameter_definitions(self, definitions: object) -> None:
        """收下参数面板刚编译好的声明，只留下「几个自由参数」这一个数。

        联合批量那行读的是「共享几个、各自还剩几个」，后半边只有编译过参数的一方算得出来。
        """
        self._free_parameters = sum(1 for item in definitions if not item.locked) or None
        if self._document is not None:
            self._apply_state(self._document.project)

    def current_step_index(self) -> int:
        return self._current_index

    def select_step(self, index: int) -> None:
        """停到走过的某一步上，比如从「参数」回到「结构」去再加一层。

        只认走过的步。往前那几步的内容项目里还没有，挪过去右栏就换成一屏空卡；运行中整块
        都不接受选择，那一刻左栏归 ``set_running`` 管（设计稿帧④）。

        手点的落点只在项目状态不变的这段时间里有效，见 ``_sync_state``。
        """
        if self._running or not 0 <= index <= self._reached_index:
            return
        if index == self._current_index:
            return
        self._selected_index = index
        self._refresh()

    def _resolve_index(self) -> int:
        """三个来源合成「现在在看第几步」：运行覆盖 > 手点 > 项目走到哪。"""
        if self._running:
            return RUNNING_STEP_INDEX
        if self._selected_index is not None:
            return min(self._selected_index, self._reached_index)
        return self._reached_index

    def set_running(self, running: bool, percent: int | None = None) -> None:
        """拟合跑起来了/停了，把左栏钉在「拟合」那一步或松开。

        设计稿帧④ 把这一步画成 current、小字写「进行中 62%」，而 ``_determine_step`` 永远算不出
        它——那个判断只看项目里存了什么，「正在跑」不是项目字段。同一次运行里右栏和画布已经跟着
        ``running_changed`` 收窄了（``window_layout.apply_step_scope`` 的 ``RUNNING_STEP_INDEX``），
        左栏此前是唯一没跟上的那一列：右边在演运行中，左边说项目停在「参数」。

        ``percent`` 省略时保留上一次的读数，这样进度推进和「开始/结束」可以分两路送进来。
        """
        if percent is not None:
            self._running_percent = max(0, min(100, int(percent)))
        if running == self._running:
            # 运行中反复送进度：不必重刷整块（会连带重算六行状态），把那行小字改掉就够。
            if running:
                self._descriptions[RUNNING_STEP_INDEX].setText(
                    RUNNING_STATE_TEMPLATE.format(percent=self._running_percent)
                )
            return
        self._running = running
        self._refresh()

    def set_running_percent(self, percent: int) -> None:
        """进度条动了。没在运行就是过路信号（复位、初始化），照旧忽略。"""
        if not self._running:
            self._running_percent = max(0, min(100, int(percent)))
            return
        self.set_running(True, percent)

    def set_sampling_walkers(self, walkers: int | None) -> None:
        """MCMC 采起来了/停了，把方法段第四行钉在「采样中」或交还给报告。

        与 ``set_running`` 是同一类覆盖，理由也一样：「链在跑」不是报告里的字段，报告要等采完才
        有 ``mcmc``。区别在于这一路只影响方法段那四行，所以没有报告（独立构造的面板）时直接递
        ``None`` 给它，而不是走整块重画。

        ``walkers`` 取采样期间那个 spin box 的值，由 ``operation_state.refresh_operation_state``
        两头推：起来时推数、停了推 ``None``。只推一头的话，链停了半小时左栏还在报「采样中」。
        """
        if walkers == self._sampling_walkers:
            return
        self._sampling_walkers = walkers
        if self._document is None:
            self._methods.set_report(None, sampling_walkers=walkers)
            return
        self._apply_methods(self._document.project)

    def step_rows(self) -> tuple[QWidget, ...]:
        """The six ``.pstep`` rows, in order.

        Exposed because a row's height is the contract: the design fits six of
        them plus a dataset list and a summary into one 264px column, so what a
        test has to be able to measure is the row, not the label inside it.
        """
        return tuple(self._step_rows)

    def step_titles(self) -> tuple[str, ...]:
        return tuple(title for title, _ in PIPELINE_STEPS)

    def method_captions(self) -> tuple[str, ...]:
        """方法段那四行小字此刻读什么。"""
        return self._methods.captions()

    def methods_visible(self) -> bool:
        """方法段此刻算不算露面。

        读的是显式隐藏位，不是 ``isVisible()``：窗口还没 show 出来时后者一律为假，而「这一段
        该不该出现」是它自己的状态，跟祖先有没有上屏无关。
        """
        return not self._methods.isHidden()

    def _on_project_changed(self, project: object) -> None:
        self._sync_state(project)

    def _apply_index(self, index: int) -> None:
        """Paint every step for a current index, glyph and colour together.

        The guided header encodes state exactly this way, and the two headers sit
        one above the other in guidance mode: if the nav kept its own ●/◦/○
        vocabulary the same project would report its progress in two different
        alphabets on one screen.

        走过的那几行同时开放点击。可点与否照 ``_reached_index`` 算，不是照 ``index``：从
        「参数」回到「结构」之后，「参数」那行仍然回得去。
        """
        for i, marker in enumerate(self._dots):
            if i < index:
                state, glyph = "done", "✓"
            elif i == index:
                state, glyph = "current", str(i + 1)
            else:
                state, glyph = "pending", str(i + 1)
            marker.setText(glyph)
            theme.set_step_state(marker, state)
            theme.set_step_state(self.step_labels[i], state)
            self._step_rows[i].set_reachable(not self._running and i <= self._reached_index)
        # 方法段只在「结果」这一步露面（设计稿帧⑤）。判据落在这里而不是 ``_apply_state``：可见性
        # 是「在看第几步」的事，而这一步在 ``_refresh`` 里没有文档时也算得出来。
        self._methods.setVisible(index == RESULT_STEP_INDEX)

    def _apply_state(self, project: api.XrrProject) -> None:
        """Rewrite each step's sub-line to what the project now says about it.

        没数可报时分两种写法。项目还没走到的那几步照设计稿写「—」（帧③／帧④ 的「结果 /
        导出」两行）：说的是「这一步现在没有可报的」。项目正好停在这一步、而这一步又还没有
        数——比如刚导完数据、结构还没建——才退回用途那句话，因为那一刻它正是下一步该做的事。

        用途说明始终留在 tooltip 里（``steps.build_step`` 已经挂在标题上），所以两种写法都不会让
        那句话在界面上消失。
        """
        for index, ((title, purpose), label) in enumerate(zip(PIPELINE_STEPS, self._descriptions, strict=True)):
            label.setText(self._caption_for(index, title, purpose, project))
            label.setToolTip(purpose)

    def _caption_for(self, index: int, title: str, purpose: str, project: api.XrrProject) -> str:
        """这一行小字最终读什么：读数、读数加「你在这儿」、用途说明，或者「—」。"""
        if self._running and index == RUNNING_STEP_INDEX:
            # ``_fit_state`` 读的是项目字段，正在跑的时候它只会说「未开始」——那一行就在屏幕
            # 中间，旁边的进度条正在动。运行中这一行归运行状态。
            return RUNNING_STATE_TEMPLATE.format(percent=self._running_percent)
        state = self._step_state_text(index, title, project)
        suffix = self._step_suffix(index, title, project)
        if state and suffix:
            state = f"{state} · {suffix}"
        return state or (purpose if index <= self._reached_index else UNREACHED_STATE)

    def _step_suffix(self, index: int, title: str, project: api.XrrProject) -> str:
        """读数后面那一截：站在这一步上时是「你在这儿」，否则是这一步自己的特征。

        两者不叠加。设计稿帧③ 站在结构步、那叠层第一行也是「SiO₂ · 表面氧化层」，写的却只是
        「4 层 · 编辑中」：264px 的栏宽装不下两截，而站在这一步上时氧化层就摆在画布第一行，
        「你在这儿」比它更急着说。
        """
        if index == self._current_index:
            return CURRENT_STEP_SUFFIX.get(title, "")
        return _structure_trait(project) if title == "结构" else ""

    def _step_state_text(self, index: int, title: str, project: api.XrrProject) -> str:
        """这一步此刻有什么数可报，没有就是空串。

        「参数」那一步比别人多两个入参（自由参数个数、走过没走过），所以它在这里单开一条，而
        不是把这两样塞进 ``STEP_STATE_READERS`` 里让另外五个读者陪着收。
        """
        if title == "参数":
            return _parameters_state(project, self._free_parameters, passed=index < self._reached_index)
        reader = STEP_STATE_READERS.get(title)
        return reader(project) if reader is not None else ""

    def _apply_methods(self, project: api.XrrProject) -> None:
        """把活动数据集那份不确定度报告递给方法段。

        报告可能不在（拟合刚跑完、四样证据一样都没算），那时递 ``None``——「站在第一行上」的
        判据在 ``methods.uncertainty_method_rows`` 里，这里只管把报告取出来。
        """
        dataset = _active_dataset(project)
        result = None if dataset is None else dataset.last_valid_result
        self._methods.set_report(
            None if result is None else result.uncertainty,
            sampling_walkers=self._sampling_walkers,
        )

    def _refresh(self) -> None:
        """按当前项目重画一遍。没有文档时（独立构造的面板）只有覆盖可画。"""
        if self._document is None:
            self._current_index = self._resolve_index()
            self._apply_index(self._current_index)
            self.step_changed.emit(self._current_index)
            return
        self._sync_state(self._document.project)

    def _sync_state(self, project: api.XrrProject) -> None:
        reached = _determine_step(project)
        if reached != self._reached_index:
            # 项目自己往前走了一步（拟合出了结果、结构立起来了），手点的落点让位：那个落点是
            # 对着上一个项目状态选的。否则拟合跑完，屏幕还停在读者半小时前点的「结构」上，而
            # 结果就在旁边没人看见。
            self._selected_index = None
        self._reached_index = reached
        # 运行中钉在「拟合」：项目状态照算不误，只是这一刻不用它。撤掉覆盖就回到它算出来的那一步。
        self._current_index = self._resolve_index()
        self._apply_index(self._current_index)
        self._apply_state(project)
        self._apply_methods(project)
        # 每次都播报，不只在索引变了的时候：同一步之内项目内容也会变（加了一层、换了活动
        # 数据集），而跟着这一步走的那几段卡里装的正是那些内容。只在跳步时播报，就会出现
        # 「步骤没变所以没人刷新」的死角。
        self.step_changed.emit(self._current_index)
