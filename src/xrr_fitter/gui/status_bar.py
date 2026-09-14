"""窗口底部状态栏：一行若干段，段与段之间只靠间距分开。

设计稿的 ``.statusbar`` 是一条 30px 高的 flex 行：``gap:var(--md)`` 分段、``.spring``
把左右两组撑开，段内行文一律是「说明文字（``--ink-muted``）＋ 取值（``--ink``、600）」，
没有任何分隔竖线。

``QStatusBar`` 自己那两块区域（``addWidget`` 与 ``addPermanentWidget``）的位置在构造时
就定了，而设计稿逐帧换段——判定在帧① 在左边写作「拟合结果：」，在帧⑤ 挪到右边写作
「判定：」。所以这里把所有段放进一个自带 ``QHBoxLayout`` 的容器，左右由一个弹簧控件分界：
段的进出是显隐，段的换边是相对弹簧的 ``insertWidget``。

容器整体作为常驻控件挂进状态栏。拒绝提示走 ``showMessage``，而它只隐藏非常驻控件，
常驻挂法能让提示落在左边而右半边照旧可读。
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QStatusBar, QWidget

from xrr_fitter.gui import theme

# ``.statusbar{height:30px;padding:0 var(--md);gap:var(--md);font-size:11.5px}``。
# ``setPixelSize`` 只收整数，11.5 取 12（见 ``theme`` 里同样的取整）。
STATUS_BAR_H = 30
STATUS_PAD_H = theme.SPACE_MD
SPAN_GAP_PX = theme.SPACE_MD
STATUS_FONT_PX = 12

# ``.statusbar .dot{width:8px;height:8px;border-radius:50%;margin-right:5px}``：真圆点，
# 不是 ``●`` 字形——字形的大小跟着字体走，圆角方块能按设计稿给准直径。直径与 QSS 里的圆角
# 共用 ``theme`` 的一个来源。
STATUS_DOT_PX = theme.STATUS_DOT_PX
STATUS_DOT_GAP_PX = 5

# ``.statusbar b{color:var(--ink);font-weight:600}``。
STATUS_VALUE_WEIGHT = QFont.Weight.DemiBold

# 判定那一段里字形与判定词同在一个加粗行内（``<b>● 可信</b>``），它们之间只有一个空格，
# 不是段间的 12px。
GLYPH_GAP_PX = 4

# 每一段的说明文字都照设计稿抄，包括全角冒号与半角空格——它们决定了说明与取值之间贴多紧。
# 判定那一段在帧① 落在左边读作「拟合结果：」，在帧⑤ 挪到右边读作「判定：」。
VERDICT_CAPTION = "拟合结果："
VERDICT_RIGHT_CAPTION = "判定："
STEP_CAPTION = "第 "
STAGE_CAPTION = "阶段 "
RHAT_CAPTION = "split-R̂ "
ESS_CAPTION = " · ESS "
SELECTED_CAPTION = "选中："
NEXT_CAPTION = "下一步："
BATCH_CAPTION = "模式："
REMAINING_CAPTION = "预计剩余 "
DATASET_CAPTION = "活动数据集："
SOURCE_CAPTION = "源校验："

# 结构那一步右半边报的下一步动作。设计稿把它写成加粗的强调色，说明文字是「下一步：」。
NEXT_ACTION_VALUE = "开始拟合"

# 一段露不露面由三问相乘：这一帧的步骤允不允许（``SPAN_ALLOWED``）、当下有没有操作在跑
# （``SPAN_RUN_SCOPE``），以及它有没有取值可报（问把关标签）。三问分开存，才不会在步骤
# 切回来时把一段空段一起唤醒，也不会让两个刷新入口互相擦掉对方的判断——步骤归管线，
# 运行归操作状态，它们各写自己那一个属性。
SPAN_ALLOWED = "spanAllowed"
SPAN_RUN_SCOPE = "spanRunScope"
SPAN_GATES = (SPAN_ALLOWED, SPAN_RUN_SCOPE)


def _status_font(*, bold: bool, tabular: bool = False) -> QFont:
    font = QFont()
    font.setPixelSize(STATUS_FONT_PX)
    if bold:
        font.setWeight(STATUS_VALUE_WEIGHT)
    if tabular:
        # ``.tnum{font-variant-numeric:tabular-nums}``：进度与阶段每帧都在变，等宽数字
        # 让取值不随位数抖动。
        font.setFeature(QFont.Tag("tnum"), 1)
    return font


class StatusSpan(QWidget):
    """设计稿里的一个 ``<span>``：段内各片贴着排，段外由容器的 12px 间距分开。

    段是一个控件而不是一串平铺的标签，为的是两件事：说明文字与取值之间不能被段间距
    拉开（``拟合结果：`` 紧跟着加粗的判定词），以及整段的进出只需一次 ``setVisible``。
    """

    def __init__(self, parent: QWidget, name: str) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._layout = layout
        # 一段该不该露面，除了步骤与运行状态那两问（见上面的门禁属性），还得问它有没有
        # 取值可报。后者由这些「把关标签」的文字回答——说明文字永远非空，问它等于不问。
        self._gates: list[QLabel] = []

    @property
    def gates(self) -> tuple[QLabel, ...]:
        return tuple(self._gates)

    def add_text(
        self,
        text: str,
        *,
        name: str,
        bold: bool = False,
        muted: bool = True,
        tabular: bool = False,
        gate: bool = False,
    ) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName(name)
        label.setFont(_status_font(bold=bold, tabular=tabular))
        if muted:
            label.setProperty("mutedText", True)
        self._layout.addWidget(label)
        if gate:
            self._gates.append(label)
        return label

    def adopt(self, label: QLabel) -> QLabel:
        """把别处造好的标签收进这一段（``sourceStatusLabel`` 归项目动作所有）。"""
        label.setParent(self)
        label.setFont(_status_font(bold=True))
        label.setProperty("mutedText", False)
        self._layout.addWidget(label)
        self._gates.append(label)
        return label

    def add_gap(self, pixels: int) -> None:
        self._layout.addSpacing(pixels)

    def add_dot(self, name: str) -> QLabel:
        """``.dot``：8×8 的实心圆，颜色由 ``statusKind`` 决定，右边留 5px。"""
        dot = QLabel(self)
        dot.setObjectName(name)
        dot.setProperty("statusDot", True)
        dot.setFixedSize(STATUS_DOT_PX, STATUS_DOT_PX)
        self._layout.addWidget(dot)
        self._layout.addSpacing(STATUS_DOT_GAP_PX)
        return dot


def _spring(parent: QWidget) -> QWidget:
    """``.spring{flex:1}``：左右两组之间那块可伸缩的空白。

    用控件而不是 ``addStretch``，因为帧⑤ 要把判定那一段从弹簧左边挪到右边——
    ``insertWidget`` 需要一个能被 ``indexOf`` 找到的锚点，而 stretch 不是控件。
    """
    spring = QWidget(parent)
    spring.setObjectName("statusSpring")
    spring.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return spring


def _left_spans(container: QWidget) -> tuple[StatusSpan, ...]:
    """弹簧左边那一组，按设计稿逐帧的先后顺序排一次就够用。

    没有哪一帧同时露出其中两段而要求相反的先后，所以一个顺序能服务五帧：就绪、判定
    （帧①）、J 与 χ²ᵥ、引导步骤、结构规模、阶段、进度、采样诊断（帧⑤）。
    """
    readiness = StatusSpan(container, "statusReadinessSpan")
    readiness.add_dot("fitReadinessDot")
    readiness.add_text("", name="fitReadinessStatus", gate=True)

    verdict = StatusSpan(container, "statusVerdictSpan")
    verdict.add_text(VERDICT_CAPTION, name="fitQualityCaption")
    # ``<b class="cf-trusted">● 可信</b>``：字形与判定词同在一个加粗行内，同一个颜色，
    # 中间只有一个空格——所以字形是文字而不是那个 8px 圆点。
    verdict.add_text(theme.CONFIDENCE_FALLBACK_GLYPH, name="fitQualityDot", bold=True, muted=False)
    verdict.add_gap(GLYPH_GAP_PX)
    verdict.add_text("", name="fitQualityStatus", bold=True, muted=False, gate=True)

    metrics = StatusSpan(container, "statusMetricsSpan")
    metrics.add_text("", name="fitMetricsStatus", tabular=True, gate=True)

    step = StatusSpan(container, "statusStepSpan")
    step.add_text(STEP_CAPTION, name="guidanceStepCaption")
    step.add_text("", name="guidanceStepStatus", bold=True, muted=False, gate=True)
    step.add_text("", name="guidanceStepTail")

    scale = StatusSpan(container, "statusScaleSpan")
    scale.add_text("", name="structureScaleStatus", gate=True)

    stage = StatusSpan(container, "statusStageSpan")
    stage.add_text(STAGE_CAPTION, name="fitStageCaption")
    stage.add_text("", name="fitStageStatus", bold=True, muted=False, tabular=True, gate=True)

    position = StatusSpan(container, "statusPositionSpan")
    position.add_text("", name="fitPositionStatus", tabular=True, gate=True)

    sampling = StatusSpan(container, "statusSamplingSpan")
    sampling.add_text(RHAT_CAPTION, name="mcmcRhatCaption")
    sampling.add_text("", name="mcmcRhatStatus", bold=True, muted=False, tabular=True, gate=True)
    sampling.add_text(ESS_CAPTION, name="mcmcEssCaption")
    sampling.add_text("", name="mcmcEssStatus", bold=True, muted=False, tabular=True, gate=True)

    return (readiness, verdict, metrics, step, scale, stage, position, sampling)


def _right_spans(container: QWidget, source: QLabel) -> tuple[StatusSpan, ...]:
    """弹簧右边那一组：选中层、下一步（帧③）、模式、预计剩余（帧④）、数据集与源校验。

    ``source`` 是项目动作那边造好的标签，校验结果由它自己刷；这里只把它收进段里排版。
    """
    selected = StatusSpan(container, "statusSelectedSpan")
    selected.add_text(SELECTED_CAPTION, name="selectedLayerCaption")
    selected.add_text("", name="selectedLayerStatus", bold=True, muted=False, gate=True)

    following = StatusSpan(container, "statusNextSpan")
    following.add_text(NEXT_CAPTION, name="nextActionCaption")
    action = following.add_text(NEXT_ACTION_VALUE, name="nextActionStatus", bold=True, muted=False, gate=True)
    # 设计稿帧③ 把这个动作名写成 accent 色的粗体（「下一步：开始拟合」），底栏里就这一个词
    # 带颜色；前缀「下一步：」留在次要色里。取值是定死的一句话，颜色也就跟着定死。
    theme.set_status_kind(action, "accent")
    # 这一段的取值是定死的一句话，永远非空，所以「有取值可报」这一问拦不住它——只有结构
    # 那一步请它出场（设计稿仅帧③ 底栏有「下一步：」）。默认先按不允许存着。
    following.setProperty(SPAN_ALLOWED, False)

    batch = StatusSpan(container, "statusBatchSpan")
    batch.add_text(BATCH_CAPTION, name="batchModeCaption")
    batch.add_text("", name="batchModeStatus", bold=True, muted=False, gate=True)

    remaining = StatusSpan(container, "statusRemainingSpan")
    remaining.add_text(REMAINING_CAPTION, name="fitRemainingCaption")
    remaining.add_text("", name="fitRemainingStatus", bold=True, muted=False, tabular=True, gate=True)

    dataset = StatusSpan(container, "statusDatasetSpan")
    dataset.add_text(DATASET_CAPTION, name="activeDatasetCaption")
    dataset.add_text("", name="activeDatasetStatus", bold=True, muted=False, gate=True)

    verification = StatusSpan(container, "statusSourceSpan")
    verification.add_text(SOURCE_CAPTION, name="sourceStatusCaption")
    verification.adopt(source)

    return (selected, following, batch, remaining, dataset, verification)


def build(window: QWidget, bar: QStatusBar) -> QWidget:
    """把设计稿那一行装进 ``bar`` 并返回装着各段的容器。"""
    container = QWidget(bar)
    container.setObjectName("statusSegments")
    layout = QHBoxLayout(container)
    layout.setContentsMargins(STATUS_PAD_H, 0, STATUS_PAD_H, 0)
    layout.setSpacing(SPAN_GAP_PX)
    for span in _left_spans(container):
        layout.addWidget(span)
    layout.addWidget(_spring(container))
    for span in _right_spans(container, window.project_actions.source_status_label):
        layout.addWidget(span)
    # 状态栏自己的边距归零：横向留白由容器统一给，段间距才是设计稿的 12px 而不是它加上
    # 样式表默认的那几像素。
    bar_layout = bar.layout()
    if bar_layout is not None:
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(0)
    bar.setFixedHeight(STATUS_BAR_H)
    # ``showMessage`` 只藏非常驻控件，常驻挂法让拒绝提示落在左边而右半边照旧可读。
    bar.addPermanentWidget(container, 1)
    return container


def _spans(window: QWidget) -> tuple[StatusSpan, ...]:
    container = window.findChild(QWidget, "statusSegments")
    if container is None:
        return ()
    return tuple(container.findChildren(StatusSpan))


def allow(window: QWidget, names: tuple[str, ...], *, allowed: bool, gate: str = SPAN_ALLOWED) -> None:
    """记下某几段在某一维上允不允许出现；真正的显隐交给 ``sync``。

    ``gate`` 选的是哪一维：``SPAN_ALLOWED`` 是管线走到哪一步的事，``SPAN_RUN_SCOPE`` 是
    有没有操作在跑的事。同一段常被两维同时管（模式那一段在结构那一步要让位，也只在拟合
    跑着时才有话可说），各维写各自的属性，谁都不会擦掉另一维刚下的判断。
    """
    for span in _spans(window):
        if span.objectName() in names:
            span.setProperty(gate, allowed)


def sync(window: QWidget) -> None:
    """按各维都允许且「有取值可报」重算每一段的显隐。

    刷新入口都汇到这里，是为了不让空段被唤醒：结构那一步退场时判定段还没有判定可报，
    单纯把之前藏起来的段一律放出来会露出一个只剩说明文字的「拟合结果：」。
    """
    for span in _spans(window):
        permitted = all(span.property(gate) is not False for gate in SPAN_GATES)
        filled = all(label.text() for label in span.gates)
        span.setVisible(filled and permitted)


def place_verdict(window: QWidget, *, on_right: bool) -> None:
    """判定那一段在弹簧的哪一边：帧① 在左读作「拟合结果：」，帧⑤ 在右读作「判定：」。

    往回挪时不能只求「落在弹簧左边」——那会排到 J 与 χ²ᵥ 后面。设计稿帧① 的次序是就绪、
    判定、指标，所以左边的落点固定跟在就绪那一段之后。
    """
    container = window.findChild(QWidget, "statusSegments")
    caption = window.findChild(QLabel, "fitQualityCaption")
    span = window.findChild(StatusSpan, "statusVerdictSpan")
    spring = window.findChild(QWidget, "statusSpring")
    readiness = window.findChild(StatusSpan, "statusReadinessSpan")
    if None in (container, caption, span, spring, readiness):
        return
    caption.setText(VERDICT_RIGHT_CAPTION if on_right else VERDICT_CAPTION)
    layout = container.layout()
    if (layout.indexOf(span) > layout.indexOf(spring)) is on_right:
        return
    layout.removeWidget(span)
    anchor = spring if on_right else readiness
    layout.insertWidget(layout.indexOf(anchor) + 1, span)
