"""Guided four-step workflow surface.

Expert mode was the only working surface: standard mode merely hid a handful of
controls, so a newcomer still faced the whole dock workspace at once. This panel
walks import to result and shows only what the current step needs.

Every gate is answered by reading the immutable project through the public API,
so the guided surface adds no API of its own and can never disagree with what a
fit would actually accept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPropertyAnimation, QSize, Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.structure import naming
from xrr_fitter.gui.structure.stack import component_fill

# 向导抬头两行字的尺寸，照设计稿的 ``.wizhead .wh``：标题 13.5px（就近取 14）、副行 11px。
STEP_TITLE_FONT_PX = 14
STEP_SUBLINE_FONT_PX = 11

# 设计稿 ``.stack .lyr``：``padding:9px 12px``、``gap:10px``。三个数都不落在 ``theme.SPACE_*``
# 那四档阶梯（4/8/12/16）上，所以写成局部常量而不是就近取一档——纵向 9 取 4 会让每行矮
# 10px，四行加起来 40px，读者会先看见一叠比设计稿紧一圈的行，再发现色块几乎顶到了行边。
STACK_ROW_PAD_H_PX = 12
STACK_ROW_PAD_V_PX = 9
STACK_ROW_GAP_PX = 10

# 一行里三段各有字号，照设计稿 511-513 行：层名 13px、读数 12px、副行 11px。三个数写在一起，
# 它们的全部意义就是互相比大小——层名最大（这一行是哪一层），读数次之（它多厚），副行最小
# （它是层名的注脚，不是与厚度同级的第二个读数）。三段同号时这一行读起来是平的，而读者扫
# 这一列要找的正是「哪一层、多厚」。
STACK_NAME_FONT_PX = 13
STACK_MEASURE_FONT_PX = 12
STACK_ROLE_FONT_PX = 11

# 量一行字占多高时拿这个串当尺，而不是量这一行实际写的那句话。串里既有汉字也有拉丁字母，
# 量出来的就是这一档字号能占的最大高度。
LINE_HEIGHT_SAMPLE = "汉Ag"


@dataclass(frozen=True, slots=True)
class StepSpec:
    """One guided step: its identity, prose, and the action it offers.

    ``title`` is the short header form ("导入数据"); the ordinal ("第 N 步") is
    derived from position at render time, so reordering the tuple renumbers the
    steps without a second edit. ``subline`` is the header's second row — one
    phrase saying what the step costs, which is what makes "开始拟合" read as
    "全自动" instead of "要我调参数". The import step's subline counts real files,
    so the value here is only what an empty project can honestly claim.

    There is no icon field: frame ②'s ``wizbody`` is eyebrow → h3 → sub → hint →
    stack → ctarow → help, with no glyph above the title.
    """

    name: str
    title: str
    subline: str
    body: str
    action_text: str
    action: str


STEP_SPECS = (
    StepSpec(
        "importStep",
        "导入数据",
        "尚未导入",
        "选择 .xy / .dat / .txt 反射率数据文件。导入时确认光路与仪器设置。",
        "导入数据文件…",
        "import_files",
    ),
    StepSpec(
        "structureStep",
        "确认样品结构",
        "检查自动建议",
        "初始化样品结构，必要时添加膜层。默认基底为 Si，可在结构面板调整。",
        "初始化样品结构",
        "initialize_structure",
    ),
    StepSpec(
        "fitStep",
        "开始拟合",
        "全自动",
        "一键拟合会先全局筛选再局部精修。拟合过程中可以随时取消。",
        "开始一键拟合",
        "start_fit",
    ),
    StepSpec(
        "resultStep",
        "查看结果",
        "置信度与参数",
        "查看候选解与可信度。需要完整的参数表、诊断图或导出时，切换到专家模式。",
        "切换到专家模式",
        "leave_guidance",
    ),
)


def _spec_by_name() -> dict[str, StepSpec]:
    """Index the declared steps so a lookup cannot drift from the tuple's order."""
    return {spec.name: spec for spec in STEP_SPECS}


SPEC_BY_NAME = _spec_by_name()

# The guided stack list restates one structure in plain language: two media that
# take no parameters, and layers named by the role they play rather than by the
# columns the expert tree spreads them across.
ROLE_FRONTING = "入射介质 · 半无限，无需参数"
ROLE_BACKING = "衬底 · 半无限"
ROLE_OXIDE = "自动建议 · 可移除"
ROLE_SOLE_LAYER = "主体层 · 你要测量的对象"
ROLE_LAYER = "薄膜层 · 参与拟合"
ROLE_GRADIENT = "渐变层 · 参与拟合"

# A semi-infinite medium has no thickness to report, and an em dash says that
# without inviting the reader to look for a number that was merely omitted.
MEDIUM_MEASURE = "—"

# A layer the software added on its own is a decision made for the user, so the
# step says why it is there and what would justify removing it.
# 首句加粗照设计稿的 ``.hint b``：正文是「为什么」和「怎么办」，先要被看到的是发生了
# 什么。粗体只改字重、不改颜色，字色仍由 ``mark_hint`` 的主题规则给，所以浅色深色两套
# 外观都跟着走——把 ink 色写进这段富文本才会锁死在一种外观上。
OXIDE_TIP_TEXT = (
    "💡 <b>已自动加入表面氧化层。</b>金属与半导体表面在空气中通常生成几纳米氧化层。"
    "若你的样品在惰性气氛中制备并即时测量，可以移除它。"
)
# 「专家模式」加粗照设计稿 585 行：这一句的正文是理由，加粗的那三个字是读者要去找的那颗
# 开关的名字。加粗它，这一句就能扫读；不加粗，这句脚注要整句读完才知道去处叫什么。
EXPERT_HINT_TEXT = "需要精细控制每个参数的边界、先验或跨数据集共享？切换到<b>专家模式</b>即可。"
MANUAL_TEXT = "＋ 我要手动加/删层"

# Step 2 confirms a structure that exists and builds one that does not:
# "初始化样品结构" run against a finished stack would discard exactly what this
# step is asking the user to approve.
STRUCTURE_ACTION_BUILD = ("初始化样品结构", "initialize_structure")
STRUCTURE_ACTION_CONFIRM = ("看起来没问题，开始拟合 →", "confirm_structure")

# The prose follows the same fork. This wording is the only place in the guided
# surface that says where the structure came from and what happens if it is not
# exact, so it appears as soon as there is a structure to say it about.
STRUCTURE_BODY_CONFIRM = (
    "我们根据你的数据和常见薄膜体系，自动搭好了一个初始结构。看一眼是否合理——"
    "不确定也没关系，拟合会在允许范围内自动修正每一层。"
)

# ``.wizbody{max-width:720px;margin:0 auto}``. With both side columns hidden the
# guided step owns the whole shell, and a 1200px-wide line of 13.5px prose is
# unreadable, so the card is capped and centred instead of stretched. The cap is a
# *maximum*: a fixed width would raise ``QStackedWidget.minimumSizeHint``, which is
# the maximum over all pages, and so would push the expert canvas column's minimum
# up by the same amount.
BODY_MAX_WIDTH = 720

# ``.wizbody{padding:24px}``（HTML 497）。裸写的 24 而不是间距档：这一页是整屏唯一的内容，
# 四周没有别的东西替它定边界，同一档 16px 放在挤在栏里的卡上是对的，放到独占整屏的读书栏上
# 就贴边了。设计稿的 ``--xl:24px`` 在七帧的 DOM 里零引用（只用在文档装帧上），所以这一处
# 不该顺手补一档 token 出来。
BODY_PAD_PX = 24

# Which steps carry a secondary 「下一步」 beside their primary.
#
# It cannot be derived from ``spec.action``: step 2's build-time action is
# ``initialize_structure`` and only ``_sync_structure_step`` swaps it for
# ``confirm_structure``. Import opens a file dialog and fit runs asynchronously, so
# on neither does the primary itself advance the step -- these two need a way past.
# Step 2's primary *is* the advance and step 4's is the exit out of the mode, so a
# forward button beside either would be a second control doing the same thing.
STEPS_OFFERING_NEXT = frozenset({"importStep", "fitStep"})


def import_subline(project: api.XrrProject) -> str:
    """Report the imported file count, or admit the project is still empty.

    This is the one header cell that states a measurement instead of a promise,
    so it is derived from the project rather than read from ``STEP_SPECS``.
    """
    count = len(project.datasets)
    if not count:
        return SPEC_BY_NAME["importStep"].subline
    return f"{count} 个文件 · 已就绪"


def _material_label(material: api.MaterialSpec) -> str:
    """Name a semi-infinite medium the way the design's stack list does.

    ``Air`` is the one material whose spec name stays English on a Chinese
    surface, and the expert tree already renders it as 空气, so the two lists
    agree on the medium a user sees in both.

    The name comes first because a formula cannot carry phase: a-Si and c-Si are
    both ``Si``, and the frame names them apart precisely because a film and the
    substrate under it are the two things this stack is about.  The formula is
    still the fallback, for a material carrying no name of its own.
    """
    if material.name == "Air":
        return "空气"
    return naming.inline_name(material.name or material.formula or "")


def _thickness_text(value_a: float) -> str:
    """State a thickness as the magnitude the judgement in this step needs."""
    return f"≈ {value_a / 10.0:.1f} nm"


def _is_surface_oxide(component: object, index: int, oxides: frozenset[str]) -> bool:
    """Say whether this component is an oxide the software proposed itself.

    Only the topmost component can be a surface oxide, and only a recorded
    acceptance makes it automatic: a hand-drawn SiO2 cap is the user's own layer
    and must not be labelled as something to remove.
    """
    if index != 0 or not isinstance(component, api.LayerSpec):
        return False
    return component.material.formula in oxides


def _sole_layer_index(components: tuple[object, ...], oxides: frozenset[str]) -> int | None:
    """Index of the only layer that is not an automatic oxide, if there is one.

    A single film is the thing being measured, which is worth naming outright;
    with two or more, only the shared fact -- the fit will move all of them --
    still holds.
    """
    candidates = [
        index for index, component in enumerate(components) if not _is_surface_oxide(component, index, oxides)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _component_row(
    component: object,
    index: int,
    oxides: frozenset[str],
    sole: int | None,
) -> tuple[str, str, str]:
    """Project one structure component onto name, role and rough thickness.

    A periodic block reports its total stack height rather than one period,
    because that is the number comparable with the plain layers above it.
    """
    if isinstance(component, api.PeriodicBlock):
        total = component.repeats * sum(layer.thickness_a for layer in component.layers)
        return (naming.inline_name(component.name), f"周期结构 ×{component.repeats} · 参与拟合", _thickness_text(total))
    if isinstance(component, api.GradientLayerSpec):
        return (naming.inline_name(component.name), ROLE_GRADIENT, _thickness_text(component.thickness_a))
    if _is_surface_oxide(component, index, oxides):
        role = ROLE_OXIDE
    elif index == sole:
        role = ROLE_SOLE_LAYER
    else:
        role = ROLE_LAYER
    return (_layer_label(component), role, _thickness_text(component.thickness_a))


def _layer_label(layer: api.LayerSpec) -> str:
    """What the guided list calls one layer: its own name, not its material.

    设计稿帧② 的名字列写「SiO₂ 表面氧化层」「a-Si 非晶硅薄膜」——名字里已经带着材料，
    再多一层是这一层被认作什么。材料式单独拿出来说不了相（a-Si 与 c-Si 同为 ``Si``），
    而层名说得了。没有名字的层退回材料，那仍好过一行空白。
    """
    return naming.inline_name(layer.name) or _material_label(layer.material)


def _is_semi_infinite(position: int, total: int) -> bool:
    """这一行是不是那两片半无限介质。

    ``structure_rows`` 总把组件夹在入射介质与衬底之间，所以首末两位就是那两片。同一个判据
    决定这一行取中性色块还是压一层淡底——写成两个判据，改一处就会让色块和底色说两件事。
    """
    return position == 0 or position == total - 1


def _row_fill(position: int, total: int) -> str:
    """The hue this row shares with the expert tree and the stack diagram.

    Derived from the position rather than carried in the row tuple: `structure_rows`
    always brackets the components with fronting and backing, so position and
    component index differ by exactly one, and going through `component_fill` is
    what keeps the three views from each inventing their own palette.
    """
    if _is_semi_infinite(position, total):
        return theme.DATA_NEUTRAL
    return component_fill(position - 1)


def _line_height(font: QFont) -> int:
    """这一档字的一行占多高，不看这一行写的是什么。

    ``QLabel`` 的 ``sizeHint`` 按它那句话实际用到的字体量：``Si`` 全走主字体，量到 16px；
    ``空气`` 与 ``SiO₂ 表面氧化层`` 里的汉字走回退字体，回退字体的 ascent/descent 更大，量到
    19px。于是同一叠里名字是纯拉丁的那一行整行矮三像素——而这一叠画的是一叠层，行高不齐读
    起来就是「这几层薄厚不同」，偏偏厚度是右边那一列已经写明的事。设计稿四行 ``padding``
    相同、字号相同，所以四行本该等高。

    换成量 ``LINE_HEIGHT_SAMPLE``：串是固定的，量出的高度于是只跟着字号走。不用
    ``QFontMetrics.height()``——那个数只问主字体（13px 给 16），拿它钉高度会把汉字裁掉三像素。
    """
    return QFontMetrics(font).size(int(Qt.TextFlag.TextSingleLine), LINE_HEIGHT_SAMPLE).height()


class _StackList(QWidget):
    """The design's stack restatement: one row per medium and layer.

    Rows are rebuilt wholesale whenever the structure changes. The list is at
    most a handful of rows, and diffing them would buy nothing except a way for
    a stale row to survive an edit.

    设计稿 ``.stack``（506-513 行）画的是一只盒子，外框由这个控件自己带（``theme`` 里挂在
    ``#structureStepStack`` 上），行与行之间那道界面由每行的下边线给。「末行不画线」与「首末
    行带外侧圆角」这三档都按 index 直接设属性，不走 ``theme.mark_last_section``：那个函数按
    ``isHidden()`` 找末段、跟着可见性改，而这一叠是 ``set_rows`` 整批重建，末行在构造期就定了。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("structureStepStack")
        # 盒子那圈框归这个控件自己画。``QWidget`` 的子类默认不走样式表的背景与边框通路
        # （只有 Qt 自带的那些类型才自动走），实测那圈框一个像素都没落下来——左边缘量到的
        # 仍是窗口底色。这个属性把它接回来；不设它，``#structureStepStack`` 那条规则就只是
        # 一段不生效的样式。
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._rows: tuple[tuple[str, str, str], ...] = ()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        # 行贴着行：中间那道缝一旦留出来，行与行之间贴着的那个面（也就是一道界面）就变成了
        # 两行之间的空地，而分隔线画的正是那个面。
        self._layout.setSpacing(0)
        self.hide()

    def rows(self) -> tuple[tuple[str, str, str], ...]:
        return self._rows

    def set_rows(self, rows: tuple[tuple[str, str, str], ...]) -> None:
        if rows == self._rows:
            return
        self._rows = rows
        self._clear()
        for position, (name, role, measure) in enumerate(rows):
            self._layout.addWidget(self._row(name, role, measure, position, len(rows)))
        self.setVisible(bool(rows))

    def _clear(self) -> None:
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)

    def _row(self, name: str, role: str, measure: str, position: int, total: int) -> QWidget:
        row = QFrame(self)
        # ``stackRow`` 而不是 ``sectionCard``：后者画的是一圈框加圆角，四行各带一圈就成了四张
        # 并列的卡片，而这四行是一叠——盒子的框已经说了「这是一叠」，行只要一道下边线说
        # 「界面在这里」。
        row.setProperty("stackRow", True)
        row.setProperty("firstStackRow", position == 0)
        row.setProperty("lastStackRow", position == total - 1)
        row.setProperty("semiInfinite", _is_semi_infinite(position, total))
        layout = QHBoxLayout(row)
        layout.setContentsMargins(STACK_ROW_PAD_H_PX, STACK_ROW_PAD_V_PX, STACK_ROW_PAD_H_PX, STACK_ROW_PAD_V_PX)
        layout.setSpacing(STACK_ROW_GAP_PX)
        swatch = QLabel(row)
        swatch.setObjectName("structureStepSwatch")
        swatch.setPixmap(theme.stack_swatch(_row_fill(position, total)))
        # 色块只搬颜色，不搬信息：行名和角色已经把这层说清了，让读屏器再念一遍
        # 「图片」只会在每行前面加一段噪音。
        swatch.setAccessibleName("")
        # 设计稿 ``.lyr`` 是 ``align-items:center``：色块对的是整行，不是行里第一行字。
        layout.addWidget(swatch, 0, Qt.AlignmentFlag.AlignVCenter)
        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel(name, row)
        # ``.nm{font-weight:600;font-size:13px}``
        title_font = title.font()
        title_font.setPixelSize(STACK_NAME_FONT_PX)
        title_font.setWeight(QFont.Weight.DemiBold)
        title.setFont(title_font)
        # 高度按字号钉，不按这一行的名字钉：见 ``_line_height``。
        title.setFixedHeight(_line_height(title_font))
        subtitle = QLabel(role, row)
        # ``.nm small{font-weight:400;color:var(--ink-faint);font-size:11px}``：最淡那一档灰给
        # 这行注脚，静音档留给还要读的次要文字——副行与厚度同色时它们读起来就是同一级的两个
        # 读数，而它只是主名的注脚。
        subtitle.setProperty("faintText", True)
        subtitle_font = subtitle.font()
        subtitle_font.setPixelSize(STACK_ROLE_FONT_PX)
        subtitle_font.setWeight(QFont.Weight.Normal)
        subtitle.setFont(subtitle_font)
        subtitle.setFixedHeight(_line_height(subtitle_font))
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text, 1)
        amount = QLabel(measure, row)
        amount.setProperty("mutedText", True)
        # ``.mv`` 带 ``font-variant-numeric:tabular-nums``：这一列是上下扫着比厚度的，比例数字
        # 里 1 比 8 窄，``3.4`` 与 ``48.7`` 于是小数点错开一两像素。Qt 的样式表没有这个属性，
        # 所以只能落在字体对象上。
        amount_font = amount.font()
        amount_font.setPixelSize(STACK_MEASURE_FONT_PX)
        amount_font.setFeature(QFont.Tag("tnum"), 1)
        amount.setFont(amount_font)
        amount.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(amount)
        return row


class _LaidOutButton(QPushButton):
    """A button that asks for what its layout holds, not for its own (empty) text.

    ``QPushButton`` computes both hints from its text and icon; a cell built from
    child widgets carries neither, so it asked for a button-sized nothing and the
    stretching tracks beside it took the row -- 「导入数据」 wrapped to two
    characters down the side of its dot.  Deferring to the layout puts the step
    name back on one line.
    """

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        layout = self.layout()
        return layout.totalSizeHint() if layout is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        layout = self.layout()
        return layout.totalMinimumSize() if layout is not None else super().minimumSizeHint()


class _StepHeader(QWidget):
    """Labelled four-step header: a marker glyph plus each step's short title.

    The cells are built from ``STEP_SPECS`` so the count follows the one source;
    a second literal would drift the instant a step is added or removed. State is
    double-encoded — a done step swaps its ordinal for a ✓ and the current step
    is bolded — so the redundant shape and weight ride on top of the theme colour
    ``set_step_state`` sets, and the state still reads on a monochrome display.

    设计稿 ``.wizhead`` 是四张等宽的分格卡贴满整行（``.wh{flex:1}``），格与格之间只有一道
    1px 的竖线（``.wh{border-right}``，末格不画）。这里原先画的是另一张图：四格各按标题宽窄
    排、中间夹会伸缩的横轨、两端再各留一段空白——读作「圆点连成的一条流水线」，而等宽分格读
    作「四个并列的阶段」，且等宽本身就是进度的刻度。当前那一步靠整格换成内容面的白底来报，
    所以状态既写给圆点和标题，也写给格子本身。

    Each cell is a button, because ``wizhead`` is the only step affordance frame ②
    draws: with the navigation rail hidden the header is how a guided user reaches
    a step they have already passed.
    """

    SEPARATOR_WIDTH = 1

    step_activated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepHeader")
        # 纵向钉死在 ``sizeHint``：抬头是条窄带，多余高度归底下的页面。装它的 ``QVBoxLayout``
        # 两项都没有拉伸因子，两项又都能长，Qt 于是把整块余量平摊——抬头有了自己的底色之后，
        # 这一摊就是半屏机架色，四格圆点浮在正中间。带子自己声明不长，页面才拿得到那半屏。
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._cells: list[QWidget] = []
        self._markers: list[QLabel] = []
        self._titles: list[QLabel] = []
        self._sublines: list[QLabel] = []
        self._separators: list[QFrame] = []
        for index, spec in enumerate(STEP_SPECS):
            # 四格同一个拉伸因子：整行的空余按因子分，四份相等，谁的标题长短都不参与。
            layout.addWidget(self._build_cell(spec, index), 1)
            if index < len(STEP_SPECS) - 1:
                layout.addWidget(self._build_separator(spec))
        self.set_current(0)

    def _build_separator(self, spec: StepSpec) -> QWidget:
        """``.wh{border-right:1px solid var(--border)}``：贴在这一格右缘、贯通整格高的细线。"""
        # QWidget 而不是 QFrame(VLine)：VLine 的 sizeHint 返回 -1，QHBoxLayout
        # 于是只给 26px 而不是行高 38px。纯 QWidget + QSS background 没有这个问题。
        separator = QWidget(self)
        separator.setObjectName(f"guidanceSeparator_{spec.name}")
        separator.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        separator.setFixedWidth(self.SEPARATOR_WIDTH)
        separator.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        separator.setProperty("pipelineRail", True)
        self._separators.append(separator)
        return separator

    def _build_cell(self, spec: StepSpec, index: int) -> QWidget:
        """One clickable header cell: ordinal marker, step title, and its subline."""
        cell = _LaidOutButton(self)
        cell.setObjectName(f"guidanceStep_{spec.name}Cell")
        cell.setProperty("ghost", True)
        cell.setCursor(Qt.CursorShape.PointingHandCursor)
        cell.setAccessibleName(f"第 {index + 1} 步 · {spec.title}")
        cell.setToolTip(f"跳到第 {index + 1} 步：{spec.title}")
        # 横向 Preferred 才让得出宽度：``QPushButton`` 默认的 Minimum 把 ``sizeHint`` 当下限，
        # 窄窗口里四格于是各守自己标题的宽度、不再等分。纵向铺满，当前那格的白底才盖住整格
        # 高度而不是在抬头上下各留一条机架色。
        cell.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        cell.clicked.connect(lambda _checked=False, position=index: self.step_activated.emit(position))
        marker = QLabel(str(index + 1), cell)
        marker.setObjectName(f"guidanceStep_{spec.name}Marker")
        marker.setFixedSize(theme.STEP_DOT_PX, theme.STEP_DOT_PX)
        marker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        marker.setProperty("stepDot", True)
        # 圆角在样式表里跟着这个属性走（见 ``theme.STEP_DOT_PX``）：两张步骤地图的圆点不同径，
        # 一条半径服务不了两种直径。
        marker.setProperty("wizardStep", True)
        title = QLabel(spec.title, cell)
        title.setObjectName(f"guidanceStep_{spec.name}Title")
        # ``.wizhead .wh .tt{font-size:13.5px;font-weight:700}``：四格都是这个粗细，跟着状
        # 态走的只有颜色，所以粗细写在字体上而不是在 ``stepState`` 的规则里。13.5 就近取 14，
        # 顺序才对得上——向导抬头这行本来就该比左栏管线的 13px 标题重一档。
        title_font = title.font()
        title_font.setPixelSize(STEP_TITLE_FONT_PX)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        subline = QLabel(spec.subline, cell)
        subline.setObjectName(f"guidanceStep_{spec.name}Subline")
        # ``.wh .dd{font-size:11px;color:var(--ink-faint)}``
        subline.setProperty("faintText", True)
        subline_font = subline.font()
        subline_font.setPixelSize(STEP_SUBLINE_FONT_PX)
        subline.setFont(subline_font)
        # Title over subline, so the second row hangs under the step name it
        # qualifies rather than starting a column of its own beside the marker.
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(title)
        text.addWidget(subline)
        inner = QHBoxLayout(cell)
        inner.setContentsMargins(theme.SPACE_SM, theme.SPACE_XS, theme.SPACE_SM, theme.SPACE_XS)
        inner.setSpacing(theme.SPACE_XS)
        inner.addWidget(marker)
        inner.addLayout(text)
        for label in (marker, title, subline):
            # The labels fill the cell, so a press must reach the button under
            # them rather than stopping at whichever line was clicked.
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._cells.append(cell)
        self._markers.append(marker)
        self._titles.append(title)
        self._sublines.append(subline)
        return cell

    def set_current(self, index: int) -> None:
        for i, marker in enumerate(self._markers):
            title = self._titles[i]
            if i < index:
                state, glyph = "done", "✓"
            elif i == index:
                state, glyph = "current", str(i + 1)
            else:
                state, glyph = "pending", str(i + 1)
            marker.setText(glyph)
            theme.set_step_state(marker, state)
            theme.set_step_state(title, state)
            # 格子自己也收这一档：``.wh.current{background:#fff}`` 是设计稿报「此刻在这一步」
            # 的手段，而样式表够不着圆点和标题的父级，状态不写到格上就没有底色可换。
            theme.set_step_state(self._cells[i], state)

    def titles(self) -> tuple[str, ...]:
        return tuple(title.text() for title in self._titles)

    def sublines(self) -> tuple[str, ...]:
        return tuple(subline.text() for subline in self._sublines)

    def set_subline(self, index: int, text: str) -> None:
        self._sublines[index].setText(text)


class GuidancePanel(QWidget):
    """Project one step at a time, gated on the real project state."""

    step_changed = Signal(str)
    leave_requested = Signal()

    def __init__(
        self,
        document: ProjectDocument,
        actions: dict[str, Callable[[], object]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("guidancePanel")
        self.setAccessibleName("引导流程")
        self._document = document
        self._actions = dict(actions)
        self._pages: dict[str, QWidget] = {}
        self._action_buttons: dict[str, QPushButton] = {}
        self._next_buttons: dict[str, QPushButton] = {}
        # Step 2's action, prose and row list all follow whether a structure
        # exists, so each is held by name for ``_sync_structure_step`` to repaint.
        self._bodies: dict[str, QLabel] = {}
        self._action_keys: dict[str, str] = {}
        self._stack_list = _StackList(self)
        self._oxide_tip = QLabel(OXIDE_TIP_TEXT, self)
        self._oxide_tip.setObjectName("structureStepOxideTip")
        theme.set_status_kind(self._oxide_tip, "info")
        theme.mark_hint(self._oxide_tip)
        self._oxide_tip.setWordWrap(True)
        self._oxide_tip.hide()
        self._step_header = _StepHeader(self)
        self._step_header.step_activated.connect(self._activate_step)
        self._stack = QStackedWidget(self)
        self._stack.setObjectName("guidanceStack")
        total = len(STEP_SPECS)
        for index, spec in enumerate(STEP_SPECS):
            page = self._build_page(spec, index, total)
            self._pages[spec.name] = page
            self._stack.addWidget(page)
        layout = QVBoxLayout(self)
        # 壳子不垫边距、不留段间距：设计稿的 ``.wizhead`` 与它下面那个背景层都是通栏零内边距
        # 的（HTML 557-564），整屏的留白全出自最里面 ``.wizbody`` 的 24px。这里再垫一圈就是
        # 两层相加，读书栏的实际留白变 40px、进度条那一条被无故内缩。
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Frame ② stacks ``wizhead`` directly on ``wizbody`` at the top of the shell:
        # the step starts where the header ends, so there is no vertical spread to
        # float the page in the middle of the window.
        layout.addWidget(self._step_header)
        layout.addWidget(self._stack)
        document.project_changed.connect(self._refresh)
        self._refresh()

    def _build_page(self, spec: StepSpec, index: int, total: int) -> QWidget:
        """Lay one step out as frame ②'s ``wizbody``: a capped, centred page of prose.

        ``eyebrow2 → h3 → sub → 💡 hint → stack → ctarow → help``, every block on one
        left edge.  Nothing is centred inside its own line: four centred sentences are
        four objects with no shared edge to read down, which is what made the rendered
        step read as a splash screen instead of a page.  There is no glyph above the
        title either -- the frame draws none, and ``plot_icon`` scales its painter
        without touching the device pixel ratio, so at 48px the pen width and the
        1.2-unit marker radii scaled 3× with it and the curve came out a black smear.
        """
        page = QWidget()
        page.setObjectName(spec.name)
        page.setAccessibleName(f"第 {index + 1} 步 · {spec.title}")
        card = QFrame(page)
        card.setObjectName(f"{spec.name}Card")
        # ``.wizbody`` sits on a plain panel background and the frame draws no border
        # around it, so this is the design's reading column rather than a sectionCard.
        card.setMaximumWidth(BODY_MAX_WIDTH)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(BODY_PAD_PX, BODY_PAD_PX, BODY_PAD_PX, BODY_PAD_PX)
        card_layout.setSpacing(theme.SPACE_SM)
        # The ordinal is derived from position, so it stays in step with the header and
        # cannot contradict a reordered STEP_SPECS.  ``.eyebrow2`` 的强调色沿用当前步骤那档
        # token，不另开一条会跟它漂移的 accent 规则；字号、字重与字距由
        # ``apply_step_eyebrow`` 一处给出——那条 ``stepState`` 规则只给颜色。
        eyebrow = QLabel(f"第 {index + 1} 步 · 共 {total} 步", card)
        eyebrow.setObjectName(f"{spec.name}Eyebrow")
        theme.apply_step_eyebrow(eyebrow)
        card_layout.addWidget(eyebrow)
        title = QLabel(spec.title, card)
        title.setObjectName(f"{spec.name}Title")
        title.setProperty("emptyTitle", True)
        card_layout.addWidget(title)
        body = QLabel(spec.body, card)
        body.setObjectName(f"{spec.name}Body")
        body.setProperty("mutedText", True)
        body.setWordWrap(True)
        card_layout.addWidget(body)
        self._bodies[spec.name] = body

        # The tip explains the one layer the software added on its own, so it is read
        # before the list it is talking about.
        if spec.name == "structureStep":
            card_layout.addWidget(self._oxide_tip)
            card_layout.addWidget(self._stack_list)
        card_layout.addSpacing(theme.SPACE_SM)
        card_layout.addLayout(self._build_cta_row(spec, card))
        if spec.name == "structureStep":
            card_layout.addWidget(self._build_expert_hint(card))
        # Block flow: the height left over belongs below the last block, not spread
        # between the paragraphs.
        card_layout.addStretch(1)
        # ``margin:0 auto``: the column takes most of the shell up to its cap, and
        # whatever the cap leaves over splits evenly between the two sides.
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(card, 8)
        row.addStretch(1)
        return page

    def _build_cta_row(self, spec: StepSpec, card: QWidget) -> QHBoxLayout:
        """Frame ②'s ``.ctarow``: the primary, then its alternative, both left-aligned.

        A flex row at ``flex-start`` with a 12px gap, so the buttons start on the same
        left edge as the prose they follow and the slack goes to the right.  Centring
        the pair instead detached both from the text that explains them.

        设计稿把这一行的按钮都写成 ``.btn.lg``（44px 高、左右 22px、14px 字），比应用里其他
        按钮高一档：这一行是这一屏唯一的出路，一行里只有它们，所以它们自己定这一行的高度。
        两枚都得挂上这一档——``.ctarow`` 垂直居中，只给一枚就会让另一枚差出边框那一两像素，
        看着像整行没对齐。
        """
        action = QPushButton(spec.action_text, card)
        action.setObjectName(f"{spec.name}Action")
        action.setProperty("primary", True)
        action.setProperty("large", True)
        action.setAccessibleName(spec.action_text)
        # Step 2 swaps its action with the project, so the key is looked up at click
        # time rather than captured here.
        self._action_keys[spec.name] = spec.action
        action.clicked.connect(lambda _checked=False, name=spec.name: self._run(self._action_keys[name]))
        self._action_buttons[spec.name] = action
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_MD)
        row.addWidget(action)
        secondary = self._build_secondary(spec, card)
        if secondary is not None:
            secondary.setProperty("large", True)
            row.addWidget(secondary)
        row.addStretch(1)
        return row

    def _build_secondary(self, spec: StepSpec, card: QWidget) -> QPushButton | None:
        """The one alternative beside the primary, or nothing where the step needs none.

        Step 2 gets the escape hatch: guided mode owns no layer editor, so a user who
        can see the suggestion is wrong would otherwise be left fitting the structure
        they just judged wrong.  Import and fit get 「下一步」 because their primary opens
        a file dialog or starts a background run rather than advancing the step; step 2's
        primary *is* the advance and step 4's is the way out of the mode, so a forward
        button beside either would be a second control doing the same thing.

        手动那一枚是描边按钮而不是无框的 ``ghost``：设计稿帧② 把它写成 ``<span class="btn
        lg">``，与主按钮同框同高，只是不填 accent 底色。它是与「开始拟合」并列的另一条路，
        无框会把它降成一句可点的说明。「下一步 →」照旧无框——设计稿六帧一次都没画过它，
        给它换档就是自己发明。
        """
        if spec.name == "structureStep":
            manual = QPushButton(MANUAL_TEXT, card)
            manual.setObjectName("structureStepManual")
            manual.setAccessibleName(MANUAL_TEXT)
            manual.clicked.connect(lambda _checked=False: self._run("leave_guidance"))
            return manual
        if spec.name not in STEPS_OFFERING_NEXT:
            return None
        forward = QPushButton("下一步 →", card)
        forward.setObjectName(f"{spec.name}Next")
        forward.setProperty("ghost", True)
        forward.clicked.connect(lambda _checked=False, name=spec.name: self._step(name, 1))
        self._next_buttons[spec.name] = forward
        return forward

    def _build_expert_hint(self, card: QWidget) -> QLabel:
        """Step 2's closing ``.help``: the pointer to expert mode, last on the page.

        It comes after the action it is an alternative to, so the step reads as one
        offer with a footnote rather than two competing ways forward.
        """
        hint = QLabel(EXPERT_HINT_TEXT, card)
        hint.setObjectName("structureStepExpertHint")
        hint.setProperty("mutedText", True)
        hint.setWordWrap(True)
        return hint

    def structure_rows(self) -> tuple[tuple[str, str, str], ...]:
        """Restate the active structure as name, role and rough thickness.

        The expert tree spreads thickness, roughness and density across columns, so
        reading it starts with learning what the columns are. This is the same
        structure projected for the single judgement step 2 asks for -- "is this
        stack plausible" -- so the semi-infinite media say they take no parameters
        and the layers give a magnitude instead of significant figures. Rows carry
        the layer's own name, matching the expert tree, with ``structure.naming``
        standing in for the generated placeholder an auto-added oxide arrives with.

        Empty before a structure exists: a placeholder list would describe a stack
        nobody built.
        """
        structure = self._active_structure()
        if structure is None:
            return ()
        oxides = self._surface_oxide_formulas()
        sole = _sole_layer_index(structure.components, oxides)
        rows = [(_material_label(structure.fronting), ROLE_FRONTING, MEDIUM_MEASURE)]
        rows.extend(
            _component_row(component, index, oxides, sole) for index, component in enumerate(structure.components)
        )
        rows.append((_material_label(structure.backing), ROLE_BACKING, MEDIUM_MEASURE))
        return tuple(rows)

    def _active_structure(self) -> api.StructureSpec | None:
        dataset = self._active_dataset(self._document.project)
        return None if dataset is None else dataset.structure

    def _surface_oxide_formulas(self) -> frozenset[str]:
        """Formulas of the surface oxides this project accepted from a suggestion.

        A decision records the oxide's formula rather than the layer it produced,
        which is exactly what distinguishes a suggested cap from an identical one
        the user drew by hand.
        """
        dataset = self._active_dataset(self._document.project)
        if dataset is None:
            return frozenset()
        return frozenset(
            decision.oxide_material
            for decision in dataset.oxide_decisions
            if decision.accepted and decision.location == "surface"
        )

    def _sync_structure_step(self) -> None:
        """Point step 2 at the project it actually has.

        Confirming a stack and building one are different offers, and the design
        makes the difference visible in three places at once: the action, the prose
        above it, and the row list. Rebuilding all three from the project on every
        change is what stops 「我们自动搭好了一个初始结构」 from appearing over an
        empty stack -- and it runs whichever step is on screen, because a caller may
        read step 2 while step 1 is current.
        """
        rows = self.structure_rows()
        self._stack_list.set_rows(rows)
        self._oxide_tip.setVisible(any(role == ROLE_OXIDE for _name, role, _measure in rows))
        built = bool(rows)
        text, key = STRUCTURE_ACTION_CONFIRM if built else STRUCTURE_ACTION_BUILD
        action = self._action_buttons["structureStep"]
        action.setText(text)
        action.setAccessibleName(text)
        self._action_keys["structureStep"] = key
        body = STRUCTURE_BODY_CONFIRM if built else SPEC_BY_NAME["structureStep"].body
        self._bodies["structureStep"].setText(body)

    def step_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in STEP_SPECS)

    def step_header_titles(self) -> tuple[str, ...]:
        """The short step titles the header shows, in order, sourced from STEP_SPECS."""
        return self._step_header.titles()

    def step_header_sublines(self) -> tuple[str, ...]:
        """The header's second row per step: one phrase on what the step costs.

        The import cell counts real files, so this reports what the header shows
        after the last project change rather than the static ``STEP_SPECS`` value.
        """
        return self._step_header.sublines()

    def current_step(self) -> str:
        return self.step_names()[self._stack.currentIndex()]

    def show_step(self, name: str) -> None:
        page = self._pages.get(name)
        if page is None:
            raise KeyError(f"unknown guidance step: {name}")
        old_page = self._stack.currentWidget()
        if old_page is not page:
            self._animate_transition(old_page, page)
        self._stack.setCurrentWidget(page)
        self._step_header.set_current(self.step_names().index(name))
        self._refresh()
        self.step_changed.emit(name)

    def _animate_transition(self, old_page: object, new_page: object) -> None:
        effect = QGraphicsOpacityEffect(new_page)
        new_page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(200)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.finished.connect(lambda: new_page.setGraphicsEffect(None))
        animation.start()

    def step_is_available(self, name: str) -> bool:
        """Answer a step's gate from the project, never from local UI state."""
        project = self._document.project
        dataset = self._active_dataset(project)
        if name == "importStep":
            return True
        if name == "structureStep":
            return bool(project.datasets)
        if name == "fitStep":
            return dataset is not None and dataset.structure is not None
        if name == "resultStep":
            return dataset is not None and dataset.last_valid_result is not None
        raise KeyError(f"unknown guidance step: {name}")

    def _active_dataset(self, project: api.XrrProject) -> object | None:
        active = project.ui_state.active_dataset_id
        return next(
            (dataset for dataset in project.datasets if dataset.dataset_id == active),
            None,
        )

    def _step(self, name: str, offset: int) -> None:
        """Move by one step, staying inside the declared sequence.

        Only import and fit carry a forward button, so the last step can no longer be
        walked off the end here; leaving the mode is what ``resultStep``'s primary is
        for.
        """
        names = self.step_names()
        target = names.index(name) + offset
        if 0 <= target < len(names):
            self.show_step(names[target])

    def _activate_step(self, index: int) -> None:
        """Jump to a step from its header cell, if the project has reached it.

        With the navigation rail hidden the header is the guided surface's only step
        affordance, which is what keeps 「导入数据」 reachable from step 2.  A cell whose
        precondition does not hold stays inert, so the header cannot walk ahead of the
        project any more than the buttons can.
        """
        names = self.step_names()
        if not 0 <= index < len(names):
            return
        name = names[index]
        if self.step_is_available(name):
            self.show_step(name)

    def _run(self, key: str) -> None:
        if key == "leave_guidance":
            self.leave_requested.emit()
            return
        # Step 2's confirming form does not touch the project: the structure is
        # already what the user just approved, so the only thing left is to advance.
        if key == "confirm_structure":
            self.show_step("fitStep")
            return
        operation = self._actions.get(key)
        if operation is not None:
            operation()

    def _refresh(self, *_args) -> None:
        current = self.current_step()
        names = self.step_names()
        index = names.index(current)
        self._step_header.set_subline(0, import_subline(self._document.project))
        self._sync_structure_step()
        # The action for a step the project is not ready for would fail, so it is
        # disabled rather than left to raise.  Forward is gated on the *next* step's
        # own precondition, so the flow cannot run ahead of the project -- the old
        # unconditional ``setEnabled(True)`` contradicted this comment.
        self._action_buttons[current].setEnabled(self.step_is_available(current))
        forward = self._next_buttons.get(current)
        following = names[index + 1] if index + 1 < len(names) else None
        if forward is not None and following is not None:
            forward.setEnabled(self.step_is_available(following))
