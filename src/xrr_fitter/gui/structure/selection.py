"""帧③ 的选中层检查器：就地编辑一层的材料与几何。

设计稿把结构编辑分成两半——画布里的层堆叠负责「选哪一层」，右栏这张卡负责
「这一层是什么」。在此之前已有层只能整层删掉重加，层序和堆叠里的选中位置都会跟着
丢；这里给的是替换，索引不动。

写回走 :class:`~xrr_fitter.gui.structure.editor.StructureEditor` 的同一条提交路径，
所以手改一个数字和从对话框新建一层受同一套校验，卡片自己不判定合法性。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from math import ceil
from typing import NamedTuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.railbar import RailBar

CommitLayer = Callable[[int, api.LayerSpec], object]

# 第 ``index`` 层每个可见数字的显示单位界限。卡片自己不生成它——界限的唯一来源是参数
# 声明，卡片和参数表因此不可能各说各的。
LayerBounds = Mapping[str, tuple[float, float]]
BoundsLookup = Callable[[int], LayerBounds]

# 同一层每个可见数字「此刻是不是被按住的」。来源同样是参数声明：一枚永远写「未锁定」的
# 方框，在这个参数真被锁住时说的就是反话。
LayerLocks = Mapping[str, bool]
LocksLookup = Callable[[int], LayerLocks]

# 设计稿的 ``.inp{width:96px}``：数字框只装得下一个长度，不跟着表单一起拉宽——省下的
# 宽度归它右边那根条。这 96px 是按设计稿自己的 12.5px 字号量的（那只框里恰好写下
# ``48.70000000``），而屏上的字号由系统给；所以它是下限，不是定值，见 :func:`_field_width`。
FIELD_WIDTH_PX = 96

# 设计稿在这只框里写的是 ``48.70000000``：两位整数、一个小数点、``NUMBER_DECIMALS`` 位小数。
FIELD_INTEGER_DIGITS = 2

# 框自己占掉的那一圈：样式表给的左右各 ``SPACE_SM`` 内缩，加两道 1px 的边。
FIELD_CHROME_PX = 2 * theme.SPACE_SM + 2

# 设计稿 ``.field{gap:3px}`` 配 ``margin-bottom:10px``：标签紧贴自己下面那个框，格与格之间
# 松一档。两个间距一样大时，四格十行字读起来是均匀的一片，哪个标签管哪个框只能靠对齐去
# 猜；3 比 10 的落差让「标签跟着它下面那个框」在扫视时自己成组。
FIELD_GAP_PX = 3
FIELD_MARGIN_PX = 10

# 设计稿这三个框写的是 ``48.70000000`` / ``0.44000000`` / ``2.28000000``。四位会截掉真实位数：
# 拟合解出来的厚度是 48.7031… 这种数，读者把框里的数改一位再提交时，被写回结构的是显示
# 出来的那份，末几位就这么悄悄丢了。八位也是本仓库另外四处数字输入的既有惯例。
NUMBER_DECIMALS = 8

# 周期块、梯度层没有单一的厚度或密度，一组填进去的数字只会是假的；卡片整体禁用并
# 说明去哪里改，而不是给一个提交出去就会被拒的表单。
UNSUPPORTED_HINT = "该组件不是普通层，请用「添加周期块」对话框编辑。"
EMPTY_HINT = "在层堆叠里选一层即可就地编辑它的材料与几何。"

# 设计稿在化学式框下摆了一行 ``.help``。它说的两件事——这个框会拿化学式去算 SLD、
# 密度也可以另给——是这个框唯一的用法说明；只挂 tooltip 等于要求使用者先猜到该悬停，
# 而不知道填什么的人恰好不会悬停。同一份文案也给 tooltip，两处不会说得不一样。
FORMULA_HELP_TEXT = "输入化学式（如 SiO2、Al2O3）由 periodictable 自动计算 SLD；也可直接给定密度。"

# 设计稿写的是「材料 / 化学式」。``MaterialSpec`` 带着名字、化学式和密度三样东西，而
# 下面那行 ``.help`` 明说了「也可直接给定密度」——只写「化学式」和它自己的说明对不上。
FORMULA_LABEL = "材料 / 化学式"

# 设计稿这三格的标签：量的符号和单位都写在标签里，框里只剩数字。
THICKNESS_LABEL = "厚度 d（nm）"
ROUGHNESS_LABEL = "粗糙度 σ（nm）"
DENSITY_LABEL = "密度 ρ（g·cm⁻³）"

# 设计稿这一格写的是 ``a-Si（自定义密度）``——名字，加上它的 SLD 是怎么来的。只印化学式，
# 这一格就答不出那行字：一层名叫 a-Si、化学式是 Si 的层，框里写 ``Si``，而堆叠里、抬头里
# 它都叫 a-Si。``MaterialSpec`` 恰好禁止「没有化学式却给了密度」，所以「自定义密度」这四个
# 字唯一能指的就是「名字与化学式不是同一个」；直接给定 SLD 的层连化学式都没有，那一格得
# 说出来，不能装作有。
CUSTOM_DENSITY_SUFFIX = "（自定义密度）"
DIRECT_SLD_SUFFIX = "（直接给定 SLD）"

# 设计稿这一格右端是一枚 ``▾``：常见材料点一下就有，不用记怎么拼。这五个是
# ``services.materials`` 的初始密度表——填表以外的化学式一样能算，但只有这几个是应用能
# 立刻给出体密度的。表在服务层，没有经 ``api`` 转出来，所以这里只抄名字：抄错了对不上，
# 点出来的化学式算不出密度，界面上一眼就能看见。
KNOWN_FORMULAS = ("Si", "SiO2", "Si3N4", "TaN", "Zr")
MENU_GLYPH = "▾"
MENU_TOOLTIP = "常见材料"

# 设计稿 ``密度 ρ（g·cm⁻³）`` 的标签后面跟着一枚 14px 的 ``.lock``。密度是三个数里最容易被按住
# 的量（薄层的 d 与 ρ 相关，读者常先把 ρ 钉在手册值上再拟合 d），方框摆在标签旁边，扫参数名
# 的时候就答了「这个数是我按住的还是求解器在动的」。
LOCK_BOX_PX = 14
LOCK_GLYPH = "✓"
LOCKED_TOOLTIP = "已锁定"
UNLOCKED_TOOLTIP = "未锁定"

# 空态数字格显示的字符。``QDoubleSpinBox`` 只有一种「不显示数字」的办法：把
# ``specialValueText`` 顶在下限上，所以这个串不能是空串——空串等于「没有特殊文本」，
# 显示的还是下限那个数。
BLANK_NUMBER_TEXT = "—"

# 声明里长度以 Å 存，可见的每一处都是 nm。``parameters.grouping`` 对参数表做的是同一
# 道换算，两处共用一个常数才不会有一处漏改。
NM_PER_ANGSTROM = 0.1


def layer_bounds(
    definitions: Sequence[api.ParameterDefinition],
    index: int,
    material: api.MaterialSpec,
) -> dict[str, tuple[float, float]]:
    """The display-unit span each of the layer's editable numbers may move in.

    密度那一行要多转一道：声明里是无量纲的 ``density_scale``（相对材料体密度的倍率），
    卡片显示的是绝对的 g·cm⁻³。换算是线性的，两端界限乘同一个体密度就是准确值。显式
    给 SLD 的材料没有体密度可乘，那一项就不给——编造一段界限等于把「我不知道」画成
    「就是这一段」。

    拿不到声明的量原样缺席（结构刚建好、曲线还没导入时整份都是空的）。
    """
    by_name = {definition.name: definition for definition in definitions}
    bounds: dict[str, tuple[float, float]] = {}
    for key, suffix, scale in (
        ("thickness", "thickness_a", NM_PER_ANGSTROM),
        ("roughness", "roughness_a", NM_PER_ANGSTROM),
        ("density", "density_scale", material.bulk_density_g_cm3),
    ):
        definition = by_name.get(f"component.{index}.{suffix}")
        if definition is None or scale is None:
            continue
        bounds[key] = (definition.lower * scale, definition.upper * scale)
    return bounds


def layer_locks(definitions: Sequence[api.ParameterDefinition], index: int) -> dict[str, bool]:
    """Whether each of the layer's editable numbers is currently held fixed.

    和 :func:`layer_bounds` 读的是同一批声明，键也一样。没有对应声明的量原样缺席：那时候
    连锁没锁都不知道，画一枚方框等于替声明说话。
    """
    by_name = {definition.name: definition for definition in definitions}
    locks: dict[str, bool] = {}
    for key, suffix in (
        ("thickness", "thickness_a"),
        ("roughness", "roughness_a"),
        ("density", "density_scale"),
    ):
        definition = by_name.get(f"component.{index}.{suffix}")
        if definition is None:
            continue
        locks[key] = bool(definition.locked)
    return locks


def material_label(material: api.MaterialSpec) -> str:
    """这一格显示的那行字：材料叫什么，以及它的 SLD 是从哪儿来的。

    名字和化学式是同一个（``Si``）时多写什么都是重复；不是同一个就说清楚密度是另给的；
    连化学式都没有的层只能靠直接给定的 SLD，那一格得承认这件事。
    """
    if material.formula is None:
        return f"{material.name}{DIRECT_SLD_SUFFIX}"
    if material.name == material.formula:
        return material.formula
    return f"{material.name}{CUSTOM_DENSITY_SUFFIX}"


class BoundState(NamedTuple):
    """绑定这一层时，四个框里各是什么。

    提交闸门比的是这个，而不是重建出来的 ``LayerSpec``：nm↔Å 来回换算差在小数第十几位上
    （``48.7 / 0.1 == 486.99999999999994``），拿重建件跟原件比，每次失焦都会看出一次「改了」。
    逐格比显示值，再只换真的动过的那几格，没碰的字段就逐位不变。
    """

    label: str
    thickness: float
    roughness: float
    density: float


def _field_width(editor: QDoubleSpinBox) -> int:
    """框宽：设计稿的 96px，但至少要写得下 ``48.70000000`` 这样一个取值。

    这三格显示八位小数是有意的——拟合解出来的厚度是 48.7031… 这种数，读者改一位再提交
    时，被写回结构的是显示出来的那份。屏上只写得出 ``48.700`` 时他读到的是一个更精确、
    而且错的数：他不知道自己没看见后面几位，也就不知道这一次提交会把它们抹掉。

    十个数字的字形不一定同宽（``1`` 常比 ``0`` 窄），按此刻恰好写着的那串去量，换一层材料
    就可能差几像素；取最宽的那个数字，一次算准所有取值。
    """
    metrics = QFontMetricsF(editor.font())
    digits = FIELD_INTEGER_DIGITS + NUMBER_DECIMALS
    widest = max(metrics.horizontalAdvance(digit) for digit in "0123456789")
    needed = ceil(digits * widest + metrics.horizontalAdvance(".")) + FIELD_CHROME_PX
    return max(FIELD_WIDTH_PX, needed)


def _number(name: str, label: str, minimum: float) -> QDoubleSpinBox:
    editor = QDoubleSpinBox()
    editor.setObjectName(name)
    editor.setAccessibleName(label)
    editor.setDecimals(NUMBER_DECIMALS)
    editor.setRange(minimum, 1_000_000.0)
    # 设计稿那只框是纯输入框，框里只有数字。默认的上下箭头要在这 96px 里占掉 18px——够写
    # 三位小数，而这三格的全部意思就是把解出来的位数摆给人看。数字仍能用上下方向键和滚轮
    # 调，少的只是那两枚在设计稿里从来没有过的箭头。
    editor.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    editor.setFixedWidth(_field_width(editor))
    return editor


def _material_field(editor: QLineEdit, button: QToolButton) -> QFrame:
    """设计稿的 ``.inp.sel``：一只框，``▾`` 在框里靠右。

    输入框和按钮各自带边，屏幕上就是两只挨着的框，读者会以为那枚 ``▾`` 是另一个控件。边
    画在外面这只 ``QFrame`` 上，里面两件去掉自己的边，点框里任何地方都在编辑同一样东西。
    """
    field = QFrame()
    field.setObjectName("selectedLayerMaterialField")
    layout = QHBoxLayout(field)
    # 设计稿 ``.inp{padding:0 10px}``：左右内缩由外框给，里面的输入框自己不再留白，否则
    # 文字会离左边框 20px。
    layout.setContentsMargins(FIELD_MARGIN_PX, 0, FIELD_MARGIN_PX, 0)
    layout.setSpacing(0)
    layout.addWidget(editor, 1)
    layout.addWidget(button)
    return field


def _lock_chip() -> QLabel:
    """设计稿的 ``.lock``：14px 的小方框，锁住时填成主色并打一个勾。

    ``QLabel`` 而不是 ``QCheckBox``——它报的是声明此刻的状态，点它不改任何东西；能点的
    复选框会让读者以为锁是在这里开关的，而锁归参数表管。
    """
    chip = QLabel()
    chip.setObjectName("selectedLayerDensityLock")
    chip.setFixedSize(LOCK_BOX_PX, LOCK_BOX_PX)
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return chip


def _edited_material(
    material: api.MaterialSpec,
    bound: BoundState,
    label: str,
    density: float,
) -> api.MaterialSpec:
    """把两格里改动过的那些写回材料，没动过的原样留着。

    那行字没动就绝不能拿它当化学式：框里显示的是材料标签，``a-Si（自定义密度）`` 不是
    periodictable 认得的串，写回去这一层的 SLD 从此算不出来。它动了才是化学式——那时还要
    撤掉可能存在的 SLD 覆盖，``MaterialSpec`` 只许一个 SLD 来源，两个都给会被当场拒掉。

    体密度是 ``None`` 的层（直接给定 SLD）那一格是空的，空格里读出来的数不是这一层的密度，
    不能顺手写进去。
    """
    if label == bound.label:
        if material.bulk_density_g_cm3 is None or density == bound.density:
            return material
        return replace(material, bulk_density_g_cm3=density)
    return replace(
        material,
        formula=label,
        bulk_density_g_cm3=None if material.bulk_density_g_cm3 is None else density,
        sld_override_a2=None,
    )


class SelectedLayerCard(QWidget):
    """Edit the currently selected plain layer, or state why it cannot."""

    def __init__(
        self,
        commit_layer: CommitLayer,
        bounds_for: BoundsLookup | None = None,
        locks_for: LocksLookup | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("selectedLayerCard")
        self.setAccessibleName("选中层")
        self._commit_layer = commit_layer
        self._bounds_for = bounds_for
        self._locks_for = locks_for
        self._index: int | None = None
        self._layer: api.LayerSpec | None = None
        # 绑定这一层时四个框里各是什么。提交闸门比的是它，见 :class:`BoundState`。
        self._bound: BoundState | None = None
        # 卡内不写层名：设计稿把它写进抬头「选中层 · a-Si 非晶硅」，卡内第一格直接是材料。
        # 抬头同时是这张卡的无障碍名称，层名写在那里比卡内再占一行更省高度。
        self.formula_editor = QLineEdit()
        self.formula_editor.setObjectName("selectedLayerFormulaInput")
        self.formula_editor.setAccessibleName(FORMULA_LABEL)
        self.formula_editor.setToolTip(FORMULA_HELP_TEXT)
        # 边归外面那只 ``QFrame``：一格里两道边就成了两只框。
        self.formula_editor.setFrame(False)
        self.material_menu_button = QToolButton()
        self.material_menu_button.setObjectName("selectedLayerMaterialMenuButton")
        self.material_menu_button.setText(MENU_GLYPH)
        self.material_menu_button.setAccessibleName(MENU_TOOLTIP)
        self.material_menu_button.setToolTip(MENU_TOOLTIP)
        self.material_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.material_menu_button)
        for formula in KNOWN_FORMULAS:
            action = menu.addAction(formula)
            # ``triggered`` 首个实参是 ``checked``；默认值绑住本轮的 ``formula``，闭包才不会
            # 全部落在最后一项上。
            action.triggered.connect(lambda _checked=False, value=formula: self._choose_formula(value))
        self.material_menu_button.setMenu(menu)
        self.material_field = _material_field(self.formula_editor, self.material_menu_button)
        self.formula_help = QLabel(FORMULA_HELP_TEXT)
        self.formula_help.setObjectName("selectedLayerFormulaHelp")
        self.formula_help.setProperty("mutedText", True)
        self.formula_help.setWordWrap(True)
        self.thickness_editor = _number("selectedLayerThicknessInput", THICKNESS_LABEL, 0.02)
        self.roughness_editor = _number("selectedLayerRoughnessInput", ROUGHNESS_LABEL, 0.0)
        self.density_editor = _number("selectedLayerDensityInput", DENSITY_LABEL, 0.000001)
        self.thickness_rail = RailBar("selectedLayerThicknessRail")
        self.roughness_rail = RailBar("selectedLayerRoughnessRail")
        self.density_rail = RailBar("selectedLayerDensityRail")
        self.density_lock = _lock_chip()
        self.hint_label = QLabel()
        self.hint_label.setObjectName("selectedLayerHint")
        self.hint_label.setProperty("mutedText", True)
        self.hint_label.setWordWrap(True)
        self.error_label = QLabel()
        self.error_label.setObjectName("selectedLayerError")
        self.error_label.setProperty("statusKind", "error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self._arrange()
        # 条画的是「当前值落在哪儿」，所以边调边看离边界多远——这是读者调这三个数时唯一
        # 想知道的事，等到失焦才更新就晚了。
        for editor in self._number_editors:
            editor.valueChanged.connect(self._refresh_rails)
        # 设计稿这张卡里没有「应用」：一格填完就写回。``editingFinished`` 在失焦和回车时都
        # 发，值没变它也发，所以提交前要过 :meth:`_commit` 里那道闸门。
        self.formula_editor.editingFinished.connect(self._commit)
        for editor in self._number_editors:
            editor.editingFinished.connect(self._commit)
        self.show_component(None, None)

    def _field(
        self,
        layout: QVBoxLayout,
        name: str,
        label: str,
        editor: QWidget,
        rail: RailBar | None,
        *,
        chip: QLabel | None = None,
    ) -> None:
        """设计稿的 ``.field``：标签独占一行，输入框和条并排在下一行。

        标签摆左边（``QFormLayout`` 的默认）会吃掉「密度 ρ（g·cm⁻³）」那么宽的一列，
        剩给条的只有几十像素——三个数加两个符号写不下，先被挤掉的是右端的上界。

        ``chip`` 跟标签同排：把它和标签装进同一个 ``QHBoxLayout`` 而不是一个容器控件，
        两者的 ``geometry()`` 才都还是相对这张卡的，量得出「方框在标签右边、和它齐平」。
        """
        caption = QLabel(label)
        caption.setObjectName(name)
        if chip is None:
            layout.addWidget(caption)
        else:
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.setSpacing(theme.SPACE_XS)
            head.addWidget(caption)
            head.addWidget(chip)
            head.addStretch(1)
            layout.addLayout(head)
        layout.addSpacing(FIELD_GAP_PX)
        if rail is None:
            layout.addWidget(editor)
            return
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACE_SM)
        row.addWidget(editor)
        row.addWidget(rail, 1)
        layout.addLayout(row)

    def _arrange(self) -> None:
        # 间距全部显式给：格内 3px、格间 10px 是设计稿量出来的两个数，交给 ``setSpacing``
        # 只能得到同一个值。末尾那根弹簧吃掉多余高度，否则卡比 sizeHint 高时富余会摊进
        # 每个 ``QLabel``，把 3 和 10 一起撑大。
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._field(layout, "selectedLayerFormulaLabel", FORMULA_LABEL, self.material_field, None)
        layout.addSpacing(FIELD_GAP_PX)
        layout.addWidget(self.formula_help)
        layout.addSpacing(FIELD_MARGIN_PX)
        self._field(layout, "selectedLayerThicknessLabel", THICKNESS_LABEL, self.thickness_editor, self.thickness_rail)
        layout.addSpacing(FIELD_MARGIN_PX)
        self._field(layout, "selectedLayerRoughnessLabel", ROUGHNESS_LABEL, self.roughness_editor, self.roughness_rail)
        layout.addSpacing(FIELD_MARGIN_PX)
        self._field(
            layout,
            "selectedLayerDensityLabel",
            DENSITY_LABEL,
            self.density_editor,
            self.density_rail,
            chip=self.density_lock,
        )
        layout.addSpacing(FIELD_MARGIN_PX)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.error_label)
        layout.addStretch(1)

    @property
    def _editors(self) -> tuple[QWidget, ...]:
        return (
            self.material_field,
            self.thickness_editor,
            self.roughness_editor,
            self.density_editor,
        )

    @property
    def _number_editors(self) -> tuple[QDoubleSpinBox, ...]:
        return (self.thickness_editor, self.roughness_editor, self.density_editor)

    @property
    def _rails(self) -> tuple[tuple[str, QDoubleSpinBox, RailBar], ...]:
        return (
            ("thickness", self.thickness_editor, self.thickness_rail),
            ("roughness", self.roughness_editor, self.roughness_rail),
            ("density", self.density_editor, self.density_rail),
        )

    def refresh_bounds(self) -> None:
        """Re-read the declarations behind the rails and the lock chip.

        换数据集、换结构都会重发一批声明，而重发和「堆叠重新选中这一层」谁先谁后没有
        保证。两条路都走这里，最后一次刷新用的就一定是新声明。
        """
        self._refresh_rails()
        self._refresh_lock()

    def _refresh_rails(self) -> None:
        bounds: LayerBounds = {}
        if self._layer is not None and self._index is not None and self._bounds_for is not None:
            bounds = self._bounds_for(self._index)
        for key, editor, rail in self._rails:
            span = bounds.get(key)
            if span is None:
                rail.clear()
                continue
            rail.set_span(span[0], editor.value(), span[1])

    def _refresh_lock(self) -> None:
        """把方框对齐到声明此刻说的话，说不出话时干脆不画。

        锁着画成实心并打勾，没锁只留一只空框；拿不到声明（结构还没提交）时连锁没锁都不
        知道，那就跟界限条一样空着——一枚永远写「未锁定」的方框，在这个参数真被按住时说
        的就是反话。
        """
        locks: LayerLocks = {}
        if self._layer is not None and self._index is not None and self._locks_for is not None:
            locks = self._locks_for(self._index)
        locked = locks.get("density")
        self.density_lock.setVisible(locked is not None)
        if locked is None:
            return
        self.density_lock.setText(LOCK_GLYPH if locked else "")
        self.density_lock.setProperty("locked", locked)
        self.density_lock.setToolTip(LOCKED_TOOLTIP if locked else UNLOCKED_TOOLTIP)
        theme.repolish(self.density_lock)

    def _blank(self) -> None:
        """Empty every field so a dead card cannot be read as a live one.

        没有层名那一行兜着「未选择」之后，卡内只剩这组数字；留着上一层的 48.7 会读成
        「当前选中层厚 48.7 nm」。

        空的办法只能是 ``specialValueText`` 顶在下限上。清 ``lineEdit()`` 不行：
        ``QAbstractSpinBox`` 在 show、换样式、换字体时都按 ``value`` 重刷显示，而卡片在
        真 app 里一定被显示过——那种空态只在没显示过的测试里是绿的。
        """
        self.formula_editor.clear()
        for editor in self._number_editors:
            editor.setSpecialValueText(BLANK_NUMBER_TEXT)
            editor.setValue(editor.minimum())

    def show_component(self, index: int | None, component: object | None) -> None:
        """Bind the card to ``component``, or shut it down when it cannot edit one."""
        self.error_label.hide()
        layer = component if isinstance(component, api.LayerSpec) else None
        self._index = index if layer is not None else None
        self._layer = layer
        for editor in self._editors:
            editor.setEnabled(layer is not None)
        if layer is None:
            self._bound = None
            self._blank()
            self.hint_label.setText(EMPTY_HINT if component is None else UNSUPPORTED_HINT)
            self.hint_label.show()
            self._refresh_rails()
            self._refresh_lock()
            return
        # 设计稿这张卡里四格之外没有别的字。「没选中」「选不了」那两句是卡里唯一的内容时才
        # 该出现；选中之后再挂一句操作说明，等于每选一层就重复一遍同样的话。
        self.hint_label.clear()
        self.hint_label.hide()
        # 空态占的是「等于下限」这个值，绑定真层时必须撤掉，否则粗糙度真为 0 的理想光滑
        # 界面会跟着显示成空。撤成空串就是 Qt 的默认：没有特殊文本，下限照常显示成数字。
        for editor in self._number_editors:
            editor.setSpecialValueText("")
        label = material_label(layer.material)
        self.formula_editor.setText(label)
        # 长度在结构里以 Å 存，可见的每一处都是 nm；换算只发生在这道边界上。
        self.thickness_editor.setValue(float(layer.thickness_a) * NM_PER_ANGSTROM)
        self.roughness_editor.setValue(float(layer.roughness_a) * NM_PER_ANGSTROM)
        density = layer.material.bulk_density_g_cm3
        if density is None:
            # 直接给定 SLD 的层没有体密度。``float(None)`` 会当场抛 TypeError，右栏整段跟着
            # 不出来；而随便填一个数就是替这一层编一个它没有的量。空着并且改不动才是真话。
            self.density_editor.setSpecialValueText(BLANK_NUMBER_TEXT)
            self.density_editor.setValue(self.density_editor.minimum())
            self.density_editor.setEnabled(False)
        else:
            self.density_editor.setValue(float(density))
        # 闸门比的是「框里此刻是什么」，所以基线也从框里读回来——八位小数的取整已经发生过，
        # 拿源数据当基线会让第一次失焦看出一次不存在的改动。
        self._bound = BoundState(
            label,
            self.thickness_editor.value(),
            self.roughness_editor.value(),
            self.density_editor.value(),
        )
        self._refresh_rails()
        self._refresh_lock()

    def _choose_formula(self, formula: str) -> None:
        """▾ 菜单里点中一个化学式：填进框里并立刻写回。

        点菜单不会让输入框失焦，``editingFinished`` 因此不发。不在这里提交的话，选完还得
        再点一次别处才生效，而读者已经做完了「选材料」这个动作。
        """
        self.formula_editor.setText(formula)
        self._commit()

    def _commit(self) -> None:
        """Write the four fields back, but only when one of them actually moved.

        ``editingFinished`` 每次失焦都发，值没变它也发。照发就提交的话，读者只是把光标移开
        就得到一句「结构已修改（未保存）」，接着这份结构还会被当成新结构去问要不要重拟合，
        而屏幕上四个数跟他进来时一模一样。
        """
        layer = self._layer
        index = self._index
        bound = self._bound
        if layer is None or index is None or bound is None:
            return
        label = self.formula_editor.text().strip()
        thickness = self.thickness_editor.value()
        roughness = self.roughness_editor.value()
        density = self.density_editor.value()
        if (label, thickness, roughness, density) == bound:
            return
        try:
            edited = replace(
                layer,
                material=_edited_material(layer.material, bound, label, density),
                # 没动过的那一格逐位照抄。nm→Å 再回来差在小数第十几位上
                # （``48.7 / 0.1 == 486.99999999999994``），换算一遍就等于替读者改了他没碰的数。
                thickness_a=layer.thickness_a if thickness == bound.thickness else thickness / NM_PER_ANGSTROM,
                roughness_a=layer.roughness_a if roughness == bound.roughness else roughness / NM_PER_ANGSTROM,
            )
            self._commit_layer(index, edited)
        except (TypeError, ValueError) as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        self.error_label.hide()
        # 提交会让结构重新发一遍下来，那条路径自己会把这张卡重新绑到同一层上。它没走到
        # （选中被挪开了）时才轮到这里记账，否则会把已经刷新过的基线覆盖回旧的一份。
        if self._index == index:
            self._layer = edited
            self._bound = BoundState(label, thickness, roughness, density)
