"""帧③ 的「选中层」检查器：选中一层，就地改它的材料与几何。

设计稿的右栏在结构编辑态是一张 ``选中层 · a-Si 非晶硅`` 卡：材料 / 化学式、厚度 d（nm）、
粗糙度 σ（nm）、密度 ρ（g·cm⁻³）各一个输入框，改完即写回结构。之前这些只能通过
「添加普通层」对话框在新建时填一次，已有层要改只能删了重加——层序会变，堆叠里的
选中位置也会丢。这里量的是那张卡：它认不认得当前选中的层，改了之后提交出去的
``LayerSpec`` 对不对，以及选不中普通层时它是不是老实地把自己关掉。

设计稿这张卡里没有「应用」按钮，四个框各自离焦或回车就提交。所以这里也量「没改就
不提交」——``editingFinished`` 每次失焦都发，不设闸门的话点进点出一次就会把一份一模
一样的结构推给文档，界面跟着报「结构已修改（未保存）」，而使用者什么都没动。
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QAbstractSpinBox, QFrame, QLabel, QLineEdit, QToolButton, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.structure.selection import BLANK_NUMBER_TEXT, EMPTY_HINT, UNSUPPORTED_HINT

pytest.importorskip("pytestqt")


def _layer(name: str = "a-Si", *, thickness_a: float = 487.0, roughness_a: float = 30.0) -> api.LayerSpec:
    return api.LayerSpec(
        name,
        api.MaterialSpec(name, "Si", 2.28),
        thickness_a,
        roughness_a=roughness_a,
    )


def _card(qtbot, *, bounds=None, locks=None):
    from xrr_fitter.gui.structure.selection import SelectedLayerCard

    commits: list[tuple[int, api.LayerSpec]] = []
    card = SelectedLayerCard(
        lambda index, layer: commits.append((index, layer)),
        bounds_for=None if bounds is None else (lambda _index: bounds),
        locks_for=None if locks is None else (lambda _index: locks),
    )
    qtbot.addWidget(card)
    return card, commits


def _finish(qtbot, editor) -> None:
    """按下回车——设计稿没有「应用」，一格填完的信号就是这个。"""
    qtbot.keyClick(editor, Qt.Key.Key_Return)


def test_the_formula_field_says_what_it_accepts_without_being_hovered(qtbot) -> None:
    """设计稿的 ``.help`` 是化学式框下的一行可见说明，不是 tooltip。

    「化学式」这个标签只说了格式的名字，没说这个框会拿它去算 SLD、也没说密度可以
    另给。这两件事是这个框唯一的用法说明，藏在 tooltip 里等于要求使用者先猜到该
    悬停——不知道填什么的人恰好不会去悬停。
    """
    card, _commits = _card(qtbot)

    help_label = card.findChild(QLabel, "selectedLayerFormulaHelp")
    assert help_label is not None
    assert "periodictable" in help_label.text()
    assert "也可直接给定密度" in help_label.text()
    assert help_label.wordWrap()
    assert help_label.isVisibleTo(card)


def test_the_card_names_the_selected_layer_and_shows_its_geometry_in_nm(qtbot) -> None:
    """卡内第一格就是化学式——层名由抬头写，长度按 nm 显示，不是内部的 Å。

    结构的存储单位是 Å，设计稿和所有可见的长度输入都是 nm；卡片直接摆 487 会让
    读者以为这层有 487 nm 厚。
    """
    card, _commits = _card(qtbot)

    card.show_component(2, _layer())

    assert card.findChild(QLabel, "selectedLayerTitle") is None
    assert card.thickness_editor.value() == pytest.approx(48.7)
    assert card.roughness_editor.value() == pytest.approx(3.0)
    assert card.density_editor.value() == pytest.approx(2.28)
    assert card.findChild(QLineEdit, "selectedLayerFormulaInput").text() == "a-Si（自定义密度）"


def test_a_field_that_is_filled_in_commits_without_waiting_for_a_button(qtbot) -> None:
    """设计稿这张卡里没有按钮：改厚度、回车，替换件就提交出去了。

    卡里四个框改的都是同一层的同一份 ``LayerSpec``，一格填完就写回；再摆一枚「应用」
    等于让每次改动都要两个动作，而设计稿把那枚按钮的位置留给了下一段抬头。
    """
    card, commits = _card(qtbot)
    card.show_component(2, _layer())

    card.thickness_editor.setValue(50.0)
    _finish(qtbot, card.thickness_editor)

    assert len(commits) == 1
    index, layer = commits[0]
    assert index == 2
    assert layer.thickness_a == pytest.approx(500.0)
    assert layer.roughness_a == pytest.approx(30.0)
    assert layer.material.formula == "Si"


def test_the_card_has_no_apply_button_left_to_press(qtbot) -> None:
    """按钮不是隐藏而是不存在——留着它就还能被 Tab 走到、被读屏念到。"""
    card, _commits = _card(qtbot)

    assert card.findChild(QWidget, "applySelectedLayerButton") is None
    assert not hasattr(card, "apply_button")


def test_leaving_a_field_untouched_commits_nothing(qtbot) -> None:
    """点进去又点出来不算改动。

    ``editingFinished`` 每次失焦都发，值没变它也发。照发就提交的话，读者只是把光标移
    开就得到一句「结构已修改（未保存）」，接着这份结构还会被当成新结构去问要不要重
    拟合——而屏幕上的四个数跟他进来时一模一样。
    """
    card, commits = _card(qtbot)
    card.show_component(2, _layer())

    for editor in (card.thickness_editor, card.roughness_editor, card.density_editor):
        _finish(qtbot, editor)

    assert commits == []


def test_a_periodic_block_leaves_the_card_disabled_rather_than_half_filled(qtbot) -> None:
    """周期块没有单一厚度，卡片整体禁用并说明原因，不给一组会骗人的数字。"""
    card, commits = _card(qtbot)
    card.show_component(0, object())

    assert not card.thickness_editor.isEnabled()
    assert card.hint_label.text() == UNSUPPORTED_HINT
    assert commits == []


def test_nothing_selected_clears_the_fields_instead_of_keeping_stale_numbers(qtbot) -> None:
    """没有选中项时字段要空着，而不是留着上一层的数字。

    层名原先在卡内单占一行，「未选择」写在那里就够；现在层名归抬头，卡内只剩这组数
    字——留着 48.7 会读成「当前选中层厚 48.7 nm」。

    卡片必须真的显示过：``QAbstractSpinBox`` 在 show 时按 ``value`` 重刷显示，只清
    ``lineEdit()`` 的空态在没显示的测试里是绿的，一到真 app 里就被刷回数字。
    """
    card, _commits = _card(qtbot)
    card.show_component(1, _layer())

    card.show_component(None, None)
    card.show()
    qtbot.waitExposed(card)

    assert card.hint_label.text() == EMPTY_HINT
    assert card.findChild(QLineEdit, "selectedLayerFormulaInput").text() == ""
    assert card.thickness_editor.text() == BLANK_NUMBER_TEXT
    assert card.roughness_editor.text() == BLANK_NUMBER_TEXT
    assert card.density_editor.text() == BLANK_NUMBER_TEXT
    assert not card.thickness_editor.isEnabled()


def test_a_perfectly_smooth_interface_still_reads_as_a_number_not_as_blank(qtbot) -> None:
    """空态占用的是「等于下限」这个值，所以真的等于下限的层不能跟着一起变空。

    粗糙度 0 是一层可以有的真实几何——理想光滑界面。空态靠 ``specialValueText`` 顶在
    下限上，绑定真层时必须把它撤掉；不撤，这一层的 0 会读成「没选中」。
    """
    card, _commits = _card(qtbot)

    card.show_component(1, _layer(roughness_a=0.0))
    card.show()
    qtbot.waitExposed(card)

    assert card.roughness_editor.value() == pytest.approx(0.0)
    assert card.roughness_editor.text() != BLANK_NUMBER_TEXT
    assert card.roughness_editor.text() == "0.00000000"


DESIGN_BOUNDS = {"thickness": (40.0, 60.0), "roughness": (0.0, 1.0), "density": (2.0, 2.5)}


def test_each_number_says_which_span_it_is_allowed_to_move_in(qtbot) -> None:
    """设计稿每个数字框右边跟着一根 ``.railbar``：``40 ≤ 48.7 ≤ 60``。

    框里的数字只说了「现在是多少」。拟合器动的正是这三个数，而它们能走多远写在声明
    里——不把界限印在旁边，读者要判断「48.7 是不是快贴上界了」就得离开这张卡去参数表
    里翻那一行，而那张表在同一栏里往下滚。
    """
    card, _commits = _card(qtbot, bounds=DESIGN_BOUNDS)

    card.show_component(2, _layer(thickness_a=487.0, roughness_a=4.4))

    assert card.thickness_rail.text() == "40 ≤ 48.7 ≤ 60"
    assert card.roughness_rail.text() == "0 ≤ 0.44 ≤ 1"
    assert card.density_rail.text() == "2 ≤ 2.28 ≤ 2.5"
    assert card.thickness_rail.fraction() == pytest.approx(0.435)


def test_the_rails_go_blank_together_with_the_fields_they_annotate(qtbot) -> None:
    """没选中普通层时，条跟着数字一起空掉。

    留着上一层的 ``40 ≤ 48.7 ≤ 60`` 会读成「当前这一层的厚度界限」，而这恰好是它此刻
    唯一不能表示的意思——数字都空了，界限却还挂着，比两个都空更容易骗人。
    """
    card, _commits = _card(qtbot, bounds=DESIGN_BOUNDS)
    card.show_component(2, _layer())

    card.show_component(None, None)

    assert card.thickness_rail.text() == ""
    assert card.roughness_rail.text() == ""
    assert card.density_rail.text() == ""


def test_a_layer_with_no_declarations_shows_the_number_without_inventing_a_span(qtbot) -> None:
    """拿不到声明就只空掉条，数字照常显示。

    结构还没提交、曲线还没导入的时候是没有声明的；那时候编造一段界限，等于把「我不
    知道」画成「就是这一段」。
    """
    card, _commits = _card(qtbot)

    card.show_component(2, _layer())

    assert card.thickness_editor.value() == pytest.approx(48.7)
    assert card.thickness_rail.text() == ""


def test_the_field_label_sits_above_its_row_so_the_rail_has_room_to_be_read(qtbot) -> None:
    """设计稿的 ``.field > label`` 是独占一行的块级标签，输入框和条并排在下一行。

    标签摆左边（``QFormLayout`` 的默认）会吃掉「密度 ρ（g·cm⁻³）」那么宽的一列，剩给
    条的只有几十像素——三个数加两个符号写不下，先被挤掉的恰好是右端的上界。
    """
    from PySide6.QtWidgets import QLabel

    card, _commits = _card(qtbot, bounds=DESIGN_BOUNDS)
    card.show_component(2, _layer())
    card.resize(300, 400)
    card.show()
    qtbot.waitExposed(card)

    label = card.findChild(QLabel, "selectedLayerThicknessLabel")
    assert label is not None
    assert label.geometry().bottom() <= card.thickness_editor.geometry().top()
    assert card.thickness_rail.geometry().left() >= card.thickness_editor.geometry().right()
    assert card.thickness_rail.width() > card.thickness_rail.minimumWidth()


def test_the_material_field_is_labelled_the_way_the_design_labels_it(qtbot) -> None:
    """设计稿写的是「材料 / 化学式」——这个框收的不止是化学式。

    ``MaterialSpec`` 带着名字、化学式和密度三样东西，而卡片下面那行 ``.help`` 明说了
    「也可直接给定密度」。只写「化学式」的标签和它自己的说明文字对不上。
    """
    from PySide6.QtWidgets import QLabel

    card, _commits = _card(qtbot)

    label = card.findChild(QLabel, "selectedLayerFormulaLabel")
    assert label is not None
    assert label.text() == "材料 / 化学式"


def test_the_bounds_are_the_ones_the_parameter_table_reads_from(qtbot, tmp_path) -> None:
    """条上的界限直接取自参数声明，卡和参数表因此不可能各说各的。

    密度那一行要多转一道：声明里是无量纲的 ``density_scale``，卡片显示的是绝对的
    g·cm⁻³。换算是线性的（乘材料的体密度），所以两端界限乘同一个数就是准确值，不是
    近似。

    厚度和粗糙度在声明里以 Å 存，可见处一律 nm——和输入框走同一道换算。
    """
    from xrr_fitter.gui.structure.selection import layer_bounds

    curve = tmp_path / "curve.xy"
    curve.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    layer = _layer(thickness_a=487.0, roughness_a=4.4)
    project = api.add_dataset(api.new_project(), curve, api.InstrumentSpec())
    project = api.set_structure(
        project,
        "curve",
        api.StructureSpec(api.MaterialSpec("Air", None, None, 0.0j), (layer,), api.MaterialSpec("Si", "Si", 2.329)),
    )
    definitions = api.describe_parameters(project, "curve")

    bounds = layer_bounds(definitions, 0, layer.material)

    thickness = next(item for item in definitions if item.name == "component.0.thickness_a")
    density = next(item for item in definitions if item.name == "component.0.density_scale")
    assert bounds["thickness"] == pytest.approx((thickness.lower / 10.0, thickness.upper / 10.0))
    assert bounds["density"] == pytest.approx((density.lower * 2.28, density.upper * 2.28))
    assert bounds["thickness"][0] < 48.7 < bounds["thickness"][1]


def test_the_numbers_carry_the_precision_the_design_writes_them_with(qtbot) -> None:
    """设计稿这三个框写的是 ``48.70000000`` / ``0.44000000`` / ``2.28000000``——八位小数。

    四位在这张卡上会截掉真实位数：拟合解出来的厚度是 48.7031 nm 这种数，四位显示还
    行，可读者把框里的数改一位再提交时，被写回结构的是显示出来的那份，末几位就这么
    悄悄丢了。八位也是本仓库另外四处数字输入的既有惯例。
    """
    card, _commits = _card(qtbot)

    card.show_component(2, _layer(thickness_a=487.0, roughness_a=4.4))
    card.show()
    qtbot.waitExposed(card)

    assert card.thickness_editor.decimals() == 8
    assert card.roughness_editor.decimals() == 8
    assert card.density_editor.decimals() == 8
    assert card.thickness_editor.text() == "48.70000000"
    assert card.roughness_editor.text() == "0.44000000"
    assert card.density_editor.text() == "2.28000000"


MATERIAL_LABELS = (
    (api.MaterialSpec("Si", "Si", 2.329), "Si"),
    (api.MaterialSpec("a-Si", "Si", 2.28), "a-Si（自定义密度）"),
    (api.MaterialSpec("Air", None, None, 0.0j), "Air（直接给定 SLD）"),
)


@pytest.mark.parametrize(("material", "label"), MATERIAL_LABELS)
def test_the_material_field_reads_as_a_material_not_as_a_bare_formula(qtbot, material, label) -> None:
    """设计稿这一格写的是 ``a-Si（自定义密度）``——名字加上它的 SLD 是怎么来的。

    只印化学式，这一格就答不出设计稿那行字：一层名叫 a-Si、化学式是 Si 的层，框里写
    ``Si``，而堆叠里、抬头里、以及设计稿这一格里它都叫 a-Si。``MaterialSpec`` 恰好禁止
    「没有化学式却给了密度」，所以「自定义密度」这四个字唯一能指的就是「名字与化学式
    不是同一个」；直接给定 SLD 的层连化学式都没有，那一格得说出来，不能装作有。
    """
    card, _commits = _card(qtbot)

    card.show_component(1, api.LayerSpec("层", material, 487.0, roughness_a=30.0))

    assert card.findChild(QLineEdit, "selectedLayerFormulaInput").text() == label


def test_a_layer_whose_sld_is_given_directly_leaves_the_density_box_blank(qtbot) -> None:
    """直接给定 SLD 的层没有体密度，那一格空着并且改不动，而不是崩在绑定的路上。

    ``MaterialSpec`` 的两种形态里，SLD 覆盖那一种的 ``bulk_density_g_cm3`` 是 ``None``；
    照着 ``float(...)`` 往框里塞会当场抛 ``TypeError``，右栏整段跟着不出来。空着才是真话：
    这一层的 SLD 不是由密度算出来的，这个框对它没有意义。
    """
    card, _commits = _card(qtbot)

    card.show_component(1, api.LayerSpec("Air", api.MaterialSpec("Air", None, None, 0.0j), 487.0, roughness_a=30.0))
    card.show()
    qtbot.waitExposed(card)

    assert card.density_editor.text() == BLANK_NUMBER_TEXT
    assert not card.density_editor.isEnabled()
    assert card.thickness_editor.isEnabled()


def test_leaving_the_material_label_alone_moves_only_the_density(qtbot) -> None:
    """没动那行字，就只改密度——名字和化学式都留着。

    框里显示的是「材料标签」而不是化学式原文。把它原样当化学式提交，一层叫 a-Si 的层
    会被写成化学式 ``a-Si（自定义密度）``，而 periodictable 认不出这串字，这层的 SLD 从此
    算不出来。
    """
    card, commits = _card(qtbot)
    card.show_component(2, _layer())

    card.density_editor.setValue(2.4)
    _finish(qtbot, card.density_editor)

    assert len(commits) == 1
    _index, layer = commits[0]
    assert layer.material.name == "a-Si"
    assert layer.material.formula == "Si"
    assert layer.material.bulk_density_g_cm3 == pytest.approx(2.4)
    # 没碰的那两格必须逐位不变：nm↔Å 来回换算差在小数第十几位上，写回结构就成了一次
    # 「结构变了」，而读者只改了密度。
    assert layer.thickness_a == 487.0
    assert layer.roughness_a == 30.0


def test_typing_a_formula_over_the_label_commits_it_as_the_formula(qtbot) -> None:
    """把那行字换成 ``SiO2``，提交的就是化学式——这一格照 ``.help`` 说的收化学式。"""
    card, commits = _card(qtbot)
    card.show_component(2, _layer())
    editor = card.findChild(QLineEdit, "selectedLayerFormulaInput")

    editor.setText("SiO2")
    _finish(qtbot, editor)

    assert len(commits) == 1
    _index, layer = commits[0]
    assert layer.material.formula == "SiO2"
    assert layer.material.name == "a-Si"


def test_the_material_field_offers_the_formulas_the_app_knows_a_density_for(qtbot) -> None:
    """设计稿这一格右端是一枚 ``▾``：常见材料点一下就有，不用记怎么拼。

    ``services/materials.py`` 里那张初始密度表是这个应用唯一「认得」的化学式集合——填表
    以外的化学式一样能算，但这五个是它能立刻给出体密度的。把它们摆在框里，比让读者
    先去别处查一遍再手打回来省一次往返。
    """
    from xrr_fitter.gui.structure.selection import KNOWN_FORMULAS

    card, commits = _card(qtbot)
    card.show_component(2, _layer())

    button = card.findChild(QToolButton, "selectedLayerMaterialMenuButton")
    assert button is not None
    menu = button.menu()
    assert menu is not None
    assert tuple(action.text() for action in menu.actions()) == KNOWN_FORMULAS
    assert KNOWN_FORMULAS == ("Si", "SiO2", "Si3N4", "TaN", "Zr")

    next(action for action in menu.actions() if action.text() == "Si3N4").trigger()

    assert card.findChild(QLineEdit, "selectedLayerFormulaInput").text() == "Si3N4"
    assert [layer.material.formula for _index, layer in commits] == ["Si3N4"]


def test_the_material_field_is_one_bordered_box_holding_a_borderless_editor(qtbot) -> None:
    """设计稿的 ``.inp.sel`` 是一只框，``▾`` 在框里靠右——不是框旁边又一只控件。

    输入框和按钮各带一道边，屏幕上就是两只挨着的框，读者会以为那枚 ``▾`` 是另一个控件；
    设计稿画的是一整格，点框里任何地方都在编辑同一样东西。
    """
    card, _commits = _card(qtbot)

    field = card.findChild(QFrame, "selectedLayerMaterialField")
    editor = card.findChild(QLineEdit, "selectedLayerFormulaInput")
    button = card.findChild(QToolButton, "selectedLayerMaterialMenuButton")

    assert field is not None
    assert editor.parentWidget() is field
    assert button.parentWidget() is field
    assert editor.hasFrame() is False


DESIGN_LOCKS = {"thickness": False, "roughness": False, "density": False}


def test_the_density_label_carries_the_lock_chip_the_design_draws_beside_it(qtbot) -> None:
    """设计稿 ``密度 ρ（g·cm⁻³）`` 的标签后面跟着一枚 14px 的 ``.lock``，标题写「未锁定」。

    这张卡上三个数里只有密度那一个在设计稿里带着这枚方框——密度是三者中最容易被按住的
    量（薄层的 d 与 ρ 相关，读者常先把 ρ 钉在手册值上再拟合 d）。方框摆在标签旁边，扫参数
    名的时候就答了「这个数是我按住的还是求解器在动的」。
    """
    card, _commits = _card(qtbot, locks=DESIGN_LOCKS)
    card.show_component(2, _layer())
    card.resize(320, 520)
    card.show()
    qtbot.waitExposed(card)

    chip = card.findChild(QLabel, "selectedLayerDensityLock")
    label = card.findChild(QLabel, "selectedLayerDensityLabel")
    assert chip is not None
    assert chip.size().width() == 14
    assert chip.size().height() == 14
    assert chip.toolTip() == "未锁定"
    assert chip.property("locked") is False
    assert chip.isVisibleTo(card)
    assert chip.geometry().left() >= label.geometry().right()
    assert label.geometry().top() <= chip.geometry().center().y() <= label.geometry().bottom()


def test_the_lock_chip_states_the_declaration_it_stands_for(qtbot) -> None:
    """方框不能只是画着好看：锁着的画成实心，没有对应声明时干脆不画。

    一枚永远写「未锁定」的方框，在这个参数真被锁住时说的就是反话——而它旁边那个数此刻
    根本不会动，读者却在等它动。拿不到声明时（结构还没提交）连锁没锁都不知道，那就跟
    界限条一样空着。
    """
    card, _commits = _card(qtbot, locks={"thickness": False, "roughness": False, "density": True})
    chip = card.findChild(QLabel, "selectedLayerDensityLock")

    card.show_component(2, _layer())

    assert chip.property("locked") is True
    assert chip.toolTip() == "已锁定"

    blind, _commits = _card(qtbot)
    blind.show_component(2, _layer())
    blind.resize(320, 520)
    blind.show()
    qtbot.waitExposed(blind)

    assert blind.findChild(QLabel, "selectedLayerDensityLock").isVisibleTo(blind) is False


def test_the_hint_line_does_not_answer_to_the_diagnostics_section_name(qtbot) -> None:
    """这行提示是选中层卡自己的，不能占着「结构诊断」那一段的名字。

    右栏另有一段真的叫结构诊断（设计稿 722），它的正文是 ``structureCorrelationHint``。两
    个不同段落用同一个 objectName，``findChild`` 拿到的是碰巧先遍历到的那一个——量的是
    A、改的是 B，这种测试绿着也不说明什么。
    """
    card, _commits = _card(qtbot)

    assert card.findChild(QLabel, "structureDiagnosticsHint") is None
    assert card.hint_label.objectName() == "selectedLayerHint"


def test_the_hint_steps_aside_once_a_layer_is_actually_bound(qtbot) -> None:
    """选中了层，提示行就收起来——设计稿这张卡里四格之外没有别的字。

    「没选中」和「选不了」这两句必须留着，它们是那时候卡里唯一的内容。选中之后再挂一句
    操作说明，等于把设计稿留给下一段抬头的位置占掉，而那句话每次选层都重复一遍。
    """
    card, _commits = _card(qtbot)
    card.resize(320, 520)
    card.show()
    qtbot.waitExposed(card)

    card.show_component(2, _layer())

    assert card.hint_label.isVisibleTo(card) is False

    card.show_component(None, None)

    assert card.hint_label.isVisibleTo(card) is True
    assert card.hint_label.text() == EMPTY_HINT


def test_the_fields_are_spaced_the_way_the_design_spaces_them(qtbot) -> None:
    """设计稿 ``.field{gap:3px}`` 配 ``margin-bottom:10px``：格内紧、格间松。

    两个间距一样大时，四格十行字读起来是均匀的一片——哪个标签管哪个框要靠对齐去猜。
    3 比 10 的落差让「标签跟着它下面那个框」在扫视时自己成组。
    """
    card, _commits = _card(qtbot, bounds=DESIGN_BOUNDS)
    card.show_component(2, _layer())
    card.resize(320, 560)
    card.show()
    qtbot.waitExposed(card)

    head = card.findChild(QLabel, "selectedLayerThicknessLabel")
    following = card.findChild(QLabel, "selectedLayerRoughnessLabel")
    row_bottom = max(card.thickness_editor.geometry().bottom(), card.thickness_rail.geometry().bottom())

    assert card.thickness_editor.geometry().top() - head.geometry().bottom() - 1 == 3
    assert following.geometry().top() - row_bottom - 1 == 10


# 右栏在 1400×900 下宽 340，卡片两侧各内缩 12——这张卡在屏上就是这么宽，三个框和三根条
# 全部要在这个宽度里放下。
INSPECTOR_CARD_WIDTH_PX = 316

# 设计稿这三格里写的数，配 ``NUMBER_DECIMALS`` 位小数：``48.70000000`` / ``0.44000000`` /
# ``2.28000000``。粗糙度用 4.4 Å 的层才写得出设计稿那个 0.44 nm。
DESIGN_NUMBERS = (
    ("thickness_editor", "48.70000000"),
    ("roughness_editor", "0.44000000"),
    ("density_editor", "2.28000000"),
)


def _sized_card(qtbot):
    """按右栏的真实宽度立起这张卡，并套上应用发货的那份样式表。

    ``fixture`` 不走 :func:`theme.apply_theme`，而一只框里能写多宽由样式表的 ``padding``
    决定；不套样式表量到的就不是应用实际发出去的那只框。
    """
    card, _commits = _card(qtbot, bounds=DESIGN_BOUNDS)
    card.setStyleSheet(theme.build_stylesheet(card.palette()))
    card.show_component(2, _layer(roughness_a=4.4))
    card.resize(INSPECTOR_CARD_WIDTH_PX, 560)
    card.show()
    qtbot.waitExposed(card)
    return card


@pytest.mark.parametrize(("attribute", "text"), DESIGN_NUMBERS)
def test_each_number_is_written_out_in_full_where_it_is_read(qtbot, attribute: str, text: str) -> None:
    """框里那个数必须整个写在屏上，末几位不能被框边裁掉。

    这三格显示八位小数是有意的：拟合解出来的厚度是 48.7031… 这种数，改一位再提交时被
    写回结构的是显示出来的那份。可屏上只写得出 ``48.700`` 时，读者读到的是一个更精确、
    而且错的数——他不知道自己没看见后面几位，也就不知道这一提交会把它们改掉。
    """
    card = _sized_card(qtbot)
    box = getattr(card, attribute)
    edit = box.findChild(QLineEdit)

    assert box.text() == text
    assert edit is not None
    assert edit.width() >= QFontMetrics(box.font()).horizontalAdvance(text)


def test_the_number_boxes_carry_no_stepper_arrows(qtbot) -> None:
    """设计稿那三只框是纯输入框，框里只有数字。

    ``QDoubleSpinBox`` 默认在右端画一对上下箭头。在设计稿给的 96px 里它们要占 18px——
    够写三位小数，而这三格的全部意思就是把解出来的位数摆给人看。数字仍能用上下方向键
    和滚轮调，少的只是那两枚在设计稿里从来没有过的箭头。
    """
    card = _sized_card(qtbot)

    for attribute, _text in DESIGN_NUMBERS:
        box = getattr(card, attribute)
        assert box.buttonSymbols() is QAbstractSpinBox.ButtonSymbols.NoButtons


def test_the_rail_beside_a_number_still_says_the_whole_span(qtbot) -> None:
    """框要的宽度是从它右边那根条借的，条上那句 ``40 ≤ 48.7 ≤ 60`` 不能因此写不下。

    先被挤掉的是右端的上界，而只写 ``40 ≤ 48.7`` 会读成「下界 40，当前 48.7，上不封顶」。
    """
    card = _sized_card(qtbot)

    for name in ("thickness_rail", "roughness_rail", "density_rail"):
        rail = getattr(card, name)
        assert rail.width() >= QFontMetrics(rail.font()).horizontalAdvance(rail.text())
