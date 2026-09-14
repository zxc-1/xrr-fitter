"""帧③ 的「参数总览」是画布列的第三张卡，右栏那一段只剩三档与两枚徽标。

设计稿帧③ 把右栏第二段整段画完了（HTML 711-720）：抬头「参数化 · 厚度 d」，下面一行三档
（自由 / 固定 / 仅范围），再一行两枚徽标（先验、共享）。就这些。而实现把整张参数表——连
「显示高级选项」勾选框、参数/共享/约束三页和底下那行状态字——一起塞进了这一段：340px 的
右栏里那张表要横着挤五列，读者改一层的厚度要在同一栏里滚过一张表才够到诊断。

表该去哪儿，设计稿只说了一次：``.canvas-top`` 的第三个 tab 就叫「参数总览」（HTML 645）。
它没画这张卡的样子——帧③ 的 ``.canvas-body`` 里只有层堆叠与 SLD 剖面两张 ``.plotcard``——
但它是整份设计稿里「参数总览」四个字唯一出现的位置。tab 是这一列的目录（点 SLD 时层堆叠
还在画面里，所以它不是换页），于是这张卡跟着目录进画布列：和层堆叠同一个滚动区，排在它
下面。放进 splitter 的下半段会让它常驻抢走一段高度，而那段高度是同一张设计稿里的 SLD
剖面在用的。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QLabel,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QWidget,
)
from tests.support.model_cases import dataset_project, final_fit_result

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)

OVERVIEW_CARD = "canvasParameterOverview"


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _window(qtbot, tmp_path, *, layers: int = 2, shown: bool = False):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        AIR,
        tuple(api.LayerSpec(f"film{index}", SIO2, 40.0, roughness_a=3.0) for index in range(layers)),
        SI,
    )
    project = api.select_active_dataset(api.set_structure(project, "curve", structure), "curve")
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    if shown:
        window.resize(1400, 900)
        window.show()
        qtbot.waitExposed(window)
    return window


def _fitted_window(qtbot):
    """跑完一轮的项目：管线停在「结果」那一步，也就是帧①。"""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    value = api.XrrProject.new((dataset_project(result=final_fit_result()),), master_seed=1201)
    window = MainWindow(ProjectDocument(replace(value, base_directory="/private/tmp")))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _card(window) -> QFrame:
    card = window.canvas_column.findChild(QFrame, OVERVIEW_CARD)
    assert card is not None, "画布列里没有「参数总览」这张卡"
    return card


def _visible_height(scroll: QScrollArea, widget: QWidget) -> int:
    """``widget`` 在 ``scroll`` 视口里露出来的高度，滚不到就是 0。"""
    viewport = scroll.viewport()
    top = widget.mapTo(viewport, widget.rect().topLeft()).y()
    return max(0, min(top + widget.height(), viewport.height()) - max(top, 0))


def test_the_parameterisation_section_holds_the_three_choices_and_nothing_else(qtbot, tmp_path) -> None:
    """右栏「参数化」这一段就是设计稿画的那两行：三档加两枚徽标。

    设计稿 711-720 把这一段画完了，里面没有表、没有勾选框、没有状态行。它们留在这里的代价
    不是多几像素：这一段和「选中层」「结构诊断」共一栏 340px，一张五列的表夹在中间，读者改
    完一层要滚过整张表才够到下面那句诊断，而那句诊断说的正是刚改的这一层。
    """
    window = _window(qtbot, tmp_path)
    section = window.inspector_column.findChild(QWidget, "inspectorParameters")

    assert section is not None
    assert section.findChild(QWidget, "parameterDisposition") is not None
    assert section.findChild(QWidget, "parameterFreeChoice") is not None
    assert section.findChild(QTabWidget, "parameterTabs") is None
    assert section.findChild(QTableWidget) is None
    assert section.findChild(QCheckBox, "expertModeToggle") is None
    assert section.findChild(QLabel, "parameterStatus") is None


def test_the_bounds_table_hangs_under_the_canvas_tab_that_names_it(qtbot, tmp_path) -> None:
    """整张参数表搬进画布列那张「参数总览」卡里——tab 指的就是它。

    设计稿只在 645 行提过「参数总览」四个字，就是那个 tab。目录指着一张不存在的卡，点它只能
    去右栏那段本不该装表的地方；这张卡建起来之后，目录第三项才有对应的东西。表连着的三页
    （参数 / 共享 / 约束）、勾选框和状态行是同一个面板的零件，跟着它一起走：拆开它们等于把
    「哪一行在哪一页」这件事分到两栏去读。
    """
    window = _window(qtbot, tmp_path)
    card = _card(window)

    assert card.findChild(QLabel, f"{OVERVIEW_CARD}Title").text() == "参数总览"
    assert card.isAncestorOf(window.parameters_panel)
    assert card.isAncestorOf(window.parameters_panel.parameter_table)
    assert card.findChild(QTabWidget, "parameterTabs") is not None
    assert card.findChild(QCheckBox, "expertModeToggle") is not None
    assert card.findChild(QLabel, "parameterStatus") is not None


def test_the_overview_card_carries_a_title_and_no_caption(qtbot, tmp_path) -> None:
    """卡有抬头，没有副题——设计稿从没写过这张卡的副题。

    画布列另一张卡（层堆叠）的副题「从空气到基底 · 点击选中 · 拖动排序」是设计稿 653 行的
    原文。这张卡设计稿只给了 tab 上那三个字，副题只能是编的；编一句出来，读者读到的是一句
    没人审过的说明，而它和抬头一样醒目。
    """
    window = _window(qtbot, tmp_path)
    card = _card(window)

    subtitle = card.findChild(QLabel, f"{OVERVIEW_CARD}Subtitle")
    assert subtitle is not None
    assert subtitle.text() == ""


def test_the_overview_card_shares_the_stacks_scroll_and_sits_below_it(qtbot, tmp_path) -> None:
    """它和层堆叠在同一个滚动区里，排在层堆叠下面。

    这张卡若改当 splitter 的一段，它就常驻占着一段高度——而 900px 高的窗口上那段高度正是
    SLD 剖面在用的，同一张设计稿里画着它。放进层堆叠那个滚动区，它只在读者滚到时才占地方。
    """
    window = _window(qtbot, tmp_path, shown=True)
    scroll = window.canvas_column.findChild(QScrollArea, "structurePanelScroll")
    card = _card(window)
    stack = window.canvas_column.findChild(QFrame, "structureStackCard")

    assert scroll is not None
    assert scroll.isAncestorOf(card)
    inner = scroll.widget()
    assert card.mapTo(inner, card.rect().topLeft()).y() >= stack.mapTo(inner, stack.rect().bottomLeft()).y()


def test_the_overview_card_does_not_bill_its_height_to_the_sld_pane(qtbot, tmp_path) -> None:
    """上半段要的高度是层堆叠那张卡的高度，多出来的三百像素不从 SLD 那边扣。

    「装在滚动区里所以不占地方」只在下限上成立：``QScrollArea`` 的 ``sizeHint`` 照的是整个被滚
    动的内容，而 splitter 按 ``sizeHint`` 分第一轮高度。卡一进来，1400×900 两层的项目上半段
    就从 322 涨到 358，空项目在 1280×760 上从 240 涨到 360——那 36 与 120 像素是同一张设计稿
    里 SLD 剖面的地方，而读者此刻并没有在看参数表。

    所以量的是「上半段没有比层堆叠自己要得更多」：这一段的高度预算归层堆叠，卡靠滚动去够。
    """
    from xrr_fitter.gui.window_layout import STRUCTURE_PANE_FLOOR_PX

    window = _window(qtbot, tmp_path, shown=True)
    canvas = window.canvas_column.findChild(QSplitter, "canvasSplitter")
    scroll = window.canvas_column.findChild(QScrollArea, "structurePanelScroll")
    stack = max(window.structure_panel.sizeHint().height(), STRUCTURE_PANE_FLOOR_PX)

    assert scroll.sizeHint().height() <= stack
    assert canvas.sizes()[canvas.indexOf(scroll)] <= stack
    # 卡确实比这个预算高——不然这条用例什么都没量。
    assert scroll.widget().sizeHint().height() > stack


def test_the_overview_tab_brings_that_card_into_the_canvas_viewport(qtbot, tmp_path) -> None:
    """点「参数总览」，卡滚进画布列的视口——右栏一动不动。

    这三个 tab 是画布列的目录：它先前把右栏滚到「参数化」那一段，因为表当时在那儿。表搬到
    本列之后还去滚右栏，读者点了目录会看见另一栏跳一下，而目录指的那张卡仍在视野外。

    只断言「露出来了」，不断言「顶边贴着视口顶」：``ensureWidgetVisible`` 对比视口还高的控件
    是居中而不是顶对齐。
    """
    from xrr_fitter.gui.window_layout import STRUCTURE_PANE_FLOOR_PX

    window = _window(qtbot, tmp_path, layers=8, shown=True)
    canvas = window.canvas_column.findChild(QSplitter, "canvasSplitter")
    scroll = window.canvas_column.findChild(QScrollArea, "structurePanelScroll")
    inspector = window.inspector_column.findChild(QScrollArea, "inspectorScroll")
    card = _card(window)
    scroll.setMinimumHeight(STRUCTURE_PANE_FLOOR_PX)
    sizes = [0] * canvas.count()
    sizes[canvas.indexOf(window.central_stack)] = canvas.height()
    canvas.setSizes(sizes)
    scroll.verticalScrollBar().setValue(0)
    QApplication.processEvents()
    parked = inspector.verticalScrollBar().value()
    assert _visible_height(scroll, card) == 0, "卡一开始就在视口里，这条用例量不到滚动"

    window.structure_panel.editor.canvas_tabs.setCurrentIndex(2)
    QApplication.processEvents()

    assert _visible_height(scroll, card) > 0, "点了「参数总览」，那张卡没有被滚出来"
    assert inspector.verticalScrollBar().value() == parked, "右栏跟着跳了一下"


def test_the_overview_card_leaves_the_canvas_together_with_the_layer_stack(qtbot) -> None:
    """结果态（帧①）画布只有反射率与残差两张图，参数总览跟着层堆叠一起收起。

    这一步的参数表装的是上一轮交给求解器的初值，而右栏此刻正在报结果值。两张表同时在屏幕上，
    读者要先判断哪张是结果——而其中一张还标着「参数总览」。
    """
    window = _fitted_window(qtbot)

    assert window.pipeline_nav.current_step_index() == 4
    assert _card(window).isVisibleTo(window.canvas_column) is False
