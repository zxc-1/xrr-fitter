"""Each inspector section holds its panel behind exactly one caption.

Under the docks this file argued the opposite way round: a dock titled 参数 that
wrapped its panel in a card titled 参数 made the reader pass two identical
headers to reach one table, so the cards were removed and the dock title was the
section's only label.

The fixed-column shell has no dock title bars, so the caption came back as
``theme.titled_card`` -- and it is now the only label the section has.  What the
old duplication argument becomes is a ceiling of one: the card captions the
section, and nothing between the card and the panel captions it a second time.

The height contract inverts too.  Each dock had to fit its own content because
each scrolled separately; the inspector scrolls as one body, so overflowing it
is how three sections share one column.  What cannot overflow is the width: the
column is a budget with horizontal scrolling off, so a section wider than it is
clipped with no way to reach the rest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QFrame, QLabel, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui.window_layout import INSPECTOR_SECTIONS, RIGHT_COLUMN_WIDTH

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path: Path, *, expert: bool = False):
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "curve.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
        SI,
    )
    project = api.set_structure(project, "curve", structure)
    if expert:
        project = api.set_expert_mode(project, True)
    return api.select_active_dataset(project, "curve")


def _window(qtbot, tmp_path: Path, *, expert: bool = False, size=(1280, 760)):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.theme import build_stylesheet

    window = MainWindow(ProjectDocument(_project(tmp_path, expert=expert)))
    qtbot.addWidget(window)
    # The theme is what gives headers and cards their padding, and the fixture
    # does not run ``apply_theme``; without it these measure a control the app
    # never ships.
    window.setStyleSheet(build_stylesheet(window.palette()))
    window.resize(*size)
    window.show()
    # The inspector is the guided surface's second column, which it drops.
    window.set_guidance_visible(False)
    qtbot.wait(1)
    return window


def _card(window, name: str) -> QFrame:
    card = window.inspector_column.findChild(QFrame, name)
    assert card is not None, name
    return card


def test_each_section_card_holds_its_panel_directly(qtbot, tmp_path) -> None:
    """One caption per section: the panel hangs off the card, not off a wrapper.

    Parentage rather than ancestry, because ancestry would also accept a second
    captioned card in between -- which is the arrangement this file was written
    against in the first place.
    """
    window = _window(qtbot, tmp_path)

    for name, _title, _subtitle, attribute in INSPECTOR_SECTIONS:
        card = _card(window, name)
        panel = getattr(window, attribute)
        assert panel.parent() is card, f"{name} holds {attribute} via {type(panel.parent()).__name__}"


def test_no_panel_repeats_the_caption_of_the_section_it_sits_in(qtbot, tmp_path) -> None:
    """A card captioned 候选解 must not hold a second widget reading 候选解."""
    window = _window(qtbot, tmp_path)

    for name, title, _subtitle, attribute in INSPECTOR_SECTIONS:
        panel = getattr(window, attribute)
        repeats = [
            child.objectName()
            for child in panel.findChildren(QWidget)
            if getattr(child, "text", None) is not None and child.text() == title
        ]
        assert repeats == [], f"{name} repeats {title!r} in {repeats}"


def test_no_two_sections_in_the_column_carry_the_same_caption(qtbot, tmp_path) -> None:
    """两张同名卡同时可见时，抬头就不再是路标。

    上面那条只核一张卡有没有重复自己的抬头，跨节重名它放得过去：判读卡叫「拟合判定」，
    而运行控件那一节也叫「拟合判定」，读者在同一栏里遇到两个同名抬头，得读完卡里的
    内容才知道刚点开的是哪一个。设计稿里「拟合判定」只指判读，运行那一节叫「控制」。
    """
    window = _window(qtbot, tmp_path)
    titles = [title for _name, title, _subtitle, _attribute in INSPECTOR_SECTIONS]

    assert len(set(titles)) == len(titles), f"sections share a caption: {titles}"

    for name, title, _subtitle, attribute in INSPECTOR_SECTIONS:
        others = set(titles) - {title}
        panel = getattr(window, attribute)
        repeats = [
            (child.objectName(), child.text())
            for child in panel.findChildren(QWidget)
            if getattr(child, "text", None) is not None and child.text() in others
        ]
        assert repeats == [], f"{name} repeats another section's caption in {repeats}"


def test_the_candidate_section_counts_its_rows_in_its_own_caption(qtbot, tmp_path) -> None:
    """设计稿的抬头是「候选解 3 个」：计数落在候选解那一段自己的抬头上。

    这个计数此前挂在外层卡的副标题上，那时外层卡就叫「候选解」。设计稿帧① 右栏是三段
    ——拟合判定 / 参数 · 结果值 / 候选解——三段都在这张卡里，所以外层那句抬头只能说这
    三段合起来是什么，计数跟着它数的那一段走。这个 fixture 没有拟合结果，核的是空态那句。
    """
    window = _window(qtbot, tmp_path)

    card = _card(window, "inspectorResults")
    subtitle = card.findChild(QLabel, "resultCandidatesCardSubtitle")

    assert subtitle is not None
    assert subtitle.text() == window.result_panel.candidate_summary()
    assert subtitle.text() == "尚未拟合"


def test_the_inspector_fits_its_column_budget(qtbot, tmp_path) -> None:
    """Every section has to fit 340px, including the ones expert mode widens.

    Horizontal scrolling is off, so a section asking for more than the column is
    clipped outright; and widening the column would take the width the design
    gives the canvas rather than making the section readable.
    """
    window = _window(qtbot, tmp_path, expert=True)

    floor = window.inspector_column.minimumSizeHint().width()
    assert floor <= RIGHT_COLUMN_WIDTH, f"inspector asks {floor}px of a {RIGHT_COLUMN_WIDTH}px column"


@pytest.mark.parametrize("expert", [False, True])
def test_the_parameter_grid_needs_no_sideways_scrolling_in_its_column(qtbot, tmp_path, expert: bool) -> None:
    """The grid's columns have to add up to the viewport it is actually given.

    ``minimumSizeHint`` is not enough to catch this: a QTableWidget's hint does not
    sum its column widths, so the section above reports it fits 340px while the
    table hides three of its columns behind a horizontal scrollbar.  Only the
    widths measured inside the real column say whether a bound is reachable.
    """
    window = _window(qtbot, tmp_path, expert=expert)
    table = window.parameters_panel.parameter_table
    viewport = table.viewport().width()

    assert viewport > 0, "table was never laid out, so this measures nothing"
    widths = [table.columnWidth(column) for column in range(table.columnCount())]
    assert sum(widths) <= viewport, f"columns {widths} sum to {sum(widths)} in a {viewport}px viewport"
    assert table.horizontalScrollBar().isVisible() is False


def _bottom_inset(card: QFrame) -> int:
    """这张卡下缘让给边框的那几个像素——扁平段是 1（只有下边线），封边的末段是 0。

    量 ``contentsRect`` 而不是 ``frameWidth()``：后者对「四边都有框」和「只有下边线」都报
    1，分不开这两种形状。
    """
    return card.rect().height() - card.contentsRect().height()


def _visible_cards(window) -> list[QFrame]:
    # ``isHidden()`` 而不是 ``isVisible()``：窗口在这套 fixture 里已经 show 过，但按步骤
    # 收起来的段落要的是「它自己被收了吗」，而不是「它此刻画在屏幕上吗」。
    return [card for card in window.inspector_cards.values() if not card.isHidden()]


def test_every_captioned_section_in_the_column_is_ruled_off_not_boxed(qtbot, tmp_path) -> None:
    """设计稿右栏是一栏被横线分节的正文（``.insp-sec``），不是一列各带圆角边框的卡片。

    五六段各画一圈框时读者看到的是一列卡片，而卡片之间的关系要靠边距去猜；横线把它们
    读成同一栏的连续小节。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window(qtbot, tmp_path)
    apply_step_scope(window, 2)
    qtbot.wait(10)

    cards = _visible_cards(window)

    assert len(cards) > 1, "这一步只露了一段，横线与封边的区别测不出来"
    for card in cards:
        assert card.property("inspectorSection") is True, card.objectName()
        assert not card.property("sectionCard"), card.objectName()
        assert card.contentsRect().width() == card.rect().width(), f"{card.objectName()} 侧边还有框"


def test_only_the_last_section_showing_gives_up_its_rule(qtbot, tmp_path) -> None:
    """末段之下没有下一段，那道线就成了整栏的封边。

    哪一段是末段随步骤变（``apply_step_scope`` 按步骤切可见性），所以这条界不能在构造期
    钉死：参数那一步末段是 控制，结构那一步是 结构诊断。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window(qtbot, tmp_path)

    for step in (1, 2):
        apply_step_scope(window, step)
        qtbot.wait(10)
        cards = _visible_cards(window)

        insets = [_bottom_inset(card) for card in cards]
        names = [card.objectName() for card in cards]
        assert insets[-1] == 0, f"step {step}: 末段 {names[-1]} 仍在封边"
        assert insets[:-1] == [1] * (len(cards) - 1), f"step {step}: {names} 的横线是 {insets}"


def test_the_result_sections_butt_together_so_a_rule_reads_as_a_border(qtbot, tmp_path) -> None:
    """帧① 右栏那三段之间只有那道横线，没有空隙。

    段与段之间留出边距时，横线离两边都远，读起来是一条装饰性的分隔线；紧贴着上下两段
    时它才是「这一节到此为止」。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window(qtbot, tmp_path)
    apply_step_scope(window, 4)
    qtbot.wait(10)
    panel = window.result_panel
    cards = [
        panel.findChild(QFrame, name) for name in ("resultConfidenceCard", "resultValuesCard", "resultCandidatesCard")
    ]
    assert all(card is not None for card in cards)

    for card in cards:
        assert card.property("inspectorSection") is True, card.objectName()
    assert cards[1].y() == cards[0].y() + cards[0].height()
    assert cards[2].y() == cards[1].y() + cards[1].height()
    # 次要内容在帧① 是收起来的，所以候选解那一段就是这一栏的末段。
    assert panel.secondary.isHidden()
    assert [_bottom_inset(card) for card in cards] == [1, 1, 0]
