"""The pipeline stepper as a band of the navigation rail.

The stepper used to be a QDockWidget of its own, which gave the design's single
264px rail a second title bar and a drag handle, and let the map of the pipeline
be pulled out of the column whose stages it describes.  It is now a plain widget
between the dataset list and the rail's summary, the way the mockup orders
``nav-sec 分析管线`` → ``.pipe`` → ``.ds-summary``.

Where it sits in the rail is a property of the rail and is asserted in
``test_workspace_columns.py``; this file is about the stepper itself.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDockWidget, QLabel, QSplitter, QWidget

from xrr_fitter.gui import theme
from xrr_fitter.gui.navigation.steps import MARKER_CELL_PX, RAIL_MIN_H, STEP_PAD_V_PX, build_step

PIPELINE_STEP_NAMES = ["数据", "结构", "参数", "拟合", "结果", "导出"]

# 一步的高度上限。设计稿的 ``.pstep`` 是 44：7 上下内边距 + 30 的正文（13px 标题 + 11.5px
# 说明，两行行盒都紧贴字号）。Qt 的 QLabel 高度是行距再加 2，同样两行量出来 19 + 17 = 36，
# 所以同一份字排成 50。放宽的是 Qt 行盒的差，不是字号——字号照设计稿钉在
# ``navigation/steps.py`` 的 ``STEP_TITLE_FONT_PX`` / ``STEP_DESC_FONT_PX``。
PSTEP_MAX_PX = 50

# 抬头也在这条带子的账上：``.nav-sec{padding:var(--md) var(--md) var(--sm)}`` 加上设计稿
# 给它的 ``margin-top:6px``，一行 11px 标题在 Qt 里量到 16px —— 6 + 12 + 16 + 8。抬头归
# stepper 自己（见 ``test_the_stepper_names_itself_above_its_first_step``），所以它落在
# ``nav.sizeHint()`` 里，不在左栏的账上。
HEADING_BLOCK_PX = 6 + 12 + 16 + 8


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def test_the_stepper_is_a_plain_band_of_the_rail(qtbot) -> None:
    """Not a dock: the rail is one column, so it carries one title bar at most."""
    window = _window(qtbot)

    assert isinstance(window.pipeline_nav, QDockWidget) is False
    assert window.nav_column.isAncestorOf(window.pipeline_nav) is True


def test_the_stepper_names_the_six_canonical_stages(qtbot) -> None:
    """The design's 六段管线, ending on the step that is always still to come.

    导出 never turns ✓ -- nothing on the project records that a file was written,
    and the mockup draws the step pending in all three of its frames, including
    the one showing a converged 可信 result.  That is the honest reading: the step
    says what remains, and exporting again is always available, so a tick would
    claim a finality the workflow does not have.

    The opening step is 数据 rather than 导入 because it covers the dataset list
    and its masks as well as the import itself, which is also the word the panel
    directly above it uses.
    """
    window = _window(qtbot)

    actual = [label.text() for label in window.pipeline_nav.step_labels]
    assert actual == PIPELINE_STEP_NAMES


def test_no_shell_command_can_dismiss_the_stepper(qtbot) -> None:
    """The one panel that says where you are cannot be the one you lose.

    As a dock it had a drag handle and no close button; the handle went with the
    docks, so what is left to check is the command that rearranges the shell: a
    layout reset must not drop it.

    Guided mode is the one exception, and it is not a dismissal.  Frame ② draws no
    ``appbody`` at all -- ``cmdbar`` → ``wizhead`` → ``wizbody`` → ``statusbar``,
    with the frame's own lead asking for 「隐藏停靠面板与高级批量选项」 -- so the rail
    steps aside for a surface that answers the same question itself, in the
    four-cell ``wizhead`` and the status bar's 第 N 步.  Leaving 六段管线 up beside
    it would put two competing numberings of one project on one screen.  What has
    to hold is that the swap is reversible.
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)

    window.reset_layout()
    assert window.pipeline_nav.isVisibleTo(window) is True

    window.set_guidance_visible(True)
    assert window.pipeline_nav.isVisibleTo(window) is False
    assert window.guidance.step_header_titles()

    window.set_guidance_visible(False)
    assert window.pipeline_nav.isVisibleTo(window) is True


def test_a_step_costs_no_more_than_the_designs_pstep(qtbot) -> None:
    """``.pstep`` is 7px of padding around a 30px body, and six of them fit 264px.

    The band was built with a separate 12px connector row between every pair of
    steps and 2px of layout spacing on top, which spent 356px on a map the design
    draws in 280 -- roughly a hundred pixels taken from the two panels above it in
    the same 264px column.
    """
    window = _window(qtbot)
    nav = window.pipeline_nav

    for row in nav.step_rows():
        assert row.sizeHint().height() <= PSTEP_MAX_PX, row.objectName()
    budget = HEADING_BLOCK_PX + len(PIPELINE_STEP_NAMES) * PSTEP_MAX_PX + 2 * theme.SPACE_SM
    assert nav.sizeHint().height() <= budget


def test_the_connector_hangs_inside_its_step_not_between_two(qtbot) -> None:
    """``.pstep .line`` lives in the step's own dot column, so it costs no height.

    ``width:2px;flex:1`` makes the line fill whatever the row's text leaves beside
    the dot; it is not a row of its own.  Drawn between the steps instead, each
    connector added its full height to the band, and ``:last-child`` hiding the
    last one was the only reason the sum was not one row taller still.
    """
    window = _window(qtbot)
    nav = window.pipeline_nav
    rows = nav.step_rows()

    for title in nav.step_titles()[:-1]:
        rail = nav.findChild(QWidget, f"pipelineRail_{title}")
        row = next(item for item in rows if item.objectName() == f"pipelineStep_{title}")
        assert row.isAncestorOf(rail), title

    window.show()
    qtbot.waitExposed(window)
    for above, below in zip(rows[:-1], rows[1:], strict=True):
        assert below.y() == above.y() + above.height(), f"gap under {above.objectName()}"


def test_every_step_is_the_same_height_whatever_its_caption_says(qtbot) -> None:
    """六行等距。``.pstep`` 的高度是 7px 内边距加两行字，和字写的是什么无关。

    实测不是：写着中文的那几行 50px，写着 ``—`` 的那几行 48px。QLabel 的高度提示会跟着
    文字所属的字体走——一行中日韩字要向后备字体借更高的行盒，一枚 em dash 用的是基准
    字体的——于是同一条管线里「未开始」那行比「—」那行高 2px，六行的间距忽宽忽窄，
    而设计稿的六个圆点是等距落下来的一列。

    钉高度而不是钉字号：字号照设计稿写在 ``STEP_DESC_FONT_PX``，变的只是行盒。
    """
    window = _window(qtbot)
    nav = window.pipeline_nav

    heights = {row.objectName(): row.sizeHint().height() for row in nav.step_rows()}
    assert len(set(heights.values())) == 1, heights


def test_the_dot_stays_at_the_top_of_its_row_in_the_last_step_too(qtbot) -> None:
    """``.pstep{align-items:flex-start}``：圆点顶边和标题顶边齐平，六行都是。

    实测最后一行的圆点低 7px。连接线只挂在前五行里（``.pstep:last-child .line`` 不画），
    而线正是那一列里唯一会伸展的东西——线一撤，QVBoxLayout 里只剩下一枚定尺圆点，剩余
    高度于是均分到它上下两侧，圆点被居中。屏幕上那一列六个点，最后一个偏下。
    """
    window = _window(qtbot)
    nav = window.pipeline_nav
    window.show()
    qtbot.waitExposed(window)

    offsets = {}
    for title, row in zip(nav.step_titles(), nav.step_rows(), strict=True):
        marker = nav.findChild(QLabel, f"pipelineDot_{title}")
        label = nav.findChild(QLabel, f"pipelineLabel_{title}")
        offsets[title] = (
            marker.mapTo(row, marker.rect().topLeft()).y(),
            label.mapTo(row, label.rect().topLeft()).y(),
        )
    assert {dot for dot, _ in offsets.values()} == {STEP_PAD_V_PX}, offsets
    assert all(dot <= text for dot, text in offsets.values()), offsets


def test_the_map_does_not_outweigh_the_panels_it_maps(qtbot) -> None:
    """In 264px the six-step map is the smaller half of the rail, as drawn.

    Frame ① stacks 数据集 → 分析管线 → ds-summary in one column, and the panels the
    map points at are what the user actually works in.  At the documented 1280×760
    the old band took 356px of 675 and left the two panels above 302 between them,
    which cut the dataset list down to a row or two and clipped the structure
    editor entirely.
    """
    window = _window(qtbot)
    window.resize(1280, 760)
    window.show()
    qtbot.waitExposed(window)
    splitter = window.nav_column.findChild(QSplitter, "leftSplitter")

    assert splitter is not None
    assert window.pipeline_nav.height() < splitter.height()


def test_the_stepper_names_itself_above_its_first_step(qtbot) -> None:
    """``nav-sec 分析管线`` 是这条管线的抬头，设计稿三帧都画着它。

    抬头归 stepper 自己而不是归左栏：六段一起滚，抬头钉在栏上就会留在原处，说明的那几段却
    已经滚走了。
    """
    window = _window(qtbot)
    heading = window.pipeline_nav.findChild(QLabel, "pipelineNavHeader")

    assert heading is not None
    assert heading.text() == "分析管线"

    window.show()
    qtbot.waitExposed(window)
    first = window.pipeline_nav.step_rows()[0]
    assert heading.mapTo(window, heading.rect().bottomLeft()).y() <= first.mapTo(window, first.rect().topLeft()).y()


def test_disconnected_step_keeps_the_rail_floor_with_compact_text(qtbot) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    rows = []
    for connected in (True, False):
        parts = build_step(parent, "数据" if connected else "导出", "—", connected=connected)
        # Model the compact text boxes that expose the hosted macOS layout gap.
        parts.label.setFixedHeight(10)
        parts.description.setFixedHeight(10)
        rows.append(parts.row)
    minimum = MARKER_CELL_PX + RAIL_MIN_H + 2 * STEP_PAD_V_PX
    assert [row.sizeHint().height() for row in rows] == [minimum, minimum]
