"""Step-by-step guidance mode.

Expert mode used to be the only working surface: standard mode merely hid some
controls, leaving a newcomer facing the full dock workspace. Guidance is a
separate surface that walks import to result, showing only what the current step
needs. Every gate is answered by existing public API reads, so the mode adds no
api surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFont, QFontInfo
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QWidget,
)

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)

STEP_NAMES = ("importStep", "structureStep", "fitStep", "resultStep")

INTERACTIVE = (QAbstractButton, QComboBox, QSpinBox, QLineEdit)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path: Path, *, structured: bool = False):
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "curve.xy"),
        api.InstrumentSpec(),
    )
    if structured:
        structure = api.StructureSpec(
            AIR,
            (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
            SI,
        )
        project = api.set_structure(project, "curve", structure)
    return api.select_active_dataset(project, "curve")


def _window(qtbot, project=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    document = ProjectDocument() if project is None else ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.resize(1280, 760)
    window.show()
    qtbot.wait(1)
    return window


def _oxidised_project(tmp_path: Path):
    """A project whose surface oxide came from a suggestion the user accepted.

    Frame ② draws the 💡 tip above the stack list, and the tip's gate is a
    recorded ``oxide_decisions`` entry -- a hand-written SiO2 cap is the user's
    own layer and correctly shows no tip -- so the layout contracts have to reach
    it through the real suggestion path.
    """
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "curve.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(AIR, (api.LayerSpec("film", SI, 480.0, roughness_a=4.0),), SI)
    project = api.set_structure(project, "curve", structure)
    suggestion = next(item for item in api.suggest_oxide_layers(structure) if item.location == "surface")
    project = api.accept_oxide_suggestion(project, "curve", suggestion)
    return api.select_active_dataset(project, "curve")


def test_the_workspace_is_the_default_surface_for_a_new_project(qtbot) -> None:
    """A newcomer meets the full workspace; guidance is toggled from the View menu."""
    window = _window(qtbot)

    assert window.guidance.isVisibleTo(window) is False
    assert window.inspector_column.isVisibleTo(window) is True


def test_guidance_declares_the_four_workflow_steps(qtbot) -> None:
    window = _window(qtbot)

    assert tuple(window.guidance.step_names()) == STEP_NAMES


def test_each_step_stays_within_the_control_budget(qtbot, tmp_path) -> None:
    """The whole point is a small surface, so the budget is asserted."""
    window = _window(qtbot, _project(tmp_path, structured=True))

    for name in STEP_NAMES:
        window.guidance.show_step(name)
        qtbot.wait(1)
        visible = [
            widget for kind in INTERACTIVE for widget in window.guidance.findChildren(kind) if widget.isVisible()
        ]
        assert len(visible) <= 10, f"{name} shows {len(visible)} controls"


def test_a_step_is_blocked_until_its_precondition_holds(qtbot) -> None:
    """An empty project cannot advance past import."""
    window = _window(qtbot)

    assert window.guidance.step_is_available("importStep") is True
    assert window.guidance.step_is_available("structureStep") is False
    assert window.guidance.step_is_available("fitStep") is False


def test_importing_data_unblocks_the_structure_step(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path))

    assert window.guidance.step_is_available("structureStep") is True
    assert window.guidance.step_is_available("fitStep") is False


def test_defining_a_structure_unblocks_the_fit_step(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path, structured=True))

    assert window.guidance.step_is_available("fitStep") is True


def test_switching_to_expert_reveals_the_inspector_column(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path, structured=True))

    window.set_guidance_visible(False)

    assert window.guidance.isVisibleTo(window) is False
    assert window.inspector_column.isVisibleTo(window) is True


def test_switching_back_and_forth_keeps_project_state(qtbot, tmp_path) -> None:
    """The surfaces are two views of one project, not two projects."""
    window = _window(qtbot, _project(tmp_path, structured=True))
    before = window.document.project

    window.set_guidance_visible(False)
    window.set_guidance_visible(True)

    assert window.document.project is before
    assert window.guidance.step_is_available("fitStep") is True


def test_guidance_toggle_lives_in_the_view_menu(qtbot) -> None:
    window = _window(qtbot)
    action = window.chrome_actions["guidanceModeAction"]

    assert action.isCheckable() is True
    assert action.isChecked() is False

    action.setChecked(True)

    assert window.guidance.isVisibleTo(window) is True


def test_the_guided_header_labels_all_four_steps(qtbot) -> None:
    """Frame ② names each step in the header; a bare position dot names nothing.

    A newcomer — and a screen reader — has to read which step is which, so the
    four-step header carries each step's short title rather than only a marker.
    The ordinal ("第 N 步") is not baked into the title here; it is derived from
    position, so the header title stays the short design form.
    """
    window = _window(qtbot)

    assert window.guidance.step_header_titles() == (
        "导入数据",
        "确认样品结构",
        "开始拟合",
        "查看结果",
    )


def test_the_guided_header_draws_one_cell_per_declared_step(qtbot, monkeypatch) -> None:
    """The header reads STEP_SPECS, never a second hardcoded step count.

    A duplicated count silently drifts the moment a step is added or removed —
    the same latent split the confidence glyphs once carried. Building the panel
    against a patched spec and counting the header cells proves the header
    follows the one source instead of a copy.
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.guidance import panel as guidance_panel

    extended = guidance_panel.STEP_SPECS + (
        guidance_panel.StepSpec(
            "reviewStep",
            "复核",
            "候选解与置信度",
            "复核候选解。",
            "复核",
            "noop",
        ),
    )
    monkeypatch.setattr(guidance_panel, "STEP_SPECS", extended)
    panel = guidance_panel.GuidancePanel(ProjectDocument(), {})
    qtbot.addWidget(panel)

    assert len(panel.step_header_titles()) == len(extended)
    assert panel.step_header_titles()[-1] == "复核"


def _guided(qtbot, project=None, step: str = "structureStep"):
    """A window sitting on one guided step, past the 200ms page fade."""
    window = _window(qtbot, project)
    window.set_guidance_visible(True)
    window.guidance.show_step(step)
    qtbot.wait(250)
    return window


def test_the_guided_step_draws_no_icon(qtbot) -> None:
    """Frame ②'s wizbody is eyebrow → h3 → sub → hint → stack → ctarow → help.

    There is no icon in it, and the one we drew was worse than redundant:
    ``plot_icon`` paints a 16-unit design and scales the *painter* to the
    requested size without touching the device pixel ratio, so at 48px the pen
    width and the 1.2-unit marker radii scale 3× too and ``data_curve`` renders
    as a black smear above the step title.
    """
    window = _window(qtbot)

    for name in STEP_NAMES:
        assert window.guidance.findChild(QLabel, f"{name}Icon") is None, name


def test_the_guided_step_reads_left_aligned_not_centred(qtbot, tmp_path) -> None:
    """``.wizbody`` is block flow: prose starts at one left edge and stays there.

    Centring every line turns four sentences into four independent objects with
    no shared edge to read down, which is why the rendered step looked like a
    splash screen rather than the design's page of prose.
    """
    window = _guided(qtbot, _oxidised_project(tmp_path))
    panel = window.guidance

    for suffix in ("Eyebrow", "Title", "Body"):
        label = panel.findChild(QLabel, f"structureStep{suffix}")
        assert label is not None, suffix
        assert not (label.alignment() & Qt.AlignmentFlag.AlignHCenter), suffix
    hint = panel.findChild(QLabel, "structureStepExpertHint")
    assert not (hint.alignment() & Qt.AlignmentFlag.AlignHCenter)


def test_the_guided_eyebrow_is_the_designs_small_accent_caption(qtbot, tmp_path) -> None:
    """设计稿 ``.wizbody .eyebrow2`` 是 12px / 700 / 字距 .6px 的强调色小字（HTML 498 行）。

    「第 2 步 · 共 4 步」这一句的全部作用是在标题之上垫一行位置感。它得比正文小一档、粗一
    档、字距松一点，读者才认得出这是页眉；跟正文同字号同字重时，它读起来就是正文的第一句。
    此前只有颜色到位：``QLabel[stepState="current"]`` 那条规则只给 ``color``，而
    ``font-weight: 700`` 挂在眉标没带的 ``ladderStep`` 变体上，于是字号、字重、字距三项全缺。

    三项都量字体对象而不是样式表：字距和字号本来就只能落字体对象（Qt 的样式表没有
    ``letter-spacing``），字重跟着一起落，这一档才是一处实现。
    """
    from xrr_fitter.gui import theme

    window = _guided(qtbot, _oxidised_project(tmp_path))
    eyebrow = window.guidance.findChild(QLabel, "structureStepEyebrow")

    assert eyebrow is not None
    # 颜色仍走 ``stepState`` 那条 QSS 通道，随明暗切换；字体三项落字体对象。
    assert eyebrow.property("stepState") == "current"
    # ``QFontInfo`` 而不是 ``font().pointSize()``：像素给的字号读 ``pointSize()`` 会拿到 -1。
    assert QFontInfo(eyebrow.font()).pixelSize() == theme.STEP_EYEBROW_FONT_PX
    assert eyebrow.font().weight() == QFont.Weight.Bold
    assert eyebrow.font().letterSpacingType() == QFont.SpacingType.AbsoluteSpacing
    # 容差是 1/64 px：Qt 把字距存成 1/64 像素的定点数，0.6px 落到最近的 38/64=0.59375。
    assert eyebrow.font().letterSpacing() == pytest.approx(theme.STEP_EYEBROW_TRACKING_PX, abs=1 / 64)


def test_the_guided_body_keeps_the_designs_reading_width(qtbot, tmp_path) -> None:
    """``.wizbody{max-width:720px;margin:0 auto}``: prose does not stretch to the shell.

    With both side columns gone the guided step owns the full window, and a
    1264px-wide line of 13.5px text is unreadable. The card is capped and
    centred instead, which is what the ``margin:0 auto`` says.
    """
    from xrr_fitter.gui.guidance.panel import BODY_MAX_WIDTH

    window = _guided(qtbot, _oxidised_project(tmp_path))
    window.resize(1600, 900)
    qtbot.wait(10)
    card = window.guidance.findChild(QFrame, "structureStepCard")

    assert card is not None
    assert card.width() <= BODY_MAX_WIDTH
    assert card.mapTo(window.guidance, QPoint(0, 0)).x() > 0


def test_the_guided_shell_adds_no_padding_of_its_own(qtbot, tmp_path) -> None:
    """留白只由 ``.wizbody`` 那一档给（HTML 497），壳子不再叠一层。

    设计稿的引导屏是命令条 → ``.wizhead`` → 一个零内边距的背景层 → ``.wizbody``（HTML
    557-565）：横向 24px、纵向 24px 全出自最里面那一层。壳子自己也垫一圈的话两层相加，
    读书栏的实际留白变成 40px 而进度条那一条被无故内缩，24 这一档就量不出来了。
    """
    window = _guided(qtbot, _oxidised_project(tmp_path))
    panel = window.guidance
    header = panel.findChild(QWidget, "stepHeader")
    margins = panel.layout().contentsMargins()

    assert header is not None
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (0, 0, 0, 0)
    assert panel.layout().spacing() == 0
    assert header.mapTo(panel, QPoint(0, 0)) == QPoint(0, 0)


def test_the_guided_body_pads_itself_the_way_wizbody_does(qtbot, tmp_path) -> None:
    """``.wizbody{padding:24px}``（HTML 497）：这一页的留白比面板里其他段落宽一档。

    它是这一屏唯一的内容，四周没有别的东西替它定边界——同一档 16px 内边距在一张挤在栏里
    的卡上是对的，放到独占整屏的读书栏上就贴边了。设计稿在这里裸写 24 而不引 ``--lg``，
    这一处也就不该跟着间距档走。
    """
    from xrr_fitter.gui.guidance.panel import BODY_PAD_PX

    window = _guided(qtbot, _oxidised_project(tmp_path))
    card = window.guidance.findChild(QFrame, "structureStepCard")

    assert card is not None
    margins = card.layout().contentsMargins()
    assert BODY_PAD_PX == 24
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (
        BODY_PAD_PX,
        BODY_PAD_PX,
        BODY_PAD_PX,
        BODY_PAD_PX,
    )


def test_the_guided_step_orders_its_blocks_the_way_frame_two_does(qtbot, tmp_path) -> None:
    """eyebrow → title → body → 💡 hint → stack → ctarow → help, top to bottom.

    The tip explains the layer the software added on its own, so it has to be read
    before the list it is talking about; the closing pointer to expert mode is the
    last thing on the page, after the action it is an alternative to.
    """
    window = _guided(qtbot, _oxidised_project(tmp_path))
    panel = window.guidance
    card = panel.findChild(QFrame, "structureStepCard")
    blocks = [
        panel.findChild(QLabel, "structureStepEyebrow"),
        panel.findChild(QLabel, "structureStepTitle"),
        panel.findChild(QLabel, "structureStepBody"),
        panel.findChild(QLabel, "structureStepOxideTip"),
        panel.findChild(QWidget, "structureStepStack"),
        panel.findChild(QPushButton, "structureStepAction"),
        panel.findChild(QLabel, "structureStepExpertHint"),
    ]
    assert None not in blocks
    assert panel.findChild(QLabel, "structureStepOxideTip").isVisibleTo(card) is True

    tops = [block.mapTo(card, QPoint(0, 0)).y() for block in blocks]
    assert tops == sorted(tops), tops


def test_the_guided_actions_sit_in_one_row_at_the_bodys_left_edge(qtbot, tmp_path) -> None:
    """Frame ②'s ``.ctarow`` is a left-aligned flex row, not a centred pair.

    The step also loses the justified ``← 上一步 / 下一步 →`` footer bar underneath
    the card: the frame draws no such bar, and on the last step it produced two
    ways out (「下一步 →」 relabelled 「进入主界面 →」 beside the primary
    「切换到专家模式」) that did the same thing.
    """
    window = _guided(qtbot, _oxidised_project(tmp_path))
    panel = window.guidance
    card = panel.findChild(QFrame, "structureStepCard")
    body = panel.findChild(QLabel, "structureStepBody")
    action = panel.findChild(QPushButton, "structureStepAction")
    manual = panel.findChild(QPushButton, "structureStepManual")

    assert action.text() == "看起来没问题，开始拟合 →"
    assert action.mapTo(card, QPoint(0, 0)).x() == body.mapTo(card, QPoint(0, 0)).x()
    assert manual.mapTo(card, QPoint(0, 0)).x() > action.mapTo(card, QPoint(0, 0)).x()
    assert action.mapTo(card, QPoint(0, 0)).y() == manual.mapTo(card, QPoint(0, 0)).y()

    for name in STEP_NAMES:
        assert panel.findChild(QPushButton, f"{name}Back") is None, name
    # Step 2's primary *is* the advance, and step 4's is the exit, so neither
    # carries a second forward button beside it.
    assert panel.findChild(QPushButton, "structureStepNext") is None
    assert panel.findChild(QPushButton, "resultStepNext") is None
    assert panel.findChild(QPushButton, "importStepNext") is not None


def test_the_step_header_is_how_a_guided_user_goes_back(qtbot, tmp_path) -> None:
    """With the footer gone, the header cells are the navigation -- as drawn.

    Frame ② gives the guided surface exactly one step affordance: the four-cell
    ``wizhead``. Making the cells activate the step they name is what keeps
    「导入数据」 reachable from step 2 now that the navigation rail is hidden, and
    it costs no button the design does not draw. A cell whose precondition does
    not hold stays inert, so the header cannot walk ahead of the project.
    """
    window = _guided(qtbot, _project(tmp_path, structured=True), step="fitStep")
    panel = window.guidance

    panel.findChild(QAbstractButton, "guidanceStep_importStepCell").click()

    assert panel.current_step() == "importStep"

    panel.findChild(QAbstractButton, "guidanceStep_resultStepCell").click()

    assert panel.step_is_available("resultStep") is False
    assert panel.current_step() == "importStep"


def test_the_header_cells_get_the_width_their_titles_need(qtbot) -> None:
    """Frame ② writes each step's name beside its dot, on one line.

    ``QPushButton`` computes its size hints from its own text and icon, and these
    cells carry neither -- the marker and the two text lines are child widgets in
    a layout the button never consults.  So every cell asked for a button-sized
    nothing while the connecting tracks, which carry the stretch, took the row:
    「导入数据」 came out as a two-character column wrapped down the side of its
    dot.  The cell has to ask for what it contains.
    """
    window = _guided(qtbot)
    header = window.guidance.findChild(QWidget, "stepHeader")

    for name in STEP_NAMES:
        cell = header.findChild(QAbstractButton, f"guidanceStep_{name}Cell")
        title = header.findChild(QLabel, f"guidanceStep_{name}Title")
        assert cell.width() >= title.sizeHint().width(), (name, cell.width(), title.sizeHint())
        assert title.width() >= title.sizeHint().width(), (name, title.width(), title.sizeHint())


def _header_cells(window):
    header = window.guidance.findChild(QWidget, "stepHeader")
    cells = tuple(header.findChild(QAbstractButton, f"guidanceStep_{name}Cell") for name in STEP_NAMES)
    return header, cells


def test_the_header_splits_the_row_into_four_equal_cells(qtbot) -> None:
    """``.wizhead .wh{flex:1}``（HTML 489）：四格等分一整行，行里没有别的东西。

    实现原先把四格按各自的 ``sizeHint`` 摆，在格与格之间插一条会伸缩的横轨、前后再各夹一个
    ``addStretch``——画出来是「圆点—长横线—圆点」的连线图，四格宽度取决于标题多长（「确认样品
    结构」那格最宽），两端还留出成片空白。设计稿是四张等宽的分格卡片贴满整行：等宽本身就是
    进度的刻度，一眼看得出走到几分之几。
    """
    window = _guided(qtbot)
    header, cells = _header_cells(window)

    widths = [cell.width() for cell in cells]
    # 整数除不尽时 Qt 把余数分给前几格，所以允许 1px 的差。
    assert max(widths) - min(widths) <= 1, widths
    assert cells[0].x() == 0, f"第一格左边留了 {cells[0].x()}px 空白"
    right = cells[-1].x() + cells[-1].width()
    assert right == header.width(), f"最后一格右边到 {right}，抬头宽 {header.width()}"


def test_the_header_separates_its_cells_with_hairlines_not_a_connecting_track(qtbot) -> None:
    """``.wh{border-right:1px solid var(--border)}`` 且 ``.wh:last-child`` 不画。

    连线图的横轨是躺着的 2px 长线，占的是格与格之间的一段宽度；设计稿的分隔是立着的 1px
    细线，贴在每格右边缘、贯通整格高，最后一格不画。三条竖线把一行切成四格，横轨则是把四个
    圆点连成一条链——前者读作「四个并列的阶段」，后者读作「一条流水线」。
    """
    window = _guided(qtbot)
    header, cells = _header_cells(window)

    rails = [w for w in header.findChildren(QWidget) if w.property("pipelineRail")]
    assert len(rails) == len(STEP_NAMES) - 1, len(rails)
    for rail in rails:
        assert rail.width() == 1, (rail.objectName(), rail.width())
        assert rail.height() >= cells[0].height(), (rail.objectName(), rail.height())


def test_the_current_header_cell_is_lit_the_way_the_design_lights_it(qtbot) -> None:
    """``.wh.current{background:#fff}`` 加 ``.n{width:30px;height:30px}``（HTML 490-495）。

    分格之后「走到哪一步」由整格底色报，不再由那一个圆点独自承担：设计稿给 current 格换成
    ``--panel`` 白底，衬在 ``--panel-2`` 的抬头底上，四格里哪一格是活的隔着屏幕就分得出。所以
    状态得写到格子本身，光写在圆点和标题上样式表够不着这块底。圆点同时从 20px 长到 30px——
    分格卡片里它是那一格的主记号，20px 在 14px 粗标题旁边显得像个附注。

    色板与样式表得自己装：这只 fixture 不走 ``apply_theme``，而离屏默认板把 Window / Base /
    AlternateBase 全给成白的——机架色与画布色相等时，「current 那格比 pending 亮」这句话不管
    实现怎么写都是假的。
    """
    import re

    from xrr_fitter.gui.theme import build_stylesheet, light_palette

    window = _guided(qtbot, step="structureStep")
    palette = light_palette()
    window.setPalette(palette)
    sheet = build_stylesheet(palette)
    window.setStyleSheet(sheet)
    qtbot.wait(1)
    header, cells = _header_cells(window)

    assert [cell.property("stepState") for cell in cells] == ["done", "current", "pending", "pending"]
    marker = header.findChild(QLabel, "guidanceStep_structureStepMarker")
    assert (marker.width(), marker.height()) == (30, 30), (marker.width(), marker.height())
    # 半径必须是直径的一半，否则 30px 的方块只是圆角方块。尺寸在 ``panel.py``、半径在
    # ``theme.py``，所以这里跨着两处量一次。左栏管线的圆点是 20px，两张地图共用 ``stepDot``
    # 的颜色规则但不能共用半径，量的是抬头这一档专有的那条。
    radius = re.search(
        r'QLabel\[stepDot="true"\]\[wizardStep="true"\]\s*\{[^}]*border-radius:\s*(\d+)px',
        sheet,
    )
    assert radius is not None, "样式表里找不到向导抬头圆点的圆角规则"
    assert int(radius.group(1)) == marker.width() // 2, (radius.group(1), marker.width())

    # 底色：current 那格比同排的 pending 亮一档。取每格右下角内侧的像素——左边是圆点、上边是
    # 标题，右下那块只有底色。
    image = header.grab().toImage()
    ratio = max(1, round(image.devicePixelRatio()))
    lit, dim = (
        image.pixelColor((cell.x() + cell.width() - 6) * ratio, (cell.y() + cell.height() - 6) * ratio).lightness()
        for cell in (cells[1], cells[2])
    )
    assert lit > dim, f"current 格底色亮度 {lit} 不比 pending 格 {dim} 高"


def test_the_header_is_a_band_above_the_page_not_half_the_panel(qtbot) -> None:
    """``.wizhead`` 之后紧接 ``.wizbody``（HTML 487、497）：抬头是条窄带，多余高度归页面。

    分格之后抬头有了自己的底色，它有多高就一眼看得见：``QVBoxLayout`` 里抬头与页面都没有拉伸
    因子，两项都能长，于是 Qt 把整panel的余量平摊——抬头量到 329px，是 ``sizeHint`` 的七倍多，
    四格圆点浮在半屏机架色的正中间，向导内容被顶到屏幕下半。抬头必须贴着自己的内容收边，页面
    拿走剩下的高度。

    上限用圆点直径推：设计稿 ``.wh{padding:13px 16px}`` 配 30px 的记号是 56px 一档，写成
    ``STEP_DOT_PX * 2`` 既盖得住这一档，也不至于把字体度量的一两个像素当回归。
    """
    from xrr_fitter.gui import theme

    window = _guided(qtbot)
    header, cells = _header_cells(window)
    stack = window.guidance.findChild(QStackedWidget, "guidanceStack")

    assert header.height() == header.sizeHint().height(), (header.height(), header.sizeHint().height())
    assert header.height() <= theme.STEP_DOT_PX * 2, (header.height(), theme.STEP_DOT_PX)
    assert stack.height() > header.height(), (stack.height(), header.height())
    # 收边不能收成把格子挤瘦：那格白底仍要铺满抬头这一条。
    assert cells[1].height() == header.height(), (cells[1].height(), header.height())


def _oxide_project(tmp_path: Path):
    """帧② 画的那份结构：空气 / 自动加入的 SiO₂ / a-Si 主体层 / c-Si 基底。

    氧化层走的是真实那条边——``suggest_oxide_layers`` 提议、``accept_oxide_suggestion``
    接受——所以它既进了层堆叠，也在数据集上留下了一条 accepted 决策。手工插一层同样材料
    的帽层不会留下决策，而这条决策正是帧② 那句「已自动加入表面氧化层」的依据。
    """
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    project = api.select_active_dataset(project, "curve")
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("a-Si · 非晶硅薄膜", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),),
        api.MaterialSpec("c-Si · 晶体硅基底", "Si", 2.329),
    )
    project = api.set_structure(project, "curve", structure)
    suggestion = next(
        item for item in api.suggest_oxide_layers(project.datasets[0].structure) if item.location == "surface"
    )
    return api.accept_oxide_suggestion(project, "curve", suggestion)


def test_the_guided_stack_names_layers_the_way_frame_two_writes_them(qtbot, tmp_path) -> None:
    """帧② 的四行写「空气 / SiO₂ 表面氧化层 / a-Si 非晶硅薄膜 / c-Si 晶体硅基底」。

    名字取层自己的名字，不取材料名：材料名是 ``SiO2``、``a-Si`` 这样的化学标签，而设计稿
    这张列表要的是「这是什么层」。自动加入的氧化层没有人给它起过名，服务层生成的
    ``"SiO2 native oxide"`` 是英文占位串，显示层把它翻回设计稿的写法。
    """
    window = _window(qtbot, _oxide_project(tmp_path))

    names = [name for name, _role, _measure in window.guidance.structure_rows()]

    assert names == ["空气", "SiO₂ 表面氧化层", "a-Si 非晶硅薄膜", "c-Si 晶体硅基底"]


def test_the_guided_stack_gives_each_row_the_subline_frame_two_gives_it(qtbot, tmp_path) -> None:
    """副行说的是角色，不重复名字：氧化层那行写「自动建议 · 可移除」。"""
    window = _window(qtbot, _oxide_project(tmp_path))

    roles = [role for _name, role, _measure in window.guidance.structure_rows()]

    assert roles == [
        "入射介质 · 半无限，无需参数",
        "自动建议 · 可移除",
        "主体层 · 你要测量的对象",
        "衬底 · 半无限",
    ]


def test_an_auto_added_oxide_shows_the_hint_frame_two_prints(qtbot, tmp_path) -> None:
    """帧② 在层列表上方印一条 💡 提示，解释这一层是谁加的、什么时候该删。"""
    window = _window(qtbot, _oxide_project(tmp_path))
    window.guidance.show_step("structureStep")

    tip = window.guidance.findChild(QLabel, "structureStepOxideTip")

    assert tip is not None
    assert tip.isVisibleTo(window.guidance) is True
