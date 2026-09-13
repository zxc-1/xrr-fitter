"""M4–M7 structural visual contracts.

These tests verify that widgets expose the correct object names, accessible
names, and structural composition expected by the design spec's layout.
They complement the existing behavioral tests with reskin-contract assertions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor, QFontInfo, QFontMetrics, QImage
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabBar,
    QToolButton,
    QWidget,
)
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

import xrr_fitter.api as api
from xrr_fitter.gui.plots.panel import ANALYSIS_MIN_H
from xrr_fitter.gui.plots.sld_state import MIN_SLD_PLOT_H
from xrr_fitter.gui.window_layout import MINIMUM_SHELL_HEIGHT, MINIMUM_SHELL_WIDTH

# -- helpers --

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path, count: int = 64) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + i * 0.02:.6f} {1000.0 / (i + 1):.12g}" for i in range(count)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path, *, structured: bool = False):
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
    window.show()
    qtbot.wait(1)
    return window


# -- M4: Structure editor structural contract --


def _card_texts(root: QWidget, name: str) -> tuple[str, str]:
    """Assert the titled-card contract and return the card's two captions.

    Every card in the mockup is headed the same way -- ``.ph > .t + .sub`` -- so
    the frame, property and header-label checks belong here once, leaving each
    test to state only the copy it expects.

    The frame itself comes in two shapes: the bordered ``.plotcard`` the canvas
    draws and the flat, rule-separated ``.insp-sec`` the inspector column draws.
    Both are ``titled_card`` output and both carry this same header, so the
    contract asserted here is "one of the two shapes"; which shape a given card
    takes is measured by the edge-pixel tests in ``test_theme_semantics``.
    """
    card = root.findChild(QFrame, name)
    assert card is not None, f"no titled card frame named {name!r}"
    assert card.property("sectionCard") is True or card.property("inspectorSection") is True, (
        f"{name} is neither a bordered card nor a flat inspector section"
    )
    title = card.findChild(QLabel, f"{name}Title")
    subtitle = card.findChild(QLabel, f"{name}Subtitle")
    assert title is not None and subtitle is not None, f"{name} lacks header labels"
    assert title.property("sectionHeader") is True
    assert subtitle.property("mutedText") is True
    return title.text(), subtitle.text()


def test_card_header_puts_the_title_and_its_hint_on_one_row(qtbot) -> None:
    """Mockup ``.insp-sec .h`` is ``display:flex; justify-content:space-between``.

    The header stacked the hint below the title and let it wrap, so every card
    spent two text rows on chrome before its content began -- and the wrap made
    that cost grow with the copy.  Two such cards in the results dock pushed it
    77px past the documented minimum window and opened a scrollbar on content
    that had fit before.
    """
    from xrr_fitter.gui import theme

    card, _body = theme.titled_card(None, "probeCard", "拟合判定", "形状 + 文字双编码 · 依据见下方")
    qtbot.addWidget(card)
    card.show()
    qtbot.wait(1)
    title = card.findChild(QLabel, "probeCardTitle")
    subtitle = card.findChild(QLabel, "probeCardSubtitle")

    assert title.y() == subtitle.y(), "the hint sits on a row of its own"
    assert subtitle.x() > title.x(), "the hint trails the title on the same row"


def test_card_header_hint_reads_smaller_than_its_title(qtbot) -> None:
    """Mockup ``.plotcard .ph`` sizes the hint at 11px against the title's 12.5px.

    The app spelled the difference in colour alone, so the hint carried the
    title's weight in width while saying less: the stack card's hint asked 232px
    of a 264px column on its own, which is what put the card 45px past the
    column it lives in.  Reading order is title, then hint; the type has to say
    so before a reader gets to the colour.

    Both sizes are measured in pixels, which is the unit the mockup states them
    in and the unit ``titled_card`` sets them in.  ``QFont.pointSize()`` reports
    ``-1`` for a font sized in pixels, so comparing point sizes here would read
    a px-sized title as *smaller* than any pt-sized hint and pass this contract
    on a header that had lost its hierarchy.
    """
    from xrr_fitter.gui import theme

    card, _body = theme.titled_card(None, "probeCard", "拟合判定", "形状 + 文字双编码 · 依据见下方")
    qtbot.addWidget(card)
    card.show()
    qtbot.wait(1)
    title = card.findChild(QLabel, "probeCardTitle")
    subtitle = card.findChild(QLabel, "probeCardSubtitle")

    assert QFontInfo(subtitle.font()).pixelSize() < QFontInfo(title.font()).pixelSize(), (
        "the hint is typeset as loud as the title"
    )


def test_card_header_hint_yields_its_width_to_the_column(qtbot) -> None:
    """The hint is the least load-bearing text in the card, so it shrinks first.

    A no-wrap ``QLabel`` reports its whole string as a minimum width, so the
    header's floor was title + hint + spacing whatever column the card sat in --
    and the hint is the longest of the two.  It has to elide against the width
    it is given while ``text()`` keeps the full copy for tooltips and tests.
    """
    from xrr_fitter.gui import theme

    card, _body = theme.titled_card(None, "probeCard", "拟合判定", "形状 + 文字双编码 · 依据见下方")
    qtbot.addWidget(card)
    subtitle = card.findChild(QLabel, "probeCardSubtitle")

    assert subtitle.minimumSizeHint().width() < subtitle.fontMetrics().horizontalAdvance(subtitle.text()), (
        "the hint demands its full string back from the column"
    )


def test_structure_commands_share_one_line_with_the_canvas_tabs(qtbot, tmp_path) -> None:
    """设计稿 ``.canvas-top``（HTML 148 行）是一条 ``display:flex`` 的行：tab 与命令同高。

    此前是六个命令排成会换行的两行，外加一条随建议进出的氧化层条——画布列一共三段高度，
    行头就占掉一段。这里核的是「命令和 tab 在同一条水平带里」：谁再往这一行塞按钮塞到换
    行，或者把命令挪到 tab 下面另起一行，这条都会挂。
    """
    from PySide6.QtWidgets import QTabBar

    from xrr_fitter.gui import theme

    window = _window(qtbot, _project(tmp_path, structured=True))
    # 必须自己套一遍发货的 stylesheet：fixture 不走 ``apply_theme``，而这一行的高度全由
    # QSS 决定（tab 的内边距、按钮的 ``min-height``）。不套的话量到的是一个没有内边距的
    # 裸 QTabBar 和一个照样带内边距的按钮，两边根本不是 app 里那两个控件。
    window.setStyleSheet(theme.build_stylesheet(window.palette()))
    qtbot.wait(1)
    row = window.structure_panel.findChild(QWidget, "structureCanvasTop")
    assert row is not None
    tabs = row.findChild(QTabBar, "structureCanvasTabs")
    buttons = [button for button in row.findChildren(QPushButton) if button.isVisibleTo(row)]
    assert [button.text() for button in buttons] == ["＋ 添加层", "建议氧化层", "周期结构…"]

    band = tabs.geometry()
    for button in buttons:
        assert band.top() <= button.geometry().center().y() <= band.bottom(), (
            f"{button.text()} 掉出了 tab 那条水平带，行头换行了"
        )
        # 设计稿这三个是 ``.btn.sm``（26px），比它们旁边的 tab 矮一档。高过 tab 的话，
        # 这条带的高度就改由按钮说了算，行头凭空粗一圈——而画布列一共只有三段高度。
        assert button.height() <= band.height(), f"{button.text()} 比它旁边的 tab 还高"
    assert row.height() <= band.height() + 2 * theme.SPACE_SM + 2, "行头比它排的那一行还高"


def test_structure_stack_sits_in_a_titled_card(qtbot, tmp_path) -> None:
    """Mockup frame ③ draws the stack inside a titled card, not bare in the dock.

    The subtitle is checked whole: it has to name the reading direction and both
    interactions, and a substring match would pass on a truncated caption.
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    stack = window.structure_panel.findChild(QWidget, "structureStack")
    assert stack is not None
    assert stack.parentWidget().objectName() == "structureStackCard"
    assert _card_texts(window.structure_panel, "structureStackCard") == (
        "层堆叠",
        "从空气到基底 · 点击选中 · 拖动排序",
    )


def test_sld_pane_sits_in_a_titled_card_with_legend(qtbot, tmp_path) -> None:
    """Mockup frame ③ card 2 pairs the SLD profile with a three-entry legend.

    The legend is what makes the band and the handles readable: without it the
    filled ribbon and the accent dots are unexplained decoration.  Swatch labels
    carry a pixmap and no text, so dropping the empty ones leaves the captions.
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    pane = window.findChild(QWidget, "sldPane")
    assert pane is not None
    assert _card_texts(pane, "sldCard") == ("SLD 深度剖面", "实时预览 · 界面手柄可拖拽")
    legend = pane.findChild(QWidget, "sldCardLegend")
    assert legend is not None, "sldCard has no legend row"
    captions = tuple(text for label in legend.findChildren(QLabel) if (text := label.text()))
    assert captions == ("当前 SLD 剖面", "16–84% 不确定带", "可拖拽界面")


def test_sld_pane_reserves_plot_height_beneath_its_card_chrome(qtbot, tmp_path) -> None:
    """The pane's floor has to leave the axes readable, not just fit the header.

    Title, subtitle, legend and controls are four stacked rows, so a hand-counted
    floor collapses the axes to a sliver the moment a row is added -- which is
    exactly what matplotlib's ``constrained_layout`` warning reports.  The floor
    must therefore be derived: whatever the chrome needs, plus a plot worth
    drawing.
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    pane = window.findChild(QWidget, "sldPane")
    assert pane is not None
    canvas = pane.findChild(QWidget, "diagnosticCanvas:sld")
    assert canvas is not None, "sldPane has no SLD canvas"
    chrome = pane.minimumSizeHint().height() - canvas.minimumSizeHint().height()
    assert pane.minimumHeight() >= chrome + MIN_SLD_PLOT_H
    assert canvas.height() >= MIN_SLD_PLOT_H


# -- M3: Pipeline navigator structural contract --


def _step_parts(nav: QWidget, title: str) -> tuple[QLabel, QLabel, QLabel]:
    """Return one nav step's dot, title and description labels.

    Mockup ``.pstep`` is three parts -- ``.rail > .dot``, ``.body > .t`` and
    ``.body > .d`` -- so locating them belongs here once, leaving each test to
    state only the glyph, state or copy it expects.
    """
    dot = nav.findChild(QLabel, f"pipelineDot_{title}")
    label = nav.findChild(QLabel, f"pipelineLabel_{title}")
    description = nav.findChild(QLabel, f"pipelineDescription_{title}")
    assert dot is not None, f"step {title!r} has no dot"
    assert label is not None, f"step {title!r} has no title label"
    assert description is not None, f"step {title!r} has no description line"
    return dot, label, description


def test_pipeline_nav_steps_carry_theme_step_state(qtbot, tmp_path) -> None:
    """Mockup ``.pstep`` colours the dot and title from the step's state.

    The nav swapped glyph text only, so ``theme.set_step_state`` -- and the three
    ``stepState`` QSS rules written for it -- never reached the navigator: the
    steps rendered in one colour and the state read from punctuation alone.
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    nav = window.pipeline_nav
    current = nav.current_step_index()
    titles = nav.step_titles()

    expected = ["done"] * current + ["current"] + ["pending"] * (len(titles) - current - 1)
    states = []
    for title in titles:
        dot, label, _description = _step_parts(nav, title)
        assert dot.property("stepState") == label.property("stepState"), f"{title}: dot/title disagree"
        states.append(dot.property("stepState"))
    assert states == expected


def test_pipeline_nav_done_steps_show_a_check_and_others_their_ordinal(qtbot, tmp_path) -> None:
    """Mockup dots read ``✓`` once done and their 1-based number before that.

    Double-encoding is the point: the guided header already does this, so a nav
    that shows ``●/◦/○`` states progress in punctuation the guided surface does
    not use, and neither glyph survives being described out loud.
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    nav = window.pipeline_nav
    current = nav.current_step_index()
    titles = nav.step_titles()

    expected = ["✓"] * current + [str(index + 1) for index in range(current, len(titles))]
    glyphs = [_step_parts(nav, title)[0].text() for title in titles]
    assert glyphs == expected


def test_pipeline_nav_steps_describe_themselves_below_their_title(qtbot) -> None:
    """Mockup ``.pstep .body .d`` states each step's job as visible faint prose.

    The descriptions existed only as tooltips, which a newcomer scanning the
    column never sees and a screen reader reaches only on hover.

    设计稿给的是 ``color:var(--ink-faint)``——比 ``mutedText`` 再浅一档，跟同一栏里
    「数据集」「分析管线」两句抬头同色。这一条原先写的是 ``mutedText``，是当时还没做分级时
    「非正文即 muted」的近似；分级做出来之后，这里就该照设计稿要最浅那档。
    """
    window = _window(qtbot)
    nav = window.pipeline_nav

    for title in nav.step_titles():
        _dot, _label, description = _step_parts(nav, title)
        assert description.text().strip(), f"{title}: empty description"
        assert description.property("faintText") is True, f"{title}: description not faint"


def test_pipeline_nav_draws_a_rail_between_steps_but_not_after_the_last(qtbot) -> None:
    """Mockup ``.pstep .line`` chains the dots and ``:last-child`` drops it.

    Without the connector the dots read as an unordered list of links rather than
    as one ordered run, which is the whole claim the left column makes.
    """
    window = _window(qtbot)
    nav = window.pipeline_nav
    titles = nav.step_titles()

    rails = [nav.findChild(QWidget, f"pipelineRail_{title}") for title in titles]
    assert all(rail is not None for rail in rails[:-1]), "steps before the last need a rail"
    assert rails[-1] is None, "the last step must not trail a rail"
    for rail in rails[:-1]:
        assert rail.frameShape() == QFrame.Shape.VLine, "the rail must draw a visible line"


# -- M5: Nine-stage progress structural contract --


def test_progress_stage_labels_cover_nine_stages() -> None:
    """STAGE_LABELS must cover all 9 canonical stages."""
    from xrr_fitter.gui.fitting.progress import STAGE_LABELS, STAGE_WEIGHTS

    assert len(STAGE_WEIGHTS) == 9
    for stage_key, _ in STAGE_WEIGHTS:
        assert stage_key in STAGE_LABELS, f"missing label for stage: {stage_key}"


def test_progress_weights_sum_to_unity() -> None:
    """Weights must sum to 1.0 for monotonic 0-1000 bar."""
    from xrr_fitter.gui.fitting.progress import STAGE_WEIGHTS

    total = sum(w for _, w in STAGE_WEIGHTS)
    assert abs(total - 1.0) < 1e-9


def test_progress_resolution_is_1000() -> None:
    """Progress bar resolution matches design spec."""
    from xrr_fitter.gui.fitting.progress import PROGRESS_RESOLUTION

    assert PROGRESS_RESOLUTION == 1000


def _ladder_rows(view: QWidget) -> tuple[tuple[QLabel, QLabel, QLabel], ...]:
    """Return every ladder row's marker, name and metric labels, in stage order.

    Mockup ``.stg`` is three parts -- ``.ic``, ``.nm`` and ``.mt`` -- so locating
    them belongs here once, leaving each test to state only the glyph, state or
    copy it expects.
    """
    from xrr_fitter.gui.fitting.progress import STAGE_WEIGHTS

    rows = []
    for key, _weight in STAGE_WEIGHTS:
        marker = view.findChild(QLabel, f"fitStageMarker_{key}")
        name = view.findChild(QLabel, f"fitStageName_{key}")
        metric = view.findChild(QLabel, f"fitStageMetric_{key}")
        assert marker is not None, f"stage {key!r} has no marker"
        assert name is not None, f"stage {key!r} has no name label"
        assert metric is not None, f"stage {key!r} has no metric label"
        rows.append((marker, name, metric))
    return tuple(rows)


def test_progress_block_sits_in_a_titled_card(qtbot) -> None:
    """Mockup frame ④ heads the progress block with 总进度 and names its scale.

    The view stacked five bare rows at zero margins, so the bar floated against
    the dock wall with nothing stating the scale: a reader had no way to tell
    ``620`` meant 620 of 1000 rather than a percent that had overshot.
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    assert _card_texts(view, "fitProgressCard") == ("总进度", "共九阶段 · 单调递增 0–1000")
    card = view.findChild(QFrame, "fitProgressCard")
    assert card.findChild(QProgressBar, "fitProgressBar") is not None, "bar sits outside the card"


def test_progress_ladder_lists_every_stage_with_its_localized_name(qtbot) -> None:
    """Mockup ``.stages`` shows all nine stages at once, not only the live one.

    One stage line says where the run is but never how far it reaches, so the
    eight stages still to come stayed invisible and the 0-1000 scale had no
    visible ladder standing behind it.
    """
    from xrr_fitter.gui.fitting.progress import STAGE_LABELS, STAGE_WEIGHTS, ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    names = [name.text() for _marker, name, _metric in _ladder_rows(view)]
    assert names == [STAGE_LABELS[key] for key, _weight in STAGE_WEIGHTS]


def test_progress_ladder_marks_done_current_and_pending_stages(qtbot) -> None:
    """Mockup ``.stg.done/.current/.pending`` is the navigator's vocabulary reused.

    ``theme.set_step_state`` and its three QSS rules already encode this state as
    colour plus weight, and the pipeline nav now speaks it; a ladder inventing its
    own marks would report one run's progress in two alphabets on one screen.
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "D", 3, 6, 0.5, "refining"))

    rows = _ladder_rows(view)
    states = [marker.property("stepState") for marker, _name, _metric in rows]
    assert states == ["done"] * 3 + ["current"] + ["pending"] * 5
    assert [name.property("stepState") for _marker, name, _metric in rows] == states
    glyphs = [marker.text() for marker, _name, _metric in rows]
    assert glyphs == ["✓"] * 3 + [str(index + 1) for index in range(3, 9)]


def test_progress_ladder_metric_reports_only_the_live_stage_count(qtbot) -> None:
    """Mockup ``.stg .mt`` carries each row's own measure, right of the name.

    Only the running stage has a measure in a progress frame -- a frame carries no
    history for the stages already finished -- so the column states the live count
    on the current row and a plain status word elsewhere, rather than inventing
    numbers the fit never reported.
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "D", 3, 6, 0.5, "refining"))

    metrics = [metric.text() for _marker, _name, metric in _ladder_rows(view)]
    assert metrics == ["已完成"] * 3 + ["3/6 (50%)"] + ["待运行"] * 5


def test_stage_names_expose_the_key_that_progress_events_and_logs_carry() -> None:
    """Mockup frame ④ writes 初筛候选 A, so the name carries the stage's own key.

    ``A``..``E`` are not ornament: they are the literal stage keys in
    ``FitProgress.stage``, in resume checkpoints and in worker log lines. A
    translated name alone leaves a user comparing this window against a log or a
    checkpoint with nothing to match on. The other four keys already read as
    words -- ``basin-recovery``, ``bootstrap``, ``profile``, ``finalizing`` --
    and pasting them beside their Chinese names would drop English into the row
    for no gain, which is why the mockup letters five stages and not nine.
    """
    from xrr_fitter.gui.fitting.progress import STAGE_LABELS, stage_text

    for key in ("A", "B", "C", "D", "E"):
        assert f" {key}" in stage_text(key), f"stage {key} hides its key: {stage_text(key)!r}"
    for key in ("basin-recovery", "bootstrap", "profile", "finalizing"):
        assert key not in STAGE_LABELS[key], f"stage {key} pastes its key into the name"


def test_stage_names_state_the_method_and_the_verdict_the_stage_produces() -> None:
    """Frame ④ names 差分进化 and Bootstrap, and ends on 汇总与判定.

    Two stages are distinguished by the method they run rather than by what they
    touch: ``B`` searches globally with ``differential_evolution`` and
    ``bootstrap`` resamples, and naming the method is how a reader knows why one
    takes a third of the run. ``finalizing`` is the stage that calls
    ``classify_result_with_evidence`` (``analysis/report.py``), so 「汇总」alone
    under-reports it -- the confidence verdict is decided there, not merely
    collected.
    """
    from xrr_fitter.gui.fitting.progress import STAGE_LABELS

    assert "差分进化" in STAGE_LABELS["B"]
    assert "Bootstrap" in STAGE_LABELS["bootstrap"]
    assert STAGE_LABELS["finalizing"] == "汇总与判定"


# -- M6: Confidence verdict and uncertainty evidence structural contract --


def _confidence_panel(qtbot, *evidence: str, confidence: str = "可用但相关"):
    """A results panel showing one downgraded result carrying the given reasons.

    ``final_fit_result`` ships ``classification_evidence=()``, so a test about the
    reason line has to inject the codes itself; assembling that here leaves each
    test stating only the one claim it makes about the rendered verdict.

    ``confidence`` 传的是判定的字面值，由 ``type(result.confidence)`` 反查成枚举：GUI
    这一侧从不导入 ``ConfidenceClass``，键全部走字符串，测试跟着同一条路才能证明那条路
    是通的。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    result = final_fit_result()
    result = replace(
        result,
        confidence=type(result.confidence)(confidence),
        classification_evidence=evidence,
    )
    value = replace(project(dataset_project(result=result)), base_directory="/private/tmp")
    panel = ResultsPanel(ProjectDocument(api.select_active_dataset(value, "curve")))
    qtbot.addWidget(panel)
    return panel


def test_confidence_verdict_sits_in_a_titled_card(qtbot) -> None:
    """Mockup ``.insp-sec`` heads 拟合判定 before showing the badge it explains.

    The verdict shipped as a bare two-label row at zero margins, so 可用但相关
    sat flush against whatever widget preceded it with nothing saying it was a
    verdict at all: a reader met one coloured word and had to infer what it judged.
    """
    panel = _confidence_panel(qtbot, "strong_correlation")

    # 设计稿的抬头是「拟合判定 aSi_ML_25C」：副标题是这份判定属于哪条曲线。多数据集
    # 的项目里这一栏投影的是当前那条，不写出来的话「可信」读不出是对谁可信。
    assert _card_texts(panel, "resultConfidenceCard") == ("拟合判定", "curve")
    card = panel.findChild(QFrame, "resultConfidenceCard")
    assert card.findChild(QLabel, "confidenceBadge") is not None, "verdict sits outside the card"
    assert card.findChild(QLabel, "confidenceMarker") is not None, "glyph sits outside the card"


def test_the_verdict_is_boxed_with_its_reason_inside(qtbot) -> None:
    """设计稿 ``.confbadge``（HTML 421）：字形、判定、理由三者同在一只圆角框里。

    实测这三个控件是平铺的——字形和判定并成一行，理由作为下一行直接加在卡的布局上。
    平铺的代价不在好看：理由和它下面那排判读徽章、再下面那三行键值指标一样都只是卡里的
    一行，读者没有线索知道「参数唯一，置信区间收敛」是在解释上面那两个字，而不是又一条
    并列的证据。框把这层归属画出来。
    """
    panel = _confidence_panel(qtbot, "strong_correlation")

    box = panel.findChild(QFrame, "confidenceVerdictBox")
    assert box is not None, "判定没有外框"
    assert box.property("verdictBox") is True, "外框没挂上样式表认的 property"
    for name in ("confidenceMarker", "confidenceBadge", "confidenceReason"):
        assert box.findChild(QLabel, name) is not None, f"{name} 落在框外"


def test_the_verdict_box_tint_follows_the_verdict(qtbot) -> None:
    """设计稿给四种判定各一种底色（``.confbadge.trusted`` 等），不只染那两个字。

    四种判定只靠文字与字形的颜色区分时，「可信」和「不可信」在缩略图里是同一个形状；
    整块底色一染，判定在扫视中就先于阅读到达。染色由样式表按 ``statusKind`` 出，所以
    这里锁的是框上那把键。
    """
    for verdict, kind in (("可信", "ok"), ("可用但相关", "info"), ("多解", "warn"), ("不可信", "error")):
        panel = _confidence_panel(qtbot, "strong_correlation", confidence=verdict)

        box = panel.findChild(QFrame, "confidenceVerdictBox")
        assert box.property("statusKind") == kind, verdict


def test_an_unclassified_verdict_keeps_the_box_but_no_state_colour(qtbot) -> None:
    """没有结果时判定读「不可用」：框还在，四种状态色一种都不能借。

    借 ok 会把「还没跑」读成「通过」，借 error 会读成「失败」。设计稿未加后缀的
    ``.confbadge`` 正是这一档：中性边框、透明底。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    panel = ResultsPanel(ProjectDocument())
    qtbot.addWidget(panel)

    box = panel.findChild(QFrame, "confidenceVerdictBox")
    assert box is not None, "空项目里判定框整只不见了"
    assert not box.property("statusKind"), "未分类的判定借了一种状态色"
    assert panel.confidence_text() == "不可用"


def test_the_verdict_box_holds_its_text_off_the_border(qtbot) -> None:
    """设计稿 ``.confbadge`` 的 ``padding:10px var(--md)``。

    内距写成布局边距而不是样式表的 ``padding``：这只框自己带布局，QSS 的 padding 会
    和布局边距叠加一次，量出来是设计稿的两倍。
    """
    from xrr_fitter.gui import theme

    panel = _confidence_panel(qtbot, "strong_correlation")
    margins = panel.findChild(QFrame, "confidenceVerdictBox").layout().contentsMargins()

    assert (margins.left(), margins.right()) == (theme.VERDICT_PAD_H_PX, theme.VERDICT_PAD_H_PX)
    assert (margins.top(), margins.bottom()) == (theme.VERDICT_PAD_V_PX, theme.VERDICT_PAD_V_PX)


def test_the_verdict_glyph_is_typeset_at_the_display_size(qtbot) -> None:
    """设计稿 ``.confbadge .glyph`` 是 22px，比判定那行的 15px 又大一档。

    字形是这块的图标：和判定同号时它读成句首的一个标点，大一档才读成一枚标记。字号在
    Python 里设，因为 22px 折算成 16.5pt，而 Qt 的样式表解析器不接受带小数的 pt——写进
    QSS 会整条规则失效，字形静默退回默认字号。
    """
    from xrr_fitter.gui import theme

    panel = _confidence_panel(qtbot, "strong_correlation")

    assert panel.confidence_marker.font().pointSizeF() == theme.VERDICT_GLYPH_PT
    assert theme.VERDICT_GLYPH_PT > theme.FONT_PT_MD


def test_confidence_glyphs_all_take_the_same_width(qtbot) -> None:
    """五个判定字形必须同宽，否则判定一变、右边那行判定文字就横移。

    标记控件早就带着 ``mono`` property，但那条样式表规则写的是 ``font-family: monospace``，
    而 Qt 只按已安装字体族匹配 ``font-family``——没有平台装了名叫 monospace 的族，所以整条
    规则静默退回比例字体。实测代价：``▲``（多解）比另外四个窄 2px。

    这里量的是字形推进宽度，不是样式表文本：规则「写了」和字体「落了」是两件事，上面那个 bug
    正是两者不一致。
    """
    from xrr_fitter.gui import theme

    panel = _confidence_panel(qtbot, "strong_correlation")
    marker = panel.findChild(QLabel, "confidenceMarker")
    marker.setStyleSheet(theme.build_stylesheet(marker.palette()))
    marker.ensurePolished()

    metrics = QFontMetrics(marker.font())
    glyphs = (*theme.CONFIDENCE_GLYPHS.values(), theme.CONFIDENCE_FALLBACK_GLYPH)
    assert all(metrics.inFont(glyph) for glyph in glyphs), "判定字形在该字体里缺字，会画成豆腐块"
    assert len({metrics.horizontalAdvance(glyph) for glyph in glyphs}) == 1


def test_confidence_card_shows_the_classification_reasons_on_screen(qtbot) -> None:
    """Mockup ``.confbadge .txt .r`` prints the reason next to the verdict.

    The reasons already reached ``_set_confidence`` but landed only on a tooltip
    and the accessible description, so on screen the verdict stood unexplained.
    Hovering is unavailable in a printed screenshot and on a touch pad, which is
    exactly where a downgraded fit most needs to say why it was downgraded.
    """
    from xrr_fitter.gui.results.uncertainty import CLASSIFICATION_LABELS

    panel = _confidence_panel(qtbot, "strong_correlation", "boundary_hit")

    reason = panel.findChild(QLabel, "confidenceReason")
    assert reason is not None, "the verdict carries no visible reason label"
    assert reason.property("mutedText") is True, "the reason must read as secondary to the verdict"
    assert reason.wordWrap() is True, "reasons run long enough to wrap inside a dock"
    assert CLASSIFICATION_LABELS["strong_correlation"] in reason.text()
    assert CLASSIFICATION_LABELS["boundary_hit"] in reason.text()


def test_uncertainty_evidence_sits_in_a_titled_card(qtbot) -> None:
    """Mockup ``.insp-sec`` heads the evidence block the way it heads every other.

    The evidence box shipped at zero margins with no header, so a wall of
    diagnostic lines began mid-dock under whatever happened to precede it, and
    nothing on screen said the lines belonged to the inspected candidate alone
    rather than to the whole fit.
    """
    from xrr_fitter.gui.results.uncertainty import UncertaintyView

    view = _confidence_panel(qtbot, "strong_correlation").uncertainty

    assert isinstance(view, UncertaintyView)
    assert view.accessibleName() == "不确定度诊断"
    assert _card_texts(view, "uncertaintyCard") == ("不确定度证据", "仅列出选中候选解自有的诊断")
    card = view.findChild(QFrame, "uncertaintyCard")
    evidence = card.findChild(QPlainTextEdit, "uncertaintyEvidence")
    assert evidence is not None, "the evidence box sits outside its own card"


def test_uncertainty_height_covers_the_card_header_as_well_as_the_evidence(qtbot) -> None:
    """The card's own header must not be taken out of the evidence lines.

    ``sizeHint`` sizes the view to the evidence it holds, and it counted text
    lines only.  Wrapping that text in a headed card added a title, a subtitle
    and 12px margins inside the same height, so the box lost roughly two of its
    three floor lines and opened a scrollbar on evidence that used to fit.
    """
    from xrr_fitter.gui.results.uncertainty import EVIDENCE_LINE_FLOOR

    view = _confidence_panel(qtbot, "strong_correlation").uncertainty
    title = view.findChild(QLabel, "uncertaintyCardTitle")
    subtitle = view.findChild(QLabel, "uncertaintyCardSubtitle")
    header = title.sizeHint().height() + subtitle.sizeHint().height()
    floor = EVIDENCE_LINE_FLOOR * view.evidence.fontMetrics().lineSpacing()

    assert view.sizeHint().height() >= floor + header, "the header eats the evidence lines"


# -- Canvas: the mockup's canvas-top row and its stacked plot cards --


def _rect_in(root: QWidget, widget: QWidget):
    """`widget`'s rectangle in `root`'s coordinates, for overlap assertions."""
    return widget.rect().translated(widget.mapTo(root, widget.rect().topLeft()))


def test_plot_modebar_shares_one_row_with_the_view_tabs(qtbot) -> None:
    """Mockup ``.canvas-top`` is one row: ``.tabs`` 紧跟 ``.modebar``，留白在行尾。

    模式条曾被抬出布局、钉在图的右上角，那是把控件盖在数据上，还要一个事件过滤器追着角
    跑。后来改用 ``QTabWidget`` 的角落位——但角落位只有「贴右边框」这一种摆法：在 950px
    宽的画布上，那是离它作用的那四个 tab 六百多像素的另一头。设计稿的 ``.spring`` 在
    ``.canvas-top``（HTML 148 行）里是零宽的，所以两者相邻，空白全落在行尾。

    断言落在结果态（帧①）：六帧里只有它的 ``.canvas-top`` 带模式条。帧③ 那一行的尾部是
    三个结构按钮，帧⑤ 尾部空着，帧②⑥ 根本没有这一行。所以要验「模式条与标签页同一行」，
    得在它真被画出来的那一帧上验。
    """
    value = replace(project(dataset_project(result=final_fit_result())), base_directory="/private/tmp")
    window = _window(qtbot, api.select_active_dataset(value, "curve"))
    panel = window.plot_panel
    row = panel.findChild(QWidget, "reflectivityCanvasTop")
    bar = panel.findChild(QTabBar, "reflectivityCanvasTabs")

    assert row is not None and bar is not None
    assert row.isAncestorOf(panel.toolbar), "模式条不在 .canvas-top 那一行里"
    # 同一行：两者的垂直区间重叠，而不是一个在另一个上下。
    assert _rect_in(row, panel.toolbar).intersected(_rect_in(row, bar)).isEmpty()
    assert _rect_in(row, panel.toolbar).center().y() == _rect_in(row, bar).center().y()
    # 紧跟：模式条与 tab 之间的空隙不超过一格间距，行尾的留白比它大。
    gap = panel.toolbar.geometry().left() - bar.geometry().right()
    assert 0 < gap <= 24, f"tab 与模式条之间空了 {gap}px"
    assert row.width() - panel.toolbar.geometry().right() > gap
    # 页面里不该再有第二条 tab 条与它抢这一行。
    page = panel.reflectivity_tabs.currentWidget()
    assert page is not None
    assert panel.reflectivity_tabs.tabBar().isVisibleTo(panel.reflectivity_tabs) is False


def test_modebar_draws_the_designs_grouping_border(qtbot) -> None:
    """Mockup ``.modebar`` is one bordered pill, not loose glyphs on the tab row.

    四枚控件裸站在四个 tab 旁边，读起来就是一行分不出组的记号；那圈边框说的是「这几枚是
    一组，作用在下面那张图上」。``QWidget`` 只有被告知自己画背景之后才会画样式表里的边
    框，否则那条一直写在主题里的规则什么也没画出来。

    断言落在结果态（帧①）：模式条现在住在 ``.canvas-top`` 那一行里，而那一行随反射率那一
    段一起按流程步收放（``STEP_PLOT_PANES``）。结构那一步整段是收着的，收着的分支不会布
    局，``grab()`` 会拿到一张零尺寸的图——量到的是「没布过局」，不是「没画边框」。设计稿
    也只在帧① 画模式条。

    样式表要照 ``apply_theme`` 在真应用里那样套上：不套的话这条没有背景也没有边框，抓下
    来是一整片不透明的底色，过或不过都不是因为它该不该有边框。
    """
    from xrr_fitter.gui.theme import build_stylesheet

    value = replace(project(dataset_project(result=final_fit_result())), base_directory="/private/tmp")
    window = _window(qtbot, api.select_active_dataset(value, "curve"))
    window.setStyleSheet(build_stylesheet(window.palette()))
    qtbot.wait(1)
    bar = window.plot_panel.toolbar
    assert bar.width() > 0 and bar.height() > 0, "模式条没有被布局，抓下来的是一张空图"
    image = bar.grab().toImage()

    # The border is 1px in the stylesheet's units, and the stylesheet's unit is a
    # logical pixel; a Retina grab hands back a device-pixel image where that one
    # stroke spans ``ratio`` rows, antialiased across them.  Indexing rows 0 and 1
    # on such a grab compares two halves of the same stroke and reads the darker
    # half as "no border".  So the stroke is the whole top logical row, and the
    # fill is the first row past it.  Both are the same black at different
    # opacities, so the alpha alone separates them, and with no border drawn the
    # two would be one wash.
    ratio = max(1, round(image.devicePixelRatio()))
    mid_x = image.width() // 2
    edge = max(QColor.fromRgba(image.pixel(mid_x, y)).alpha() for y in range(ratio))
    fill = QColor.fromRgba(image.pixel(mid_x, ratio)).alpha()
    assert edge > fill, f"the mode bar draws no grouping border (edge {edge}, fill {fill})"


def test_no_two_modebar_controls_wear_the_same_glyph(qtbot, tmp_path) -> None:
    """Every control on the bar is told apart by its glyph alone.

    The buttons carry no text, so two controls painted the same picture are one
    control as far as a reader can tell -- and the overlay toggle wore the reset
    view's house, which sits four cells away on the same bar.

    设计稿 ``.modebar`` 只摆四枚字形，另外五条退到条上的右键菜单；菜单项自带文字，但它们
    的图标和条上那四枚同处一份图标集，所以这里连菜单一起验。范围那一条例外：它同时是条上
    的 ``▭`` 和菜单里的「范围」，是一条命令两个入口（见
    ``test_canvas_top.py::test_the_range_glyph_and_its_menu_entry_are_one_command``），
    共用一枚字形恰恰是对的，所以按命令去重而不是按控件。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    toolbar = window.plot_panel.toolbar
    buttons = toolbar.findChildren(QToolButton)

    assert sorted(button.objectName() for button in buttons) == sorted(
        ("plotNavPan", "plotNavZoom", "plotNavHome", "plotModeRange")
    ), "the mode bar lost its controls"

    def _glyph(icon) -> bytes:
        image = icon.pixmap(16, 16).toImage().convertToFormat(QImage.Format.Format_ARGB32)
        return image.constBits().tobytes()

    controls: dict[str, bytes] = {button.objectName(): _glyph(button.icon()) for button in buttons}
    for action in toolbar.tool_actions():
        if action.isSeparator():
            continue
        key = action.objectName().removeprefix("plotToolAction:")
        # 「范围」那一条已经以 ``plotModeRange`` 的身份记过一次。
        controls.setdefault("plotModeRange" if key == "range" else key, _glyph(action.icon()))

    seen: dict[bytes, str] = {}
    for name, glyph in controls.items():
        assert glyph not in seen, f"{name} wears {seen[glyph]}'s glyph"
        seen[glyph] = name


PLOT_CARD_COPY = {
    "log": ("反射率 R vs 入射角 2θ", "curve · 归一化"),
    "raw": ("原始强度 vs 存储角度", "curve · 拟合点与排除点"),
    "qz4": ("qz⁴·R vs 散射矢量 qz", "curve · 抑制菲涅尔衰减 · 非拟合数据"),
    # 残差卡的第二句写这次拟合的判读，不冠数据集名，也就没有 ``curve · ``；还没拟合时
    # 那句判读无从写起，于是这张卡先说「尚未拟合」。见
    # ``test_the_residual_card_reads_out_its_own_verdict``。
    "residual": ("加权残差 vs 散射矢量 qz", "尚未拟合"),
}


def test_reflectivity_views_sit_in_titled_plot_cards(qtbot, tmp_path) -> None:
    """Mockup ``.canvas-body`` stacks ``.plotcard``s, each headed ``.t`` + ``.sub``.

    A bare canvas behind a two-word tab label does not say what is plotted
    against what: the axes carry units but not the quantity, and the tab name is
    the only caption on screen.  The card header is where the mockup names the
    pair and the dataset it belongs to, so each view carries the same header the
    SLD companion already does.
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    panel = window.plot_panel

    observed = {key: _card_texts(panel, f"plotCard:{key}") for key in PLOT_CARD_COPY}
    assert observed == PLOT_CARD_COPY


def test_canvas_stacks_the_selected_view_over_a_permanent_residual_card(qtbot) -> None:
    """Mockup ``.canvas-body`` holds two plotcards at once: a view and its residual.

    Whether a fit is good is not a question the curve alone answers -- two curves
    laid over four decades read as agreement long after the residual has gone
    structured -- so the design draws the residual beneath the curve instead of
    only behind a tab.  Behind a tab alone, checking it costs a click and the
    view being checked, which in practice means it goes unchecked.

    帧① 的 ``.tabs`` 同时列着「加权残差」，所以这张卡有两个家，而不是从 tab 条上搬走了：
    默认这一段（对数反射率）选中时它钉在 tab 组下面，选中「加权残差」时它搬去当那一页的
    正文。断言落在默认那一态上——两个家之间的搬家由
    ``test_canvas_top.py::test_the_residual_tab_moves_the_one_card_instead_of_drawing_it_twice`` 钉。

    断言落在结果态（帧①）而不是结构态：设计稿帧③ 的画布是「层堆叠 + SLD 深度剖面」，一张
    残差都没有——那一步的残差画的是上一轮的旧曲线。所以「残差常驻」这件事要在它真被画出来
    的那一步上验，见 ``STEP_PLOT_PANES``。
    """
    value = replace(project(dataset_project(result=final_fit_result())), base_directory="/private/tmp")
    window = _window(qtbot, api.select_active_dataset(value, "curve"))
    panel = window.plot_panel

    # 一张卡，两个位置：tab 条上有「加权残差」这一段，而此刻选中的是别的段，所以卡该在
    # splitter 那一格里。控件只有一个父件，所以「两处都有」永远只能是搬家，不是两份。
    assert "residual" in panel.reflectivity_tab_keys()
    cards = panel.findChildren(QFrame, "plotCard:residual")
    assert len(cards) == 1, "the residual is drawn twice"
    card = cards[0]
    assert card.isVisible(), "the canvas has no permanent residual card"
    selected = panel.reflectivity_tabs.currentWidget()
    assert selected is not None and selected.isVisible(), "the selected view lost its place"
    # Stacked, not merely both present: the residual belongs directly below the
    # view it judges, which is what the vertical splitter's child order encodes.
    # splitter 里那一格装的是整组（``.canvas-top`` 那一行 + 页面），页面壳自己不在 splitter
    # 里，问它拿到的是 −1。
    splitter = panel.plot_splitter
    assert splitter.indexOf(card) == splitter.indexOf(panel.reflectivity_group) + 1


def test_reflectivity_card_subtitle_follows_the_active_dataset(qtbot, tmp_path) -> None:
    """The mockup's ``.sub`` opens with the dataset name (``aSi_ML_25C · 归一化``).

    Which curve is on screen is the first thing a reader checks against a
    batch of imports, and the tab row cannot say it -- one tab bar serves every
    dataset.  With no dataset the caption is the note alone rather than a stray
    separator.
    """
    from xrr_fitter.gui.plots.diagnostics import plot_card_subtitle

    window = _window(qtbot, _project(tmp_path, structured=True))
    assert _card_texts(window.plot_panel, "plotCard:log")[1].startswith("curve · ")
    assert plot_card_subtitle("log", None) == "归一化"


def test_reflectivity_card_carries_the_designs_legend(qtbot, tmp_path) -> None:
    """The mockup keys every mark drawn on the reflectivity card.

    Four marks share one pair of axes -- points, a line, a shaded window and the
    clipped-point glyph -- and only the first two are conventional enough to read
    unlabelled.  Swatch labels carry a pixmap and no text, so dropping the empty
    ones leaves the captions.
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    legend = window.plot_panel.findChild(QWidget, "plotCard:logLegend")

    assert legend is not None, "the reflectivity card has no legend row"
    captions = tuple(text for label in legend.findChildren(QLabel) if (text := label.text()))
    assert captions == ("观测数据", "当前拟合模型", "拟合窗口", "▽ 截断点（低于本底）")


def _uncertainty(*, systematic: bool = False, candidate_id: str | None = None) -> api.UncertaintyReport:
    return api.UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.eye(1),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=systematic,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _fit_pair(
    *,
    residuals: tuple[float, ...] = (1.0, -1.0, 2.0, 0.0),
    report: bool = True,
    systematic: bool = False,
    report_candidate_id: str | None = None,
):
    """一份拟合结果与它的候选解：Σr²=6，自由参数 0，所以 χ²ᵥ = 6/4 = 1.5。"""
    candidate = replace(fit_candidate("candidate-0"), weighted_residuals=np.array(residuals, dtype=float))
    result = final_fit_result(candidate)
    if report:
        result = replace(result, uncertainty=_uncertainty(systematic=systematic, candidate_id=report_candidate_id))
    return result, candidate


def _fitted_project(tmp_path, *, systematic: bool = False):
    """一份磁盘上真的能读回来的拟合工程，点数与候选解的数组对得上。

    画布不直接读工程里的 ``last_valid_result``：它先把源文件 ``api.import_data`` 读回来，再
    ``validate_result`` 逐个候选核对点数。源文件打不开时那一步会静默丢掉整个数据集，于是
    「有结果」被读成「还没拟合」——卡片会正确地写「尚未拟合」，而要验的那句根本没机会出现。
    所以这里过一遍 ``api.add_dataset``（它自己算 sha256 与 base_directory），曲线只写四行，
    正好配 ``fit_candidate`` 的四点数组。
    """
    result, _candidate = _fit_pair(systematic=systematic)
    value = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy", 4), api.InstrumentSpec())
    dataset = value.datasets[0]
    value = replace(value, datasets=(replace(dataset, last_valid_result=result),))
    return api.select_active_dataset(value, dataset.dataset_id)


def test_the_residual_card_reads_out_its_own_verdict(qtbot, tmp_path) -> None:
    """设计稿这张卡的第二句是判读，不是数据集名：``χ²ᵥ = 1.14 · 无系统性结构``。

    另外三张卡的副标题冠着数据集名，因为一条 tab 条服务所有数据集，读者得知道屏幕上是哪
    一条曲线——但那个名字这样已经在画布上出现三遍了。残差图问的是另一个问题：这次拟合到底
    行不行。χ²ᵥ 给量级（1 附近是噪声量级），「有没有系统性结构」给形状（残差该是白噪声，
    成片同号就说明模型缺东西）。两句合起来正好是读者看完残差图想得到的结论，所以这一格
    换成判读，而纵轴的定义 ``(数据−模型)/σ`` 交给图例。
    """
    window = _window(qtbot, _fitted_project(tmp_path))

    assert _card_texts(window.plot_panel, "plotCard:residual")[1] == "χ²ᵥ = 1.5 · 无系统性结构"


def test_the_residual_card_keys_the_sigma_band_beside_the_points(qtbot, tmp_path) -> None:
    """设计稿残差卡的图例是两条：``(数据−模型)/σ`` 与 ``±1σ 区间``。

    带子是图上唯一的绝对刻度——纵轴随这一轮残差自动缩放，同一张形状在量级差十倍时看起来
    一模一样，只有那条常数 ±1 的浅带能把「偏离了几个 σ」读出来。所以它必须在图例里有名字，
    否则读者只会把它当装饰底色。零参考线不进图例：一条穿过 0 的水平实线不需要人教。
    """
    window = _window(qtbot, _fitted_project(tmp_path))
    legend = window.plot_panel.findChild(QWidget, "plotCard:residualLegend")

    assert legend is not None, "残差卡没有图例行"
    captions = tuple(text for label in legend.findChildren(QLabel) if (text := label.text()))
    assert captions == ("(数据−模型)/σ", "±1σ 区间")


def test_the_residual_verdict_only_claims_what_this_fit_supports() -> None:
    """判读的每一句都得有出处，凑不出来就少说一句，而不是补一句默认值。

    三种缺口各有各的说法：还没拟合过，卡片先说「尚未拟合」——写 ``χ²ᵥ = —`` 会让读者以为
    算过而算不出来。窗口里一个点都没参与（残差整条是 ``nan``）时 χ²ᵥ 没有定义，说「不可用」。
    报告属于另一个候选解时（读者切了候选、不确定度还是上一个的）只留 χ²ᵥ：χ²ᵥ 是从当前
    候选自己的残差算的，「有没有系统性结构」却是那份旧报告的结论，两句混在一行会把旧结论
    说成这个候选的。
    """
    from xrr_fitter.gui.plots.diagnostics import residual_card_subtitle

    result, candidate = _fit_pair()

    assert residual_card_subtitle(None, None) == "尚未拟合"
    assert residual_card_subtitle(result, None) == "尚未拟合"
    assert residual_card_subtitle(result, candidate) == "χ²ᵥ = 1.5 · 无系统性结构"

    blind_result, blind_candidate = _fit_pair(residuals=(float("nan"),) * 4)
    assert residual_card_subtitle(blind_result, blind_candidate) == "χ²ᵥ 不可用"

    stale_result, stale_candidate = _fit_pair(report_candidate_id="candidate-9")
    assert residual_card_subtitle(stale_result, stale_candidate) == "χ²ᵥ = 1.5"

    structured_result, structured_candidate = _fit_pair(systematic=True)
    # 「无系统性结构」的反面要写成检出，而不是把那句删掉：残差有结构是这次拟合最该被读到的
    # 一件事，沉默会读成通过。措辞跟不确定度面板的「检出系统性残差」同源。
    assert residual_card_subtitle(structured_result, structured_candidate) == "χ²ᵥ = 1.5 · 检出系统性结构"

    unreported_result, unreported_candidate = _fit_pair(report=False)
    assert residual_card_subtitle(unreported_result, unreported_candidate) == "χ²ᵥ = 1.5"


def test_reflectivity_panes_leave_the_naming_to_their_card(qtbot, tmp_path) -> None:
    """The mockup heads a ``.plotcard`` once, so the pane draws no title inside it.

    pyqtgraph can title a pane from within the plot, and every draw used to: with
    the tab label above and the card header beside it, one picture carried three
    names at three different alignments -- the in-plot one indented by the y-axis
    width -- and two of them were the same words.  The caveat that title carried
    alone, that qz⁴·R is a diagnostic transform rather than fitted data, moves
    into that card's subtitle instead of going with it.
    """
    from xrr_fitter.gui.plots.diagnostics import plot_card_subtitle

    window = _window(qtbot, _project(tmp_path, structured=True))
    panel = window.plot_panel

    for key in ("log", "raw"):  # the two panes a freshly built window draws
        label = panel.view(key).plot_item.titleLabel
        assert (label.text, label.isVisible()) == ("", False), f"{key} pane titled itself"
    assert "非拟合数据" in plot_card_subtitle("qz4", None)


def test_the_analysis_pane_takes_the_whole_canvas_at_the_smallest_shipped_window(qtbot, tmp_path) -> None:
    """帧⑤：选到分析页，中栏整段归它，两张竖排子图在最小窗口里各自都还有轴可读。

    此前钉这一屏的断言只读逻辑——``canvas_pane_keys() == ("analysis",)`` 加残差让位——而这
    两句在「画布四段全隐藏、屏幕整块空白」时同样成立：``_sync_pane_scope`` 给分析组多加了
    一道 ``self._result is not None``，而那个结果是 ``prepare_project_plots`` 把源文件从磁盘
    读回来之后才有的。读不回来时它静默丢掉整个数据集，于是分析组高 0、splitter 四格全 0，
    旧断言照绿。所以这里改成量真实几何：可见、拿到整段高度、两张图各有轴高。

    尺寸取 ``MINIMUM_SHELL_*``，也就是发货允许的最小窗口。更大的窗口只会把画布拉高
    （1400×900 量到 790×739、1920×1200 量到 1310×1039），在这一档读得出来别的档就都读
    得出来，所以下限钉在这里。
    """
    window = _window(qtbot, _fitted_project(tmp_path))
    window.resize(MINIMUM_SHELL_WIDTH, MINIMUM_SHELL_HEIGHT)
    panel = window.plot_panel

    panel.select_view("uncertainty")
    qtbot.wait(50)

    assert panel.canvas_pane_keys() == ("analysis",)
    assert panel.analysis_group.isVisibleTo(panel) is True
    # 整段中栏，而不是与别的段分成两截；再加一道下限，否则「四格全 0」时两边都是 0 也相等。
    assert panel.analysis_group.height() == panel.plot_splitter.height(), panel.plot_splitter.sizes()
    assert panel.analysis_group.height() > ANALYSIS_MIN_H, panel.plot_splitter.sizes()

    view = panel.view("uncertainty")
    view.canvas.draw()
    renderer = view.figure.canvas.get_renderer()
    correlation, profile = (axes.get_window_extent(renderer) for axes in view.figure.axes[:2])

    # 上下排，不是并肩：并肩会把方阵压到半栏宽，剩给刻度、图例和色标的余地只有一百多像素。
    assert profile.y1 <= correlation.y0, (correlation.bounds, profile.bounds)
    # 最小窗口实测两张各 145px 高；120 是「刻度加轴标签之外还剩一条曲线」的下限。
    for box in (correlation, profile):
        assert box.height >= 120, box.bounds
