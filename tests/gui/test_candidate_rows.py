"""帧① 候选解一节的行形状。

设计稿的每个候选是一行三段：左边一个状态形状，中间加粗的名字，右边贴着右缘的
``J=…`` 加一枚「当前」胶囊。实测是一行散文——``candidate-best · 局部目标值
J=0.2 · 全局排序目标值 J=0.4 · 有效 · 推荐 · 查看中``，六个字段用点号串成一句，
要读者自己在里面找目标值。这里锁定它被拆成可分别对齐的段。

形状照设计稿画可信度（●◆▲■○）。可信度是整份结果的判定，``FitCandidate`` 上没有
这个字段，但采信的那一行正是被判定的那一行——判定挂在它身上。其余候选没有被判定过，
它们的存在本身就是「目标值相近的候选不止一个」，也就是 ``多解 ▲``；解不出来的那些是
``不可信 ■``，这一条压过采信标记，因为一个不该采信的解不因为被打开而变得可信。

右侧那枚标记只有两种：打开着的那行写「当前」，其余每行写「切换」。「推荐」曾经是第三
种，但推荐与「点它就能打开」是同一件事，两枚标记并排时读者要先分辨哪一枚是可点的。
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontInfo

from xrr_fitter.gui import theme


def _candidate(
    candidate_id: str = "candidate-a",
    *,
    objective: float = 1.83,
    valid: bool = True,
    stop_reason: str = "converged",
    ranking: float | None = None,
):
    from dataclasses import replace

    from tests.support.model_cases import fit_candidate

    return replace(
        fit_candidate(candidate_id=candidate_id, objective=objective),
        valid=valid,
        stop_reason=stop_reason,
        ranking_objective=ranking,
    )


def _row(candidate, *, selected: bool = False, adopted: bool = False, confidence: str | None = None):
    from xrr_fitter.gui.results.candidate_row import candidate_row

    return candidate_row(candidate, selected=selected, adopted=adopted, confidence=confidence)


def test_the_adopted_candidate_carries_the_verdict_passed_down_to_it() -> None:
    """采信的那一行就是被判定的那一行，所以判定的形状画在它身上。"""
    row = _row(_candidate(), adopted=True, confidence="可信")

    assert row.glyph == "●"
    assert row.kind == "ok"
    assert row.state == "可信"


def test_a_correlated_verdict_reaches_the_row_as_its_own_shape() -> None:
    """``可用但相关`` 在状态栏折到 warn，这里不折：四种判定四种形状。"""
    row = _row(_candidate(), adopted=True, confidence="可用但相关")

    assert row.glyph == "◆"
    assert row.kind == "info"
    assert row.state == "可用但相关"


def test_a_candidate_nobody_adopted_reads_as_one_of_several_solutions() -> None:
    """没被采信的可用解就是「目标值相近的候选不止一个」，也就是设计稿的 ``多解``。"""
    row = _row(_candidate())

    assert row.glyph == "▲"
    assert row.kind == "warn"
    assert row.state == "多解"


def test_an_unusable_candidate_is_shaped_as_the_verdict_that_refuses_it() -> None:
    row = _row(_candidate(valid=False, stop_reason="max_nfev"))

    assert row.glyph == "■"
    assert row.kind == "error"
    assert row.state == "不可信"


def test_being_adopted_does_not_make_an_unusable_candidate_trustworthy() -> None:
    """打开一个算不出来的解是允许的（检查用），把它画成 ● 不是。"""
    row = _row(_candidate(valid=False), selected=True, adopted=True, confidence="可信")

    assert row.glyph == "■"
    assert row.kind == "error"


def test_an_adopted_row_without_a_verdict_draws_the_hollow_ring() -> None:
    """旧存档没有判定字段。空心圈读作「还没测」，不是第五档可信度。"""
    row = _row(_candidate(), adopted=True, confidence=None)

    assert row.glyph == theme.CONFIDENCE_FALLBACK_GLYPH
    assert row.kind == theme.CONFIDENCE_FALLBACK_KIND
    assert row.state == "不可用"


def test_the_shapes_are_distinct_so_colour_is_never_the_only_signal() -> None:
    """四档四形。两档共用一个形状就把双编码退回成纯色编码。"""
    shapes = {_row(_candidate(), adopted=True, confidence=grade).glyph for grade in theme.CONFIDENCE_GLYPHS}

    assert len(shapes) == len(theme.CONFIDENCE_GLYPHS)


def test_the_objective_is_its_own_right_aligned_field() -> None:
    """设计稿右缘是 ``J=1.83``：一个能上下对齐的短字段，不是句子里的一截。"""
    row = _row(_candidate(objective=1.83))

    assert row.objective == "J=1.83"


def test_the_global_ranking_objective_stays_on_the_row_when_it_differs() -> None:
    """两个目标值都在，排序用的那个不能丢——它决定了谁被采信。"""
    row = _row(_candidate(objective=0.2, ranking=0.4))

    assert row.objective == "J=0.2"
    assert row.ranking == "排序 J=0.4"


def test_a_ranking_objective_equal_to_the_local_one_is_not_written_twice() -> None:
    """一次单阶段求解里两者相等，写两遍等于让读者去比较两个一样的数。"""
    row = _row(_candidate(objective=1.83, ranking=1.83))

    assert row.objective == "J=1.83"
    assert row.ranking == ""


def test_a_candidate_without_a_ranking_objective_leaves_that_field_empty() -> None:
    row = _row(_candidate(objective=0.2))

    assert row.ranking == ""


def test_the_inspected_candidate_is_marked_current_rather_than_described() -> None:
    row = _row(_candidate(), selected=True)

    assert row.mark == "当前"


def test_every_other_row_offers_the_switch_the_design_draws_on_it() -> None:
    """设计稿候选 B/C 右缘各有一枚「切换」：这一行不是当前，但点它就能是。"""
    row = _row(_candidate(), selected=False)

    assert row.mark == "切换"


def test_an_unusable_candidate_can_still_be_opened_for_inspection() -> None:
    """无效解不能被采信，但读者要能打开它看为什么算不出来，所以照样写「切换」。"""
    row = _row(_candidate(valid=False), selected=False)

    assert row.mark == "切换"


def test_the_adopted_row_is_not_marked_twice_while_it_is_the_one_open() -> None:
    """一行只有一枚标记的宽度：当前压过一切，它回答「我在看哪个」。"""
    row = _row(_candidate(), selected=True, adopted=True, confidence="可信")

    assert row.mark == "当前"


def test_the_title_is_the_position_the_row_holds() -> None:
    """行首的名字是序号，求解器 ID 见 ``test_the_solver_id_stays_reachable_on_the_row_it_names``。"""
    row = _row(_candidate("candidate-best"))

    assert row.title == "候选 A"


@pytest.mark.parametrize("kind", ["ok", "info", "warn", "error", "muted_text"])
def test_every_row_kind_resolves_to_a_token_in_both_appearances(kind: str) -> None:
    """行的颜色走 token，不写死十六进制，否则深色下要么刺眼要么看不见。"""
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        assert getattr(tokens, kind)


@pytest.mark.parametrize("kind", ["ok", "info", "warn", "error", "muted_text"])
def test_the_glyph_colour_survives_a_token_written_as_rgba(kind: str) -> None:
    """``QColor("rgba(0, 0, 0, 140)")`` 解不出来，画出来是不透明的黑。

    空心圈那一档取的是 ``muted_text``，而它在两套外观里都写成 ``rgba(...)``——形状色
    直接交给 ``QColor`` 时，这一行会画成一枚墨黑的圈，比四档判定里最重的 ■ 还重。
    """
    from dataclasses import replace as replace_row

    from xrr_fitter.gui.results.candidate_row import glyph_colour

    for tokens, palette in ((theme.LIGHT_TOKENS, _light_palette()), (theme.DARK_TOKENS, _dark_palette())):
        colour = glyph_colour(replace_row(_row(_candidate()), kind=kind), palette)
        assert colour.isValid()
        assert colour.rgba() == theme.token_colour(getattr(tokens, kind)).rgba()


def test_the_hollow_ring_keeps_the_muted_alpha_instead_of_painting_solid() -> None:
    """空心圈取 ``muted_text``，设计稿里是 #5B6270（不透明灰）。"""
    from xrr_fitter.gui.results.candidate_row import glyph_colour

    colour = glyph_colour(_row(_candidate(), adopted=True, confidence=None), _light_palette())

    assert (colour.red(), colour.green(), colour.blue(), colour.alpha()) == (91, 98, 112, 255)


def _light_palette():
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#FFFFFF"))
    return palette


def _dark_palette():
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1E1E1E"))
    return palette


def test_the_row_typesets_its_four_segments_at_the_designs_four_sizes(qapp) -> None:
    """设计稿这一行里四段各有字号：形状 14px、名字 12.5px、读数 12px、胶囊 11px。

    四个数写在同一行上，所以它们的全部意义就是互相比大小——形状最大（它是这一行的判定），
    名字次之（它是这一行是谁），读数再小一档（它是拿来扫着比的），胶囊最小（它只是一枚
    标记）。此前四段里有三段借的是点数档（形状 ``FONT_PT_MD`` 实测 15px，读数与胶囊同为
    ``FONT_PT_SM`` 的 12px），名字则一档没设、跟着应用默认字号也是 12px：于是名字、读数、
    胶囊三段量出来一样大，四级层次塌成两级。

    按像素给而不是折算成 pt：这四档要直接比大小，混用两套单位就得先猜当前 DPI 才知道谁
    大——实测 11pt 是 15px 而不是设计稿的 14px，正是这么差出来的。名字取整到 13：
    ``setPixelSize`` 只吃整数，而 12 会让名字与读数同号，「谁是主字段」就只剩字重在说。
    """
    from xrr_fitter.gui.results import candidate_row as row_module

    sizes = (
        QFontInfo(row_module.glyph_font(QFont())).pixelSize(),
        QFontInfo(row_module.title_font(QFont(), selected=False)).pixelSize(),
        QFontInfo(row_module.value_font(QFont())).pixelSize(),
        QFontInfo(row_module.badge_font(QFont())).pixelSize(),
    )

    assert sizes == (14, 13, 12, theme.BADGE_FONT_PX)
    # 顺序本身是断言：四段读起来的层次由字号说出来，量到的必须是严格递减。
    assert list(sizes) == sorted(sizes, reverse=True)


def test_only_the_open_row_carries_the_heavier_name(qapp) -> None:
    """设计稿选中那行的名字是 ``<b>``（700），其余每行显式写 ``font-weight:600``。

    三行并排时读者要一眼看出「我在看哪一个」。选中行已经有主色边框与一层底色，名字再重
    半档是同一件事的第三重编码；反过来，每一行都写 700 就等于没写——加粗成了这一节的常态，
    它就不再指向任何一行。
    """
    from xrr_fitter.gui.results.candidate_row import title_font

    assert title_font(QFont(), selected=True).weight() == QFont.Weight.Bold
    assert title_font(QFont(), selected=False).weight() == QFont.Weight.DemiBold


def test_the_objective_reading_is_typeset_in_tabular_figures(qapp) -> None:
    """设计稿这一段带 ``.tnum``：三行的 ``J=…`` 是拿来上下扫着比大小的。

    比例数字里 1 比 8 窄，``J=1.83`` 与 ``J=2.41`` 于是等宽不同——右对齐只让末位齐，小数点
    仍错开一两像素。代理自己的 docstring 早就写着「读者扫这一列就能比大小」，等宽数字才是
    那句话的实现。
    """
    from xrr_fitter.gui.results.candidate_row import value_font

    assert value_font(QFont()).isFeatureSet(QFont.Tag("tnum")) is True


def test_the_objective_reading_takes_the_fainter_of_the_two_grey_steps() -> None:
    """设计稿这一段是 ``.faint``（``--ink-faint``），不是 ``.muted``（``--ink-muted``）。

    两档灰的分工是固定的：静音那档给还要读的次要文字，最淡那档给读者扫过去、只在需要时
    停下来看的数。这一行的主角是名字与判定形状，``J=…`` 属于后者——给它静音档会让它跟名字
    抢同一级注意力。两档在两套外观里都写成 ``rgba(...)``，所以取色仍走 ``token_colour``。
    """
    from xrr_fitter.gui.results.candidate_row import value_colour

    for tokens, palette in ((theme.LIGHT_TOKENS, _light_palette()), (theme.DARK_TOKENS, _dark_palette())):
        assert value_colour(palette).rgba() == theme.token_colour(tokens.faint_text).rgba()
        assert value_colour(palette).rgba() != theme.token_colour(tokens.muted_text).rgba()


def test_neither_mark_is_heavier_than_the_design_draws_it(qapp) -> None:
    """行尾那两枚标记都是 600：``.badge`` 写 600，``.btn.ghost`` 也写 600。

    自绘那条路把两枚都调到了 700，比同一行的名字（未选中 600）还重——于是「当前」这枚纯状态
    标记压过了它所标记的那一行是谁。字号同理：胶囊是 11px 的一档小字，切换按钮是
    ``.btn.sm`` 的 12px。
    """
    from xrr_fitter.gui.results.candidate_row import badge_font, switch_font

    assert badge_font(QFont()).weight() == theme.BADGE_FONT_WEIGHT
    assert QFontInfo(badge_font(QFont())).pixelSize() == theme.BADGE_FONT_PX
    assert switch_font(QFont()).weight() == QFont.Weight.DemiBold
    assert QFontInfo(switch_font(QFont())).pixelSize() == 12


def test_the_pill_on_the_row_is_the_same_badge_the_stylesheet_paints() -> None:
    """胶囊一档字只有一处实现：自绘与 ``QLabel[badge="true"]`` 读同两个常量。

    这一枚胶囊在两条路上被画：结果检视器的行是自绘，别处的「当前」是带 ``badge`` 属性的
    ``QLabel`` 走样式表。两边各写一份 11px/600 就会在版面调整时分叉——代理里那句禁止再抄一份
    局部胶囊常量的注释说的正是这件事，它当时管住了内边距，字号字重却仍是两份。
    """
    sheet = theme.build_stylesheet(theme.light_palette())
    body = sheet.split('QLabel[badge="true"] {', 1)[1].split("}", 1)[0]

    assert theme.BADGE_FONT_PX == 11
    assert f"font-size: {theme.BADGE_FONT_PX}px" in body
    assert f"font-weight: {theme.BADGE_FONT_WEIGHT.value}" in body


def test_the_list_paints_its_rows_through_the_row_delegate(qtbot) -> None:
    """段落对齐要靠自绘：QListWidgetItem 的一行文本没法分段右对齐。"""
    from xrr_fitter.gui.results.candidate_row import CandidateRowDelegate
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)

    assert isinstance(widget.itemDelegate(), CandidateRowDelegate)


def test_every_loaded_row_hands_the_delegate_its_segments(qtbot) -> None:
    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    widget.load(
        (_candidate("candidate-a"), _candidate("candidate-b", objective=1.97)),
        selected_id="candidate-a",
        recommended_id="candidate-a",
    )

    rows = tuple(widget.item(index).data(CANDIDATE_ROW_ROLE) for index in range(widget.count()))

    assert tuple(row.title for row in rows) == ("候选 A", "候选 B")
    assert tuple(row.objective for row in rows) == ("J=1.83", "J=1.97")
    assert tuple(row.mark for row in rows) == ("当前", "切换")


def test_the_verdict_reaches_the_adopted_row_and_only_that_row(qtbot) -> None:
    """判定是整份结果的，画在被采信的那一行上；另一行照旧是「多解」。

    设计稿候选 A 是 ●（可信），B 是 ▲（多解）。同一份结果里两行取同一个判定的话，
    三行都会画成 ●，读者于是读到「三个都可信」——而这份结果恰恰只推荐了一个。
    """
    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    widget.load(
        (_candidate("candidate-a"), _candidate("candidate-b", objective=1.97)),
        selected_id="candidate-a",
        recommended_id="candidate-a",
        confidence="可信",
    )

    rows = tuple(widget.item(index).data(CANDIDATE_ROW_ROLE) for index in range(widget.count()))

    assert tuple(row.glyph for row in rows) == ("●", "▲")
    assert tuple(row.kind for row in rows) == ("ok", "warn")


def test_projecting_a_result_reads_the_verdict_off_the_result_itself(qtbot) -> None:
    """判定不必由调用方转述：``project_result`` 手上就有整份结果。"""
    from tests.support.model_cases import final_fit_result

    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    result = final_fit_result(_candidate("candidate-a"), _candidate("candidate-b", objective=1.97))

    widget.project_result("curve", result, None)

    rows = tuple(widget.item(index).data(CANDIDATE_ROW_ROLE) for index in range(widget.count()))
    assert rows[0].state == "可信"
    assert rows[1].state == "多解"


def test_the_inspected_row_keeps_the_verdict_after_a_candidate_switch(qtbot) -> None:
    """切到 B 去看时 A 仍是被采信的那一行，它的 ● 不能因为焦点移开就没了。"""
    from tests.support.model_cases import final_fit_result

    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    result = final_fit_result(_candidate("candidate-a"), _candidate("candidate-b", objective=1.97))
    widget.project_result("curve", result, None)

    widget.inspect("candidate-b")

    rows = tuple(widget.item(index).data(CANDIDATE_ROW_ROLE) for index in range(widget.count()))
    assert (rows[0].glyph, rows[0].mark) == ("●", "切换")
    assert (rows[1].glyph, rows[1].mark) == ("▲", "当前")


def test_the_list_is_exactly_as_tall_as_the_rows_it_holds(qtbot) -> None:
    """设计稿这三行整段展开：候选解是这一栏的末段，列表内再套一层滚动条就把末行藏了。"""
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    widget.load(
        (_candidate("candidate-a"), _candidate("candidate-b"), _candidate("candidate-c")),
        selected_id="candidate-a",
        recommended_id="candidate-a",
    )

    expected = sum(widget.sizeHintForRow(row) for row in range(widget.count()))
    assert widget.sizeHint().height() == expected


def test_the_rows_are_separated_by_the_gap_the_design_draws(qtbot) -> None:
    """设计稿 ``gap:6px``：每行是一只有边框的盒子，盒子之间要留出缝。

    缝算进行高而不是交给 ``setSpacing``：那个开关四边都插，首行上方和末行下方于是
    各多出三像素，而设计稿这一段紧贴着上一节的横线开始。
    """
    from xrr_fitter.gui.results.candidate_row import ROW_GAP_PX
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    widget.load((_candidate("candidate-a"), _candidate("candidate-b")), selected_id="candidate-a", recommended_id=None)

    assert widget.spacing() == 0
    assert widget.sizeHintForRow(1) - widget.sizeHintForRow(0) == ROW_GAP_PX


def test_the_candidate_list_draws_no_frame_of_its_own(qtbot) -> None:
    """外框是双线的来源：这张列表本来就装在检视器那一段里。"""
    from PySide6.QtWidgets import QFrame

    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)

    assert widget.frameShape() == QFrame.Shape.NoFrame
    assert widget.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_the_app_stylesheet_leaves_the_candidate_list_without_a_frame(qtbot) -> None:
    """通用规则给每张列表一道 1px 外框，这张列表必须按 id 把它抹掉。

    ``setFrameShape(NoFrame)`` 只管 Qt 自己画的那道框，样式表里的 ``border`` 照旧生效：
    检视器里于是出现双线，而按内容算出来的列表高被这道框吃掉两像素，末行被裁。
    """
    from PySide6.QtWidgets import QListWidget

    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    widget.load((_candidate("candidate-a"), _candidate("candidate-b")), selected_id="candidate-a", recommended_id=None)
    plain = QListWidget()
    qtbot.addWidget(plain)
    sheet = theme.build_stylesheet(widget.palette())
    widget.setStyleSheet(sheet)
    plain.setStyleSheet(sheet)

    assert plain.frameWidth() == 1, "通用规则已不给列表加框，本测试的前提失效"
    assert widget.frameWidth() == 0
    expected = sum(widget.sizeHintForRow(row) for row in range(widget.count()))
    assert widget.sizeHint().height() == expected


def test_the_row_still_reads_as_prose_to_a_screen_reader(qtbot) -> None:
    """自绘接管的是画面，不是可访问文本：读屏仍要拿到完整那句话。"""
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    widget.load((_candidate("candidate-best", objective=0.2, ranking=0.4),), selected_id=None, recommended_id=None)

    text = widget.candidate_text(0)

    assert "candidate-best" in text
    assert "局部目标值 J=0.2" in text
    assert "全局排序目标值 J=0.4" in text
    assert "有效" in text


def test_the_panel_reports_the_candidate_count_for_its_own_caption(qtbot) -> None:
    """设计稿抬头右侧是「3 个」：不用数行数就知道这次搜出了几个解。

    面板报的是措辞，挂在哪一行抬头由 ``window_layout`` 决定——这一节的抬头是外层
    卡片，面板自己再套一张同名卡就会让读者穿过两个「候选解」才到列表。
    """
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, final_fit_result, project

    import xrr_fitter.api as api
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    result = final_fit_result(_candidate("candidate-a"), _candidate("candidate-b"), _candidate("candidate-c"))
    value = replace(project(dataset_project(result=result)), base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)

    assert panel.candidate_count() == 3
    assert panel.candidate_summary() == "3 个"


def test_the_verdict_names_the_dataset_it_judges(qtbot) -> None:
    """设计稿抬头是「拟合判定 aSi_ML_25C」：判定属于哪条曲线要写出来。"""
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, final_fit_result, project

    import xrr_fitter.api as api
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    result = final_fit_result(_candidate("candidate-a"))
    value = replace(project(dataset_project(result=result)), base_directory="/private/tmp")
    panel = ResultsPanel(ProjectDocument(api.select_active_dataset(value, "curve")))
    qtbot.addWidget(panel)

    assert panel.verdict_subtitle() == "curve"


def test_the_verdict_says_there_is_no_dataset_rather_than_naming_none(qtbot) -> None:
    """没有数据集时副标题留空会剩一行空白，而这一节本来就没有判定可言。"""
    from dataclasses import replace

    from tests.support.model_cases import project

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    panel = ResultsPanel(ProjectDocument(replace(project(), base_directory="/private/tmp")))
    qtbot.addWidget(panel)

    assert panel.verdict_subtitle() == "未选择数据集"


def test_the_empty_section_says_so_instead_of_counting_zero(qtbot) -> None:
    """还没拟合时报「0 个」像是搜索失败了；空态该说的是还没跑。"""
    from dataclasses import replace

    from tests.support.model_cases import project

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    panel = ResultsPanel(ProjectDocument(replace(project(), base_directory="/private/tmp")))
    qtbot.addWidget(panel)

    assert panel.candidate_summary() == "尚未拟合"


def test_the_panel_announces_a_changed_count_so_the_caption_can_follow(qtbot) -> None:
    """计数变了要广播出去：卡片比面板先构造不完，抬头只能靠信号跟上。"""
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, final_fit_result, project

    import xrr_fitter.api as api
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    result = final_fit_result(_candidate("candidate-a"), _candidate("candidate-b"))
    value = replace(project(dataset_project(result=result)), base_directory="/private/tmp")
    document = ProjectDocument(api.select_active_dataset(value, "curve"))
    panel = ResultsPanel(document)
    qtbot.addWidget(panel)

    seen: list[str] = []
    panel.candidate_summary_changed.connect(seen.append)
    panel.clear_results()

    assert seen == ["尚未拟合"]
    assert panel.candidate_summary() == "尚未拟合"


def test_a_candidate_is_named_by_position_not_by_its_solver_id() -> None:
    """设计稿的候选叫「候选 A」，不叫求解器给的 ID。

    真实 ID 是 ``E-0``、``automatic-refit-2`` 这种求解器内部编号：它写的是这个候选
    从哪条种子来，回答不了读者在这一节要问的「有几个解、我在看第几个」。设计稿把
    这一节写成 候选 A / B / C，位置就是名字——三行并排时读者数的是第几行，不是
    去比两串编号谁大。ID 仍在同一行的悬停里，跟日志对照时拿得到。
    """
    row = _row(_candidate("E-0"))

    assert row.title == "候选 A"


def test_the_ordinal_walks_the_alphabet_and_falls_back_past_it() -> None:
    """第 27 个候选没有第 27 个字母，所以序号用完了要退回编号。

    自动拟合一次能留下几十个候选，A–Z 只够二十六个。越界时写「候选 27」而不是
    循环回 A——两行同名会让「我在看第几个」这个问题彻底没答案。
    """
    from xrr_fitter.gui.results.candidate_row import candidate_row

    assert candidate_row(_candidate("E-1"), selected=False, ordinal=1).title == "候选 B"
    assert candidate_row(_candidate("E-25"), selected=False, ordinal=25).title == "候选 Z"
    assert candidate_row(_candidate("E-26"), selected=False, ordinal=26).title == "候选 27"


def test_the_solver_id_stays_reachable_on_the_row_it_names(qtbot) -> None:
    """名字换成序号之后，求解器 ID 挂在同一行的悬停上，没有丢。"""
    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE
    from xrr_fitter.gui.results.candidates import CandidateList

    view = CandidateList()
    qtbot.addWidget(view)
    view.load(
        (_candidate("E-0"), _candidate("automatic-refit-2", objective=1.97)),
        selected_id="E-0",
        recommended_id="E-0",
    )

    assert view.item(0).data(CANDIDATE_ROW_ROLE).title == "候选 A"
    assert view.item(1).data(CANDIDATE_ROW_ROLE).title == "候选 B"
    assert "E-0" in view.item(0).toolTip()
    assert "automatic-refit-2" in view.item(1).toolTip()
