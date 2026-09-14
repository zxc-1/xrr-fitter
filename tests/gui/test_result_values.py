"""帧① 检视器「参数 · 结果值」表的投影契约。

设计稿把拟合结果的每个参数排成 ``参数 | 结果值 | ±1σ | 单位`` 四列，并用跨列
的分组行把一层的厚度、粗糙度、密度收在该层名下。这与设置区的边界表是两张不同
的表：那张写的是用户交给求解器的初值与上下限，这张写的是求解器交回来的值。

数值取自候选解自身的 ``ParameterValue``，误差取自 ``UncertaintyReport`` 的
``parameter_sigma``——它与 ``correlation_names`` 同序，是真正的逐参数 1σ。旧存档
没有这个字段，此时误差列写「不可用」而不是拿 bootstrap 区间的半宽冒充 1σ。

长度量在这里与设置表用同一套换算：声明是 Å，屏幕上是 nm，误差随值一起换算，
否则一个 nm 的值配一个 Å 的误差会读成两个数量级的谎报。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QTableWidget
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    project,
)

import xrr_fitter.api as api
from xrr_fitter.model.parameters import ParameterValue

CAPTIONS = {"component.0": "表面氧化层", "component.1": "非晶硅层"}


def _definition(name: str, display_name: str, unit: str, **changes) -> api.ParameterDefinition:
    values = {
        "name": name,
        "display_name": display_name,
        "unit": unit,
        "category": "structure",
        "initial": 1.0,
        "lower": 0.0,
        "upper": 100.0,
        "transform": "linear",
        "locked": False,
    }
    values.update(changes)
    return api.ParameterDefinition(**values)


def _definitions() -> tuple[api.ParameterDefinition, ...]:
    return (
        _definition("component.0.thickness_a", "表面氧化层 厚度", "Å"),
        _definition("component.0.roughness_a", "表面氧化层 粗糙度", "Å"),
        _definition("component.1.density_scale", "非晶硅层 密度", "g/cm³", locked=True),
        _definition("instrument.scale", "强度标度", "", category="instrument"),
    )


def _values() -> tuple[ParameterValue, ...]:
    return (
        ParameterValue("component.0.thickness_a", 34.2, 0.0, 100.0),
        ParameterValue("component.0.roughness_a", 5.1, 0.0, 100.0),
        ParameterValue("component.1.density_scale", 2.19, 0.0, 100.0),
        ParameterValue("instrument.scale", 0.982, 0.0, 100.0),
    )


def _report(sigma: np.ndarray | None, *, names: tuple[str, ...] | None = None) -> api.UncertaintyReport:
    names = tuple(value.name for value in _values()) if names is None else names
    return api.UncertaintyReport(
        correlation_names=names,
        correlation_matrix=np.eye(len(names)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id="candidate-a",
        parameter_sigma=sigma,
    )


def _result(*, sigma: np.ndarray | None = None):
    candidate = replace(
        fit_candidate("candidate-a", 0.2),
        parameters=_values(),
        unit_vector=np.zeros(len(_values())),
    )
    result = final_fit_result(candidate)
    return replace(
        result,
        parameter_definitions=_definitions(),
        uncertainty=_report(sigma),
    )


def _table(qtbot, result=None, *, candidate_id: str = "candidate-a", captions: dict[str, str] | None = None):
    from xrr_fitter.gui.results.values import ResultValueTable

    table = ResultValueTable()
    qtbot.addWidget(table)
    if result is not None:
        table.project_result(result, candidate_id, CAPTIONS if captions is None else captions)
    return table


def _rows(table) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            "" if table.item(row, column) is None else table.item(row, column).text()
            for column in range(table.columnCount())
        )
        for row in range(table.rowCount())
    )


def test_the_result_table_reads_parameter_value_sigma_and_unit(qtbot) -> None:
    table = _table(qtbot)

    headers = tuple(table.horizontalHeaderItem(column).text() for column in range(table.columnCount()))

    assert headers == ("参数", "结果值", "±1σ", "单位")


def test_each_owner_opens_with_a_caption_row_naming_it(qtbot) -> None:
    """设计稿的 ``<tr class="grouprow">``：一层的参数收在该层名下。"""
    table = _table(qtbot, _result())

    captions = tuple(row[0] for row in _rows(table) if table.columnSpan(_rows(table).index(row), 0) > 1)

    assert captions == ("表面氧化层", "非晶硅层", "仪器")


def test_length_results_and_their_sigma_are_both_shown_in_nanometres(qtbot) -> None:
    """值换算而误差不换算，会把 0.8 Å 读成 0.8 nm——差一个数量级。"""
    sigma = np.array([0.8, 0.6, 0.05, 0.004])
    table = _table(qtbot, _result(sigma=sigma))

    rows = {row[0]: row for row in _rows(table)}

    assert rows["厚度 d"][1:] == ("3.420", "0.080", "nm")
    assert rows["粗糙度 σ"][1:] == ("0.510", "0.060", "nm")


def test_a_dimensionless_result_writes_an_em_dash_rather_than_an_empty_unit(qtbot) -> None:
    sigma = np.array([0.8, 0.6, 0.05, 0.004])
    table = _table(qtbot, _result(sigma=sigma))

    rows = {row[0]: row for row in _rows(table)}

    assert rows["强度标度"][1:] == ("0.982", "0.004", "—")


def test_a_locked_parameter_reports_being_locked_instead_of_an_uncertainty(qtbot) -> None:
    """锁定的参数没有参与求解，给它一个误差数字等于凭空报告了一次测量。"""
    sigma = np.array([0.8, 0.6, 0.05, 0.004])
    table = _table(qtbot, _result(sigma=sigma))

    rows = {row[0]: row for row in _rows(table)}

    assert rows["密度 ρ"][1] == "2.190"
    assert rows["密度 ρ"][2] == "锁定"


def test_a_report_without_per_parameter_sigma_says_so_rather_than_reusing_bootstrap(qtbot) -> None:
    """bootstrap 区间是分位区间，其半宽不是 1σ；旧存档缺 sigma 时必须留白。"""
    table = _table(qtbot, _result(sigma=None))

    rows = {row[0]: row for row in _rows(table)}

    assert rows["厚度 d"][1] == "3.420"
    assert rows["厚度 d"][2] == "不可用"


def _tiny_result():
    """一个本底量级的结果：3.1e-7 用定点写就是 0.000，只能改科学记数。"""
    definitions = (_definition("instrument.background", "本底", "", category="instrument"),)
    values = (ParameterValue("instrument.background", 3.1e-7, 0.0, 1.0),)
    candidate = replace(
        fit_candidate("candidate-a", 0.2),
        parameters=values,
        unit_vector=np.zeros(len(values)),
    )
    return replace(
        final_fit_result(candidate),
        parameter_definitions=definitions,
        uncertainty=_report(np.array([4e-8]), names=("instrument.background",)),
    )


def test_a_value_too_small_for_three_decimals_is_written_in_scientific_notation(qtbot) -> None:
    """设计稿的本底行写 ``3.1e-7``：定点三位小数会把它压成 ``0.000``。"""
    table = _table(qtbot, _tiny_result())

    rows = {row[0]: row for row in _rows(table)}

    assert rows["本底"][1] == "3.1e-7"


def test_a_tiny_values_sigma_shares_the_values_exponent(qtbot) -> None:
    """设计稿把 4e-8 写成 ``0.4e-7``，跟着值的指数走。

    两列各自取自己的指数时，屏幕上是 ``3.1e-7`` 配 ``4.0e-8``——同一行里两个指数，
    读者得先在心里对齐一次幂次才知道误差是值的百分之几。共用值的指数之后，两个尾数
    直接可比。
    """
    table = _table(qtbot, _tiny_result())

    rows = {row[0]: row for row in _rows(table)}

    assert rows["本底"][2] == "0.4e-7"


def test_a_group_row_names_the_role_before_the_material(qtbot) -> None:
    """设计稿第 436 行：分组行写「表面氧化层 · SiO₂」，与层堆叠第 659 行正好相反。

    列表回答「这一层是什么材料」，所以材料在前；表的分组行回答「下面这几个数属于样品
    的哪一部分」，所以角色在前。
    """
    captions = {"component.0": "SiO₂ · 表面氧化层", "component.1": "a-Si · 非晶硅薄膜"}
    table = _table(qtbot, _result(), captions=captions)
    rows = _rows(table)

    written = tuple(row[0] for index, row in enumerate(rows) if table.columnSpan(index, 0) > 1)

    assert written == ("表面氧化层 · SiO₂", "非晶硅薄膜 · a-Si", "仪器")


def test_reversing_the_caption_still_strips_the_owner_from_each_row(qtbot) -> None:
    """分组行调了顺序，行名照旧只剩「厚度 d」——否则每行都重复一遍层名。"""
    captions = {"component.0": "表面氧化层", "component.1": "非晶硅层"}
    table = _table(qtbot, _result(), captions=captions)

    assert "厚度 d" in {row[0] for row in _rows(table)}


def test_the_values_table_draws_no_grid_and_no_frame_of_its_own(qtbot) -> None:
    """设计稿的 ``table.grid`` 只有横线：``td`` 各带一道下边框，没有竖线也没有外框。

    竖线把四列切成四只格子，读者的眼睛于是横着走一格停一下；设计稿靠右对齐的数字列
    自己就成列了，不需要线。外框则是双线的来源——这张表本来就装在检视器那一段里。
    """
    table = _table(qtbot, _result())

    assert table.showGrid() is False
    assert table.frameShape() == QFrame.Shape.NoFrame


def test_the_values_table_never_shows_a_scrollbar(qtbot) -> None:
    """设计稿这张表整段展开：一次结果十二行，滚动条会把最后几层藏在折叠线下。"""
    table = _table(qtbot, _result())

    assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_the_values_table_is_exactly_as_tall_as_its_rows(qtbot) -> None:
    """不滚动就得把高度让给内容：表高 = 表头 + 每行行高，一像素不多。

    留着 ``QTableWidget`` 默认的那份高度，检视器里这张表会先撑出一大片空白，再把
    候选解那一段顶到视口以下。
    """
    table = _table(qtbot, _result())

    expected = table.horizontalHeader().sizeHint().height() + sum(
        table.rowHeight(row) for row in range(table.rowCount())
    )
    assert table.sizeHint().height() == expected
    assert table.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed


def test_the_table_grows_when_a_longer_result_replaces_a_shorter_one(qtbot) -> None:
    """高度跟着行数走：投影换一份结果后不重算，表还按上一份的行数占位。"""
    table = _table(qtbot, _tiny_result())
    short = table.sizeHint().height()

    table.project_result(_result(), "candidate-a", CAPTIONS)

    assert table.sizeHint().height() > short


def test_a_group_row_is_shaded_the_way_the_design_shades_it(qtbot) -> None:
    """设计稿 ``.grouprow td{background:var(--panel-2)}``：分组行比参数行深一档。

    抬头靠加粗和字号已经分出层级，但十二行里插三条分组行时，仅靠字重读者仍要逐行辨认；
    一层底色让「哪几行属于同一层」在扫一眼时就成块。
    """
    from xrr_fitter.gui.results.values import GROUP_SHADE_ALPHA

    table = _table(qtbot, _result())
    rows = _rows(table)
    group = next(index for index in range(len(rows)) if table.columnSpan(index, 0) > 1)
    plain = next(index for index in range(len(rows)) if table.columnSpan(index, 0) == 1)

    assert table.item(group, 0).background().color().alphaF() == pytest.approx(GROUP_SHADE_ALPHA, abs=0.01)
    assert table.item(plain, 0).background().style() == Qt.BrushStyle.NoBrush


def test_the_locked_row_carries_the_lock_mark_the_design_draws(qtbot) -> None:
    """设计稿在 ``密度 ρ`` 后面画一枚打勾的方框（``.lock.on``）。

    误差列的「锁定」二字要读到第三列才见到；行名旁的方框在扫参数名时就答了「这个数是
    我按住的，不是求解器解出来的」。
    """
    from xrr_fitter.gui.results.values import LOCKED_ROLE

    sigma = np.array([0.8, 0.6, 0.05, 0.004])
    table = _table(qtbot, _result(sigma=sigma))
    rows = _rows(table)

    locked = {row[0] for index, row in enumerate(rows) if table.item(index, 0).data(LOCKED_ROLE)}

    assert locked == {"密度 ρ"}


def test_clearing_the_projection_empties_every_row(qtbot) -> None:
    table = _table(qtbot, _result())

    table.clear_projection()

    assert table.rowCount() == 0


@pytest.mark.parametrize("candidate_id", ("candidate-missing", None))
def test_an_unknown_candidate_leaves_no_stale_values_on_screen(qtbot, candidate_id) -> None:
    """候选切换到一个结果里不存在的 ID 时，上一次的数字必须先消失。"""
    table = _table(qtbot, _result())

    table.project_result(_result(), candidate_id, CAPTIONS)

    assert table.rowCount() == 0


def test_the_app_stylesheet_leaves_this_one_table_without_a_frame(qtbot) -> None:
    """通用规则给每张表一道 1px 外框，这张表必须按 id 把它抹掉。

    ``setFrameShape(NoFrame)`` 只管 Qt 自己画的那道框，样式表里的 ``border`` 照旧生效：
    检视器里于是出现双线，而按内容算出来的表高被这道框吃掉两像素，末行被裁。
    """
    from xrr_fitter.gui import theme

    table = _table(qtbot, _result())
    plain = QTableWidget(1, 1)
    qtbot.addWidget(plain)
    sheet = theme.build_stylesheet(table.palette())
    table.setStyleSheet(sheet)
    plain.setStyleSheet(sheet)

    assert plain.frameWidth() == 1, "通用规则已不给表加框，本测试的前提失效"
    assert table.frameWidth() == 0
    expected = table.horizontalHeader().sizeHint().height() + sum(
        table.rowHeight(row) for row in range(table.rowCount())
    )
    assert table.sizeHint().height() == expected


def _fitted_panel(qtbot):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    value = project(dataset_project(result=_result(sigma=np.array([0.8, 0.6, 0.05, 0.004]))))
    value = replace(value, base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    return panel


def test_the_inspector_shows_the_result_values_under_the_verdict(qtbot) -> None:
    """帧①的检视器顺序：拟合判定，然后参数·结果值，然后候选解。"""
    panel = _fitted_panel(qtbot)

    layout = panel.layout()
    order = tuple(layout.itemAt(index).widget().objectName() for index in range(layout.count()))

    assert order.index("resultConfidenceCard") < order.index("resultValuesCard")
    assert order.index("resultValuesCard") < order.index("resultCandidatesCard")
    assert panel.result_values.rowCount() > 0


def test_each_frame_one_section_carries_the_caption_the_design_gives_it(qtbot) -> None:
    """设计稿帧① 右栏是三段，每段一句抬头：拟合判定 / 参数 · 结果值 / 候选解。

    这三句此前只有第一句落在它该落的地方。结果读数表没有抬头，而「参数 · 结果值」
    挂在设置区那张边界表上——那张表写的是初值与上下限，一个结果值都不写。读者照抬头
    找结果值会读到自己交出去的初值，两者在收敛的拟合里数量级相同，很难当场看出读错了。
    """
    panel = _fitted_panel(qtbot)

    captions = {
        "resultConfidenceCard": "拟合判定",
        "resultValuesCard": "参数 · 结果值",
        "resultCandidatesCard": "候选解",
    }
    for name, title in captions.items():
        card = panel.findChild(QFrame, name)
        assert card is not None, name
        heading = card.findChild(QLabel, f"{name}Title")
        assert heading is not None, name
        assert heading.text() == title, name


def test_the_result_values_caption_counts_the_free_parameters(qtbot) -> None:
    """设计稿在这句抬头右侧写「12 自由」：读者不数行就知道这次求解了几个数。

    自由数由 ``result.parameter_definitions`` 现算：锁定与被约束的参数没有参与求解，
    数进去会把「求解了几个」报成「表里有几行」。这个数不必调 ``describe_parameters``，
    结果自己带着当时那份声明。
    """
    panel = _fitted_panel(qtbot)

    subtitle = panel.findChild(QLabel, "resultValuesCardSubtitle")

    assert subtitle is not None
    assert subtitle.text() == "3 自由"


def test_the_candidate_count_rides_its_own_caption(qtbot) -> None:
    """「3 个」跟着候选解那句抬头走，因为它数的是那一段里的行。"""
    panel = _fitted_panel(qtbot)

    subtitle = panel.findChild(QLabel, "resultCandidatesCardSubtitle")

    assert subtitle is not None
    assert subtitle.text() == panel.candidate_summary()
    assert subtitle.text() == "1 个"
