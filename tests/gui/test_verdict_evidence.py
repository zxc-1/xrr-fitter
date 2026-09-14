"""帧① 拟合判定卡的徽章行与指标行。

设计稿在可信度大徽标下面挂两块东西：一排判读徽章（自助收敛率、边界命中、强相关）
和一组键值指标（目标值 J、约化 χ²ᵥ、自助失败率）。实测这些数字全在证据面板的散文里——
读者要在八行 prose 里找「Bootstrap 失败率：0.02」。这里锁定它们各自被渲染成
独立控件，且措辞与设计稿一致。
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QColor, QPalette

import xrr_fitter.api as api
from xrr_fitter.gui import theme


def _palette(window: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(window))
    return palette


def _report(
    *,
    failure_rate: float = 0.02,
    boundary_hits: tuple[str, ...] = (),
    strong: tuple[tuple[str, str, float], ...] = (),
    performed: bool = True,
) -> api.UncertaintyReport:
    return api.UncertaintyReport(
        correlation_names=("a", "b"),
        correlation_matrix=np.eye(2),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=failure_rate,
        boundary_hits=boundary_hits,
        strong_correlations=strong,
        systematic_residual=False,
        diagnostics=(),
        bootstrap_performed=performed,
    )


def _badges(qtbot, report):
    from xrr_fitter.gui.results.verdict import VerdictEvidence

    widget = VerdictEvidence()
    qtbot.addWidget(widget)
    widget.set_report(report)
    return widget


def _texts(widget) -> tuple[str, ...]:
    return tuple(badge.text() for badge in widget.badges() if badge.isVisible())


def _kinds(widget) -> dict[str, str]:
    return {badge.text(): badge.property("statusKind") for badge in widget.badges()}


def test_bootstrap_convergence_is_stated_as_a_rate_not_a_failure_count(qtbot) -> None:
    """设计稿写「✓ 自助收敛 98%」，是收敛率；报告里存的是失败率 0.02。"""
    widget = _badges(qtbot, _report(failure_rate=0.02))

    assert "✓ 自助收敛 98%" in _texts(widget)


def test_a_high_bootstrap_failure_rate_stops_claiming_convergence(qtbot) -> None:
    """失败率高时仍打勾会把一次没收敛的拟合报成收敛。"""
    widget = _badges(qtbot, _report(failure_rate=0.4))

    texts = _texts(widget)
    assert "⚠ 自助收敛 60%" in texts
    assert _kinds(widget)["⚠ 自助收敛 60%"] == "warn"


def test_a_skipped_bootstrap_is_not_reported_as_full_convergence(qtbot) -> None:
    """没跑自助时失败率是 0，照 1-0 算会打出「收敛 100%」。"""
    widget = _badges(qtbot, _report(failure_rate=0.0, performed=False))

    assert "— 未运行自助" in _texts(widget)


def test_no_boundary_hit_is_stated_positively(qtbot) -> None:
    widget = _badges(qtbot, _report(boundary_hits=()))

    assert "✓ 无边界命中" in _texts(widget)
    assert _kinds(widget)["✓ 无边界命中"] == "ok"


def test_boundary_hits_are_counted_on_the_badge(qtbot) -> None:
    widget = _badges(qtbot, _report(boundary_hits=("component.0.thickness_a", "instrument.scale")))

    texts = _texts(widget)
    assert "⚠ 2 处边界命中" in texts
    assert _kinds(widget)["⚠ 2 处边界命中"] == "warn"


def test_the_strongest_correlation_coefficient_is_shown_on_the_badge(qtbot) -> None:
    """设计稿写「ℹ 1 处相关 ρ=0.71」——数量加最强的那个系数。"""
    widget = _badges(
        qtbot,
        _report(strong=(("a", "b", 0.71),)),
    )

    texts = _texts(widget)
    assert "ℹ 1 处相关 ρ=0.71" in texts
    assert _kinds(widget)["ℹ 1 处相关 ρ=0.71"] == "info"


def test_the_reported_correlation_is_the_largest_in_magnitude(qtbot) -> None:
    """负相关一样纠缠；按带符号取最大会把 -0.93 让给 0.71。"""
    widget = _badges(
        qtbot,
        _report(strong=(("a", "b", 0.71), ("a", "c", -0.93))),
    )

    assert "ℹ 2 处相关 ρ=0.93" in _texts(widget)


def test_no_strong_correlation_is_stated_positively(qtbot) -> None:
    widget = _badges(qtbot, _report(strong=()))

    assert "✓ 无强相关" in _texts(widget)


def test_the_badges_carry_a_pill_shape_from_the_theme(qtbot) -> None:
    """设计稿 ``.badge`` 是 border-radius:999px 的胶囊，不是一行彩色文字。"""
    widget = _badges(qtbot, _report())

    for badge in widget.badges():
        assert badge.property("badge") is True


def test_the_objective_and_failure_rate_are_shown_as_labelled_metrics(qtbot) -> None:
    """设计稿的 kv 行：左键名右数值，而不是埋在散文里。

    三行的顺序照设计稿：``目标值 J`` / ``约化 χ²ᵥ`` / ``自助失败率``。χ²ᵥ 夹在中间不是
    排版偏好——J 是这套代码自己的目标函数值（带正则项、带标度自由度），χ²ᵥ 才是能跟
    文献里「拟合好不好」直接比的那个数。把它紧贴 J 放，读者一眼就知道后者是前者的
    可比版本；放到最后就变成一条脚注。
    """
    from xrr_fitter.gui.results.verdict import VerdictEvidence

    widget = VerdictEvidence()
    qtbot.addWidget(widget)
    widget.set_report(_report(failure_rate=0.02))
    widget.set_objective(1.83)
    widget.set_reduced_chi_squared(1.14)

    assert widget.metrics() == (("目标值 J", "1.83"), ("约化 χ²ᵥ", "1.14"), ("自助失败率", "2%"))


def test_an_absent_reduced_chi_squared_reads_as_unavailable(qtbot) -> None:
    """没有残差数组时这一行留白会读成 χ²ᵥ=0，也就是一次完美拟合。"""
    from xrr_fitter.gui.results.verdict import VerdictEvidence

    widget = VerdictEvidence()
    qtbot.addWidget(widget)
    widget.set_report(_report())
    widget.set_reduced_chi_squared(None)

    assert widget.metrics()[1] == ("约化 χ²ᵥ", "不可用")


def test_the_reduced_chi_squared_skips_points_outside_the_fit_window() -> None:
    """候选解的 ``weighted_residuals`` 是全长数组，窗口外是 ``nan``。

    ``fit/candidates.py`` 先 ``np.full(..., np.nan)`` 再只往 ``fit_mask`` 里写，所以整条
    曲线上没参与拟合的点留着 ``nan``。直接 ``np.sum(r**2)`` 得到 ``nan``，那一行就永远是
    「不可用」——而屏幕上明明画着残差。
    """
    import numpy as np

    from xrr_fitter.gui.results.verdict import reduced_chi_squared

    residuals = np.array([np.nan, 1.0, -1.0, 2.0, 1.0, np.nan])

    # χ² = 1+1+4+1 = 7，ν = 4 − 2 = 2
    assert reduced_chi_squared(residuals, free_parameters=2) == pytest.approx(3.5)


def test_the_reduced_chi_squared_divides_by_the_degrees_of_freedom() -> None:
    """约化的意思是除以自由度 ν = N − p，不是除以点数。

    除以 N 会让自由参数越多、报出的 χ²ᵥ 越好看——正好把过拟合报成拟合得更好。
    """
    import numpy as np

    from xrr_fitter.gui.results.verdict import reduced_chi_squared

    residuals = np.full(6, np.sqrt(0.76))  # χ² = 4.56

    assert reduced_chi_squared(residuals, free_parameters=2) == pytest.approx(1.14)
    assert reduced_chi_squared(residuals, free_parameters=4) == pytest.approx(2.28)


@pytest.mark.parametrize("free", [6, 7])
def test_a_fit_without_degrees_of_freedom_has_no_reduced_chi_squared(free: int) -> None:
    """ν ≤ 0 时约化 χ² 没有定义；除下去会得到 inf 或负数。"""
    import numpy as np

    from xrr_fitter.gui.results.verdict import reduced_chi_squared

    assert reduced_chi_squared(np.ones(6), free_parameters=free) is None


def test_an_all_masked_residual_array_has_no_reduced_chi_squared() -> None:
    """一个点都没参与拟合时没有可报的 χ²ᵥ，而不是 0/0。"""
    import numpy as np

    from xrr_fitter.gui.results.verdict import reduced_chi_squared

    assert reduced_chi_squared(np.full(4, np.nan), free_parameters=2) is None
    assert reduced_chi_squared(None, free_parameters=2) is None


def test_an_absent_objective_reads_as_unavailable_rather_than_zero(qtbot) -> None:
    from xrr_fitter.gui.results.verdict import VerdictEvidence

    widget = VerdictEvidence()
    qtbot.addWidget(widget)
    widget.set_report(_report())
    widget.set_objective(None)

    assert widget.metrics()[0] == ("目标值 J", "不可用")


def test_clearing_the_evidence_hides_the_whole_block(qtbot) -> None:
    """没有结果时留一排「✓ 无边界命中」会读成一次干净的拟合。"""
    widget = _badges(qtbot, _report())
    widget.clear_report()

    assert _texts(widget) == ()
    assert widget.metrics() == ()


def test_the_metric_values_are_typeset_in_tabular_figures(qtbot) -> None:
    """kv 的 .v 是 tabular-nums；等宽数字才能让上下两行的小数点对齐。"""
    widget = _badges(qtbot, _report())
    widget.set_objective(1.83)
    widget.ensurePolished()

    for value in widget.metric_labels():
        assert value.property("mono") is True


@pytest.mark.parametrize("kind", ["ok", "info", "warn", "error"])
def test_every_status_kind_paints_the_pill_rather_than_only_the_text(kind: str) -> None:
    """胶囊要有底色和边框；只给文字上色就退回成现在的散文。"""
    for window in ("#FFFFFF", "#1E1F22"):
        sheet = theme.build_stylesheet(_palette(window))

        rule = f'QLabel[badge="true"][statusKind="{kind}"]'
        assert rule in sheet
        body = sheet.split(rule, 1)[1].split("}", 1)[0]
        assert "background" in body
        assert "border" in body


def test_a_badge_without_a_status_kind_is_painted_neutral_not_left_bare() -> None:
    """设计稿的 ``.badge.mut``：没有状态的结论走基础规则，同样是一枚胶囊。

    这条规则不能挂在第五个 statusKind 上——``set_status_kind`` 的白名单是共享的，
    只画四种状态色。所以「未运行自助」用空状态，由基础规则给中性底色。
    """
    for window, tokens in (("#FFFFFF", theme.LIGHT_TOKENS), ("#1E1F22", theme.DARK_TOKENS)):
        sheet = theme.build_stylesheet(_palette(window))

        body = sheet.split('QLabel[badge="true"] {', 1)[1].split("}", 1)[0]
        assert f"color: {tokens.muted_text}" in body
        assert f"background: {tokens.surface}" in body
        assert f"border: 1px solid {tokens.surface_border}" in body


def test_the_verdict_card_hosts_the_badges_and_metrics(qtbot) -> None:
    """帧①：判定卡里是大徽标、判读徽章、指标行三层，缺一层就退回旧版。"""
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, project

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    value = replace(project(dataset_project()), base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)

    card = panel.findChild(object, "resultConfidenceCard")
    assert panel.verdict_evidence is not None
    assert panel.verdict_evidence.parent() is card


def test_the_panel_feeds_the_chi_squared_from_the_visible_candidate(qtbot) -> None:
    """χ²ᵥ 得从屏幕上那个候选解算出来，否则这一行永远是「不可用」。

    指标行的两个数同源：J 是候选解的 ``objective``，χ²ᵥ 是同一个候选解的
    ``weighted_residuals``。喂 J 而不喂残差，等于把设计稿画着的那行留空。
    """
    from dataclasses import replace

    import numpy as np
    from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    # χ² = 4×1.2 = 4.8；``parameter_definitions`` 为空所以 ν = 4，χ²ᵥ = 1.2
    candidate = replace(fit_candidate(objective=1.83), weighted_residuals=np.sqrt(np.full(4, 1.2)))
    result = replace(final_fit_result(candidate), uncertainty=_report())
    value = replace(project(dataset_project(result=result)), base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)

    metrics = dict(panel.verdict_evidence.metrics())
    assert metrics["目标值 J"] == "1.83"
    assert metrics["约化 χ²ᵥ"] == "1.2"
