"""Placement of the MCMC sampling controls.

MCMC is an opt-in deep dive rather than part of the fit loop, so its seven
inputs must not occupy the analysis column for every project. The panel keeps
ownership of the controls, which is what lets candidate configuration and
operation state keep tracking the selection, but an on-demand dialog holds them.
"""

from __future__ import annotations

from dataclasses import replace
from math import log

import numpy as np
import pytest
from PySide6.QtCore import Qt
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

import xrr_fitter.api as api


def _uncertainty(candidate_id: str = "candidate-a") -> api.UncertaintyReport:
    return api.UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.array([[1.0]]),
        profiles=(),
        bootstrap_intervals=(("scale", 0.8, 1.2),),
        bootstrap_failure_rate=0.125,
        boundary_hits=("scale",),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _result():
    first = replace(
        fit_candidate("candidate-a", 0.2),
        ranking_objective=0.4,
        unit_vector=np.zeros(17),
    )
    second = replace(
        fit_candidate("candidate-b", 0.3),
        ranking_objective=0.8,
        unit_vector=np.zeros(3),
    )
    return replace(final_fit_result(first, second), uncertainty=_uncertainty())


def _panel(qtbot, *, expert: bool):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    value = project(dataset_project(result=_result()))
    value = replace(value, base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    if expert:
        value = api.set_expert_mode(value, True)
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    panel.show()
    return panel


def test_mcmc_controls_stay_out_of_the_panel_layout(qtbot) -> None:
    panel = _panel(qtbot, expert=True)

    assert panel.mcmc_group.isVisibleTo(panel) is False
    for control in (panel.walkers, panel.burn_in, panel.production, panel.thin):
        assert control.isVisibleTo(panel) is False


def test_uncertainty_dialog_button_is_expert_only(qtbot) -> None:
    """One entry point replaces the inline group, and only in expert mode."""
    expert = _panel(qtbot, expert=True)
    assert expert.uncertainty_button.isVisibleTo(expert) is True

    plain = _panel(qtbot, expert=False)
    assert plain.uncertainty_button.isVisibleTo(plain) is False


def test_uncertainty_dialog_hosts_the_mcmc_controls(qtbot) -> None:
    """Opening the entry point reveals the same owned controls, not copies."""
    panel = _panel(qtbot, expert=True)

    dialog = panel.open_uncertainty_dialog()
    qtbot.addWidget(dialog)

    assert panel.mcmc_group.isVisibleTo(dialog) is True
    assert panel.mcmc_group.window() is dialog
    assert panel.mcmc_config() == api.McmcConfig.standard(17)


def test_uncertainty_dialog_hosts_the_evidence_view(qtbot) -> None:
    """The dialog carries the evidence as well, or the four pages ship unreachable.

    ``window_layout`` keeps the panel's ``secondary`` container out of the
    inspector column and ``UncertaintyView`` lives in it, so a dialog holding only
    the sampling controls would offer a place to start a chain and nowhere to read
    what it said.
    """
    panel = _panel(qtbot, expert=True)
    panel.set_secondary_visible(False)

    dialog = panel.open_uncertainty_dialog()
    qtbot.addWidget(dialog)

    assert panel.uncertainty.window() is dialog
    assert panel.uncertainty.isVisibleTo(dialog) is True
    assert panel.uncertainty.pages.isVisibleTo(dialog) is True


def _mcmc_report(**changes) -> api.McmcReport:
    values = {
        "config": api.McmcConfig(walkers=4, burn_in=2, production_steps=4),
        "child_seed": 7,
        "parameter_names": ("component.0.thickness_a", "instrument.scale"),
        "samples_physical": np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]),
        "log_probability": np.zeros(4),
        "acceptance_fraction": np.array([0.2, 0.6, 0.4, 0.8]),
        "split_rhat": np.array([1.05, 1.12]),
        "effective_sample_size": np.array([120.0, 80.0]),
        "boundary_hits": (),
        "candidate_id": "candidate-a",
    }
    values.update(changes)
    return api.McmcReport(**values)


def test_report_lines_show_prior_conflicts() -> None:
    from xrr_fitter.gui.results.uncertainty import _report_lines

    report = replace(_uncertainty(), prior_conflicts=("slab1.thickness",))
    text = "\n".join(_report_lines(report))

    assert "先验冲突" in text
    assert "slab1.thickness" in text


def test_report_lines_show_no_conflict_when_empty() -> None:
    from xrr_fitter.gui.results.uncertainty import _report_lines

    report = replace(_uncertainty(), prior_conflicts=())
    conflict_line = next(line for line in _report_lines(report) if "先验冲突" in line)

    assert conflict_line.endswith("无")


def test_mcmc_lines_show_prior_conflicts() -> None:
    from xrr_fitter.gui.results.uncertainty import _mcmc_lines

    mcmc = _mcmc_report(prior_conflicts=("component.0.thickness_a",))
    report = replace(_uncertainty("candidate-a"), mcmc=mcmc)
    text = "\n".join(_mcmc_lines(report, "candidate-a"))

    assert "MCMC 先验冲突" in text
    assert "component.0.thickness_a" in text


def test_boundary_and_prior_conflict_are_distinct_lines() -> None:
    from xrr_fitter.gui.results.uncertainty import _report_lines

    report = replace(
        _uncertainty(),
        boundary_hits=("scale",),
        prior_conflicts=("slab1.thickness",),
    )
    lines = _report_lines(report)
    boundary_line = next(line for line in lines if line.startswith("边界命中"))
    conflict_line = next(line for line in lines if line.startswith("先验冲突"))

    assert boundary_line != conflict_line
    assert boundary_line.startswith("边界命中（可疑）：")
    assert conflict_line.startswith("先验冲突（信息）：")
    assert "scale" in boundary_line
    assert "slab1.thickness" in conflict_line


def test_mcmc_boundary_and_prior_conflict_explain_signal_nature() -> None:
    from xrr_fitter.gui.results.uncertainty import _mcmc_lines

    mcmc = _mcmc_report(
        boundary_hits=("instrument.scale",),
        prior_conflicts=("component.0.thickness_a",),
    )
    report = replace(_uncertainty("candidate-a"), mcmc=mcmc)
    lines = _mcmc_lines(report, "candidate-a")

    assert any(line.startswith("MCMC 边界命中（可疑）：") for line in lines)
    assert any(line.startswith("MCMC 先验冲突（信息）：") for line in lines)


def _prior_definition(
    name: str = "component.0.thickness_a",
    **changes,
) -> api.ParameterDefinition:
    values = {
        "name": name,
        "display_name": name,
        "unit": "Å",
        "category": "structure",
        "initial": 40.0,
        "lower": 10.0,
        "upper": 100.0,
        "transform": "log",
        "locked": False,
    }
    values.update(changes)
    return api.ParameterDefinition(**values)


def _prior_dialog(
    qtbot,
    definition: api.ParameterDefinition | None = None,
    existing_prior: api.PriorSpec | None = None,
):
    from xrr_fitter.gui.parameters.dialogs import PriorDialog

    dialog = PriorDialog(
        definition or _prior_definition(),
        existing_prior=existing_prior,
    )
    qtbot.addWidget(dialog)
    return dialog


def test_prior_dialog_builds_spec_from_selection(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(1.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() == api.PriorSpec("normal", (10.0, 2.0))


def test_prior_dialog_rejects_invalid_and_stays_open(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(1.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.0)  # sigma must be positive

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is None
    error = dialog.findChild(QLabel, "priorDialogError")
    assert error.isVisibleTo(dialog)


def test_prior_dialog_commit_callback_receives_validated_core_unit_spec(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox

    from xrr_fitter.gui.parameters.dialogs import PriorDialog

    committed: list[api.PriorSpec] = []
    dialog = PriorDialog(_prior_definition(), commit_spec=committed.append)
    qtbot.addWidget(dialog)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(4.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.5)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert committed == [api.PriorSpec("normal", (40.0, 5.0))]


@pytest.mark.parametrize(
    ("kind", "display_values", "persisted_values"),
    (
        ("normal", (4.0, 0.5), (40.0, 5.0)),
        ("soft_range", (2.0, 8.0, 0.5), (20.0, 80.0, 5.0)),
        ("lognormal", (log(4.0), 0.2), (log(40.0), 0.2)),
    ),
)
def test_prior_dialog_converts_length_values_from_nm_to_core_units(
    qtbot,
    kind: str,
    display_values: tuple[float, ...],
    persisted_values: tuple[float, ...],
) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText(kind)
    for index, value in enumerate(display_values):
        dialog.findChild(QDoubleSpinBox, f"priorParam{index}").setValue(value)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is not None
    assert dialog.spec().parameters == pytest.approx(persisted_values)


@pytest.mark.parametrize(
    ("existing", "display_values"),
    (
        (api.PriorSpec("normal", (40.0, 5.0)), (4.0, 0.5)),
        (api.PriorSpec("soft_range", (20.0, 80.0, 5.0)), (2.0, 8.0, 0.5)),
        (api.PriorSpec("lognormal", (log(40.0), 0.2)), (log(4.0), 0.2)),
    ),
)
def test_prior_dialog_prefills_existing_length_prior_in_display_units(
    qtbot,
    existing: api.PriorSpec,
    display_values: tuple[float, ...],
) -> None:
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

    dialog = _prior_dialog(qtbot, existing_prior=existing)

    assert dialog.findChild(QComboBox, "priorKindSelect").currentText() == existing.kind
    actual = tuple(
        dialog.findChild(QDoubleSpinBox, f"priorParam{index}").value() for index in range(len(display_values))
    )
    assert actual == pytest.approx(display_values)


def test_prior_dialog_roughness_fraction_uses_unscaled_fraction_fields(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    definition = _prior_definition(
        "component.0.roughness_a",
        initial=3.0,
        lower=0.0,
        upper=50.0,
        transform="roughness_fraction",
    )
    existing = api.PriorSpec("normal", (0.5, 0.1))
    dialog = _prior_dialog(qtbot, definition, existing)

    assert dialog.findChild(QDoubleSpinBox, "priorParam0").value() == pytest.approx(0.5)
    assert "分数" in dialog.findChild(QLabel, "priorParamLabel0").text()
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(0.6)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() == api.PriorSpec("normal", (0.6, 0.2))


def test_prior_dialog_validates_converted_spec_against_definition(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    dialog = _prior_dialog(qtbot)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("normal")
    # 11 nm becomes 110 Å, beyond the declaration's 100 Å upper bound.
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(11.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is None
    error = dialog.findChild(QLabel, "priorDialogError")
    assert error.isVisibleTo(dialog)
    assert "within bounds" in error.text()


def test_prior_dialog_rejects_lognormal_for_zero_lower_bound(qtbot) -> None:
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QDoubleSpinBox, QLabel

    definition = _prior_definition(
        "instrument.scale",
        unit="1",
        initial=1.0,
        lower=0.0,
        upper=2.0,
        transform="linear",
    )
    dialog = _prior_dialog(qtbot, definition)
    dialog.findChild(QComboBox, "priorKindSelect").setCurrentText("lognormal")
    dialog.findChild(QDoubleSpinBox, "priorParam0").setValue(0.0)
    dialog.findChild(QDoubleSpinBox, "priorParam1").setValue(0.2)

    buttons = dialog.findChild(QDialogButtonBox, "priorDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.spec() is None
    assert "positive lower bound" in dialog.findChild(QLabel, "priorDialogError").text()


def test_parameter_panel_opens_prior_dialog_with_definition_and_existing_prior(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters import dialogs
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    source = tmp_path / "sample.xy"
    source.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    value = api.add_dataset(api.new_project(), source, api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.2), 40.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    value = api.set_structure(value, "sample", structure)
    name = "component.0.thickness_a"
    existing = api.PriorSpec("normal", (40.0, 5.0))
    value = api.set_parameter_priors(value, "sample", (api.ParameterPrior(name, existing),))
    panel = ParametersPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    captured: list[tuple[object, object]] = []

    class StubDialog:
        def __init__(self, definition, parent, *, existing_prior):
            captured.append((definition, existing_prior))

        def exec(self):
            return 0

    monkeypatch.setattr(dialogs, "PriorDialog", StubDialog)

    panel._edit_prior_row(name)

    definition, prior = captured[0]
    assert definition.name == name
    assert prior == existing


# -- G21 + G14: the four evidence pages --


def _bands() -> api.SldUncertaintyBands:
    """A degenerate band whose five faces are five constant levels.

    A band test only needs the faces to be tellable apart on screen; keeping each
    quantile flat means an assertion about which face was drawn does not also have
    to restate a depth-dependent profile.
    """
    depth = np.linspace(0.0, 40.0, 4)
    levels = (0.025, 0.16, 0.5, 0.84, 0.975)
    real = np.tile(np.arange(len(levels), dtype=float)[:, None], (1, depth.size))
    return api.SldUncertaintyBands(
        depth_a=depth,
        quantiles=levels,
        real=real,
        imaginary=real * 0.5,
        align_label="基底界面",
        sample_count=500,
        total_samples=2000,
        failure_rate=0.0,
    )


def _full_report(candidate_id: str = "candidate-a") -> api.UncertaintyReport:
    """Owned evidence of all four kinds, so every page has something to draw."""
    profile = api.ParameterProfile(
        name="component.0.thickness_a",
        values=np.array([30.0, 40.0, 50.0]),
        objectives=np.array([0.30, 0.20, 0.32]),
        lower_closed=True,
        upper_closed=True,
    )
    return replace(
        _uncertainty(candidate_id),
        correlation_names=("component.0.thickness_a", "instrument.scale"),
        correlation_matrix=np.array([[1.0, -0.65], [-0.65, 1.0]]),
        profiles=(profile,),
        sld_bands=_bands(),
        mcmc=_mcmc_report(candidate_id=candidate_id),
    )


def _view(qtbot, report: api.UncertaintyReport | None = None):
    """An ``UncertaintyView`` holding one candidate's evidence, already shown.

    The pages draw on ``set_result``, so a case that reads artists needs the view
    realized: a matplotlib canvas that never became visible defers its draw.
    """
    from xrr_fitter.gui.results.uncertainty import UncertaintyView

    view = UncertaintyView()
    qtbot.addWidget(view)
    view.show()
    if report is not None:
        result = replace(_result(), uncertainty=report)
        view.set_result(result, "candidate-a")
    return view


def test_uncertainty_view_carries_the_four_evidence_pages(qtbot) -> None:
    """四种不确定度证据各占一页，读者不必离开这张卡去别处凑齐判读。

    这张卡原先只有一个 ``QPlainTextEdit``：相关矩阵和可信带画在诊断 tab 组里，
    Profile 和 MCMC 只剩几行文字，于是「这个厚度到底定没定住」要在两处之间来回对。
    """
    from xrr_fitter.gui.plots.posterior import UNCERTAINTY_PAGE_TITLES

    view = _view(qtbot)
    pages = view.pages

    titles = tuple(pages.tabText(index) for index in range(pages.count()))
    assert titles == UNCERTAINTY_PAGE_TITLES
    # 逐字取设计稿帧⑤ 的 ``.canvas-top``（HTML L911）。第二页叫「Profile 似然」而不是
    # 「参数剖面」：设计稿八处都用前者（帧⑤ 标题、左栏子管线那一步、画布内标题、能力对照
    # 表），G22 的方法表 ``navigation/methods.py`` 也是，标签写成别的名字会让同一份证据在
    # 导航里和画布里叫两个名。
    assert UNCERTAINTY_PAGE_TITLES == ("相关矩阵", "Profile 似然", "SLD 可信带", "MCMC 后验")


def test_the_evidence_text_stays_in_the_card_beside_the_pages(qtbot) -> None:
    """加了四页图不等于把那段文字挤走：边界命中、失败率、归属都只在文字里说得清。"""
    from PySide6.QtWidgets import QFrame, QPlainTextEdit

    view = _view(qtbot, _full_report())

    card = view.findChild(QFrame, "uncertaintyCard")
    assert card.findChild(QPlainTextEdit, "uncertaintyEvidence") is not None
    assert card.findChild(type(view.pages), "uncertaintyPages") is not None
    assert "边界命中（可疑）：scale" in view.text()


def test_each_page_draws_the_owned_evidence(qtbot) -> None:
    """四页各自画出自己那份证据：矩阵有图像、剖面有曲线、可信带有填充、后验有直方。"""
    view = _view(qtbot, _full_report())

    correlation, profile, bands, posterior = view.page_figures()

    assert any(axes.images for axes in correlation.axes), "相关矩阵没有画出图像"
    assert any(axes.lines for axes in profile.axes), "剖面页没有画出曲线"
    assert any(axes.collections for axes in bands.axes), "可信带页没有填充区"
    assert any(axes.patches for axes in posterior.axes), "后验页没有直方"


def test_the_profile_page_marks_where_the_interval_closed(qtbot) -> None:
    """闭合阈值画成参考线，曲线才读得出区间在哪里收口。

    ``lower_closed``/``upper_closed`` 是把目标函数同一个数比出来的；只画曲线不画那条
    线，读者手里就只剩两个布尔值和一条看不出交点的曲线。这条线标「区间闭合阈值」而
    不标 Δχ²=1，因为 ``robust_log_cost`` 下后者没有统计含义。

    线画在阈值减掉这条曲线自己最优值的位置：纵轴是相对最优的增量（那件事归
    ``tests/gui/test_profile_likelihood_readings.py`` 钉），照搬 profile 上发布的那个绝对目标值
    会把线画到画面外。减完之后仍是扫描做过的那次比较，只是换成了同一个基准。
    """
    report = _full_report()
    threshold = 0.25
    profile = replace(report.profiles[0], objective_threshold=threshold)
    view = _view(qtbot, replace(report, profiles=(profile,)))

    figure = view.page_figures()[1]

    increment = threshold - float(np.min(profile.objectives))
    axes = next(item for item in figure.axes if item.lines)
    reference = tuple(line for line in axes.lines if np.allclose(np.asarray(line.get_ydata(), dtype=float), increment))
    assert len(reference) == 1, "剖面页没有画出闭合阈值参考线"
    legend = axes.get_legend()
    assert legend is not None
    assert any("区间闭合阈值" in text.get_text() for text in legend.get_texts())


def test_the_profile_page_omits_the_reference_line_without_a_threshold(qtbot) -> None:
    """没有阈值就不画线：凭空补一条参考线等于替扫描编一个它没算过的数。"""
    report = _full_report()
    view = _view(qtbot, report)

    figure = view.page_figures()[1]

    assert report.profiles[0].objective_threshold is None
    axes = next(item for item in figure.axes if item.lines)
    assert len(axes.lines) == len(report.profiles)


def test_the_pages_refuse_evidence_owned_by_another_candidate(qtbot) -> None:
    """归属别的候选就四页全空：拿 candidate-b 的证据标在 candidate-a 上就是伪造。"""
    view = _view(qtbot)
    result = replace(_result(), uncertainty=_full_report("candidate-b"))
    view.set_result(result, "candidate-a")

    for figure in view.page_figures():
        assert not any(axes.images for axes in figure.axes)
        assert not any(axes.patches for axes in figure.axes)
        assert not any(axes.collections for axes in figure.axes)


def test_the_posterior_page_gives_every_parameter_its_own_panel(qtbot) -> None:
    """每个采样参数一格，且每格三条分位线（P16/P50/P84）。

    把两个参数叠在一格里，两条边缘分布的宽度就没法比；而只画中位数的话，
    「定住了」和「先验宽度原样返回」在屏幕上看着一样。
    """
    report = _full_report()
    view = _view(qtbot, report)
    figure = view.page_figures()[3]

    panels = tuple(axes for axes in figure.axes if axes.patches)
    assert len(panels) == len(report.mcmc.parameter_names)

    samples = np.asarray(report.mcmc.samples_physical, dtype=float)
    for index, axes in enumerate(panels):
        expected = np.quantile(samples[:, index], (0.16, 0.5, 0.84))
        drawn = sorted(float(line.get_xdata()[0]) for line in axes.lines)
        assert np.allclose(drawn, expected), axes.get_title()


def test_the_posterior_page_says_so_when_no_sampling_was_run(qtbot) -> None:
    """没跑过 MCMC 的报告，后验页要写明「没跑」而不是留一张空白坐标系。"""
    view = _view(qtbot, replace(_full_report(), mcmc=None))
    figure = view.page_figures()[3]

    texts = tuple(text.get_text() for axes in figure.axes for text in axes.texts)
    assert any("未运行" in text for text in texts), texts


def test_clearing_the_result_also_clears_the_pages(qtbot) -> None:
    """清结果时四页一起清：留着上一次拟合的矩阵会被读成这一次的证据。"""
    view = _view(qtbot, _full_report())
    view.clear_result("尚无拟合结果")

    for figure in view.page_figures():
        assert not any(axes.images for axes in figure.axes)
        assert not any(axes.patches for axes in figure.axes)


def test_the_inspector_states_how_many_resamples_the_bootstrap_asked_for(qtbot) -> None:
    """设计稿帧⑤ 右栏「自助抽样」段的第一个读数是「重采样次数 200」，失败率排在它后面。

    顺序不是排版偏好：失败率是个比例，先给基数才读得出到底丢了几次。这一段此前只报比例。
    """
    view = _view(qtbot, replace(_full_report(), bootstrap_sample_count=200))

    text = view.text()
    lines = text.splitlines()

    assert "Bootstrap 重采样次数：200" in text
    count_at = next(index for index, line in enumerate(lines) if "重采样次数" in line)
    rate_at = next(index for index, line in enumerate(lines) if "失败率" in line)
    assert count_at < rate_at


def test_the_inspector_says_the_resample_count_is_unrecorded_when_it_is(qtbot) -> None:
    """这个字段加进来之前存的工程文件不带次数，此时报「未记录」——报 0 会被读成一次都没抽。"""
    view = _view(qtbot, _full_report())

    assert "Bootstrap 重采样次数：未记录" in view.text()
