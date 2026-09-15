"""Current shell layouts consume v2 evidence without inventing statistical claims."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel
from tests.gui.plot_support import _candidate, _panel
from tests.support.model_cases import dataset_project, final_fit_result, prepared_data, project

import xrr_fitter.api as api
from xrr_fitter.gui.data.import_dialog import ImportDialog
from xrr_fitter.gui.plots.diagnostics import residual_card_subtitle
from xrr_fitter.gui.plots.reflectivity import _prepared_dataset


def _counts_source(tmp_path):
    source = tmp_path / "counts.xy"
    source.write_text("\n".join(f"{0.05 + i * 0.03} {i}" for i in range(40)) + "\n", encoding="utf-8")
    return source


def test_batch_preview_retains_theta_and_poisson_together(qtbot, tmp_path, monkeypatch):
    observed = []
    import_data = api.import_data

    def capture(*args, **kwargs):
        data = import_data(*args, **kwargs)
        observed.append(data)
        return data

    monkeypatch.setattr(api, "import_data", capture)
    dialog = ImportDialog((_counts_source(tmp_path),), noise_model="poisson")
    qtbot.addWidget(dialog)
    dialog.select_beam_kind("monochromatic")
    dialog.theta_convention.setChecked(True)

    assert observed, "Preview must use the supported API with both declarations"
    np.testing.assert_allclose(observed[-1].two_theta_deg[:3], [0.1, 0.16, 0.22])
    np.testing.assert_array_equal(observed[-1].intensity_raw[:3], [0, 1, 2])
    assert np.all(observed[-1].validation_mask[:3])
    assert dialog.preview_table.item(0, 1).text() == "θ"
    assert dialog.import_button().isEnabled()


def test_project_plot_reread_keeps_both_source_declarations(tmp_path):
    project = api.set_fit_config(api.new_project(), replace(api.FitConfig.fast(7), noise_model="poisson"))
    project = api.add_dataset(project, _counts_source(tmp_path), api.InstrumentSpec(), angle_convention="theta")

    prepared = _prepared_dataset(project, project.datasets[0])

    assert prepared is not None, "Positional noise must not become an angle convention"
    data, mask = prepared
    np.testing.assert_allclose(data.two_theta_deg[:3], [0.1, 0.16, 0.22])
    np.testing.assert_array_equal(data.intensity_raw[:3], [0, 1, 2])
    assert np.all(mask[:3])


@pytest.mark.parametrize("mode", ("robust_log", "poisson"))
def test_non_gaussian_residual_card_does_not_claim_reduced_chi_square(mode):
    candidate = _candidate(prepared_data(size=24), noise_model=mode)
    result = final_fit_result(candidate)

    text = residual_card_subtitle(result, candidate)

    assert "χ²" not in text
    assert mode in text
    assert candidate.residual_unit in text


@pytest.mark.parametrize("systematic", (None, False, True))
def test_residual_card_preserves_unavailable_diagnostic_state(systematic):
    candidate = _candidate(prepared_data(size=24), noise_model="gaussian")
    report = api.UncertaintyReport(
        (), np.empty((0, 0)), (), (), 0.0, (), (), systematic, (), candidate_id=candidate.candidate_id
    )
    result = replace(final_fit_result(candidate), uncertainty=report)

    text = residual_card_subtitle(result, candidate)

    expected = {None: "系统性残差：未执行/不可用", False: "无系统性结构", True: "检出系统性结构"}
    assert expected[systematic] in text
    if systematic is None:
        assert "无系统性结构" not in text


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_only_standardized_gaussian_residuals_show_a_sigma_band(qtbot, mode):
    data = prepared_data(size=24)
    candidate = _candidate(data, noise_model=mode)
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))
    view = panel.view("residual")

    assert view.sigma_label_item.isVisible() is (mode == "gaussian")
    legend = panel.findChild(object, "plotCard:residualLegend")
    captions = tuple(label.text() for label in legend.findChildren(QLabel) if label.isVisibleTo(legend))
    assert any("±1σ" in caption for caption in captions) is (mode == "gaussian")


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_verdict_metric_does_not_relabel_a_non_gaussian_loss_as_chi_square(qtbot, tmp_path, mode):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    candidate = _candidate(prepared_data(size=4), noise_model=mode)
    report = api.UncertaintyReport(
        (), np.empty((0, 0)), (), (), 0.0, (), (), None, (), candidate_id=candidate.candidate_id
    )
    result = replace(final_fit_result(candidate), uncertainty=report)
    value = api.select_active_dataset(
        replace(project(dataset_project(result=result)), base_directory=str(tmp_path)), "curve"
    )
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)

    metric = dict(panel.verdict_evidence.metrics())["约化 χ²ᵥ"]

    assert (metric != "不可用") is (mode == "gaussian")


def test_generic_profile_navigation_does_not_claim_likelihood_calibration():
    from xrr_fitter.gui.navigation.methods import UNCERTAINTY_METHODS
    from xrr_fitter.gui.plots.posterior import UNCERTAINTY_PAGE_TITLES
    from xrr_fitter.gui.plots.sld import PROFILE_TITLE

    captions = (UNCERTAINTY_METHODS[2][0], UNCERTAINTY_PAGE_TITLES[1], PROFILE_TITLE)

    assert all("似然" not in caption for caption in captions)
    assert all("剖面" in caption for caption in captions)


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_status_bar_retains_noise_units_without_inventing_chi_square(mode):
    from xrr_fitter.gui.chrome import _fit_metrics_text

    candidate = _candidate(prepared_data(size=4), noise_model=mode)
    result = final_fit_result(candidate)
    window = SimpleNamespace(
        document=SimpleNamespace(project=project(dataset_project(result=result)), active_dataset_id="curve")
    )

    text = _fit_metrics_text(window)

    assert f"J = {candidate.objective:g}" in text
    assert mode in text
    assert f"[{candidate.residual_unit}]" in text
    assert ("χ²" in text) is (mode == "gaussian")


def _evidence_window(qtbot, tmp_path, expert):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    candidate = _candidate(prepared_data(size=4))
    report = api.UncertaintyReport(
        ("scale",), np.ones((1, 1)), (), (), 0.0, (), (), None, (), candidate_id=candidate.candidate_id
    )
    result = replace(final_fit_result(candidate), uncertainty=report)
    value = replace(project(dataset_project(result=result)), base_directory=str(tmp_path))
    value = api.set_expert_mode(api.select_active_dataset(value, "curve"), expert)
    window = MainWindow(ProjectDocument(value))
    qtbot.addWidget(window)
    return window


def _open_read_only_evidence(qtbot, window):
    action = window.chrome_actions.get("viewInferenceEvidenceAction")
    assert action is not None, "Ordinary mode must retain a real read-only route to the full evidence"
    assert action.isVisible() and action.isEnabled()
    action.trigger()
    dialog = window.result_panel._uncertainty_dialog
    qtbot.addWidget(dialog)
    return action, dialog


@pytest.mark.parametrize("expert", (False, True))
def test_read_only_evidence_action_retains_access_without_enabling_sampling(qtbot, tmp_path, expert):
    from xrr_fitter.gui.results.uncertainty import UncertaintyView

    window = _evidence_window(qtbot, tmp_path, expert)
    original = window.document.project
    _action, dialog = _open_read_only_evidence(qtbot, window)
    panel = window.result_panel
    assert dialog.isVisible()
    assert dialog.findChild(UncertaintyView) is panel.uncertainty
    assert "稳健对数（探索）" in panel.uncertainty.text()
    assert "系统性残差：未执行/不可用" in panel.uncertainty.text()
    assert not panel.mcmc_group.isVisibleTo(dialog)
    assert not panel.controller.is_running
    assert window.document.project is original
    assert window.chrome_actions["openUncertaintyAction"].isVisible() is expert


def test_shared_evidence_dialog_refreshes_title_and_sampling_controls(qtbot, tmp_path):
    window = _evidence_window(qtbot, tmp_path, True)
    original = window.document.project
    action, dialog = _open_read_only_evidence(qtbot, window)
    panel = window.result_panel
    assert dialog.windowTitle() == "推断证据"
    window.chrome_actions["openUncertaintyAction"].trigger()
    assert panel._uncertainty_dialog is dialog
    assert panel.mcmc_group.isVisibleTo(dialog)
    assert dialog.windowTitle() == "不确定度分析"
    action.trigger()
    assert not panel.mcmc_group.isVisibleTo(dialog)
    assert dialog.windowTitle() == "推断证据"
    assert window.document.project is original


def test_profile_tooltip_points_to_the_actual_read_only_evidence_menu(qtbot, tmp_path):
    from xrr_fitter.gui.navigation.methods import UncertaintyMethodNav

    window = _evidence_window(qtbot, tmp_path, False)
    nav = window.findChild(UncertaintyMethodNav)
    pointer = nav.labels[2].toolTip()

    assert "拟合" in pointer
    assert window.chrome_actions["viewInferenceEvidenceAction"].text() in pointer


def test_empty_correlation_evidence_is_not_drawn_as_a_singular_matrix(qtbot):
    from xrr_fitter.gui.results.uncertainty import UncertaintyView

    candidate = _candidate(prepared_data(size=4))
    report = api.UncertaintyReport(
        (), np.empty((0, 0)), (), (), 0.0, (), (), None, (), candidate_id=candidate.candidate_id
    )
    view = UncertaintyView()
    qtbot.addWidget(view)
    view.set_result(replace(final_fit_result(candidate), uncertainty=report), candidate.candidate_id)

    figure = view.page_figures()[0]

    assert not any(axes.images for axes in figure.axes)
    assert any("不可用" in note.get_text() for axes in figure.axes for note in axes.texts)
