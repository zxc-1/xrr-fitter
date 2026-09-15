"""GUI text must not turn unavailable or exploratory evidence into confidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.bootstrap_cases import bootstrap_evidence
from tests.support.model_cases import final_fit_result, fit_candidate

import xrr_fitter.api as api
from xrr_fitter.gui.results.uncertainty import UncertaintyView
from xrr_fitter.model.bootstrap import BootstrapResult
from xrr_fitter.model.inference import CovarianceEvidence, ResidualEvidence


def _report(**changes):
    values = dict(
        correlation_names=("scale",),
        correlation_matrix=np.ones((1, 1)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=None,
        diagnostics=(),
        candidate_id="candidate-a",
    )
    values.update(changes)
    return api.UncertaintyReport(**values)


def _view(qtbot, report, *, mode="robust_log"):
    candidate = replace(fit_candidate("candidate-a"), noise_model=mode)
    result = replace(final_fit_result(candidate), uncertainty=report)
    view = UncertaintyView()
    qtbot.addWidget(view)
    view.set_result(result, "candidate-a")
    return view


def _profile(**changes):
    values = dict(
        name="scale",
        values=np.array([0.8, 1.0, 1.2]),
        objectives=np.array([0.3, 0.1, 0.4]),
        lower_closed=True,
        upper_closed=True,
    )
    values.update(changes)
    return api.ParameterProfile(**values)


@pytest.mark.parametrize(("value", "label"), ((None, "未执行/不可用"), (False, "否"), (True, "是")))
def test_diagnostic_unknown_is_not_displayed_as_a_negative_conclusion(qtbot, value, label):
    report = _report(systematic_residual=value, residual_autocorrelation=value)
    text = _view(qtbot, report).text()
    assert f"系统性残差：{label}" in text
    assert f"残差 ACF：{label}" in text


def test_unexecuted_member_diagnostics_keep_the_dataset_and_reason(qtbot):
    evidence = ResidualEvidence("short-curve", False, None, None, 3, unavailable_reason="insufficient_points")
    text = _view(qtbot, _report(member_residuals=(evidence,))).text()
    assert "short-curve" in text
    assert "未执行" in text
    assert "insufficient_points" in text
    assert "3" in text


def test_unexecuted_bootstrap_does_not_display_a_measured_zero_failure_rate(qtbot):
    text = _view(qtbot, _report()).text()
    assert "Bootstrap：未执行" in text
    assert "Bootstrap 失败率：0" not in text


def test_loss_support_profile_shows_its_actual_method_without_nominal_coverage(qtbot):
    profile = _profile(method="objective_tolerance", delta_total=0.6, objective_point_count=40)
    text = _view(qtbot, _report(profiles=(profile,))).text()
    assert "loss_support" in text
    assert "objective_tolerance" in text
    assert "探索" in text
    assert "95%" not in text


def test_likelihood_profile_displays_its_declared_confidence_and_method(qtbot):
    profile = _profile(
        interval_kind="likelihood_ratio",
        confidence_level=0.95,
        method="chi_square_1df",
        delta_total=3.841458820694124,
        objective_point_count=40,
    )
    text = _view(qtbot, _report(profiles=(profile,)), mode="gaussian").text()
    assert "likelihood_ratio" in text
    assert "chi_square_1df" in text
    assert "置信水平：95%" in text


def test_unavailable_profile_exposes_the_regular_condition_failure(qtbot):
    profile = _profile(interval_kind="unavailable", method="likelihood_ratio", unavailable_reason="rank_deficient")
    text = _view(qtbot, _report(profiles=(profile,)), mode="poisson").text()
    assert "unavailable" in text
    assert "rank_deficient" in text
    assert "95%" not in text


def test_fast_bootstrap_displays_sampling_evidence_as_exploratory(qtbot):
    bootstrap = BootstrapResult(
        ("scale",),
        np.ones((8, 1)),
        (),
        0.0,
        8,
        method="wild_block_residual",
        unavailable_reason="insufficient_successful_samples",
    )
    report = _report(bootstrap_performed=True, bootstrap_evidence=bootstrap)
    text = _view(qtbot, report).text()
    assert "exploratory_bootstrap" in text
    assert "wild_block_residual" in text
    assert "insufficient_successful_samples" in text
    assert "8/8" in text
    assert "95%" not in text


def test_calibrated_bootstrap_projects_saved_interval_metadata(qtbot):
    intervals = (("scale", 0.8, 1.2),)
    evidence = replace(bootstrap_evidence(intervals), method="parametric_gaussian")
    report = _report(bootstrap_performed=True, bootstrap_intervals=intervals, bootstrap_evidence=evidence)
    text = _view(qtbot, report, mode="gaussian").text()
    assert "percentile_bootstrap" in text
    assert "parametric_gaussian" in text
    assert "置信水平：95%" in text
    assert "[0.8, 1.2]" in text


def test_unavailable_covariance_projects_method_and_reason_not_zero_sigma(qtbot):
    evidence = CovarianceEvidence(("scale",), None, "gaussian_information", 0, ("scale",), "rank_deficient")
    text = _view(qtbot, _report(covariance_evidence=evidence), mode="gaussian").text()
    assert "gaussian_information" in text
    assert "rank_deficient" in text
    assert "不可用" in text


@pytest.mark.parametrize(
    ("mode", "label", "residual", "unit"),
    (
        ("robust_log", "稳健对数（探索）", "log_reflectivity", "decade"),
        ("gaussian", "Gaussian（已知标准差）", "standardized_intensity", "1"),
        ("poisson", "Poisson（原始整数计数）", "signed_poisson_deviance", "1"),
    ),
)
def test_result_mode_and_residual_units_come_from_the_selected_candidate(qtbot, mode, label, residual, unit):
    text = _view(qtbot, _report(), mode=mode).text()
    assert f"结果模式：{label}" in text
    assert f"{residual} / {unit}" in text
