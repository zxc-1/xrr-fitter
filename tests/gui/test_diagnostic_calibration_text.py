"""Chinese UI preserves raw warnings without treating nonrejection as proof."""

from __future__ import annotations

from dataclasses import replace

from tests.gui.test_inference_evidence import _report, _view
from tests.unit.io.test_diagnostic_calibration_codec import _calibration, _residual
from tests.unit.io.test_export_v2_evidence import evidence_context

from xrr_fitter.model.inference import ResidualEvidence
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _calibrated_text(qtbot, calibration):
    member = _residual(calibration)
    report = _report(
        member_residuals=(member,),
        systematic_residual=member.systematic,
        residual_autocorrelation=member.autocorrelation,
    )
    return _view(qtbot, report, mode="poisson").text()


def test_nonrejection_is_not_rendered_as_model_correctness(qtbot):
    text = _calibrated_text(qtbot, _calibration())
    expected = (
        "校准未拒绝",
        "原始筛查",
        "suspected_diffuse_background",
        "raw tail screen",
        "poisson_refit_null_rms_v4",
        "declared_sobol4_lbfgsb_trf_v3",
        "999/999",
        "p=0.2",
    )
    missing = tuple(fragment for fragment in expected if fragment not in text)
    assert not missing
    assert "模型正确" not in text


def test_rejection_keeps_the_raw_and_calibrated_decision_distinct(qtbot):
    text = _calibrated_text(qtbot, _calibration(p_value=0.008))
    assert "校准拒绝" in text
    assert "p=0.008" in text
    assert "原始筛查" in text
    assert "系统性残差：是" in text


def test_unavailable_calibration_does_not_display_a_negative_diagnostic(qtbot):
    text = _calibrated_text(qtbot, _calibration("unavailable"))
    assert "校准不可用" in text
    assert "diagnostic_refit_failed" in text
    assert "原始筛查" in text
    assert "2/3" in text
    assert "999" in text
    assert "p=" not in text
    assert "无系统误差" not in text
    assert "系统性残差：否" not in text


def test_no_raw_trigger_is_explicitly_not_a_calibration(qtbot):
    member = ResidualEvidence(
        "curve",
        True,
        False,
        False,
        32,
        raw_systematic=False,
        raw_autocorrelation=False,
        owner_sha256="a" * 64,
    )
    report = _report(member_residuals=(member,), systematic_residual=False, residual_autocorrelation=False)
    text = _view(qtbot, report, mode="poisson").text()
    assert "未触发校准" in text
    assert "校准未拒绝" not in text


def test_missing_calibration_is_not_claimed_to_have_passed(qtbot):
    member = ResidualEvidence(
        "curve",
        False,
        None,
        None,
        32,
        unavailable_reason="poisson_diagnostic_calibration_required",
        raw_systematic=True,
        raw_autocorrelation=False,
    )
    text = _view(qtbot, _report(member_residuals=(member,)), mode="poisson").text()
    assert "校准未执行" in text
    assert "poisson_diagnostic_calibration_required" in text
    assert "校准未拒绝" not in text


def test_bootstrap_exposes_diagnostic_reason_even_when_sample_gate_has_priority(qtbot):
    original = evidence_context().selected_uncertainty.bootstrap_evidence
    bootstrap = replace(original, diagnostic_unavailable_reason="diagnostic_refit_failed")
    report = _report(
        correlation_names=bootstrap.parameter_names,
        correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        bootstrap_evidence=bootstrap,
        bootstrap_performed=True,
    )
    text = _view(qtbot, report, mode="poisson").text()
    assert "insufficient_successful_samples" in text
    assert "diagnostic_refit_failed" in text
    assert "95%" not in text


def test_saved_advisory_remains_visible_when_raw_flags_are_unknown(qtbot):
    member = ResidualEvidence(
        "curve",
        False,
        None,
        None,
        32,
        unavailable_reason="nonfinite_residuals",
        advisories=(PhysicsDiagnostic("suspected_diffuse_background", "retained original screen"),),
    )
    text = _view(qtbot, _report(member_residuals=(member,)), mode="poisson").text()
    assert "suspected_diffuse_background" in text
    assert "retained original screen" in text
    assert "nonfinite_residuals" in text
