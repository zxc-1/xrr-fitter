"""Current covariance and sampling evidence stays honest on every GUI surface."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from tests.support.bootstrap_cases import bootstrap_evidence
from tests.support.model_cases import final_fit_result, fit_candidate

import xrr_fitter.api as api
from xrr_fitter.analysis.covariance import covariance_summary
from xrr_fitter.gui.navigation.methods import uncertainty_method_rows
from xrr_fitter.gui.plots.correlation import _draw_correlation, _ranked_pairs
from xrr_fitter.gui.plots.diagnostics import apply_figure_font
from xrr_fitter.gui.results.inspector import bootstrap_readings, correlation_note
from xrr_fitter.gui.results.uncertainty import UncertaintyView, _report_lines
from xrr_fitter.gui.results.verdict import FAILURE_RATE_LABEL, VerdictEvidence
from xrr_fitter.model.inference import CovarianceEvidence

NAMES = ("scale", "thickness")


def _result(report):
    return replace(final_fit_result(fit_candidate("candidate-a")), uncertainty=report)


def _report(*, covariance=None, bootstrap=None):
    if covariance is None:
        covariance = CovarianceEvidence(NAMES, np.eye(2), "gaussian_known_sigma", 2)
    correlation, sigma = covariance_summary(covariance)
    return api.UncertaintyReport(
        covariance.names,
        correlation,
        (),
        () if bootstrap is None else bootstrap.intervals,
        0.0 if bootstrap is None else bootstrap.failure_rate,
        (),
        (),
        None,
        (),
        candidate_id="candidate-a",
        covariance_evidence=covariance,
        parameter_sigma=sigma,
        bootstrap_performed=bootstrap is not None,
        bootstrap_evidence=bootstrap,
    )


@pytest.mark.parametrize(
    ("rate", "attempts", "metric"),
    ((None, 0, "不可用"), (0.0, 200, "0%"), (1.0, 7, "100%")),
    ids=("absent", "executed-zero-failures", "all-attempts-failed"),
)
def test_bootstrap_metric_distinguishes_absent_zero_and_failed_attempts(qtbot, rate, attempts, metric):
    evidence = (
        None
        if rate is None
        else bootstrap_evidence(tuple((name, 0.8, 1.2) for name in NAMES), failure_rate=rate, attempted_count=attempts)
    )
    report = _report(bootstrap=evidence)
    widget = VerdictEvidence()
    qtbot.addWidget(widget)
    widget.set_report(report)

    assert dict(widget.metrics())[FAILURE_RATE_LABEL] == metric
    assert report.bootstrap_sample_count == attempts
    readings = bootstrap_readings(report)
    if evidence is None:
        assert readings[:2] == (("未运行", ""), ("不可用", ""))
    else:
        assert readings[0][0] == str(attempts)
        assert f"（{len(evidence.failure_reasons)}/{attempts}）" in readings[1][0]


@pytest.fixture(params=("rank-deficient", "unavailable-finite", "missing", "nonfinite", "empty"))
def withheld_covariance(request):
    case = request.param
    if case in ("rank-deficient", "unavailable-finite"):
        covariance = CovarianceEvidence(NAMES, None, "gaussian_known_sigma", 1, NAMES, "rank_deficient")
        report = _report(covariance=covariance)
        if case == "unavailable-finite":
            report = replace(report, correlation_matrix=np.eye(2))
        return report, "rank_deficient"
    if case == "empty":
        covariance = CovarianceEvidence((), np.empty((0, 0)), "gaussian_known_sigma", 0)
        return _report(covariance=covariance), "参数轴为空"
    matrix = np.full((2, 2), np.nan) if case == "nonfinite" else np.eye(2)
    return replace(_report(), covariance_evidence=None, correlation_matrix=matrix), "缺少协方差证据"


def _draw_matrix(report, *, wide_layout):
    figure = Figure(figsize=(8, 6))
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    _draw_correlation(figure, axes, report, wide_layout=wide_layout)
    apply_figure_font(figure)
    canvas.draw()
    return figure, axes


@pytest.mark.parametrize("wide_layout", (True, False))
def test_withheld_covariance_has_no_matrix_numbers_colour_key_or_ranked_claim(withheld_covariance, wide_layout):
    report, reason = withheld_covariance
    figure, axes = _draw_matrix(report, wide_layout=wide_layout)

    assert not axes.images
    assert len(figure.axes) == 1
    assert not axes.child_axes
    text = "\n".join(item.get_text() for item in figure.findobj(Text))
    assert "不可用" in text
    assert reason in text
    assert not any(claim in text for claim in ("+nan", "强正相关", "强负相关", "无强相关"))
    assert not axes.get_xticks().size
    assert not axes.get_yticks().size


def test_withheld_covariance_is_not_a_completed_navigation_method(withheld_covariance):
    report, reason = withheld_covariance
    row = uncertainty_method_rows(report)[0]

    assert row.state != "done"
    assert row.glyph != "✓"
    assert "不可用" in row.caption
    assert reason in row.caption
    assert "Hessian" not in row.caption


def test_withheld_covariance_has_a_neutral_verdict_with_an_explanatory_tooltip(qtbot, withheld_covariance):
    report, reason = withheld_covariance
    widget = VerdictEvidence()
    qtbot.addWidget(widget)
    widget.set_report(report)
    badge = widget.badges()[2]

    assert "不可用" in badge.text()
    assert "无强相关" not in badge.text()
    assert badge.property("statusKind") == ""
    assert reason in badge.toolTip()
    assert reason in badge.accessibleDescription()


def test_withheld_covariance_is_explicit_in_full_evidence_and_inspector(withheld_covariance):
    report, reason = withheld_covariance
    line = next(line for line in _report_lines(report) if line.startswith("强相关："))
    note, kind = correlation_note(report)

    assert "不可用" in line and reason in line
    assert "不可用" in note and reason in note
    assert "无强相关" not in line + note
    assert kind == ""


def test_withheld_covariance_cannot_activate_a_strong_correlation_callout(qtbot, withheld_covariance):
    report, _reason = withheld_covariance
    report = replace(report, strong_correlations=(("scale", "thickness", 0.99),))
    view = UncertaintyView()
    qtbot.addWidget(view)
    view.set_result(_result(report), "candidate-a")

    assert view.correlation_callout.isHidden()
    assert "scale/thickness=0.99" not in view.text()


@pytest.mark.parametrize("method", ("gaussian_known_sigma", "poisson_expected_information", "robust_sandwich_iid"))
def test_available_covariance_keeps_its_actual_method_matrix_and_signed_labels(qtbot, method):
    matrix = np.array([[1.0, -0.7], [-0.7, 1.0]])
    report = _report(covariance=CovarianceEvidence(NAMES, matrix, method, 2))
    report = replace(report, strong_correlations=(("scale", "thickness", -0.7),))
    view = UncertaintyView()
    qtbot.addWidget(view)
    view.set_result(_result(report), "candidate-a")
    figure = view.page_figures()[0]
    figure.set_size_inches(8, 6)
    figure.canvas.draw()

    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), matrix)
    assert "-0.70" in {item.get_text() for item in figure.axes[0].texts}
    assert any(axes.get_label() == "<colorbar>" for axes in figure.axes)
    assert uncertainty_method_rows(report)[0].state == "done"
    assert uncertainty_method_rows(report)[0].caption == method
    assert not view.correlation_callout.isHidden()


@pytest.mark.parametrize("owner", (None, "candidate-b"))
def test_finite_covariance_is_not_borrowed_from_another_candidate(qtbot, owner):
    report = replace(_report(), candidate_id=owner, strong_correlations=(("scale", "thickness", 0.99),))
    view = UncertaintyView()
    qtbot.addWidget(view)
    view.set_result(_result(report), "candidate-a")

    assert all(not axes.images for axes in view.page_figures()[0].axes)
    assert view.correlation_callout.isHidden()
    assert "强相关：" not in view.text()


def test_ranked_pairs_never_rank_nonfinite_coefficients():
    matrix = np.array([[1.0, np.nan, np.inf], [np.nan, 1.0, -0.65], [np.inf, -0.65, 1.0]])

    assert _ranked_pairs(matrix) == ((1, 2, -0.65),)
