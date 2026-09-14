"""Saved single/joint evidence is read, not recalculated or reclassified."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.unit.tools.poisson_diagnostic_fixtures import _calibration, _residual
from tests.unit.tools.poisson_diagnostic_fixtures import tool as tool


def test_saved_screen_negative_is_explicitly_not_triggered(tool):
    state = tool.evidence.diagnostic_state((_residual(),), target="background")

    assert state["status"] == "not_triggered"
    assert state["calibration"] is None
    assert state["rejected"] is False


def test_saved_unavailable_and_target_localization_are_not_coerced_false(tool):
    unavailable = _residual(executed=False, raw=True, reason="calibration_required")
    state = tool.evidence.diagnostic_state((unavailable,), target="background")
    assert state["status"] == "unavailable"
    assert state["unavailable_reason"] == "calibration_required"
    positive = _residual(calibration=_calibration(p=0.01, target_p=0.02), raw=True)
    state = tool.evidence.diagnostic_state((positive,), target="background")
    assert state["rejected"] is True
    assert state["target_detected"] is False


def test_joint_diagnostic_counts_one_family_and_rejects_incoherent_saved_families(tool):
    cal = _calibration()
    first = _residual(calibration=cal, raw=True)
    second = _residual(calibration=cal)
    second.dataset_id = "member-2"
    state = tool.evidence.diagnostic_state((first, second), target=None)
    assert state["calibration"]["attempted_count"] == 999
    other = _calibration()
    other.provenance_sha256 = "d" * 64
    second.calibration = other
    with pytest.raises(ValueError, match="family"):
        tool.evidence.diagnostic_state((first, second), target=None)


def test_missing_joint_profile_is_never_renamed_bootstrap(tool):
    member = SimpleNamespace(dataset_id="member", parameter_name=tool.cases.TARGET)
    report = SimpleNamespace(
        profiles=(),
        bootstrap_evidence=None,
        correlation_names=("shared-thickness",),
        parameter_members=((member,),),
    )
    intervals = tool.evidence.saved_intervals(report, dataset_id="member", joint=True)

    assert intervals["profile"]["available"] is False
    assert intervals["profile"]["unavailable_reason"] == "joint_profile_not_provided"
    assert intervals["bootstrap"]["available"] is False
    assert intervals["bootstrap"]["source"] == "bootstrap"


def test_missing_raw_evidence_is_not_reported_as_a_negative_screen(tool):
    residual = _residual()
    residual.raw_systematic = None

    state = tool.evidence.diagnostic_state((residual,), target=None)
    assert state["status"] == "unavailable"
    assert state["unavailable_reason"] == "raw_residual_evidence_missing"


def test_available_calibration_keeps_all_counts_seals_and_localization(tool):
    residual = _residual(calibration=_calibration(p=0.01, target_p=0.01), raw=True)
    state = tool.evidence.diagnostic_state((residual,), target="background")

    assert state["target_detected"] is True
    assert state["calibration"]["refit_nfev"] == 15
    assert state["calibration"]["sample_count"] == 999
    assert state["calibration"]["provenance_sha256"] == "c" * 64
    assert state["calibration"]["statistics"][0]["adjusted_p_value"] == 0.01


def test_product_failure_cannot_invent_a_negative_raw_screen(tool):
    outcome = tool.evidence.failed_outcome(ValueError("product unavailable"))

    assert outcome["diagnostic"]["raw_triggered"] is None
    summary = tool.protocol.summarize([outcome], required_count=1)
    assert summary["raw_unavailable"] == 1
    assert summary["raw_trigger_rate"] == 0.0


@pytest.mark.parametrize(
    ("observed", "scale", "score", "p_value"),
    [
        (0.0, 0.0, 0.0, 1.0),
        (0.25, 0.125, 2.0, 0.25),
        (float.fromhex("0x0.0000000000005p-1022"), float.fromhex("0x0.0000000000002p-1022"), 2.0, 0.25),
    ],
)
def test_rms_saved_scales_and_factorized_score_are_projected_without_recalculation(
    tool,
    observed,
    scale,
    score,
    p_value,
):
    family = _calibration(p=p_value, target_p=p_value)
    statistic = family.statistics[0]
    statistic.observed, statistic.center, statistic.scale = observed, 0.0, scale
    family.observed_score = score
    if scale == 0.0:
        family.tie_count = family.tail_count
    state = tool.evidence.diagnostic_state((_residual(calibration=family, raw=True),), target="background")

    saved = state["calibration"]
    assert saved["statistics"][0] == {
        "dataset_id": "member",
        "kind": "background",
        "observed": observed,
        "center": 0.0,
        "scale": scale,
        "adjusted_p_value": p_value,
    }
    assert saved["observed_score"] == score
    assert saved["p_value"] == p_value
    assert state["status"] == "not_rejected"
    assert state["target_detected"] is False
