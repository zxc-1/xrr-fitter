"""V2 inference evidence survives persistence without fabricating calibration."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import joint_scale_searches, scale_candidate, scale_problem

from xrr_fitter.analysis.report import build_uncertainty_report
from xrr_fitter.io.codec_results import _uncertainty_from_dict, _uncertainty_to_dict
from xrr_fitter.model.analysis import CovarianceEvidence, ResidualEvidence
from xrr_fitter.model.parameters import ParameterReference
from xrr_fitter.services.fitting import _analyze_joint_searches


def _assert_compiled_member_axis(report, problem) -> None:
    assert hasattr(report, "parameter_members"), "joint reports must retain their compiled parameter membership"
    assert report.parameter_members == tuple(variable.members for variable in problem.global_variables)


@pytest.mark.parametrize("systematic", [False, True])
def test_joint_covariance_method_and_member_diagnostics_roundtrip(systematic: bool) -> None:
    problem, searches = joint_scale_searches(systematic=systematic)
    report = _analyze_joint_searches(problem, searches, ((), ()))[0].uncertainty
    _assert_compiled_member_axis(report, problem)
    payload = _uncertainty_to_dict(report)
    restored = _uncertainty_from_dict(payload)
    assert restored.parameter_members == report.parameter_members
    assert restored.covariance_evidence is not None
    assert restored.covariance_evidence.method == report.covariance_evidence.method
    assert restored.covariance_evidence.unavailable_reason == report.covariance_evidence.unavailable_reason
    assert restored.member_residuals == report.member_residuals
    np.testing.assert_array_equal(restored.search_parameter_spread, report.search_parameter_spread)
    if report.covariance is not None:
        np.testing.assert_array_equal(restored.covariance, report.covariance)
        assert restored.covariance.flags.writeable is False
    else:
        assert restored.covariance is None
        assert restored.parameter_sigma is None


def test_covariance_and_residual_evidence_are_immutable_after_pickle() -> None:
    problem = scale_problem("gaussian")
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    copied = pickle.loads(pickle.dumps(report))
    assert copied.covariance.flags.writeable is False
    assert copied.parameter_sigma.flags.writeable is False
    assert copied.member_residuals == report.member_residuals


def test_report_rejects_sigma_inconsistent_with_covariance_evidence() -> None:
    problem = scale_problem("gaussian")
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    with pytest.raises(ValueError, match="sigma.*covariance|covariance.*sigma"):
        replace(report, parameter_sigma=report.parameter_sigma * 10)


def test_report_rejects_contradictory_member_diagnostic_summary() -> None:
    problem = scale_problem(systematic=True)
    report = build_uncertainty_report(problem, (scale_candidate(problem),))
    with pytest.raises(ValueError, match="residual.*summary|summary.*residual"):
        replace(report, systematic_residual=False)


def test_unexecuted_residual_evidence_cannot_claim_no_problem() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ResidualEvidence("sample", False, False, False, 80, (), "not_run")


@pytest.mark.parametrize("matrix", [np.zeros((2, 2)), np.array([[1, 2], [2, 1]])])
def test_covariance_matrix_cannot_contradict_its_rank_or_positivity(matrix) -> None:
    with pytest.raises(ValueError, match="covariance"):
        CovarianceEvidence(("a", "b"), matrix, "gaussian_known_sigma", 1)


def test_v2_codec_requires_inference_fields_instead_of_reconstructing_old_reports() -> None:
    problem = scale_problem("gaussian")
    payload = _uncertainty_to_dict(build_uncertainty_report(problem, (scale_candidate(problem),)))
    payload.pop("covariance_evidence", None)
    with pytest.raises(ValueError, match="covariance_evidence"):
        _uncertainty_from_dict(payload)


def _local_report():
    problem = scale_problem("gaussian")
    return build_uncertainty_report(problem, (scale_candidate(problem),))


def _assert_immutable_member_report(report, expected) -> None:
    assert report.parameter_members == expected
    assert isinstance(report.parameter_members, tuple)
    assert isinstance(report.parameter_members[0], tuple)
    assert report.covariance.flags.writeable is False
    assert report.parameter_sigma.flags.writeable is False


def test_parameter_members_roundtrip_is_immutable_and_preserves_all_existing_evidence() -> None:
    original = _local_report()
    assert hasattr(original, "parameter_members"), "reports need explicit local or global axis identity"
    assert original.parameter_members is None
    members = [[ParameterReference("small", "instrument.scale"), ParameterReference("large", "instrument.scale")]]
    report = replace(original, parameter_members=members)
    members[0].clear()
    expected = ((ParameterReference("small", "instrument.scale"), ParameterReference("large", "instrument.scale")),)

    restored = _uncertainty_from_dict(_uncertainty_to_dict(report))
    copied = pickle.loads(pickle.dumps(restored))

    _assert_immutable_member_report(copied, expected)
    before, after = _uncertainty_to_dict(original), _uncertainty_to_dict(copied)
    assert before.pop("parameter_members") is None
    assert after.pop("parameter_members") == [
        [
            {"dataset_id": "small", "parameter_name": "instrument.scale"},
            {"dataset_id": "large", "parameter_name": "instrument.scale"},
        ]
    ]
    assert after == before


@pytest.mark.parametrize(
    ("members", "error", "message"),
    (
        ((), ValueError, "parameter_members.*axis"),
        (((),), ValueError, "parameter_members.*empty"),
        ((("not-a-reference",),), TypeError, "ParameterReference"),
        (((ParameterReference("same", "scale"), ParameterReference("same", "scale")),), ValueError, "unique"),
    ),
)
def test_parameter_members_rejects_invalid_axis_declarations(members, error, message) -> None:
    report = _local_report()
    assert hasattr(report, "parameter_members"), "reports need validated parameter membership"
    with pytest.raises(error, match=message):
        replace(report, parameter_members=members)


def test_parameter_members_rejects_one_member_claimed_by_two_axes() -> None:
    report = _local_report()
    assert hasattr(report, "parameter_members"), "reports need validated parameter membership"
    reference = ParameterReference("curve", "scale")
    with pytest.raises(ValueError, match="unique"):
        replace(
            report,
            correlation_names=("axis-one", "axis-two"),
            correlation_matrix=np.eye(2),
            covariance_evidence=None,
            parameter_sigma=None,
            parameter_members=((reference,), (reference,)),
        )


def test_v2_codec_requires_parameter_members_instead_of_guessing_a_joint_axis() -> None:
    payload = _uncertainty_to_dict(_local_report())
    payload.pop("parameter_members", None)
    with pytest.raises(ValueError, match="parameter_members"):
        _uncertainty_from_dict(payload)


@pytest.mark.parametrize(
    "member",
    (
        None,
        {"dataset_id": None, "parameter_name": "instrument.scale"},
        {"dataset_id": "curve", "parameter_name": None},
        {"dataset_id": "curve", "parameter_name": "instrument.scale", "extra": True},
    ),
)
def test_parameter_members_codec_rejects_null_or_undeclared_reference_fields(member) -> None:
    payload = _uncertainty_to_dict(_local_report())
    payload["parameter_members"] = [[member]]
    with pytest.raises(ValueError, match="parameter member"):
        _uncertainty_from_dict(payload)
