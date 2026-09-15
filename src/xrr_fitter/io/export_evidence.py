"""Pure projections of saved inference; no fitting or statistical recomputation.

The result codecs own the evidence field semantics. Export summaries omit the
large sample/trace/covariance arrays that already live in the complete result
JSON and dedicated table or ORSO fields. Unexecuted diagnostics remain unknown, and
sampling counts never acquire an export-specific confidence interpretation.
"""

from __future__ import annotations

from xrr_fitter.io.codec_inference import (
    bootstrap_to_dict,
    covariance_to_dict,
    parameter_members_to_list,
    residual_to_dict,
)
from xrr_fitter.io.codec_results import _profile_to_dict
from xrr_fitter.model.analysis import (
    BootstrapResult,
    CovarianceEvidence,
    ParameterProfile,
    ResidualEvidence,
    UncertaintyReport,
)
from xrr_fitter.model.fitting import FitCandidate
from xrr_fitter.model.parameters import ParameterReference

COVARIANCE_ABSENT_REASON = "covariance not estimated for this fit result"
PARAMETER_UNCERTAINTY_FIELDS = (
    "sigma",
    "sigma_unavailable_reason",
    "covariance_method",
    "covariance_rank",
    "covariance_unavailable_reason",
)
RESIDUAL_METADATA_FIELDS = ("noise_model", "residual_name", "residual_unit")


def residual_metadata(candidate: FitCandidate) -> dict[str, str]:
    """Use the executed candidate mode, not a guessed transform of its curve."""
    return {field: getattr(candidate, field) for field in RESIDUAL_METADATA_FIELDS}


def profile_metadata(profile: ParameterProfile) -> dict[str, object]:
    payload = _profile_to_dict(profile)
    return {key: value for key, value in payload.items() if key not in {"values", "objectives"}}


def bootstrap_metadata(evidence: BootstrapResult | None) -> dict[str, object] | None:
    payload = bootstrap_to_dict(evidence)
    if payload is None:
        return None
    return {key: value for key, value in payload.items() if key not in {"samples", "failure_reasons"}}


def covariance_metadata(evidence: CovarianceEvidence | None) -> dict[str, object] | None:
    payload = covariance_to_dict(evidence)
    if payload is None:
        return None
    return {key: value for key, value in payload.items() if key != "matrix"}


def inference_metadata(report: UncertaintyReport | None, absent_reason: str | None) -> dict[str, object]:
    """Share one candidate-owned inference projection across all serializers."""
    return {
        "candidate_id": None if report is None else report.candidate_id,
        "parameter_members": parameter_members_to_list(None if report is None else report.parameter_members),
        "unavailable_reason": absent_reason,
        "covariance_evidence": covariance_metadata(None if report is None else report.covariance_evidence),
        "profiles": [] if report is None else [profile_metadata(profile) for profile in report.profiles],
        "bootstrap_evidence": bootstrap_metadata(None if report is None else report.bootstrap_evidence),
        "bootstrap_performed": None if report is None else report.bootstrap_performed,
        **_residual_report_metadata(report),
    }


def _residual_report_metadata(report: UncertaintyReport | None) -> dict[str, object]:
    return {
        "systematic_residual": None if report is None else report.systematic_residual,
        "residual_autocorrelation": None if report is None else report.residual_autocorrelation,
        "member_residuals": [] if report is None else [_member_metadata(item) for item in report.member_residuals],
    }


def _member_metadata(evidence: ResidualEvidence) -> dict[str, object]:
    payload = residual_to_dict(evidence)
    return {
        **{key: value for key, value in payload.items() if key not in {"diagnostics", "advisories"}},
        "diagnostic_count": len(evidence.diagnostics),
        "advisories": [
            {"code": item.code, "message": item.message, "point_count": len(item.point_indices)}
            for item in evidence.advisories
        ],
    }


def covariance_absent_reason(report: UncertaintyReport | None, absent_reason: str | None) -> str | None:
    if report is None or report.covariance_evidence is None:
        return absent_reason or COVARIANCE_ABSENT_REASON
    return report.covariance_evidence.unavailable_reason


def _parameter_axis_index(report: UncertaintyReport, name: str, dataset_id: str) -> int | None:
    if report.parameter_members is None:
        return report.correlation_names.index(name) if name in report.correlation_names else None
    reference = ParameterReference(dataset_id, name)
    return next((index for index, members in enumerate(report.parameter_members) if reference in members), None)


def _parameter_sigma(
    report: UncertaintyReport | None,
    name: str,
    absent_reason: str | None,
    dataset_id: str,
) -> tuple[float | None, str | None]:
    reason = covariance_absent_reason(report, absent_reason)
    if reason is not None:
        return None, reason
    index = _parameter_axis_index(report, name, dataset_id)
    if index is None:
        return None, "parameter not in covariance axis"
    return float(report.parameter_sigma[index]), None


def parameter_uncertainty(
    report: UncertaintyReport | None,
    name: str,
    absent_reason: str | None,
    *,
    dataset_id: str,
) -> dict[str, object]:
    """Project saved physical sigma; search spread and parameter bounds are not errors."""
    evidence = None if report is None else report.covariance_evidence
    sigma, reason = _parameter_sigma(report, name, absent_reason, dataset_id)
    return {
        "sigma": sigma,
        "sigma_unavailable_reason": reason,
        "covariance_method": None if evidence is None else evidence.method,
        "covariance_rank": None if evidence is None else evidence.rank,
        "covariance_unavailable_reason": covariance_absent_reason(report, absent_reason),
    }
