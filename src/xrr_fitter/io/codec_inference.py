"""Strict V2 covariance and per-member diagnostic payloads."""

from __future__ import annotations

from dataclasses import replace

from xrr_fitter.io.codec_candidates import _diagnostic_from_dict, _diagnostic_to_dict
from xrr_fitter.io.codec_common import (
    ProjectSchemaError,
    _mapping,
    _real_array_from_list,
    _real_array_to_list,
    _sequence,
    _square_array_from_list,
)
from xrr_fitter.io.codec_diagnostic_calibration import calibration_from_dict, calibration_to_dict
from xrr_fitter.model.analysis import BootstrapResult, CovarianceEvidence, ResidualEvidence
from xrr_fitter.model.joint_bootstrap_provenance import validate_joint_bootstrap_content
from xrr_fitter.model.parameters import ParameterReference


def parameter_members_to_list(value: tuple[tuple[ParameterReference, ...], ...] | None) -> list | None:
    if value is None:
        return None
    return [
        [{"dataset_id": member.dataset_id, "parameter_name": member.parameter_name} for member in group]
        for group in value
    ]


def _parameter_member_from_dict(value: object) -> ParameterReference:
    payload = _mapping(value, {"dataset_id", "parameter_name"}, "parameter member")
    if any(not isinstance(item, str) for item in payload.values()):
        raise ProjectSchemaError("parameter member fields must be strings")
    return ParameterReference(**payload)


def parameter_members_from_list(value: object) -> tuple[tuple[ParameterReference, ...], ...] | None:
    if value is None:
        return None
    return tuple(
        tuple(_parameter_member_from_dict(member) for member in _sequence(group, "parameter member group"))
        for group in _sequence(value, "parameter_members")
    )


def covariance_to_dict(value: CovarianceEvidence | None) -> dict | None:
    if value is None:
        return None
    return {
        "names": list(value.names),
        "matrix": None if value.matrix is None else _real_array_to_list(value.matrix),
        "method": value.method,
        "rank": value.rank,
        "unidentifiable_names": list(value.unidentifiable_names),
        "unavailable_reason": value.unavailable_reason,
    }


def covariance_from_dict(value: object) -> CovarianceEvidence | None:
    if value is None:
        return None
    payload = _mapping(
        value,
        {"names", "matrix", "method", "rank", "unidentifiable_names", "unavailable_reason"},
        "covariance evidence",
    )
    return CovarianceEvidence(
        tuple(_sequence(payload["names"], "covariance names")),
        None if payload["matrix"] is None else _square_array_from_list(payload["matrix"]),
        payload["method"],
        payload["rank"],
        tuple(_sequence(payload["unidentifiable_names"], "unidentifiable names")),
        payload["unavailable_reason"],
    )


def _validate_residual_owner(value: ResidualEvidence) -> None:
    if value.calibration is not None and value.owner_sha256 != value.calibration.owner_sha256:
        raise ProjectSchemaError("residual owner must match its diagnostic calibration owner")


def residual_to_dict(value: ResidualEvidence) -> dict:
    value = replace(value)
    _validate_residual_owner(value)
    return {
        "dataset_id": value.dataset_id,
        "executed": value.executed,
        "systematic": value.systematic,
        "autocorrelation": value.autocorrelation,
        "point_count": value.point_count,
        "diagnostics": [_diagnostic_to_dict(item) for item in value.diagnostics],
        "unavailable_reason": value.unavailable_reason,
        "raw_systematic": value.raw_systematic,
        "raw_autocorrelation": value.raw_autocorrelation,
        "advisories": [_diagnostic_to_dict(item) for item in value.advisories],
        "calibration": calibration_to_dict(value.calibration),
        "owner_sha256": value.owner_sha256,
    }


def residual_from_dict(value: object) -> ResidualEvidence:
    fields = set(ResidualEvidence.__dataclass_fields__)
    payload = _mapping(value, fields, "residual evidence")
    sequences = {"diagnostics", "advisories"}
    evidence = ResidualEvidence(
        **{field: payload[field] for field in fields - sequences - {"calibration"}},
        **{
            field: tuple(_diagnostic_from_dict(item) for item in _sequence(payload[field], f"residual {field}"))
            for field in sequences
        },
        calibration=calibration_from_dict(payload["calibration"]),
    )
    _validate_residual_owner(evidence)
    return evidence


def bootstrap_to_dict(value: BootstrapResult | None) -> dict | None:
    if value is None:
        return None
    value = replace(value)
    if value.joint_owner_sha256 is not None or value.method.startswith("joint_"):
        validate_joint_bootstrap_content(value)
    return {
        "parameter_names": list(value.parameter_names),
        "samples": _real_array_to_list(value.samples),
        "intervals": [list(item) for item in value.intervals],
        "failure_rate": value.failure_rate,
        "attempted_count": value.attempted_count,
        "failure_reasons": [list(item) for item in value.failure_reasons],
        "method": value.method,
        "unavailable_reason": value.unavailable_reason,
        "candidate_id": value.candidate_id,
        "provenance_sha256": value.provenance_sha256,
        "interval_kind": value.interval_kind,
        "confidence_level": value.confidence_level,
        "successful_samples": value.successful_samples,
        "diagnostic_unavailable_reason": value.diagnostic_unavailable_reason,
        "joint_owner_sha256": value.joint_owner_sha256,
        "interval_method": value.interval_method,
        "interval_ranks": None if value.interval_ranks is None else list(value.interval_ranks),
        "bootstrap_content_target": value.bootstrap_content_target,
        "monte_carlo_assurance": value.monte_carlo_assurance,
        "diagnostic_error_budget": value.diagnostic_error_budget,
    }


def bootstrap_from_dict(value: object) -> BootstrapResult | None:
    if value is None:
        return None
    fields = set(BootstrapResult.__dataclass_fields__)
    summaries = {
        "interval_kind",
        "confidence_level",
        "successful_samples",
        "bootstrap_content_target",
        "monte_carlo_assurance",
        "diagnostic_error_budget",
    }
    payload = _mapping(value, fields | summaries, "bootstrap evidence")
    names = tuple(_sequence(payload["parameter_names"], "bootstrap names"))
    samples = _real_array_from_list(payload["samples"])
    if samples.size == 0:
        samples = samples.reshape((0, len(names)))
    rows = {
        key: tuple(tuple(_sequence(row, key)) for row in _sequence(payload[key], key))
        for key in ("intervals", "failure_reasons")
    }
    result = BootstrapResult(
        **{key: payload[key] for key in fields - {"parameter_names", "samples", "intervals", "failure_reasons"}},
        parameter_names=names,
        samples=samples,
        **rows,
    )
    if result.joint_owner_sha256 is not None or result.method.startswith("joint_"):
        validate_joint_bootstrap_content(result)
    if any(payload[key] != getattr(result, key) for key in summaries):
        raise ValueError("bootstrap interval metadata must match sampling evidence")
    return result
