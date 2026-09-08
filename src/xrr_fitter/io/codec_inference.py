"""Strict V2 covariance and per-member diagnostic payloads."""

from __future__ import annotations

from xrr_fitter.io.codec_candidates import _diagnostic_from_dict, _diagnostic_to_dict
from xrr_fitter.io.codec_common import _mapping, _real_array_to_list, _sequence, _square_array_from_list
from xrr_fitter.model.analysis import CovarianceEvidence, ResidualEvidence


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


def residual_to_dict(value: ResidualEvidence) -> dict:
    return {
        "dataset_id": value.dataset_id,
        "executed": value.executed,
        "systematic": value.systematic,
        "autocorrelation": value.autocorrelation,
        "point_count": value.point_count,
        "diagnostics": [_diagnostic_to_dict(item) for item in value.diagnostics],
        "unavailable_reason": value.unavailable_reason,
    }


def residual_from_dict(value: object) -> ResidualEvidence:
    payload = _mapping(
        value,
        {"dataset_id", "executed", "systematic", "autocorrelation", "point_count", "diagnostics", "unavailable_reason"},
        "residual evidence",
    )
    return ResidualEvidence(
        payload["dataset_id"],
        payload["executed"],
        payload["systematic"],
        payload["autocorrelation"],
        payload["point_count"],
        tuple(_diagnostic_from_dict(item) for item in _sequence(payload["diagnostics"], "residual diagnostics")),
        payload["unavailable_reason"],
    )
