"""Strict saved Poisson family evidence; serialization never regenerates it.

Every key belongs to the current schema, including explicit nullable values.
The pure content seal is verified at both boundaries, never invented on save.
Derived summaries are carried for readers but must agree with the immutable
sample accounting rather than becoming a second source of inference truth.
"""

from __future__ import annotations

from dataclasses import replace

from xrr_fitter.io.codec_common import ProjectSchemaError, _finite_number, _mapping, _sequence
from xrr_fitter.model.diagnostic_calibration import DiagnosticCalibration, DiagnosticStatistic
from xrr_fitter.model.diagnostic_work import DiagnosticPathSummary, DiagnosticRefitWork, DiagnosticSolverExit
from xrr_fitter.model.provenance import diagnostic_calibration_sha256

CALIBRATION_SUMMARIES = frozenset({"p_value", "rejected", "resolution"})
CALIBRATION_STRUCTURES = frozenset({"statistics", "failure_reasons", "refit_discrepancy", "observed_work", "null_work"})


def statistic_to_dict(value: DiagnosticStatistic) -> dict[str, object]:
    validated = replace(value)
    return {field: getattr(validated, field) for field in DiagnosticStatistic.__dataclass_fields__}


def statistic_from_dict(value: object) -> DiagnosticStatistic:
    payload = _mapping(value, set(DiagnosticStatistic.__dataclass_fields__), "diagnostic statistic")
    return DiagnosticStatistic(**payload)


def work_to_dict(value: DiagnosticRefitWork) -> dict[str, object]:
    value = replace(value)
    return {
        "declared_paths": value.declared_paths,
        "paths": [
            {
                "budget": path.budget,
                "count": path.count,
                "stages": [
                    {field: getattr(stage, field) for field in DiagnosticSolverExit.__dataclass_fields__}
                    for stage in path.stages
                ],
            }
            for path in value.paths
        ],
    }


def _path_from_dict(value):
    payload = _mapping(value, set(DiagnosticPathSummary.__dataclass_fields__), "diagnostic path work")
    stages = tuple(
        DiagnosticSolverExit(**_mapping(stage, set(DiagnosticSolverExit.__dataclass_fields__), "diagnostic stage exit"))
        for stage in _sequence(payload["stages"], "diagnostic stage exits")
    )
    return DiagnosticPathSummary(payload["budget"], stages, payload["count"])


def work_from_dict(value: object) -> DiagnosticRefitWork:
    payload = _mapping(value, set(DiagnosticRefitWork.__dataclass_fields__), "diagnostic work")
    paths = tuple(_path_from_dict(path) for path in _sequence(payload["paths"], "diagnostic path histogram"))
    return DiagnosticRefitWork(payload["declared_paths"], paths)


def _validate_seal(value: DiagnosticCalibration) -> None:
    if value.provenance_sha256 != diagnostic_calibration_sha256(value):
        raise ProjectSchemaError("diagnostic calibration provenance seal does not match saved evidence")


def calibration_to_dict(value: DiagnosticCalibration | None) -> dict[str, object] | None:
    if value is None:
        return None
    validated = replace(value)
    _validate_seal(validated)
    fields = set(DiagnosticCalibration.__dataclass_fields__) - CALIBRATION_STRUCTURES
    return {
        **{field: getattr(validated, field) for field in sorted(fields)},
        "statistics": [statistic_to_dict(item) for item in validated.statistics],
        "failure_reasons": [list(item) for item in validated.failure_reasons],
        "observed_work": work_to_dict(validated.observed_work),
        "null_work": work_to_dict(validated.null_work),
        "refit_discrepancy": None if validated.refit_discrepancy is None else list(validated.refit_discrepancy),
        **{field: getattr(validated, field) for field in sorted(CALIBRATION_SUMMARIES)},
    }


def _optional_number_matches(value: object, expected: float | None) -> bool:
    if expected is None:
        return value is None
    return _finite_number(value) and value == expected


def _validate_summaries(payload: dict[str, object], evidence: DiagnosticCalibration) -> None:
    numeric = ("p_value", "resolution")
    valid = all(_optional_number_matches(payload[field], getattr(evidence, field)) for field in numeric)
    if not valid or payload["rejected"] is not evidence.rejected:
        raise ProjectSchemaError("diagnostic calibration summary metadata must match saved sample evidence")


def calibration_from_dict(value: object) -> DiagnosticCalibration | None:
    if value is None:
        return None
    fields = set(DiagnosticCalibration.__dataclass_fields__)
    payload = _mapping(value, fields | CALIBRATION_SUMMARIES, "diagnostic calibration")
    statistics = tuple(statistic_from_dict(item) for item in _sequence(payload["statistics"], "diagnostic statistics"))
    failures = tuple(
        tuple(_sequence(row, "diagnostic failure"))
        for row in _sequence(payload["failure_reasons"], "diagnostic failure reasons")
    )
    discrepancy = payload["refit_discrepancy"]
    evidence = DiagnosticCalibration(
        **{field: payload[field] for field in fields - CALIBRATION_STRUCTURES},
        statistics=statistics,
        failure_reasons=failures,
        observed_work=work_from_dict(payload["observed_work"]),
        null_work=work_from_dict(payload["null_work"]),
        refit_discrepancy=(
            None if discrepancy is None else tuple(_sequence(discrepancy, "diagnostic refit discrepancy"))
        ),
    )
    _validate_seal(evidence)
    _validate_summaries(payload, evidence)
    return evidence
