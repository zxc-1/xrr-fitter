"""Canonical candidate and domain-signoff parsing."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import re

from approved_data_model import (
    CASE_IDS,
    CANDIDATE_SCHEMA,
    HEX,
    SIGNOFF_SCHEMA,
    ApprovedRun,
    CandidateCase,
    CandidateReport,
    CanonicalEnvironment,
    DomainSignoff,
    RelativeFileRecord,
    SignedCase,
)


def _object_pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _json_value(content: bytes, label: str) -> object:
    if not isinstance(content, bytes):
        raise TypeError(f"{label} content must be bytes")
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error


def canonical_json_bytes(value: object) -> bytes:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value is not canonical JSON data") from error
    return (serialized + "\n").encode("utf-8")


def _fields(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} field set drift")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(item not in HEX for item in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _seed(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("run seed must be a nonnegative integer")
    return value


def _relative_path(value: object, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return text


def _file_record(value: object, label: str) -> RelativeFileRecord:
    data = _fields(value, {"path", "size", "sha256"}, label)
    return RelativeFileRecord(
        _relative_path(data["path"], f"{label} path"),
        _positive_int(data["size"], f"{label} size"),
        _sha256(data["sha256"], f"{label} hash"),
    )


def _file_value(value: RelativeFileRecord) -> dict[str, object]:
    return {"path": value.path, "size": value.size, "sha256": value.sha256}


def _environment(value: object) -> CanonicalEnvironment:
    names = {
        "python_version",
        "platform",
        "dependency_lock_sha256",
        "production_tree_sha256",
        "acceptance_test_tree_sha256",
        "qt_runtime_identity",
    }
    data = _fields(value, names, "candidate environment")
    version = _text(data["python_version"], "python_version")
    if re.fullmatch(r"3\.12\.\d+", version) is None:
        raise ValueError("python_version must be a normalized Python 3.12 patch version")
    if data["platform"] != "macos-arm64":
        raise ValueError("candidate platform must be macos-arm64")
    return CanonicalEnvironment(
        version,
        "macos-arm64",
        _sha256(data["dependency_lock_sha256"], "dependency lock"),
        _sha256(data["production_tree_sha256"], "production tree"),
        _sha256(data["acceptance_test_tree_sha256"], "acceptance test tree"),
        _text(data["qt_runtime_identity"], "Qt runtime identity"),
    )


def _environment_value(value: CanonicalEnvironment) -> dict[str, object]:
    return {
        "python_version": value.python_version,
        "platform": value.platform,
        "dependency_lock_sha256": value.dependency_lock_sha256,
        "production_tree_sha256": value.production_tree_sha256,
        "acceptance_test_tree_sha256": value.acceptance_test_tree_sha256,
        "qt_runtime_identity": value.qt_runtime_identity,
    }


def _normalized_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise ValueError("normalized_result must be a nonempty JSON object")
    canonical_json_bytes(value)
    return value


def _ordered_files(value: object, label: str) -> tuple[RelativeFileRecord, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    records = tuple(_file_record(item, label) for item in value)
    paths = tuple(record.path for record in records)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError(f"{label} paths must be unique and sorted")
    return records


def _run(value: object) -> ApprovedRun:
    names = {"ordinal", "seed", "project", "exports", "plots", "normalized_result"}
    data = _fields(value, names, "candidate run")
    ordinal = _positive_int(data["ordinal"], "run ordinal")
    return ApprovedRun(
        ordinal,
        _seed(data["seed"]),
        _file_record(data["project"], "run project"),
        _ordered_files(data["exports"], "run exports"),
        _ordered_files(data["plots"], "run plots"),
        _normalized_result(data["normalized_result"]),
    )


def _run_value(value: ApprovedRun) -> dict[str, object]:
    return {
        "ordinal": value.ordinal,
        "seed": value.seed,
        "project": _file_value(value.project),
        "exports": [_file_value(item) for item in value.exports],
        "plots": [_file_value(item) for item in value.plots],
        "normalized_result": value.normalized_result,
    }


def _evidence_paths(runs: tuple[ApprovedRun, ...]) -> tuple[str, ...]:
    return tuple(
        record.path
        for run in runs
        for record in (run.project, *run.exports, *run.plots)
    )


def _require_unique_evidence_paths(paths: tuple[str, ...]) -> None:
    if len(paths) != len(set(paths)):
        raise ValueError("candidate evidence paths must be globally unique")


def _runs(value: object) -> tuple[ApprovedRun, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("candidate case must contain exactly four runs")
    runs = tuple(_run(item) for item in value)
    if tuple(run.ordinal for run in runs) != (1, 2, 3, 4):
        raise ValueError("run ordinals must be canonical")
    seeds = tuple(run.seed for run in runs)
    if len(set(seeds[:3])) != 1 or seeds[3] == seeds[0]:
        raise ValueError("runs 1-3 must share a seed and run 4 must use a new seed")
    _require_unique_evidence_paths(_evidence_paths(runs))
    return runs


def _operations(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("operations must be a nonempty list")
    operations = tuple(_text(item, "operation") for item in value)
    if len(operations) != len(set(operations)):
        raise ValueError("operations must be unique")
    return operations


def _candidate_case(value: object) -> CandidateCase:
    names = {
        "case_id",
        "source",
        "configuration_sha256",
        "operations",
        "runs",
        "normalized_result",
        "conclusion",
    }
    data = _fields(value, names, "candidate case")
    case_id = _text(data["case_id"], "case_id")
    if case_id not in CASE_IDS:
        raise ValueError(f"unsupported approved-data case: {case_id}")
    return CandidateCase(
        case_id,
        _file_record(data["source"], "case source"),
        _sha256(data["configuration_sha256"], "configuration"),
        _operations(data["operations"]),
        _runs(data["runs"]),
        _normalized_result(data["normalized_result"]),
        _text(data["conclusion"], "case conclusion"),
    )


def _candidate_case_value(value: CandidateCase) -> dict[str, object]:
    return {
        "case_id": value.case_id,
        "source": _file_value(value.source),
        "configuration_sha256": value.configuration_sha256,
        "operations": list(value.operations),
        "runs": [_run_value(item) for item in value.runs],
        "normalized_result": value.normalized_result,
        "conclusion": value.conclusion,
    }


def _candidate(value: object) -> CandidateReport:
    data = _fields(value, {"schema", "environment", "workflow_contract_sha256", "cases"}, "candidate report")
    if data["schema"] != CANDIDATE_SCHEMA:
        raise ValueError("candidate report schema drift")
    supplied = data["cases"]
    if not isinstance(supplied, list):
        raise ValueError("candidate cases must be a list")
    cases = tuple(_candidate_case(item) for item in supplied)
    if tuple(case.case_id for case in cases) != CASE_IDS:
        raise ValueError("candidate cases must contain the exact sorted case registry")
    _require_unique_evidence_paths(
        tuple(path for case in cases for path in _evidence_paths(case.runs))
    )
    return CandidateReport(
        CANDIDATE_SCHEMA,
        _environment(data["environment"]),
        _sha256(data["workflow_contract_sha256"], "workflow contract"),
        cases,
    )


def candidate_value(value: CandidateReport) -> dict[str, object]:
    return {
        "schema": value.schema,
        "environment": _environment_value(value.environment),
        "workflow_contract_sha256": value.workflow_contract_sha256,
        "cases": [_candidate_case_value(item) for item in value.cases],
    }


def parse_candidate_report(content: bytes) -> CandidateReport:
    report = _candidate(_json_value(content, "candidate report"))
    if content != canonical_json_bytes(candidate_value(report)):
        raise ValueError("candidate report is not canonical JSON")
    return report


def _signed_case(value: object) -> SignedCase:
    data = _fields(value, {"case_id", "approved", "conclusion"}, "signed case")
    case_id = _text(data["case_id"], "signed case_id")
    if data["approved"] is not True:
        raise ValueError("every signed case must be approved")
    return SignedCase(case_id, True, _text(data["conclusion"], "signed conclusion"))


def _domain_signoff(value: object) -> DomainSignoff:
    names = {"schema", "reviewer", "role", "candidate_report_sha256", "cases"}
    data = _fields(value, names, "domain signoff")
    if data["schema"] != SIGNOFF_SCHEMA:
        raise ValueError("domain signoff schema drift")
    supplied = data["cases"]
    if not isinstance(supplied, list):
        raise ValueError("signed cases must be a list")
    cases = tuple(_signed_case(item) for item in supplied)
    if tuple(item.case_id for item in cases) != CASE_IDS:
        raise ValueError("signed cases must contain the exact sorted case registry")
    return DomainSignoff(
        SIGNOFF_SCHEMA,
        _text(data["reviewer"], "reviewer"),
        _text(data["role"], "reviewer role"),
        _sha256(data["candidate_report_sha256"], "candidate report"),
        cases,
    )


def signoff_value(value: DomainSignoff) -> dict[str, object]:
    return {
        "schema": value.schema,
        "reviewer": value.reviewer,
        "role": value.role,
        "candidate_report_sha256": value.candidate_report_sha256,
        "cases": [
            {"case_id": item.case_id, "approved": item.approved, "conclusion": item.conclusion}
            for item in value.cases
        ],
    }


def parse_domain_signoff(content: bytes) -> DomainSignoff:
    signoff = _domain_signoff(_json_value(content, "domain signoff"))
    if content != canonical_json_bytes(signoff_value(signoff)):
        raise ValueError("domain signoff is not canonical JSON")
    return signoff
