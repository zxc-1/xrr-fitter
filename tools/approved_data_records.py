"""Committed record, manifest, source, and tree validation."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Sequence

from approved_data_candidate import (
    _environment,
    _environment_value,
    _fields,
    _file_record,
    _file_value,
    _json_value,
    _normalized_result,
    _operations,
    _run_value,
    _runs,
    _sha256,
    _text,
    candidate_value,
    canonical_json_bytes,
    parse_candidate_report,
    signoff_value,
)
from approved_data_model import (
    CASE_IDS,
    CANDIDATE_SCHEMA,
    MANIFEST_SCHEMA,
    RECORD_SCHEMA,
    SIGNOFF_SCHEMA,
    ApprovedCaseIndex,
    ApprovedCaseRecord,
    ApprovedDataManifest,
    CandidateCase,
    CandidateReport,
    DomainSignoff,
    EmbeddedCaseSignoff,
    RelativeFileRecord,
    SignedCase,
)


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _source_file(root: Path, record: RelativeFileRecord) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("approved-data root must be a regular directory")
    target = root.joinpath(*PurePosixPath(record.path).parts)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"approved source must be a regular file: {record.path}")
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError(f"approved source escapes data root: {record.path}")
    for parent in target.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"approved source traverses a symlink: {record.path}")
    return target


def _verify_source(root: Path, record: RelativeFileRecord) -> None:
    content = _source_file(root, record).read_bytes()
    if len(content) != record.size or hashlib.sha256(content).hexdigest() != record.sha256:
        raise ValueError(f"approved source content drift: {record.path}")


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _tree_hash(records: Sequence[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for path, size, sha256 in records:
        digest.update(_frame(path.encode("utf-8")))
        digest.update(_frame(str(size).encode("ascii")))
        digest.update(_frame(sha256.encode("ascii")))
    return digest.hexdigest()


def _source_tree(cases: Sequence[CandidateCase | ApprovedCaseRecord]) -> str:
    records = tuple((item.source.path, item.source.size, item.source.sha256) for item in cases)
    return _tree_hash(records)


def check_candidate(
    candidate_report: str | Path,
    approved_data_root: str | Path,
    r22_reference: str | Path,
) -> CandidateReport:
    report_path = Path(candidate_report)
    report = parse_candidate_report(_read_regular(report_path, "candidate report"))
    root = Path(approved_data_root)
    for case in report.cases:
        _verify_source(root, case.source)
    _read_regular(Path(r22_reference), "R22 reference")
    return report


def _record(case: CandidateCase, signoff: DomainSignoff) -> ApprovedCaseRecord:
    signed = next(item for item in signoff.cases if item.case_id == case.case_id)
    if signed.conclusion != case.conclusion:
        raise ValueError(f"signed conclusion drift: {case.case_id}")
    embedded = EmbeddedCaseSignoff(signoff.reviewer, signoff.role, True, signed.conclusion)
    return ApprovedCaseRecord(
        RECORD_SCHEMA,
        case.case_id,
        case.source,
        case.configuration_sha256,
        case.operations,
        case.runs,
        case.normalized_result,
        embedded,
    )


def record_value(value: ApprovedCaseRecord) -> dict[str, object]:
    return {
        "schema": value.schema,
        "case_id": value.case_id,
        "source": _file_value(value.source),
        "configuration_sha256": value.configuration_sha256,
        "operations": list(value.operations),
        "runs": [_run_value(item) for item in value.runs],
        "normalized_result": value.normalized_result,
        "signoff": {
            "reviewer": value.signoff.reviewer,
            "role": value.signoff.role,
            "approved": value.signoff.approved,
            "conclusion": value.signoff.conclusion,
        },
    }


def _record_from_value(value: object) -> ApprovedCaseRecord:
    names = {
        "schema",
        "case_id",
        "source",
        "configuration_sha256",
        "operations",
        "runs",
        "normalized_result",
        "signoff",
    }
    data = _fields(value, names, "approved case record")
    if data["schema"] != RECORD_SCHEMA:
        raise ValueError("approved case record schema drift")
    signoff_data = _fields(
        data["signoff"],
        {"reviewer", "role", "approved", "conclusion"},
        "embedded signoff",
    )
    if signoff_data["approved"] is not True:
        raise ValueError("embedded signoff must be approved")
    case_id = _text(data["case_id"], "approved case_id")
    if case_id not in CASE_IDS:
        raise ValueError("approved case_id is not registered")
    return ApprovedCaseRecord(
        RECORD_SCHEMA,
        case_id,
        _file_record(data["source"], "approved source"),
        _sha256(data["configuration_sha256"], "approved configuration"),
        _operations(data["operations"]),
        _runs(data["runs"]),
        _normalized_result(data["normalized_result"]),
        EmbeddedCaseSignoff(
            _text(signoff_data["reviewer"], "embedded reviewer"),
            _text(signoff_data["role"], "embedded role"),
            True,
            _text(signoff_data["conclusion"], "embedded conclusion"),
        ),
    )


def parse_approved_case_record(content: bytes) -> ApprovedCaseRecord:
    record = _record_from_value(_json_value(content, "approved case record"))
    if content != canonical_json_bytes(record_value(record)):
        raise ValueError("approved case record is not canonical JSON")
    return record


def _candidate_from_records(
    manifest: ApprovedDataManifest,
    records: Sequence[ApprovedCaseRecord],
) -> CandidateReport:
    cases = tuple(
        CandidateCase(
            item.case_id,
            item.source,
            item.configuration_sha256,
            item.operations,
            item.runs,
            item.normalized_result,
            item.signoff.conclusion,
        )
        for item in records
    )
    return CandidateReport(CANDIDATE_SCHEMA, manifest.environment, manifest.workflow_contract_sha256, cases)


def _signoff_from_records(
    manifest: ApprovedDataManifest,
    records: Sequence[ApprovedCaseRecord],
) -> DomainSignoff:
    first = records[0].signoff
    if any((item.signoff.reviewer, item.signoff.role) != (first.reviewer, first.role) for item in records):
        raise ValueError("embedded reviewer identity drift")
    cases = tuple(SignedCase(item.case_id, True, item.signoff.conclusion) for item in records)
    return DomainSignoff(
        SIGNOFF_SCHEMA,
        first.reviewer,
        first.role,
        manifest.candidate_report_sha256,
        cases,
    )


def _record_index(case: CandidateCase, content: bytes) -> ApprovedCaseIndex:
    path = f"records/{case.case_id}.json"
    record = RelativeFileRecord(path, len(content), hashlib.sha256(content).hexdigest())
    return ApprovedCaseIndex(case.case_id, case.source, record, case.conclusion)


def _index_value(value: ApprovedCaseIndex) -> dict[str, object]:
    return {
        "case_id": value.case_id,
        "source": _file_value(value.source),
        "record": _file_value(value.record),
        "conclusion": value.conclusion,
    }


def manifest_value(value: ApprovedDataManifest) -> dict[str, object]:
    return {
        "schema": value.schema,
        "candidate_schema": value.candidate_schema,
        "r22_reference_sha256": value.r22_reference_sha256,
        "workflow_contract_sha256": value.workflow_contract_sha256,
        "environment": _environment_value(value.environment),
        "candidate_report_sha256": value.candidate_report_sha256,
        "domain_signoff_sha256": value.domain_signoff_sha256,
        "approved_source_tree_sha256": value.approved_source_tree_sha256,
        "records_tree_sha256": value.records_tree_sha256,
        "cases": [_index_value(item) for item in value.cases],
    }


def _index(value: object) -> ApprovedCaseIndex:
    data = _fields(value, {"case_id", "source", "record", "conclusion"}, "approved case index")
    case_id = _text(data["case_id"], "manifest case_id")
    if case_id not in CASE_IDS:
        raise ValueError("manifest case_id is not registered")
    record = _file_record(data["record"], "manifest record")
    if record.path != f"records/{case_id}.json":
        raise ValueError("manifest record path drift")
    return ApprovedCaseIndex(
        case_id,
        _file_record(data["source"], "manifest source"),
        record,
        _text(data["conclusion"], "manifest conclusion"),
    )


def _manifest(value: object) -> ApprovedDataManifest:
    names = {
        "schema",
        "candidate_schema",
        "r22_reference_sha256",
        "workflow_contract_sha256",
        "environment",
        "candidate_report_sha256",
        "domain_signoff_sha256",
        "approved_source_tree_sha256",
        "records_tree_sha256",
        "cases",
    }
    data = _fields(value, names, "approved-data manifest")
    if (data["schema"], data["candidate_schema"]) != (MANIFEST_SCHEMA, CANDIDATE_SCHEMA):
        raise ValueError("approved-data manifest schema drift")
    supplied = data["cases"]
    if not isinstance(supplied, list):
        raise ValueError("approved-data manifest cases must be a list")
    cases = tuple(_index(item) for item in supplied)
    if tuple(item.case_id for item in cases) != CASE_IDS:
        raise ValueError("approved-data manifest cases must match the registry")
    return ApprovedDataManifest(
        MANIFEST_SCHEMA,
        CANDIDATE_SCHEMA,
        _sha256(data["r22_reference_sha256"], "R22 reference"),
        _sha256(data["workflow_contract_sha256"], "workflow contract"),
        _environment(data["environment"]),
        _sha256(data["candidate_report_sha256"], "candidate report"),
        _sha256(data["domain_signoff_sha256"], "domain signoff"),
        _sha256(data["approved_source_tree_sha256"], "approved source tree"),
        _sha256(data["records_tree_sha256"], "records tree"),
        cases,
    )


def parse_approved_data_manifest(content: bytes) -> ApprovedDataManifest:
    manifest = _manifest(_json_value(content, "approved-data manifest"))
    if content != canonical_json_bytes(manifest_value(manifest)):
        raise ValueError("approved-data manifest is not canonical JSON")
    return manifest
