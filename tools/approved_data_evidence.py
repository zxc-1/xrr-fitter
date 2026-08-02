"""Atomic approved-data freezing and binding orchestration."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from approved_data_candidate import (
    candidate_value,
    canonical_json_bytes,
    parse_candidate_report,
    parse_domain_signoff,
    signoff_value,
)
from approved_data_model import (
    APPROVED_STATUS,
    CASE_IDS,
    CANDIDATE_SCHEMA,
    MANIFEST_SCHEMA,
    ApprovedCaseRecord,
    ApprovedDataBinding,
    ApprovedDataManifest,
    CandidateReport,
    DomainSignoff,
    RelativeFileRecord,
    RepoFileRecord,
)
from approved_data_records import (
    _candidate_from_records,
    _read_regular,
    _record,
    _record_index,
    _signoff_from_records,
    _source_tree,
    _tree_hash,
    _verify_source,
    check_candidate,
    manifest_value,
    parse_approved_case_record,
    parse_approved_data_manifest,
    record_value,
)


def _validate_projection(
    manifest: ApprovedDataManifest,
    records: Sequence[ApprovedCaseRecord],
) -> None:
    candidate = canonical_json_bytes(candidate_value(_candidate_from_records(manifest, records)))
    signoff = canonical_json_bytes(signoff_value(_signoff_from_records(manifest, records)))
    if hashlib.sha256(candidate).hexdigest() != manifest.candidate_report_sha256:
        raise ValueError("committed candidate projection hash drift")
    if hashlib.sha256(signoff).hexdigest() != manifest.domain_signoff_sha256:
        raise ValueError("committed signoff projection hash drift")


def _build_manifest(
    report: CandidateReport,
    report_content: bytes,
    signoff: DomainSignoff,
    signoff_content: bytes,
    record_contents: Sequence[bytes],
) -> ApprovedDataManifest:
    indexes = tuple(_record_index(case, content) for case, content in zip(report.cases, record_contents, strict=True))
    record_tree = _tree_hash(tuple((item.record.path, item.record.size, item.record.sha256) for item in indexes))
    return ApprovedDataManifest(
        MANIFEST_SCHEMA,
        CANDIDATE_SCHEMA,
        report.workflow_contract_sha256,
        report.environment,
        hashlib.sha256(report_content).hexdigest(),
        hashlib.sha256(signoff_content).hexdigest(),
        _source_tree(report.cases),
        record_tree,
        indexes,
    )


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(output: Path, manifest: bytes, records: Sequence[bytes]) -> None:
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("approved-data output parent must be a regular directory")
    if os.path.lexists(output):
        raise ValueError("approved-data output already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        record_dir = staging / "records"
        record_dir.mkdir()
        _write_file(staging / "manifest.json", manifest)
        for case_id, content in zip(CASE_IDS, records, strict=True):
            _write_file(record_dir / f"{case_id}.json", content)
        _fsync_directory(record_dir)
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def freeze_approved_data(
    candidate_report: str | Path,
    domain_signoff: str | Path,
    approved_data_root: str | Path,
    output: str | Path,
) -> ApprovedDataManifest:
    report_path = Path(candidate_report)
    report_content = _read_regular(report_path, "candidate report")
    report = parse_candidate_report(report_content)
    root = Path(approved_data_root)
    for case in report.cases:
        _verify_source(root, case.source)
    signoff_content = _read_regular(Path(domain_signoff), "domain signoff")
    signoff = parse_domain_signoff(signoff_content)
    if signoff.candidate_report_sha256 != hashlib.sha256(report_content).hexdigest():
        raise ValueError("domain signoff candidate report hash drift")
    records = tuple(_record(case, signoff) for case in report.cases)
    record_contents = tuple(canonical_json_bytes(record_value(item)) for item in records)
    manifest = _build_manifest(
        report,
        report_content,
        signoff,
        signoff_content,
        record_contents,
    )
    _validate_projection(manifest, records)
    _publish(Path(output), canonical_json_bytes(manifest_value(manifest)), record_contents)
    return manifest


def _exact_directory(path: Path, expected: set[str], label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    if {item.name for item in path.iterdir()} != expected:
        raise ValueError(f"{label} member drift")


def _evidence_files(root: Path) -> tuple[bytes, tuple[bytes, ...]]:
    _exact_directory(root, {"manifest.json", "records"}, "approved evidence root")
    record_dir = root / "records"
    expected_records = {f"{case_id}.json" for case_id in CASE_IDS}
    _exact_directory(record_dir, expected_records, "approved records")
    manifest = _read_regular(root / "manifest.json", "approved-data manifest")
    records = tuple(
        _read_regular(record_dir / f"{case_id}.json", "approved case record")
        for case_id in CASE_IDS
    )
    return manifest, records


def _validate_record_indexes(
    manifest: ApprovedDataManifest,
    records: Sequence[ApprovedCaseRecord],
    contents: Sequence[bytes],
) -> None:
    if tuple(item.case_id for item in records) != CASE_IDS:
        raise ValueError("approved record case registry drift")
    for index, record, content in zip(manifest.cases, records, contents, strict=True):
        observed = RelativeFileRecord(index.record.path, len(content), hashlib.sha256(content).hexdigest())
        if index.record != observed or index.source != record.source or index.conclusion != record.signoff.conclusion:
            raise ValueError(f"approved record index drift: {record.case_id}")


def _validate_tree_hashes(
    manifest: ApprovedDataManifest,
    records: Sequence[ApprovedCaseRecord],
) -> None:
    tree = _tree_hash(tuple((item.record.path, item.record.size, item.record.sha256) for item in manifest.cases))
    if tree != manifest.records_tree_sha256 or _source_tree(records) != manifest.approved_source_tree_sha256:
        raise ValueError("approved records or source tree hash drift")


def _load_evidence(
    evidence_root: Path,
) -> tuple[ApprovedDataManifest, tuple[ApprovedCaseRecord, ...], bytes, tuple[bytes, ...]]:
    manifest_content, record_contents = _evidence_files(evidence_root)
    manifest = parse_approved_data_manifest(manifest_content)
    records = tuple(parse_approved_case_record(content) for content in record_contents)
    _validate_record_indexes(manifest, records, record_contents)
    _validate_tree_hashes(manifest, records)
    _validate_projection(manifest, records)
    return manifest, records, manifest_content, record_contents


def calculate_approved_data_binding(
    evidence_root: str | Path,
    approved_data_root: str | Path,
) -> ApprovedDataBinding:
    root = Path(evidence_root)
    manifest, records, manifest_content, record_contents = _load_evidence(root)
    source_root = Path(approved_data_root)
    for record in records:
        _verify_source(source_root, record.source)
    evidence_entries = [
        (
            "verification/approved-data/manifest.json",
            len(manifest_content),
            hashlib.sha256(manifest_content).hexdigest(),
        )
    ]
    evidence_entries.extend(
        (
            f"verification/approved-data/records/{case_id}.json",
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
        for case_id, content in zip(CASE_IDS, record_contents, strict=True)
    )
    manifest_record = RepoFileRecord(*evidence_entries[0])
    return ApprovedDataBinding(
        APPROVED_STATUS,
        manifest_record,
        _tree_hash(tuple(evidence_entries)),
        manifest.approved_source_tree_sha256,
        manifest.candidate_report_sha256,
        manifest.domain_signoff_sha256,
    )


def validate_approved_data(
    evidence_root: str | Path,
    approved_data_root: str | Path,
) -> ApprovedDataBinding:
    return calculate_approved_data_binding(evidence_root, approved_data_root)
