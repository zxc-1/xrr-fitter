"""Immutable values for the approved-data evidence contract."""

from __future__ import annotations

from dataclasses import dataclass

CASE_IDS = (
    "known_single_layer",
    "unstable_multilayer",
    "workable_mo_si_multilayer",
)
CANDIDATE_SCHEMA = "xrr-r23-approved-data-candidate-v1"
SIGNOFF_SCHEMA = "xrr-r23-domain-signoff-v1"
MANIFEST_SCHEMA = "xrr-r23-approved-data-manifest-v1"
RECORD_SCHEMA = "xrr-r23-approved-case-record-v1"
APPROVED_STATUS = "PASS"
HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class RelativeFileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalEnvironment:
    python_version: str
    platform: str
    dependency_lock_sha256: str
    production_tree_sha256: str
    acceptance_test_tree_sha256: str
    qt_runtime_identity: str


@dataclass(frozen=True, slots=True)
class ApprovedRun:
    ordinal: int
    seed: int
    project: RelativeFileRecord
    exports: tuple[RelativeFileRecord, ...]
    plots: tuple[RelativeFileRecord, ...]
    normalized_result: dict[str, object]


@dataclass(frozen=True, slots=True)
class CandidateCase:
    case_id: str
    source: RelativeFileRecord
    configuration_sha256: str
    operations: tuple[str, ...]
    runs: tuple[ApprovedRun, ...]
    normalized_result: dict[str, object]
    conclusion: str


@dataclass(frozen=True, slots=True)
class CandidateReport:
    schema: str
    environment: CanonicalEnvironment
    workflow_contract_sha256: str
    cases: tuple[CandidateCase, ...]


@dataclass(frozen=True, slots=True)
class SignedCase:
    case_id: str
    approved: bool
    conclusion: str


@dataclass(frozen=True, slots=True)
class DomainSignoff:
    schema: str
    reviewer: str
    role: str
    candidate_report_sha256: str
    cases: tuple[SignedCase, ...]


@dataclass(frozen=True, slots=True)
class EmbeddedCaseSignoff:
    reviewer: str
    role: str
    approved: bool
    conclusion: str


@dataclass(frozen=True, slots=True)
class ApprovedCaseRecord:
    schema: str
    case_id: str
    source: RelativeFileRecord
    configuration_sha256: str
    operations: tuple[str, ...]
    runs: tuple[ApprovedRun, ...]
    normalized_result: dict[str, object]
    signoff: EmbeddedCaseSignoff


@dataclass(frozen=True, slots=True)
class ApprovedCaseIndex:
    case_id: str
    source: RelativeFileRecord
    record: RelativeFileRecord
    conclusion: str


@dataclass(frozen=True, slots=True)
class ApprovedDataManifest:
    schema: str
    candidate_schema: str
    workflow_contract_sha256: str
    environment: CanonicalEnvironment
    candidate_report_sha256: str
    domain_signoff_sha256: str
    approved_source_tree_sha256: str
    records_tree_sha256: str
    cases: tuple[ApprovedCaseIndex, ...]


@dataclass(frozen=True, slots=True)
class RepoFileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ApprovedDataBinding:
    status: str
    manifest: RepoFileRecord
    approved_evidence_tree_sha256: str
    approved_source_tree_sha256: str
    candidate_report_sha256: str
    domain_signoff_sha256: str
