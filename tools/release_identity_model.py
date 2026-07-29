"""Immutable values for the R23 release identity."""

from __future__ import annotations

from dataclasses import dataclass

from verify_distribution import ArtifactRecord


SCHEMA = "xrr-r23-release-identity-v1"
STATUS = "PASS"
NOT_RUN = "NOT_RUN: owner post-delivery acceptance"
FREEZE_SCHEMA = "xrr-r23-final-freeze-v1"


@dataclass(frozen=True, slots=True)
class RepoFileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TestManifestBinding:
    file: RepoFileRecord
    source_commit: str
    collection_sha256: str


@dataclass(frozen=True, slots=True)
class ApprovedDataStatus:
    status: str


@dataclass(frozen=True, slots=True)
class ExternalFileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class R23ReleaseIdentity:
    schema: str
    status: str
    head_commit: str
    head_tree: str
    release_spec: RepoFileRecord
    dependency_lock: RepoFileRecord
    r22_oracle_tree_sha256: str
    test_manifest: TestManifestBinding
    approved_data: ApprovedDataStatus
    artifact_manifest: ExternalFileRecord
    artifacts: tuple[ArtifactRecord, ArtifactRecord]
