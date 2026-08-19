"""Strict canonical parser for the R23 release identity."""

from __future__ import annotations

import json

from release_identity_model import (
    NOT_RUN,
    SCHEMA,
    STATUS,
    ApprovedDataStatus,
    ExternalFileRecord,
    R23ReleaseIdentity,
    RepoFileRecord,
    TestManifestBinding,
)
from verify_distribution import (
    ARTIFACT_KINDS,
    artifact_record,
    artifact_value,
    git_oid,
)

HEX = frozenset("0123456789abcdef")
IDENTITY_FIELDS = {
    "schema",
    "status",
    "head_commit",
    "head_tree",
    "release_spec",
    "dependency_lock",
    "test_manifest",
    "approved_data",
    "artifact_manifest",
    "artifacts",
}


def _object_pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _fields(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} field set drift")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(item not in HEX for item in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _size(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} size must be a positive integer")
    return value


def _repo_file(value: object, expected_path: str, label: str) -> RepoFileRecord:
    data = _fields(value, {"path", "size", "sha256"}, label)
    if data["path"] != expected_path:
        raise ValueError(f"{label} path drift")
    return RepoFileRecord(
        expected_path,
        _size(data["size"], label),
        _sha256(data["sha256"], label),
    )


def _external_file(value: object) -> ExternalFileRecord:
    record = _repo_file(value, "artifact-manifest.json", "artifact manifest")
    return ExternalFileRecord(record.path, record.size, record.sha256)


def _test_binding(value: object) -> TestManifestBinding:
    data = _fields(value, {"file", "source_commit", "collection_sha256"}, "test manifest binding")
    return TestManifestBinding(
        _repo_file(data["file"], "verification/r23/tests.json", "test manifest"),
        git_oid(data["source_commit"], "test source commit"),
        _sha256(data["collection_sha256"], "test collection"),
    )


def _approved_status(value: object) -> ApprovedDataStatus:
    data = _fields(value, {"status"}, "approved-data status")
    if data["status"] != NOT_RUN:
        raise ValueError("approved-data status drift")
    return ApprovedDataStatus(NOT_RUN)


def _artifacts(value: object):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("identity requires one wheel and one sdist")
    records = tuple(artifact_record(item) for item in value)
    if tuple(item.kind for item in records) != ARTIFACT_KINDS:
        raise ValueError("identity artifacts must be unique and sorted")
    return records


def _identity(value: object) -> R23ReleaseIdentity:
    data = _fields(value, IDENTITY_FIELDS, "release identity")
    if (data["schema"], data["status"]) != (SCHEMA, STATUS):
        raise ValueError("release identity schema or status drift")
    return R23ReleaseIdentity(
        SCHEMA,
        STATUS,
        git_oid(data["head_commit"], "head commit"),
        git_oid(data["head_tree"], "head tree"),
        _repo_file(data["release_spec"], "verification/release-spec.json", "release spec"),
        _repo_file(data["dependency_lock"], "requirements-macos-arm64-py312.lock", "dependency lock"),
        _test_binding(data["test_manifest"]),
        _approved_status(data["approved_data"]),
        _external_file(data["artifact_manifest"]),
        _artifacts(data["artifacts"]),
    )


def _file_value(value: RepoFileRecord | ExternalFileRecord) -> dict[str, object]:
    return {"path": value.path, "size": value.size, "sha256": value.sha256}


def identity_value(value: R23ReleaseIdentity) -> dict[str, object]:
    if not isinstance(value, R23ReleaseIdentity):
        raise TypeError("identity must be an R23ReleaseIdentity")
    return {
        "schema": value.schema,
        "status": value.status,
        "head_commit": value.head_commit,
        "head_tree": value.head_tree,
        "release_spec": _file_value(value.release_spec),
        "dependency_lock": _file_value(value.dependency_lock),
        "test_manifest": {
            "file": _file_value(value.test_manifest.file),
            "source_commit": value.test_manifest.source_commit,
            "collection_sha256": value.test_manifest.collection_sha256,
        },
        "approved_data": {"status": value.approved_data.status},
        "artifact_manifest": _file_value(value.artifact_manifest),
        "artifacts": [artifact_value(item) for item in value.artifacts],
    }


def canonical_identity_bytes(value: R23ReleaseIdentity) -> bytes:
    return (
        json.dumps(
            identity_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def parse_release_identity(content: bytes) -> R23ReleaseIdentity:
    if not isinstance(content, bytes):
        raise TypeError("release identity content must be bytes")
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid release identity JSON") from error
    identity = _identity(value)
    if content != canonical_identity_bytes(identity):
        raise ValueError("release identity is not canonical JSON")
    return identity
