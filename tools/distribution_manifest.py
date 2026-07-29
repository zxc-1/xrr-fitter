"""Strict artifact bundle selection and canonical manifest identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


SCHEMA = "xrr-r23-artifact-manifest-v1"
STATUS = "PASS"
MANIFEST_FIELDS = {"schema", "status", "head_commit", "head_tree", "artifacts"}
RECORD_FIELDS = {"kind", "path", "filename", "size", "sha256"}
ARTIFACT_KINDS = ("sdist", "wheel")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    kind: str
    path: str
    filename: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema: str
    status: str
    head_commit: str
    head_tree: str
    artifacts: tuple[ArtifactRecord, ArtifactRecord]


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def git_oid(value: object, label: str) -> str:
    valid = (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
    if not valid:
        raise ValueError(f"invalid {label}")
    return value


def _sha256(value: object, label: str) -> str:
    valid = (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    if not valid:
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _positive_size(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("artifact size must be a positive integer")
    return value


def _artifact_filename(kind: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact filename must be nonempty")
    path = PurePosixPath(value)
    if path.name != value or value in {".", ".."}:
        raise ValueError("artifact filename must be a basename")
    valid_suffix = value.endswith(".whl") if kind == "wheel" else value.endswith(".tar.gz")
    if not valid_suffix:
        raise ValueError(f"artifact filename does not match kind: {kind}")
    return value


def _record(value: object) -> ArtifactRecord:
    if not isinstance(value, dict) or set(value) != RECORD_FIELDS:
        raise ValueError("artifact record field set drift")
    kind = value["kind"]
    if kind not in ARTIFACT_KINDS:
        raise ValueError("artifact kind drift")
    filename = _artifact_filename(kind, value["filename"])
    expected_path = f"artifacts/{filename}"
    if value["path"] != expected_path:
        raise ValueError("artifact path must be canonical under artifacts/")
    return ArtifactRecord(
        kind,
        expected_path,
        filename,
        _positive_size(value["size"]),
        _sha256(value["sha256"], "artifact"),
    )


def _manifest(value: object) -> ArtifactManifest:
    if not isinstance(value, dict) or set(value) != MANIFEST_FIELDS:
        raise ValueError("artifact manifest field set drift")
    if value["schema"] != SCHEMA:
        raise ValueError("artifact manifest schema drift")
    if value["status"] != STATUS:
        raise ValueError("artifact manifest status drift")
    supplied = value["artifacts"]
    if not isinstance(supplied, list) or len(supplied) != 2:
        raise ValueError("artifact manifest requires one wheel and one sdist")
    records = tuple(_record(item) for item in supplied)
    if tuple(record.kind for record in records) != ARTIFACT_KINDS:
        raise ValueError("artifact records must be unique and sorted by kind")
    return ArtifactManifest(
        SCHEMA,
        STATUS,
        git_oid(value["head_commit"], "head commit"),
        git_oid(value["head_tree"], "head tree"),
        records,
    )


def _record_value(record: ArtifactRecord) -> dict[str, object]:
    return {
        "kind": record.kind,
        "path": record.path,
        "filename": record.filename,
        "size": record.size,
        "sha256": record.sha256,
    }


def manifest_value(manifest: ArtifactManifest) -> dict[str, object]:
    if not isinstance(manifest, ArtifactManifest):
        raise TypeError("manifest must be an ArtifactManifest")
    return {
        "schema": manifest.schema,
        "status": manifest.status,
        "head_commit": manifest.head_commit,
        "head_tree": manifest.head_tree,
        "artifacts": [_record_value(record) for record in manifest.artifacts],
    }


def canonical_manifest_bytes(manifest: ArtifactManifest) -> bytes:
    try:
        text = json.dumps(
            manifest_value(manifest),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("artifact manifest is not JSON serializable") from error
    return (text + "\n").encode("utf-8")


def parse_artifact_manifest(content: bytes) -> ArtifactManifest:
    if not isinstance(content, bytes):
        raise TypeError("artifact manifest content must be bytes")
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid artifact manifest JSON") from error
    manifest = _manifest(value)
    if content != canonical_manifest_bytes(manifest):
        raise ValueError("artifact manifest is not canonical JSON")
    return manifest


def read_artifact_manifest(path: str | Path) -> ArtifactManifest:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("artifact manifest must be a regular file")
    return parse_artifact_manifest(target.read_bytes())


def _artifact_kind(path: Path) -> str | None:
    if path.name.endswith(".whl"):
        return "wheel"
    if path.name.endswith(".tar.gz"):
        return "sdist"
    return None


def _artifact_entries(directory: Path) -> tuple[Path, Path]:
    if directory.name != "artifacts":
        raise ValueError("artifact directory must be named artifacts")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("artifact directory must be a regular directory")
    entries = tuple(directory.iterdir())
    valid = len(entries) == 2 and all(
        not path.is_symlink() and path.is_file() for path in entries
    )
    if not valid:
        raise ValueError("artifact directory must contain exactly two regular files")
    return entries


def select_artifacts(directory: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in _artifact_entries(directory):
        kind = _artifact_kind(path)
        if kind is None or kind in selected:
            raise ValueError("artifact directory must contain one wheel and one sdist")
        selected[kind] = path
    if tuple(sorted(selected)) != ARTIFACT_KINDS:
        raise ValueError("artifact directory must contain one wheel and one sdist")
    return selected


def _calculated_record(kind: str, path: Path) -> ArtifactRecord:
    content = path.read_bytes()
    return ArtifactRecord(
        kind,
        f"artifacts/{path.name}",
        path.name,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def calculate_artifact_manifest(
    artifact_dir: str | Path,
    *,
    head_commit: str,
    head_tree: str,
) -> ArtifactManifest:
    commit = git_oid(head_commit, "head commit")
    tree = git_oid(head_tree, "head tree")
    selected = select_artifacts(Path(artifact_dir))
    records = tuple(
        _calculated_record(kind, selected[kind]) for kind in ARTIFACT_KINDS
    )
    return ArtifactManifest(SCHEMA, STATUS, commit, tree, records)


def validate_artifact_manifest(
    manifest: ArtifactManifest,
    artifact_dir: str | Path,
    *,
    head_commit: str,
    head_tree: str,
) -> None:
    expected = calculate_artifact_manifest(
        artifact_dir,
        head_commit=head_commit,
        head_tree=head_tree,
    )
    if manifest != expected:
        raise ValueError("artifact manifest or artifact content drift")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_artifact_manifest(
    path: str | Path,
    manifest: ArtifactManifest,
) -> Path:
    target = Path(path)
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("artifact manifest parent must be a regular directory")
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("artifact manifest target must be a regular file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_manifest_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
