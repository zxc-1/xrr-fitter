#!/usr/bin/env python3
"""Build and validate the canonical R23 software release identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from release_identity_model import (  # noqa: E402
    FREEZE_SCHEMA,
    NOT_RUN,
    SCHEMA,
    STATUS,
    ApprovedDataStatus,
    ExternalFileRecord,
    R23ReleaseIdentity,
    RepoFileRecord,
    TestManifestBinding,
)
from release_identity_schema import (  # noqa: E402
    _object_pairs,
    _sha256,
    canonical_identity_bytes,
    identity_value,  # noqa: F401
    parse_release_identity,
)
from verify_distribution import (  # noqa: E402
    clean_head_identity,
    committed_blob,
    committed_tree_files,
    git_oid,
    read_artifact_manifest,
    validate_artifact_manifest,
)
from verify_report import (  # noqa: E402
    _make_report_anchor,
    _prepare_report_directory,
    _require_same_directory,
    _resolve_output_path,
    _write_new_file_in_anchored_directory,
)

RELEASE_SPEC_PATH = "verification/release-spec.json"
LOCK_PATH = "requirements-macos-arm64-py312.lock"
TEST_MANIFEST_PATH = "verification/r23/tests.json"
TEST_MANIFEST_FIELDS = {
    "schema",
    "source_commit",
    "suite",
    "test_tree",
    "node_count",
    "nodes",
    "python_version",
    "platform",
    "lock_sha256",
    "collection_sha256",
}
PRUNED_TEST_PARTS = {".pytest_cache", "__pycache__"}


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _canonical_value(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("identity input is not canonical JSON data") from error
    return (text + "\n").encode("utf-8")


def _json_content(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict) or content != _canonical_value(value):
        raise ValueError(f"{label} must be a canonical JSON object")
    return value


def _repo_file(repository: Path, commit: str, relative: str) -> RepoFileRecord:
    content = committed_blob(repository, commit, relative)
    return RepoFileRecord(relative, len(content), hashlib.sha256(content).hexdigest())


def _external_file(path: Path, expected_name: str) -> ExternalFileRecord:
    if path.name != expected_name:
        raise ValueError(f"external file must be named {expected_name}")
    content = _read_regular(path, expected_name)
    return ExternalFileRecord(expected_name, len(content), hashlib.sha256(content).hexdigest())


def _release_spec(repository: Path, commit: str, lock: RepoFileRecord) -> RepoFileRecord:
    content = committed_blob(repository, commit, RELEASE_SPEC_PATH)
    value = _json_content(content, "release spec")
    if value.get("schema") != "xrr-r23-release-spec-v1":
        raise ValueError("release spec schema drift")
    if value.get("lock_sha256") != lock.sha256:
        raise ValueError("release spec dependency lock hash drift")
    return RepoFileRecord(RELEASE_SPEC_PATH, len(content), hashlib.sha256(content).hexdigest())


def _file_record(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise ValueError(f"{label} field set drift")
    path = value["path"]
    size = value["size"]
    sha256 = value["sha256"]
    if not isinstance(path, str) or not path.startswith("tests/") or ".." in Path(path).parts:
        raise ValueError(f"{label} path drift")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"{label} size drift")
    _sha256(sha256, label)
    return {"path": path, "size": size, "sha256": sha256}


def _declared_test_tree(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("test_tree must be a list")
    records = [_file_record(item, "test tree record") for item in value]
    paths = tuple(str(item["path"]) for item in records)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("test_tree paths must be unique and sorted")
    return records


def _current_test_tree(repository: Path, commit: str) -> list[dict[str, object]]:
    records = []
    for relative, content in committed_tree_files(repository, commit, PurePosixPath("tests")).items():
        if any(part in PRUNED_TEST_PARTS for part in relative.parts):
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return sorted(records, key=lambda item: str(item["path"]))


def _validate_test_nodes(value: dict[str, object]) -> None:
    nodes = value.get("nodes")
    count = value.get("node_count")
    if not isinstance(nodes, list) or not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("test manifest node fields drift")
    if count != len(nodes):
        raise ValueError("test manifest node count drift")
    nodeids = tuple(item.get("nodeid") for item in nodes if isinstance(item, dict))
    if len(nodeids) != count or nodeids != tuple(sorted(nodeids)) or len(set(nodeids)) != count:
        raise ValueError("test manifest node registry drift")


def _validate_test_source(repository: Path, source_commit: str, head_commit: str) -> None:
    try:
        object_type = _git(repository, "cat-file", "-t", source_commit)
    except ValueError as error:
        raise ValueError("test source commit must be an existing commit object") from error
    if object_type != "commit":
        raise ValueError("test source commit must be a commit object")
    if _git(repository, "merge-base", "--is-ancestor", source_commit, head_commit):
        raise ValueError("unexpected merge-base output")
    if _git(repository, "diff", "--name-only", source_commit, head_commit, "--", "tests"):
        raise ValueError("test tree differs from test manifest source commit")


def _test_manifest(repository: Path, commit: str, lock: RepoFileRecord) -> TestManifestBinding:
    content = committed_blob(repository, commit, TEST_MANIFEST_PATH)
    value = _json_content(content, "test manifest")
    if set(value) != TEST_MANIFEST_FIELDS or value.get("schema") != "xrr-test-manifest-v1":
        raise ValueError("test manifest schema or field set drift")
    if value.get("suite") != "tests" or value.get("lock_sha256") != lock.sha256:
        raise ValueError("test manifest suite or lock drift")
    expected_collection = hashlib.sha256(
        _canonical_value({key: item for key, item in value.items() if key != "collection_sha256"})
    ).hexdigest()
    collection = _sha256(value.get("collection_sha256"), "test collection")
    if collection != expected_collection:
        raise ValueError("test manifest collection hash drift")
    if _declared_test_tree(value.get("test_tree")) != _current_test_tree(repository, commit):
        raise ValueError("test manifest filesystem tree drift")
    _validate_test_nodes(value)
    source_commit = git_oid(value.get("source_commit"), "test source commit")
    _validate_test_source(repository, source_commit, commit)
    file = RepoFileRecord(TEST_MANIFEST_PATH, len(content), hashlib.sha256(content).hexdigest())
    return TestManifestBinding(file, source_commit, collection)


def _artifact_bundle(
    artifact_dir: Path,
    artifact_manifest: Path,
    head_commit: str,
    head_tree: str,
):
    if artifact_dir.name != "artifacts" or artifact_manifest.name != "artifact-manifest.json":
        raise ValueError("artifact bundle names drift")
    if artifact_dir.parent != artifact_manifest.parent:
        raise ValueError("artifact manifest and artifacts must share a parent")
    manifest = read_artifact_manifest(artifact_manifest)
    validate_artifact_manifest(
        manifest,
        artifact_dir,
        head_commit=head_commit,
        head_tree=head_tree,
    )
    return manifest


def calculate_release_identity(
    repo_root: str | Path,
    artifact_dir: str | Path,
    artifact_manifest: str | Path,
) -> R23ReleaseIdentity:
    repository = Path(repo_root).resolve()
    artifacts = Path(artifact_dir).resolve()
    manifest_path = Path(artifact_manifest).resolve()
    git = clean_head_identity(repository)
    lock = _repo_file(repository, git.head_commit, LOCK_PATH)
    release_spec = _release_spec(repository, git.head_commit, lock)
    tests = _test_manifest(repository, git.head_commit, lock)
    manifest = _artifact_bundle(artifacts, manifest_path, git.head_commit, git.head_tree)
    return R23ReleaseIdentity(
        SCHEMA,
        STATUS,
        git.head_commit,
        git.head_tree,
        release_spec,
        lock,
        tests,
        ApprovedDataStatus(NOT_RUN),
        _external_file(manifest_path, "artifact-manifest.json"),
        manifest.artifacts,
    )


def _validate_tag(repository: Path, expected_tag: str, head_commit: str) -> str:
    reference = f"refs/tags/{expected_tag}"
    if _git(repository, "cat-file", "-t", reference) != "tag":
        raise ValueError("release tag must be annotated")
    if _git(repository, "rev-parse", f"{reference}^{{commit}}") != head_commit:
        raise ValueError("release tag points to a different commit")
    return _git(repository, "rev-parse", reference)


def validate_release_identity(
    identity: R23ReleaseIdentity,
    repo_root: str | Path,
    artifact_dir: str | Path,
    artifact_manifest: str | Path,
    *,
    expected_tag: str | None = None,
) -> None:
    expected = calculate_release_identity(repo_root, artifact_dir, artifact_manifest)
    if identity != expected:
        raise ValueError("release identity or bound input drift")
    if expected_tag is not None:
        _validate_tag(Path(repo_root).resolve(), expected_tag, identity.head_commit)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_distribution_bundle(
    report: Path,
    artifact_dir: Path,
    artifact_manifest: Path,
) -> bool:
    if report.is_symlink() or not report.is_dir():
        raise ValueError("existing identity report must be a regular directory")
    members = {path.name for path in report.iterdir()}
    if not members:
        return False
    if artifact_dir.parent != report or artifact_manifest.parent != report:
        raise ValueError("existing identity report must own the artifact bundle")
    if members != {"artifacts", "artifact-manifest.json"}:
        raise ValueError("existing identity report member drift")
    return True


def _safe_output_path(value: str | Path, label: str) -> Path:
    raw = Path(value).absolute()
    current = raw
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} must not contain symlink components")
        current = current.parent
    return _resolve_output_path(raw, label)


def _cleanup_created_identity_report(report: Path, report_identity) -> None:
    try:
        _require_same_directory(report, report_identity, "identity report directory")
    except ValueError:
        return
    if report.is_dir() and not any(report.iterdir()):
        report.rmdir()


def build_release_identity(
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path,
    artifact_manifest: str | Path,
) -> Path:
    repository = Path(repo_root).resolve()
    report = _safe_output_path(report_dir, "identity report directory")
    artifacts = _safe_output_path(artifact_dir, "artifact directory")
    manifest = _safe_output_path(artifact_manifest, "artifact manifest")
    if report == repository or report.is_relative_to(repository):
        raise ValueError("identity report directory must be outside the repository")
    anchor = _make_report_anchor(report)
    identity = calculate_release_identity(repository, artifacts, manifest)
    created = False
    report_identity = None
    try:
        report_identity = _prepare_report_directory(report, anchor)
        created = anchor.leaf_identity is None
        if not created:
            _is_distribution_bundle(report, artifacts, manifest)
        _require_same_directory(report, report_identity, "identity report directory")
        _write_new_file_in_anchored_directory(
            report,
            report_identity,
            "release-identity.json",
            canonical_identity_bytes(identity),
            directory_label="identity report directory",
            file_label="release identity",
        )
    except BaseException:
        if created and report_identity is not None:
            _cleanup_created_identity_report(report, report_identity)
        raise
    return report / "release-identity.json"


def _atomic_file(path: Path, content: bytes) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or os.path.lexists(path):
        raise ValueError("atomic output must be a new regular-file path")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _freeze_value(
    identity: R23ReleaseIdentity,
    identity_path: Path,
    artifact_manifest: Path,
    tag: str,
    tag_object: str,
) -> dict[str, object]:
    identity_content = _read_regular(identity_path, "release identity")
    manifest_content = _read_regular(artifact_manifest, "artifact manifest")
    return {
        "schema": FREEZE_SCHEMA,
        "status": STATUS,
        "tag": tag,
        "tag_object": tag_object,
        "head_commit": identity.head_commit,
        "head_tree": identity.head_tree,
        "release_identity": {
            "path": "release-identity.json",
            "size": len(identity_content),
            "sha256": hashlib.sha256(identity_content).hexdigest(),
        },
        "artifact_manifest": {
            "path": "artifact-manifest.json",
            "size": len(manifest_content),
            "sha256": hashlib.sha256(manifest_content).hexdigest(),
        },
        "artifacts": [
            {
                "kind": item.kind,
                "path": item.path,
                "filename": item.filename,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in identity.artifacts
        ],
    }


def validate_identity_file(
    repo_root: str | Path,
    release_identity: str | Path,
    artifact_dir: str | Path,
    artifact_manifest: str | Path,
    *,
    expected_tag: str | None = None,
    write_freeze_receipt: str | Path | None = None,
) -> R23ReleaseIdentity:
    identity_path = Path(release_identity).resolve()
    identity = parse_release_identity(_read_regular(identity_path, "release identity"))
    validate_release_identity(
        identity,
        repo_root,
        artifact_dir,
        artifact_manifest,
        expected_tag=expected_tag,
    )
    if write_freeze_receipt is not None:
        if expected_tag is None:
            raise ValueError("freeze receipt requires --expected-tag")
        repository = Path(repo_root).resolve()
        tag_object = _validate_tag(repository, expected_tag, identity.head_commit)
        value = _freeze_value(
            identity,
            identity_path,
            Path(artifact_manifest).resolve(),
            expected_tag,
            tag_object,
        )
        _atomic_file(Path(write_freeze_receipt).resolve(), _canonical_value(value))
    return identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--report-dir", type=Path, required=True)
    build.add_argument("--artifact-dir", type=Path, required=True)
    build.add_argument("--artifact-manifest", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--release-identity", type=Path, required=True)
    validate.add_argument("--artifact-dir", type=Path, required=True)
    validate.add_argument("--artifact-manifest", type=Path, required=True)
    validate.add_argument("--expected-tag")
    validate.add_argument("--write-freeze-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        path = build_release_identity(
            args.repo_root,
            args.report_dir,
            args.artifact_dir,
            args.artifact_manifest,
        )
    else:
        validate_identity_file(
            args.repo_root,
            args.release_identity,
            args.artifact_dir,
            args.artifact_manifest,
            expected_tag=args.expected_tag,
            write_freeze_receipt=args.write_freeze_receipt,
        )
        path = args.release_identity
    print(json.dumps({"status": "PASS", "path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
