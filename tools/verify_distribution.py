#!/usr/bin/env python3
"""Build, validate, smoke-test, and publish the canonical distribution bundle.

This entrypoint remains the public owner of the artifact-manifest contract. Its
imported names are intentionally reusable by release identity tooling, while
the CLI coordinates clean-HEAD builds without weakening those pure checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from distribution_archive import (  # noqa: E402
    canonicalize_sdist,
    verify_archives,
    verify_reproducible_artifacts,
    verify_wheel_dependencies,
)
from distribution_manifest import (  # noqa: E402
    ARTIFACT_KINDS,
    SCHEMA,
    STATUS,
    ArtifactManifest,
    ArtifactRecord,
    calculate_artifact_manifest,
    canonical_manifest_bytes,
    fsync_directory,
    git_oid,
    manifest_value,
    parse_artifact_manifest,
    read_artifact_manifest,
    select_artifacts,
    validate_artifact_manifest,
    write_artifact_manifest,
)
from distribution_manifest import (  # noqa: E402
    _record as artifact_record,
)
from distribution_manifest import (  # noqa: E402
    _record_value as artifact_value,
)
from distribution_smoke import (  # noqa: E402
    smoke_installed,
    smoke_sdist,
    smoke_wheel,
)
from distribution_source import (  # noqa: E402
    GitIdentity,
    build_once,
    check_build_versions,
    clean_head_identity,
    committed_blob,
    committed_tree_files,
    distribution_inputs,
    release_spec,
    validate_bundle_paths,
)
from verify_report import (  # noqa: E402
    _directory_identity,
    _ensure_cache_directory,
    _make_report_anchor,
    _make_report_guard,
    _prepare_report_directory,
    _require_same_directory,
    _write_new_file_in_anchored_directory,
)

__all__ = (
    "ARTIFACT_KINDS",
    "SCHEMA",
    "STATUS",
    "ArtifactManifest",
    "ArtifactRecord",
    "GitIdentity",
    "artifact_record",
    "artifact_value",
    "build_once",
    "calculate_artifact_manifest",
    "canonical_manifest_bytes",
    "canonicalize_sdist",
    "check_build_versions",
    "clean_head_identity",
    "committed_blob",
    "committed_tree_files",
    "distribution_inputs",
    "fsync_directory",
    "git_oid",
    "manifest_value",
    "parse_artifact_manifest",
    "read_artifact_manifest",
    "release_spec",
    "select_artifacts",
    "smoke_installed",
    "smoke_sdist",
    "smoke_wheel",
    "validate_artifact_manifest",
    "validate_bundle_paths",
    "verify_archives",
    "verify_distribution",
    "verify_reproducible_artifacts",
    "verify_wheel_dependencies",
    "write_artifact_manifest",
)


def _artifact_record_from_content(
    kind: str,
    filename: str,
    content: bytes,
) -> ArtifactRecord:
    return ArtifactRecord(
        kind,
        f"artifacts/{filename}",
        filename,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def _publish_artifacts(
    source: Path,
    destination: Path,
    destination_identity,
) -> tuple[ArtifactRecord, ArtifactRecord]:
    _require_same_directory(destination, destination_identity, "artifact directory")
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("artifact directory must be a regular directory")
    if any(destination.iterdir()):
        raise ValueError("artifact directory must be empty")
    selected = select_artifacts(source)
    records: list[ArtifactRecord] = []
    for kind in ARTIFACT_KINDS:
        path = selected[kind]
        content = path.read_bytes()
        _write_new_file_in_anchored_directory(
            destination,
            destination_identity,
            path.name,
            content,
            directory_label="artifact directory",
            file_label=f"{kind} artifact",
        )
        records.append(_artifact_record_from_content(kind, path.name, content))
    _require_same_directory(destination, destination_identity, "artifact directory")
    return records[0], records[1]


def _prepare_report(report: Path, artifacts: Path) -> bool:
    if artifacts.exists() or (report / "artifact-manifest.json").exists():
        raise ValueError("distribution output paths must not already exist")
    if report.exists():
        if report.is_symlink() or not report.is_dir() or any(report.iterdir()):
            raise ValueError("distribution report directory must be new or empty")
        return False
    report.mkdir(parents=True)
    return True


def _cleanup_failed_publish(
    report: Path,
    artifacts: Path,
    report_created: bool,
    report_identity,
    artifact_identity,
) -> None:
    report_matches = _directory_still_matches(report, report_identity, "report directory")
    if _directory_still_matches(artifacts, artifact_identity, "artifact directory"):
        shutil.rmtree(artifacts)
    if report_matches:
        manifest = report / "artifact-manifest.json"
        if os.path.lexists(manifest) and not manifest.is_dir():
            manifest.unlink()
        if report_created and not any(report.iterdir()):
            report.rmdir()


def _directory_still_matches(path: Path, identity, label: str) -> bool:
    try:
        _require_same_directory(path, identity, label)
    except ValueError:
        return False
    return True


def _captured_staging_root(artifacts: Path) -> Path:
    source = artifacts.parent / "staging"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("captured staging root must be a regular directory")
    return source


def _validate_build(
    _repository: Path,
    first: Path,
    second: Path,
    inputs,
    spec: dict[str, object],
    epoch: int,
) -> None:
    source_root = _captured_staging_root(first)
    verify_reproducible_artifacts(first, second)
    verify_archives(source_root, first, inputs, spec)
    selected = select_artifacts(first)
    smoke_wheel(source_root, selected["wheel"])
    smoke_sdist(source_root, selected["sdist"], epoch)


def _publish_manifest(
    source: Path,
    report: Path,
    artifacts: Path,
    identity: GitIdentity,
    report_anchor,
) -> ArtifactManifest:
    if os.path.lexists(artifacts) or os.path.lexists(report / "artifact-manifest.json"):
        raise ValueError("distribution output paths must not already exist")
    report_created = report_anchor.leaf_identity is None
    report_identity = _prepare_report_directory(report, report_anchor)
    if not report_created and any(report.iterdir()):
        raise ValueError("distribution report directory must be new or empty")
    artifact_identity = _ensure_cache_directory(report, report_identity, "artifacts")
    if artifact_identity != _directory_identity(artifacts, "artifact directory"):
        raise ValueError("artifact directory changed during validation")
    report_guard = _make_report_guard(
        report,
        report_identity,
        watched=((artifacts, "artifact directory"),),
    )
    try:
        report_guard()
        records = _publish_artifacts(source, artifacts, artifact_identity)
        report_guard()
        manifest = ArtifactManifest(
            SCHEMA,
            STATUS,
            git_oid(identity.head_commit, "head commit"),
            git_oid(identity.head_tree, "head tree"),
            records,
        )
        _write_new_file_in_anchored_directory(
            report,
            report_identity,
            "artifact-manifest.json",
            canonical_manifest_bytes(manifest),
            directory_label="report directory",
            file_label="artifact manifest",
        )
        report_guard()
    except Exception:
        _cleanup_failed_publish(
            report,
            artifacts,
            report_created,
            report_identity,
            artifact_identity,
        )
        raise
    return manifest


def verify_distribution(
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path,
) -> ArtifactManifest:
    repository = Path(repo_root).resolve()
    report, artifacts = validate_bundle_paths(repository, report_dir, artifact_dir)
    report_anchor = _make_report_anchor(report)
    identity = clean_head_identity(repository)
    check_build_versions()
    spec = release_spec(repository, identity.head_commit)
    inputs = distribution_inputs(
        repository,
        spec.get("sdist_content_policy"),
        identity.head_commit,
    )
    with tempfile.TemporaryDirectory(prefix="xrr-r23-distribution-build-") as directory:
        build_root = Path(directory)
        first = build_once(
            repository,
            build_root / "first",
            inputs,
            identity.source_date_epoch,
            identity.head_commit,
        )
        second = build_once(
            repository,
            build_root / "second",
            inputs,
            identity.source_date_epoch,
            identity.head_commit,
        )
        _validate_build(
            repository,
            first,
            second,
            inputs,
            spec,
            identity.source_date_epoch,
        )
        return _publish_manifest(first, report, artifacts, identity, report_anchor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = verify_distribution(args.repo_root, args.report_dir, args.artifact_dir)
    print(
        json.dumps(
            {
                "status": manifest.status,
                "head_commit": manifest.head_commit,
                "head_tree": manifest.head_tree,
                "artifact_count": len(manifest.artifacts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
