#!/usr/bin/env python3
"""Build, validate, smoke-test, and publish the canonical distribution bundle.

This entrypoint remains the public owner of the artifact-manifest contract. Its
imported names are intentionally reusable by release identity tooling, while
the CLI coordinates clean-HEAD builds without weakening those pure checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence


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
    _record as artifact_record,
    _record_value as artifact_value,
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
    distribution_inputs,
    release_spec,
    validate_bundle_paths,
)


def _publish_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir()
    for path in select_artifacts(source).values():
        target = destination / path.name
        with target.open("xb") as handle:
            handle.write(path.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    fsync_directory(destination)


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
) -> None:
    if artifacts.exists():
        shutil.rmtree(artifacts)
    manifest = report / "artifact-manifest.json"
    if manifest.exists():
        manifest.unlink()
    if report_created and report.exists() and not any(report.iterdir()):
        report.rmdir()


def _validate_build(
    repository: Path,
    first: Path,
    second: Path,
    inputs,
    spec: dict[str, object],
    epoch: int,
) -> None:
    verify_reproducible_artifacts(first, second)
    verify_archives(repository, first, inputs, spec)
    selected = select_artifacts(first)
    smoke_wheel(repository, selected["wheel"])
    smoke_sdist(repository, selected["sdist"], epoch)


def _publish_manifest(
    source: Path,
    report: Path,
    artifacts: Path,
    identity: GitIdentity,
) -> ArtifactManifest:
    report_created = _prepare_report(report, artifacts)
    try:
        _publish_artifacts(source, artifacts)
        manifest = calculate_artifact_manifest(
            artifacts,
            head_commit=identity.head_commit,
            head_tree=identity.head_tree,
        )
        write_artifact_manifest(report / "artifact-manifest.json", manifest)
    except Exception:
        _cleanup_failed_publish(report, artifacts, report_created)
        raise
    return manifest


def verify_distribution(
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path,
) -> ArtifactManifest:
    repository = Path(repo_root).resolve()
    report, artifacts = validate_bundle_paths(repository, report_dir, artifact_dir)
    identity = clean_head_identity(repository)
    check_build_versions()
    spec = release_spec(repository)
    inputs = distribution_inputs(repository, spec.get("sdist_content_policy"))
    with tempfile.TemporaryDirectory(prefix="xrr-r23-distribution-build-") as directory:
        build_root = Path(directory)
        first = build_once(
            repository,
            build_root / "first",
            inputs,
            identity.source_date_epoch,
        )
        second = build_once(
            repository,
            build_root / "second",
            inputs,
            identity.source_date_epoch,
        )
        _validate_build(
            repository,
            first,
            second,
            inputs,
            spec,
            identity.source_date_epoch,
        )
        return _publish_manifest(first, report, artifacts, identity)


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
