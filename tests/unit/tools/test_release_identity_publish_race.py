from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from tests.unit.tools.test_release_identity import NOT_RUN, _fixture_repo, _git


def test_build_is_atomic_and_records_not_run_status(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = tmp_path / "identity-report"

    target = module.build_release_identity(root, report, artifacts, artifact_manifest)
    assert target == report / "release-identity.json"
    assert json.loads(target.read_text(encoding="utf-8"))["approved_data"] == {"status": NOT_RUN}

    failed = tmp_path / "failed-report"
    monkeypatch.setattr(
        module,
        "_write_new_file_in_anchored_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("publish failed")),
    )
    with pytest.raises(OSError, match="publish failed"):
        module.build_release_identity(root, failed, artifacts, artifact_manifest)
    assert not failed.exists()


def test_build_rejects_a_report_path_resolving_inside_the_repository(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    external = tmp_path / "external"
    external.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlink|outside the repository"):
        module.build_release_identity(
            root,
            external / "release",
            artifacts,
            artifact_manifest,
        )

    assert not (root / "release").exists()


def test_build_rejects_regular_directory_replacement_before_identity_publish(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = tmp_path / "identity-report"
    moved = tmp_path / "moved-identity-report"
    original_check = module._require_same_directory
    swapped = False

    def check(path, identity, label):
        nonlocal swapped
        original_check(path, identity, label)
        if label == "identity report directory" and not swapped:
            report.rename(moved)
            report.mkdir()
            swapped = True

    monkeypatch.setattr(module, "_require_same_directory", check)

    with pytest.raises(ValueError, match="identity report directory|report directory"):
        module.build_release_identity(root, report, artifacts, artifact_manifest)

    assert swapped
    assert report.is_dir()
    assert not (report / "release-identity.json").exists()
    assert not (moved / "release-identity.json").exists()


def test_build_pathname_fallback_rechecks_report_identity_after_publish(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = tmp_path / "identity-report"
    moved = tmp_path / "moved-identity-report"
    helper_globals = module._write_new_file_in_anchored_directory.__globals__
    original_fsync = helper_globals["os"].fsync
    fsync_calls = 0
    swapped = False

    monkeypatch.setitem(helper_globals, "DIRECTORY_FD_SUPPORTED", False)
    monkeypatch.setitem(helper_globals, "ANCHORED_FILE_FD_SUPPORTED", False)

    def fsync(descriptor):
        nonlocal fsync_calls, swapped
        fsync_calls += 1
        if fsync_calls == 2 and not swapped:
            report.rename(moved)
            report.mkdir()
            swapped = True
        return original_fsync(descriptor)

    monkeypatch.setattr(helper_globals["os"], "fsync", fsync)

    with pytest.raises(ValueError, match="identity report directory|report directory"):
        module.build_release_identity(root, report, artifacts, artifact_manifest)

    assert swapped
    assert report.is_dir()
    assert not (report / "release-identity.json").exists()
    assert not (moved / "release-identity.json").exists()


def test_build_adds_identity_to_an_existing_distribution_bundle(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = artifacts.parent

    target = module.build_release_identity(root, report, artifacts, artifact_manifest)

    assert target == report / "release-identity.json"
    assert {path.name for path in report.iterdir()} == {
        "artifact-manifest.json",
        "artifacts",
        "release-identity.json",
    }
    module.validate_identity_file(root, target, artifacts, artifact_manifest)


def test_annotated_tag_validation_writes_bound_freeze_receipt(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = tmp_path / "identity"
    identity_path = module.build_release_identity(root, report, artifacts, artifact_manifest)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-a",
            "R23-final",
            "-m",
            "fixture final",
        ),
        cwd=root,
        check=True,
    )
    receipt = tmp_path / "r23-final-freeze.json"

    module.validate_identity_file(
        root,
        identity_path,
        artifacts,
        artifact_manifest,
        expected_tag="R23-final",
        write_freeze_receipt=receipt,
    )

    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["schema"] == "xrr-r23-final-freeze-v1"
    assert value["status"] == "PASS"
    assert value["tag"] == "R23-final"
    assert value["head_commit"] == _git(root, "rev-parse", "HEAD")
    assert value["release_identity"]["sha256"] == hashlib.sha256(identity_path.read_bytes()).hexdigest()


def test_freeze_receipt_publish_is_atomic_on_replace_failure(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    identity_path = module.build_release_identity(
        root,
        tmp_path / "identity",
        artifacts,
        artifact_manifest,
    )
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-a",
            "R23-final",
            "-m",
            "fixture final",
        ),
        cwd=root,
        check=True,
    )
    receipt = tmp_path / "r23-final-freeze.json"
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        module.validate_identity_file(
            root,
            identity_path,
            artifacts,
            artifact_manifest,
            expected_tag="R23-final",
            write_freeze_receipt=receipt,
        )

    assert not receipt.exists()
    assert not tuple(tmp_path.glob(".r23-final-freeze.json.*.tmp"))
