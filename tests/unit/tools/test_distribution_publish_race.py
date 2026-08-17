from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from tests.unit.tools.test_verify_distribution import COMMIT, TREE, _artifact_dir, _manifest


def test_atomic_writer_is_repeatable_and_leaves_no_partial_on_replace_failure(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    _directory, manifest = _manifest(module, tmp_path)
    report = tmp_path / "report"
    report.mkdir()
    target = report / "artifact-manifest.json"

    assert module.write_artifact_manifest(target, manifest) == target
    first = target.read_bytes()
    assert module.write_artifact_manifest(target, manifest) == target
    assert target.read_bytes() == first

    target.unlink()
    monkeypatch.setattr(module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        module.write_artifact_manifest(target, manifest)
    assert not target.exists()
    assert tuple(report.iterdir()) == ()


def test_pathname_fallback_rejects_target_created_before_hardlink_publish(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    report = tmp_path / "report"
    report.mkdir()
    helper = module._write_new_file_in_anchored_directory
    helper_globals = helper.__globals__
    target = report / "artifact-manifest.json"
    identity = helper_globals["_directory_identity"](report, "report directory")
    original_link = helper_globals["os"].link

    monkeypatch.setitem(helper_globals, "DIRECTORY_FD_SUPPORTED", False)
    monkeypatch.setitem(helper_globals, "ANCHORED_FILE_FD_SUPPORTED", False)

    def create_target_then_link(source, destination, *args, **kwargs):
        Path(destination).write_bytes(b"replacement")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(helper_globals["os"], "link", create_target_then_link)

    with pytest.raises(ValueError, match="appeared during validation"):
        helper(
            report,
            identity,
            target.name,
            b"published",
            directory_label="report directory",
            file_label="manifest",
        )

    assert target.read_bytes() == b"replacement"


def test_publish_manifest_does_not_cleanup_replacement_artifact_directory(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    source = _artifact_dir(tmp_path / "source")
    report = tmp_path / "bundle"
    artifacts = report / "artifacts"
    moved_artifacts = tmp_path / "captured-artifacts"
    replacement_sentinel = artifacts / "replacement-sentinel.txt"
    original_select = module.select_artifacts
    swapped = False

    def select_then_swap(path: Path):
        nonlocal swapped
        selected = original_select(path)
        if path == source and not swapped:
            artifacts.rename(moved_artifacts)
            artifacts.mkdir()
            replacement_sentinel.write_text("replacement\n", encoding="utf-8")
            swapped = True
        return selected

    monkeypatch.setattr(module, "select_artifacts", select_then_swap)

    with pytest.raises(ValueError, match="artifact directory"):
        module._publish_manifest(
            source,
            report,
            artifacts,
            module.GitIdentity(COMMIT, TREE, 123),
            module._make_report_anchor(report),
        )

    assert swapped
    assert replacement_sentinel.read_text(encoding="utf-8") == "replacement\n"


def test_publish_manifest_does_not_reread_artifact_pathname_for_manifest_bytes(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    source = _artifact_dir(tmp_path / "source")
    report = tmp_path / "bundle"
    artifacts = report / "artifacts"
    moved_artifacts = tmp_path / "published-artifacts"
    original_read_bytes = Path.read_bytes
    armed = True
    swapped = False

    def read_bytes_during_artifact_aba(path: Path) -> bytes:
        nonlocal swapped
        target = Path(path)
        if armed and target.parent == artifacts and not swapped:
            artifacts.rename(moved_artifacts)
            artifacts.mkdir()
            for published in moved_artifacts.iterdir():
                (artifacts / published.name).write_bytes(f"replacement:{published.name}".encode())
            try:
                return original_read_bytes(target)
            finally:
                for child in artifacts.iterdir():
                    child.unlink()
                artifacts.rmdir()
                moved_artifacts.rename(artifacts)
                swapped = True
        return original_read_bytes(target)

    monkeypatch.setattr(Path, "read_bytes", read_bytes_during_artifact_aba)
    try:
        manifest = module._publish_manifest(
            source,
            report,
            artifacts,
            module.GitIdentity(COMMIT, TREE, 123),
            module._make_report_anchor(report),
        )
    finally:
        armed = False

    assert not swapped
    for record in manifest.artifacts:
        published = artifacts / record.filename
        content = published.read_bytes()
        assert record.size == len(content)
        assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert module.read_artifact_manifest(report / "artifact-manifest.json") == manifest


def test_publish_manifest_does_not_cleanup_replacement_report_directory(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    source = _artifact_dir(tmp_path / "source")
    report = tmp_path / "bundle"
    artifacts = report / "artifacts"
    moved_report = tmp_path / "captured-report"
    replacement_manifest = report / "artifact-manifest.json"
    replacement_artifact_sentinel = report / "artifacts" / "replacement-sentinel.txt"
    original_publish = module._publish_artifacts
    swapped = False

    def publish_then_swap(*args):
        nonlocal swapped
        result = original_publish(*args)
        report.rename(moved_report)
        (report / "artifacts").mkdir(parents=True)
        replacement_manifest.write_bytes(b"replacement manifest\n")
        replacement_artifact_sentinel.write_text("replacement artifact\n", encoding="utf-8")
        swapped = True
        return result

    monkeypatch.setattr(module, "_publish_artifacts", publish_then_swap)

    with pytest.raises(ValueError, match="report directory|artifact directory"):
        module._publish_manifest(
            source,
            report,
            artifacts,
            module.GitIdentity(COMMIT, TREE, 123),
            module._make_report_anchor(report),
        )

    assert swapped
    assert replacement_manifest.read_bytes() == b"replacement manifest\n"
    assert replacement_artifact_sentinel.read_text(encoding="utf-8") == "replacement artifact\n"


def test_publish_manifest_uses_anchored_helper_for_artifacts_and_manifest(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    source = _artifact_dir(tmp_path / "source")
    report = tmp_path / "bundle"
    artifacts = report / "artifacts"
    original_helper = module._write_new_file_in_anchored_directory
    calls: list[tuple[Path, str]] = []

    def record_helper(directory: Path, _identity, name: str, content: bytes, **kwargs) -> None:
        assert content
        calls.append((directory, name))
        original_helper(directory, _identity, name, content, **kwargs)

    monkeypatch.setattr(module, "_write_new_file_in_anchored_directory", record_helper)

    module._publish_manifest(
        source,
        report,
        artifacts,
        module.GitIdentity(COMMIT, TREE, 123),
        module._make_report_anchor(report),
    )

    assert calls == [
        (artifacts, "xrr_fitter-0.2.0.tar.gz"),
        (artifacts, "xrr_fitter-0.2.0-py3-none-any.whl"),
        (report, "artifact-manifest.json"),
    ]


def test_bundle_paths_must_be_external_and_use_exact_layout(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    repository = tmp_path / "repo"
    repository.mkdir()
    report = tmp_path / "bundle"
    artifacts = report / "artifacts"

    assert module.validate_bundle_paths(repository, report, artifacts) == (
        report.resolve(),
        artifacts.resolve(),
    )
    with pytest.raises(ValueError, match="artifact|report"):
        module.validate_bundle_paths(repository, report, tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="outside"):
        module.validate_bundle_paths(repository, repository / "report", repository / "report/artifacts")


def test_bundle_paths_reject_report_symlink_before_resolving(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    repository = tmp_path / "repo"
    repository.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    report = tmp_path / "report"
    report.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.validate_bundle_paths(repository, report, report / "artifacts")
