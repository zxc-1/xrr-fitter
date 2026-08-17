from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest


def _commit_all(path: Path, message: str, when: str = "2024-01-02T03:04:05+00:00") -> str:
    subprocess.run(("git", "add", "-A"), cwd=path, check=True)
    environment = os.environ.copy()
    environment.update({"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when})
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            message,
        ),
        cwd=path,
        check=True,
        env=environment,
    )
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=path, text=True).strip()


def _write_release_spec(path: Path, input_files: tuple[str, ...]) -> None:
    release_spec = path / "verification" / "release-spec.json"
    release_spec.parent.mkdir(exist_ok=True)
    release_spec.write_text(
        json.dumps(
            {"sdist_content_policy": {"input_directories": ["verification"], "input_files": list(input_files)}},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_clean_head_identity_binds_tree_and_epoch_to_captured_commit_oid(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    (repository / "tracked.txt").write_text("captured\n", encoding="utf-8")
    captured_commit = _commit_all(repository, "captured")
    captured_tree = subprocess.check_output(
        ("git", "rev-parse", f"{captured_commit}^{{tree}}"), cwd=repository, text=True
    ).strip()
    captured_epoch = int(
        subprocess.check_output(("git", "show", "-s", "--format=%ct", captured_commit), cwd=repository, text=True)
    )
    (repository / "tracked.txt").write_text("later\n", encoding="utf-8")
    later_commit = _commit_all(repository, "later", "2024-01-03T03:04:05+00:00")
    subprocess.run(("git", "checkout", "-q", captured_commit), cwd=repository, check=True)
    original_git = module._git

    def checkout_after_head_capture(path: Path, *args: str) -> str:
        value = original_git(path, *args)
        if args == ("rev-parse", "HEAD"):
            subprocess.run(("git", "checkout", "-q", later_commit), cwd=path, check=True)
        return value

    monkeypatch.setattr(module, "_git", checkout_after_head_capture)
    assert module.clean_head_identity(repository) == module.GitIdentity(captured_commit, captured_tree, captured_epoch)


def test_verify_distribution_uses_release_spec_and_paths_from_captured_commit(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    (repository / "tracked.txt").write_text("captured\n", encoding="utf-8")
    _write_release_spec(repository, ("tracked.txt",))
    captured_commit = _commit_all(repository, "captured")
    (repository / "tracked.txt").unlink()
    (repository / "other.txt").write_text("later\n", encoding="utf-8")
    _write_release_spec(repository, ("other.txt",))
    later_commit = _commit_all(repository, "later")
    subprocess.run(("git", "checkout", "-q", captured_commit), cwd=repository, check=True)
    original_clean_head_identity = module.clean_head_identity
    build_calls: list[tuple[tuple[PurePosixPath, ...], str]] = []

    def racing_clean_head_identity(path: Path):
        identity = original_clean_head_identity(path)
        assert identity.head_commit == captured_commit
        subprocess.run(("git", "checkout", "-q", later_commit), cwd=path, check=True)
        return identity

    def record_build_once(
        _repository: Path,
        destination: Path,
        inputs: tuple[PurePosixPath, ...],
        _epoch: int,
        commit: str,
    ) -> Path:
        build_calls.append((inputs, commit))
        artifacts = destination / "artifacts"
        artifacts.mkdir(parents=True)
        return artifacts

    monkeypatch.setattr(module, "clean_head_identity", racing_clean_head_identity)
    monkeypatch.setattr(module, "check_build_versions", lambda: None)
    monkeypatch.setattr(module, "build_once", record_build_once)
    monkeypatch.setattr(module, "_validate_build", lambda *_args: None)
    monkeypatch.setattr(module, "_publish_manifest", lambda *_args: "published")
    report = tmp_path / "bundle"
    result = module.verify_distribution(repository, report, report / "artifacts")
    expected_inputs = (PurePosixPath("tracked.txt"), PurePosixPath("verification/release-spec.json"))
    assert result == "published"
    assert build_calls == [(expected_inputs, captured_commit), (expected_inputs, captured_commit)]


def test_verify_distribution_rejects_report_parent_replacement_before_publish(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    (repository / "tracked.txt").write_text("captured\n", encoding="utf-8")
    _commit_all(repository, "captured")
    external = tmp_path / "external"
    external.mkdir()
    report = external / "release"
    artifacts = report / "artifacts"
    original_clean_head_identity = module.clean_head_identity

    def replace_parent_after_validation(path: Path):
        identity = original_clean_head_identity(path)
        external.rmdir()
        external.symlink_to(repository, target_is_directory=True)
        return identity

    def fake_build_once(
        _repository: Path,
        destination: Path,
        _inputs: tuple[PurePosixPath, ...],
        _epoch: int,
        _commit: str,
    ) -> Path:
        output = destination / "artifacts"
        output.mkdir(parents=True)
        (output / "xrr_fitter-0.2.0.tar.gz").write_bytes(b"sdist")
        (output / "xrr_fitter-0.2.0-py3-none-any.whl").write_bytes(b"wheel")
        return output

    monkeypatch.setattr(module, "clean_head_identity", replace_parent_after_validation)
    monkeypatch.setattr(module, "check_build_versions", lambda: None)
    monkeypatch.setattr(module, "release_spec", lambda *_args: {})
    monkeypatch.setattr(module, "distribution_inputs", lambda *_args: ())
    monkeypatch.setattr(module, "build_once", fake_build_once)
    monkeypatch.setattr(module, "_validate_build", lambda *_args: None)

    with pytest.raises(ValueError, match="report parent (changed|must not be a symlink)"):
        module.verify_distribution(repository, report, artifacts)

    assert not (repository / "release").exists()


def _git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=path, check=True)
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=path, check=True)
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2024-01-02T03:04:05+00:00",
            "GIT_COMMITTER_DATE": "2024-01-02T03:04:05+00:00",
        }
    )
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=path,
        check=True,
        env=environment,
    )


def test_git_identity_and_source_date_epoch_come_from_clean_head(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify_distribution")
    repository = tmp_path / "repo"
    _git_repository(repository)

    identity = module.clean_head_identity(repository)
    expected_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repository, text=True).strip()
    expected_tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=repository, text=True).strip()
    expected_epoch = int(
        subprocess.check_output(("git", "show", "-s", "--format=%ct", "HEAD"), cwd=repository, text=True).strip()
    )
    assert identity == module.GitIdentity(expected_commit, expected_tree, expected_epoch)

    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        module.clean_head_identity(repository)


def test_validate_build_uses_captured_staging_root_for_metadata_and_smoke(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify_distribution")
    repository = tmp_path / "repo"
    repository.mkdir()
    first = tmp_path / "first" / "artifacts"
    second = tmp_path / "second" / "artifacts"
    staging = first.parent / "staging"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    staging.mkdir()
    wheel = first / "xrr_fitter-0.2.0-py3-none-any.whl"
    sdist = first / "xrr_fitter-0.2.0.tar.gz"
    inputs = (PurePosixPath("pyproject.toml"),)
    spec = {"sdist_content_policy": {}, "wheel_content_policy": {}}
    observed_roots: list[Path] = []

    monkeypatch.setattr(module, "verify_reproducible_artifacts", lambda *_args: None)
    monkeypatch.setattr(module, "select_artifacts", lambda _artifacts: {"wheel": wheel, "sdist": sdist})

    def record_archives(root: Path, artifact_dir: Path, *_args) -> None:
        assert artifact_dir == first
        observed_roots.append(root)

    def record_wheel(root: Path, selected_wheel: Path) -> None:
        assert selected_wheel == wheel
        observed_roots.append(root)

    def record_sdist(root: Path, selected_sdist: Path, epoch: int) -> None:
        assert selected_sdist == sdist
        assert epoch == 123
        observed_roots.append(root)

    monkeypatch.setattr(module, "verify_archives", record_archives)
    monkeypatch.setattr(module, "smoke_wheel", record_wheel)
    monkeypatch.setattr(module, "smoke_sdist", record_sdist)

    module._validate_build(repository, first, second, inputs, spec, 123)

    assert observed_roots == [staging, staging, staging]


def test_copy_inputs_reads_head_blobs_instead_of_mutable_worktree(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    _git_repository(repository)
    identity = module.clean_head_identity(repository)
    staging = tmp_path / "staging"
    staging.mkdir()

    # Simulate a worktree race after the clean-HEAD identity was captured.
    (repository / "tracked.txt").write_text("raced\n", encoding="utf-8")
    module._copy_inputs(
        repository,
        staging,
        (PurePosixPath("tracked.txt"),),
        identity.source_date_epoch,
        identity.head_commit,
    )

    assert (staging / "tracked.txt").read_text(encoding="utf-8") == "tracked\n"


def test_copy_inputs_uses_raw_git_blob_bytes_without_export_substitution(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    (repository / ".gitattributes").write_text("substituted.txt export-subst\n", encoding="utf-8")
    (repository / "substituted.txt").write_text("$Format:%H$\n", encoding="utf-8")
    commit = _commit_all(repository, "export-subst")
    raw_blob = subprocess.check_output(
        ("git", "cat-file", "blob", f"{commit}:substituted.txt"),
        cwd=repository,
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    module._copy_inputs(
        repository,
        staging,
        (PurePosixPath("substituted.txt"),),
        123,
        commit,
    )

    assert (staging / "substituted.txt").read_bytes() == raw_blob


def test_copy_inputs_preserves_executable_git_blob_mode(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    script = repository / "run-tool"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    commit = _commit_all(repository, "executable")
    staging = tmp_path / "staging"
    staging.mkdir()

    module._copy_inputs(
        repository,
        staging,
        (PurePosixPath("run-tool"),),
        123,
        commit,
    )

    mode = stat.S_IMODE((staging / "run-tool").stat().st_mode)
    assert mode == 0o755


def test_distribution_source_rejects_tree_oid_as_captured_commit(
    tmp_path: Path,
    load_tool_module,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    _git_repository(repository)
    tree_oid = subprocess.check_output(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=repository,
        text=True,
    ).strip()
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="commit"):
        module._copy_inputs(
            repository,
            staging,
            (PurePosixPath("tracked.txt"),),
            123,
            tree_oid,
        )
    with pytest.raises(ValueError, match="commit"):
        module.build_once(
            repository,
            tmp_path / "build",
            (PurePosixPath("tracked.txt"),),
            123,
            tree_oid,
        )


def test_build_once_runs_builder_from_captured_staging_root(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    _git_repository(repository)
    identity = module.clean_head_identity(repository)
    destination = tmp_path / "build"
    build_cwds: list[Path] = []
    original_run = module.subprocess.run

    def record_build_run(args, **kwargs):
        if tuple(args[:3]) == (sys.executable, "-m", "build"):
            build_cwds.append(Path(kwargs["cwd"]))
            return subprocess.CompletedProcess(args, 0, "", "")
        return original_run(args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", record_build_run)
    monkeypatch.setattr(
        module,
        "select_artifacts",
        lambda artifacts: {
            "sdist": artifacts / "xrr_fitter-0.2.0.tar.gz",
            "wheel": artifacts / "xrr_fitter-0.2.0-py3-none-any.whl",
        },
    )
    monkeypatch.setattr(module, "canonicalize_sdist", lambda *_args: None)

    module.build_once(
        repository,
        destination,
        (PurePosixPath("tracked.txt"),),
        identity.source_date_epoch,
        identity.head_commit,
    )

    assert build_cwds == [destination / "staging"]


@pytest.mark.parametrize(
    ("content", "match"),
    (
        (b"\xff", "release spec.*UTF-8"),
        (b"not-json\n", "release spec JSON"),
    ),
)
def test_release_spec_reports_decode_and_parse_errors_as_value_errors(
    tmp_path: Path,
    load_tool_module,
    content: bytes,
    match: str,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    release_spec = repository / "verification" / "release-spec.json"
    release_spec.parent.mkdir()
    release_spec.write_bytes(content)
    commit = _commit_all(repository, "invalid-release-spec")

    with pytest.raises(ValueError, match=match):
        module.release_spec(repository, commit)


@pytest.mark.parametrize("entrypoint", ("copy-inputs", "build-once"))
@pytest.mark.parametrize(
    "commit",
    (
        "HEAD~1",
        "A" * 40,
        "1" * 39,
        "g" * 40,
    ),
)
def test_copy_inputs_and_build_once_require_lowercase_forty_hex_commit(
    tmp_path: Path,
    load_tool_module,
    entrypoint: str,
    commit: str,
) -> None:
    load_tool_module("verify_distribution")
    module = load_tool_module("distribution_source")
    repository = tmp_path / "repo"
    _git_repository(repository)

    with pytest.raises(ValueError, match="commit"):
        if entrypoint == "copy-inputs":
            staging = tmp_path / "staging"
            staging.mkdir()
            module._copy_inputs(
                repository,
                staging,
                (PurePosixPath("tracked.txt"),),
                123,
                commit,
            )
        else:
            module.build_once(
                repository,
                tmp_path / "build",
                (PurePosixPath("tracked.txt"),),
                123,
                commit,
            )
