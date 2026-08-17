from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_verify_report_reexports_anchored_publish_helper(load_tool_module) -> None:
    report = load_tool_module("verify_report")

    assert report._write_new_file_in_anchored_directory.__module__ == "verify_publish"


def _swap_existing_leaf(report: Path, target: Path) -> None:
    moved = report.with_name(f"{report.name}-moved")
    report.rename(moved)
    report.symlink_to(target, target_is_directory=True)


def _redirect_leaf_to_moved_directory(report: Path, target: Path) -> None:
    report.rename(target)
    report.symlink_to(target, target_is_directory=True)


def _replace_leaf_with_directory(report: Path) -> None:
    moved = report.with_name(f"{report.name}-moved")
    report.rename(moved)
    report.mkdir()


def _swap_materialize(module, monkeypatch: pytest.MonkeyPatch, report: Path, target: Path) -> None:
    original = module._materialize
    swapped = False

    def materialize(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            if os.path.lexists(report):
                _swap_existing_leaf(report, target)
            else:
                report.symlink_to(target, target_is_directory=True)
            swapped = True
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_materialize", materialize)


def test_regular_mode_rejects_swap_after_prepare_before_cache_creation(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    root.mkdir()
    report = tmp_path / "report"
    target = tmp_path / "target"
    original_check = module._require_same_directory
    swapped = False

    def check(path, identity, label):
        nonlocal swapped
        original_check(path, identity, label)
        if not swapped:
            _redirect_leaf_to_moved_directory(report, target)
            swapped = True

    monkeypatch.setattr(module, "_require_same_directory", check)
    with pytest.raises(ValueError, match="report directory"):
        module.run_mode(
            "fixture",
            module.Mode(commands=(("would-run",),)),
            repo_root=root,
            report_dir=report,
            runner=lambda *_args, **_kwargs: pytest.fail("runner executed"),
        )

    assert not (target / "mpl-cache").exists()
    assert not (target / "xdg-cache").exists()


@pytest.mark.parametrize("mode", ("quality", "approved-data", "identity"))
def test_modes_reject_leaf_swap_before_workflow_runner(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    report = tmp_path / f"{mode}-report"
    target = tmp_path / f"{mode}-target"
    target.mkdir()
    calls: list[tuple[str, ...]] = []
    _swap_materialize(module, monkeypatch, report, target)

    kwargs: dict[str, object] = {}
    if mode == "approved-data":
        owner = tmp_path / "owner-data"
        owner.mkdir()
        kwargs.update(approved_data_root=owner, capture_candidate=True)
    elif mode == "identity":
        bundle = tmp_path / "bundle"
        (bundle / "artifacts").mkdir(parents=True)
        (bundle / "artifact-manifest.json").write_bytes(b"manifest")
        kwargs.update(
            artifact_dir=bundle / "artifacts",
            artifact_manifest=bundle / "artifact-manifest.json",
        )

    with pytest.raises(ValueError, match="symlink|report directory"):
        module.run_mode(
            mode,
            module.MODE_REGISTRY[mode] if mode != "quality" else module.Mode(commands=(("would-run",),)),
            repo_root=root,
            report_dir=report,
            runner=lambda args, **_kwargs: calls.append(tuple(args)),
            **kwargs,
        )

    assert not any(
        "would-run" in args or "tests/acceptance/" in args or "tools/release_identity.py" in args for args in calls
    )


@pytest.mark.parametrize("mode", ("fixture", "approved-data"))
def test_modes_reject_parent_swap_after_input_validation(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    parent = tmp_path / "external-parent"
    parent.mkdir()
    report = parent / "report"
    original_validate = module._validate_mode_inputs

    def validate_then_swap(*args, **kwargs) -> None:
        original_validate(*args, **kwargs)
        parent.rmdir()
        parent.symlink_to(root, target_is_directory=True)

    monkeypatch.setattr(module, "_validate_mode_inputs", validate_then_swap)
    kwargs: dict[str, object] = {}
    if mode == "approved-data":
        owner = tmp_path / "owner-data"
        owner.mkdir()
        kwargs.update(approved_data_root=owner, capture_candidate=True)

    with pytest.raises(ValueError, match="report directory|parent"):
        module.run_mode(
            mode,
            module.Mode(commands=(("would-run",),)),
            repo_root=root,
            report_dir=report,
            runner=lambda *_args, **_kwargs: None,
            **kwargs,
        )

    assert not (root / "report").exists()


def test_identity_rejects_regular_directory_replacement_after_build(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    bundle = tmp_path / "bundle"
    artifacts = bundle / "artifacts"
    artifacts.mkdir(parents=True)
    manifest = bundle / "artifact-manifest.json"
    manifest.write_bytes(b"manifest")
    report = tmp_path / "identity-report"
    calls: list[tuple[str, ...]] = []

    def runner(args, **_kwargs) -> None:
        command = tuple(args)
        calls.append(command)
        if command == ("build",):
            report.mkdir(exist_ok=True)
            (report / "release-identity.json").write_bytes(b"built")
            _replace_leaf_with_directory(report)
            (report / "release-identity.json").write_bytes(b"replacement")

    with pytest.raises(ValueError, match="report directory"):
        module.run_mode(
            "identity",
            module.Mode(commands=(("build",), ("validate",))),
            repo_root=root,
            report_dir=report,
            artifact_dir=artifacts,
            artifact_manifest=manifest,
            runner=runner,
        )

    assert ("validate",) not in calls


def test_release_rejects_bundle_replacement_between_distribution_and_identity(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    root.mkdir()
    report = tmp_path / "release"
    artifacts = report / "artifacts"
    manifest = report / "artifact-manifest.json"
    calls: list[tuple[str, ...]] = []
    original_run_mode = module.run_mode
    monkeypatch.setattr(module, "RELEASE_ORDER", ("distribution", "identity"))

    def run_mode(name, mode, **kwargs) -> None:
        if name == "distribution":
            artifacts.mkdir(parents=True)
            manifest.write_bytes(b"original")
            return
        _replace_leaf_with_directory(report)
        artifacts.mkdir()
        manifest.write_bytes(b"replacement")
        original_run_mode(
            name,
            module.Mode(commands=(("identity-business",),)),
            runner=lambda args, **_kwargs: calls.append(tuple(args)),
            **{key: value for key, value in kwargs.items() if key != "runner"},
        )

    monkeypatch.setattr(module, "run_mode", run_mode)
    with pytest.raises(ValueError, match="report directory"):
        module.run_release(root, report, artifacts)

    assert ("identity-business",) not in calls


def test_build_environment_rejects_report_symlink_without_expected_identity(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    report = tmp_path / "report"
    report.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.build_environment(root, report)

    assert tuple(target.iterdir()) == ()


@pytest.mark.parametrize("cache_name", ("mpl-cache", "xdg-cache"))
def test_regular_mode_rejects_cache_directory_swap_before_runner(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
    cache_name: str,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    root.mkdir()
    report = tmp_path / "report"
    target = tmp_path / "target"
    target.mkdir()
    original_materialize = module._materialize

    def materialize(*args, **kwargs):
        cache = report / cache_name
        cache.rename(report / f"{cache_name}-moved")
        cache.symlink_to(target, target_is_directory=True)
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(module, "_materialize", materialize)
    with pytest.raises(ValueError, match="cache directory"):
        module.run_mode(
            "fixture",
            module.Mode(commands=(("would-run",),)),
            repo_root=root,
            report_dir=report,
            runner=lambda *_args, **_kwargs: None,
        )

    assert tuple(target.iterdir()) == ()
