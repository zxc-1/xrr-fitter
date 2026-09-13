from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/macos_environment.py").is_file(), "missing isolated macOS environment lifecycle"
    return load_tool_module("macos_environment")


def _owned(module, tmp_path):
    job = tmp_path / "xrr-macos-1-1-test.abcdefgh"
    (job / "venv").mkdir(parents=True)
    job.chmod(0o700)
    (job / "venv/keep-until-cleanup").write_bytes(b"owned")
    module.own_environment(ROOT, job, tmp_path)
    return job


def test_combined_key_binds_all_three_input_chains_and_trust_domain(load_tool_module):
    module = _module(load_tool_module)
    key = module.combined_key(ROOT, "pr-26")
    assert key.startswith("xrr-macos-inputs-v1-pr-26-python-3.12-pip-26.1.2-")
    assert key != module.combined_key(ROOT, "trusted")
    assert key == module.combined_key(ROOT, "pr-26")
    from package_cache import cache_key
    from package_downloads import read_manifest
    from qt_cocoa_inputs import input_cache_key as qt_key
    from qt_cocoa_inputs import read_inputs as qt_inputs
    from refnx_build_inputs import input_cache_key, read_inputs

    ordinary = cache_key(read_manifest(ROOT, ROOT / "tools/package-manifests/macos-arm64-py312.json"), "pr-26")
    builder = input_cache_key(read_inputs(ROOT), "pr-26")
    qt = qt_key(qt_inputs(ROOT), "pr-26")
    expected = hashlib.sha256(f"{ordinary}\n{builder}\n{qt}\n".encode()).hexdigest()
    assert key.endswith(expected)


def test_cleanup_removes_only_the_owned_environment_and_retains_evidence(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    job = _owned(module, tmp_path)
    (job / "reports").mkdir()
    (job / "reports/evidence.json").write_bytes(b"{}\n")
    unrelated = tmp_path / "other-venv"
    unrelated.mkdir()
    module.cleanup_environment(ROOT, job, tmp_path)
    assert not (job / "venv").exists()
    assert (job / "reports/evidence.json").read_bytes() == b"{}\n"
    assert unrelated.is_dir()
    assert json.loads((job / "cleanup.json").read_text())["state"] == "PASS"


@pytest.mark.parametrize("mutation", ["root", "venv", "symlink", "marker", "parent"])
def test_cleanup_refuses_unowned_or_replaced_paths(load_tool_module, tmp_path, mutation):
    module = _module(load_tool_module)
    job = _owned(module, tmp_path)
    runner = tmp_path
    if mutation == "root":
        job.rename(tmp_path / "old-root")
        job.mkdir()
    elif mutation == "venv":
        (job / "venv").rename(job / "old-venv")
        (job / "venv").mkdir()
    elif mutation == "symlink":
        (job / "venv").rename(job / "old-venv")
        (job / "venv").symlink_to(job / "old-venv", target_is_directory=True)
    elif mutation == "marker":
        (job / "owner.json").write_text("{}\n")
    else:
        runner = tmp_path / "different-runner"
        runner.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.cleanup_environment(ROOT, job, runner)
    assert (job / "venv").exists() or (tmp_path / "old-root/venv").exists()
    assert not (job / "cleanup.json").exists()


def test_ownership_requires_private_directory_and_current_user(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    job = _owned(module, tmp_path)
    marker = json.loads((job / "owner.json").read_text())
    assert marker["uid"] == os.getuid()
    assert marker["root_identity"] == [job.stat().st_dev, job.stat().st_ino]
    with pytest.raises(ValueError, match="external"):
        module.own_environment(ROOT, ROOT / "xrr-macos-not-owned", ROOT)


@pytest.mark.parametrize("failure", [0, 1, 2, 3, 4, 5])
def test_setup_stops_at_first_failure_without_fake_success(load_tool_module, failure):
    module = _module(load_tool_module)
    calls = []

    def operation(index):
        calls.append(index)
        return 31 if index == failure else 0

    steps = [(str(index), lambda index=index: operation(index)) for index in range(6)]
    assert module.run_setup_steps(steps) == (str(failure), 31)
    assert calls == list(range(failure + 1))


def test_unrelated_cache_members_are_rejected_before_install(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "credentials").write_bytes(b"not a build input")
    with pytest.raises(ValueError, match="unrelated"):
        module.prepare_cache(ROOT, cache, tmp_path / "reports")
    assert (cache / "credentials").read_bytes() == b"not a build input"


def test_qt_original_inputs_have_their_own_cache_namespace(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    cache = tmp_path / "cache"
    for name in ("wheels", "refnx", "qt"):
        (cache / name).mkdir(parents=True)
    assert module.prepare_cache(ROOT, cache, tmp_path / "report") == cache


def test_default_setup_builds_qt_before_install_and_passes_its_receipt(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    job = _owned(module, tmp_path)
    import package_cache
    import package_downloads
    import qt_cocoa_build
    import qt_cocoa_inputs
    import refnx_build
    import refnx_build_inputs

    calls = []
    monkeypatch.setattr(module.sys, "prefix", str(job / "venv"))
    monkeypatch.setattr(package_downloads, "require_install_target", lambda target: None)

    def step(label):
        def run(*args, **kwargs):
            calls.append((label, args, kwargs))
            return 0

        return run

    monkeypatch.setattr(package_cache, "cached_download", step("ordinary-inputs"))
    monkeypatch.setattr(refnx_build_inputs, "cached_inputs", step("refnx-inputs"))
    monkeypatch.setattr(qt_cocoa_inputs, "cached_inputs", step("qt-cocoa-inputs"))
    monkeypatch.setattr(qt_cocoa_build, "build_cocoa", step("qt-cocoa-build"))
    monkeypatch.setattr(package_downloads, "install_packages", step("ordinary-install"))
    monkeypatch.setattr(refnx_build, "build_refnx", step("refnx-build"))
    assert module.setup_environment(ROOT, job, tmp_path, tmp_path / "cache", "pr-26") == 0
    assert [call[0] for call in calls] == [
        "ordinary-inputs",
        "refnx-inputs",
        "qt-cocoa-inputs",
        "qt-cocoa-build",
        "ordinary-install",
        "refnx-build",
    ]
    assert calls[4][2]["qt_build"] == job / "reports/qt-cocoa-build/build.json"
    assert calls[3][1][1:3] == (job / "reports/qt-cocoa-inputs/inputs", job / "reports/packages/wheels")
