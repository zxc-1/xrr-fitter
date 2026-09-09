from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/package_downloads.py").is_file(), "missing reproducible package download entry point"
    return load_tool_module("package_downloads")


@pytest.mark.parametrize(
    "target,platform", [("macos-arm64-py312", "macosx_15_0_arm64"), ("windows-x64-py312", "win_amd64")]
)
def test_resolution_does_not_resolve_host_dependencies(load_tool_module, tmp_path, target, platform):
    module = _module(load_tool_module)
    command = module.resolution_command(target, tmp_path)
    assert command[:5] == (sys.executable, "-m", "pip", "--isolated", "install")
    assert {"--dry-run", "--no-deps", "--ignore-installed", "--only-binary=:all:", platform} <= set(command)
    assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
    assert command[command.index("--python-version") + 1] == "3.12"


def test_download_is_hash_required_without_dependency_resolution(load_tool_module, tmp_path):
    command = _module(load_tool_module).download_command("windows-x64-py312", tmp_path)
    assert command[:5] == (sys.executable, "-m", "pip", "--isolated", "download")
    assert {"--require-hashes", "--no-deps", "--only-binary=:all:"} <= set(command)
    assert command[command.index("--dest") + 1] == str(tmp_path / "wheels")
    assert command[command.index("--requirement") + 1] == str(tmp_path / "download.requirements")


@pytest.mark.parametrize("operation", ["resolution_command", "download_command"])
def test_isolated_pip_does_not_reuse_the_global_cache(load_tool_module, tmp_path, operation):
    command = getattr(_module(load_tool_module), operation)("windows-x64-py312", tmp_path)
    assert "--no-cache-dir" in command


def test_package_reports_reject_checkout_paths_before_creating_files(load_tool_module):
    module = _module(load_tool_module)
    with pytest.raises(ValueError, match="external"):
        module.prepare_report(ROOT, ROOT / "package-report")


def test_package_reports_reject_existing_evidence(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    (tmp_path / "old.json").write_text("{}")
    with pytest.raises(ValueError):
        module.prepare_report(ROOT, tmp_path)
    assert (tmp_path / "old.json").read_text() == "{}"


def test_failed_resolution_does_not_publish_a_manifest(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    monkeypatch.setattr(module, "_require_pip", lambda: None)
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: 2)
    report = tmp_path / "report"
    assert module.resolve_packages(ROOT, "windows-x64-py312", report) == 2
    assert not (report / "manifest.json").exists()
    assert json.loads((report / "summary.json").read_text())["state"] == "FAIL"


def test_resolver_records_exact_pins_without_fetching_vcs(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    monkeypatch.setattr(module, "_require_pip", lambda: None)
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: 2)
    report = tmp_path / "report"
    module.resolve_packages(ROOT, "macos-arm64-py312", report)
    pins = (report / "locked.requirements").read_text().splitlines()
    assert len(pins) == 44
    assert "numpy==2.5.2" in pins
    assert not any("refnx" in line for line in pins)


def test_cli_checks_manifest_without_running_pip(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    monkeypatch.setattr(module, "read_manifest", lambda root, path: {"wheels": [], "target": "windows-x64-py312"})
    monkeypatch.setattr(module, "verify_wheels", lambda path, wheels: [])
    assert module.main(["verify", "--manifest", str(tmp_path / "manifest.json"), "--wheel-dir", str(tmp_path)]) == 0


def _fake_failed_download(root, manifest, stage, environment):
    (stage / "wheels").mkdir()
    (stage / "wheels" / "partial.whl").write_bytes(b"incomplete")
    (stage / "download.stderr").write_text("download failed\n")
    return 23


def test_download_failure_cleans_partial_wheels_but_retains_evidence(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    manifest = {"target": "windows-x64-py312", "wheels": []}
    monkeypatch.setattr(module, "read_manifest", lambda root, path: manifest)
    monkeypatch.setattr(module, "_require_pip", lambda: None)
    monkeypatch.setattr(module, "_download_stage", _fake_failed_download)
    report = tmp_path / "report"
    assert module.download_packages(ROOT, tmp_path / "manifest.json", report) == 23
    assert not (report / "wheels").exists()
    assert not list(report.glob("download-stage-*"))
    assert (report / "download.stderr").read_text() == "download failed\n"


def test_post_publication_verification_failure_removes_only_owned_wheels(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    manifest = {"target": "windows-x64-py312", "wheels": []}
    monkeypatch.setattr(module, "read_manifest", lambda root, path: manifest)
    monkeypatch.setattr(module, "_require_pip", lambda: None)

    def downloaded(root, manifest, stage, environment):
        (stage / "wheels").mkdir()
        (stage / "wheels" / "wheel.whl").write_bytes(b"changed")
        (stage / "download.stdout").write_text("download succeeded\n")
        return 0

    def reject(directory, wheels):
        raise ValueError("package wheel bytes do not match")

    monkeypatch.setattr(module, "_download_stage", downloaded)
    monkeypatch.setattr(module, "verify_wheels", reject)
    report = tmp_path / "report"
    with pytest.raises(ValueError, match="bytes do not match"):
        module.download_packages(ROOT, tmp_path / "manifest.json", report)
    assert not (report / "wheels").exists()
    assert not list(report.glob("download-stage-*"))
    assert (report / "download.stdout").read_text() == "download succeeded\n"
    assert not (report / "summary.json").exists()


def test_install_uses_only_hash_checked_local_wheels(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    assert hasattr(module, "install_command"), "missing verified offline install"
    command = module.install_command(tmp_path)
    assert {"--no-index", "--no-deps", "--require-hashes", "--no-cache-dir", "--force-reinstall"} <= set(command)
    assert command[command.index("--requirement") + 1] == str(tmp_path / "install.requirements")


def test_install_checks_bytes_before_invoking_pip(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    assert hasattr(module, "install_packages"), "missing verified offline install"
    manifest = {"target": "windows-x64-py312", "wheels": []}
    monkeypatch.setattr(module, "read_manifest", lambda root, path: manifest)
    monkeypatch.setattr(module, "require_install_target", lambda target: None)
    calls = []
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: calls.append(args))

    def reject(path, wheels):
        raise ValueError("package bytes mismatch")

    monkeypatch.setattr(module, "verify_wheels", reject)
    with pytest.raises(ValueError, match="bytes mismatch"):
        module.install_packages(ROOT, tmp_path / "manifest.json", tmp_path / "wheels", tmp_path / "report")
    assert not calls


def test_install_requires_an_external_virtual_environment(load_tool_module, monkeypatch):
    module = _module(load_tool_module)
    assert hasattr(module, "require_install_target"), "missing install environment boundary"
    monkeypatch.setattr(module.sys, "prefix", module.sys.base_prefix)
    with pytest.raises(ValueError, match="virtual environment"):
        module.require_install_target("macos-arm64-py312")
