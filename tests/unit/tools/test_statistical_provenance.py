from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
INPUTS = (
    "pyproject.toml",
    "requirements-macos-arm64-py312.lock",
    "tools/bootstrap-requirements.lock",
    "tools/package-manifests/macos-arm64-py312.json",
    "tools/package-manifests/refnx-source.json",
    "tools/package-manifests/refnx-build-macos-arm64-py312.json",
)
WORKFLOW_KEYS = ("GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA")


@pytest.fixture
def source(tmp_path, monkeypatch):
    for key in WORKFLOW_KEYS:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    for name in INPUTS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    subprocess.run(("git", "-C", str(root), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Statistical Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )
    return root


def _runtime(module, monkeypatch):
    manifest = json.loads((ROOT / "tools/package-manifests/macos-arm64-py312.json").read_text())
    versions = {wheel["name"]: wheel["version"] for wheel in manifest["wheels"]}
    versions["refnx"] = "0.1.65.dev0"
    monkeypatch.setattr(module.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.12.13")
    return versions


def test_execution_identity_binds_clean_source_inputs_and_locked_runtime(source, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("statistical_provenance")
    versions = _runtime(module, monkeypatch)

    identity = module.capture_identity(source)

    assert (
        identity["source_commit"]
        == subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=source, text=True).strip()
    )
    assert (
        identity["source_tree"]
        == subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=source, text=True).strip()
    )
    assert identity["input_sha256"] == {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in INPUTS
    }
    assert identity["runtime"] == {
        "python": "3.12.13",
        "platform": "darwin",
        "machine": "arm64",
        "packages": versions,
    }
    assert identity["workflow"] == {"provider": "local"}


def test_execution_identity_rejects_dirty_source(source, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("statistical_provenance")
    _runtime(module, monkeypatch)
    (source / "pyproject.toml").write_text("changed")
    with pytest.raises(ValueError, match="clean"):
        module.capture_identity(source)


def test_execution_identity_rejects_a_wrong_installed_numerical_pin(source, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("statistical_provenance")
    versions = _runtime(module, monkeypatch)
    versions["numpy"] = "0.0"
    with pytest.raises(ValueError, match="runtime"):
        module.capture_identity(source)


@pytest.mark.parametrize("field,value", [("machine", "x86_64"), ("python_version", "3.13.1")])
def test_sharded_execution_requires_the_locked_target(source, monkeypatch, field, value, load_tool_module) -> None:
    module = load_tool_module("statistical_provenance")
    _runtime(module, monkeypatch)
    monkeypatch.setattr(module.platform, field, lambda: value)
    with pytest.raises(ValueError, match="target"):
        module.capture_identity(source)


def test_hosted_execution_identity_uses_the_exact_run_attempt_and_source(source, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("statistical_provenance")
    _runtime(module, monkeypatch)
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=source, text=True).strip()
    for key, value in zip(WORKFLOW_KEYS, ("true", "owner/repository", "1234", "2", commit), strict=True):
        monkeypatch.setenv(key, value)

    identity = module.capture_identity(source)

    assert identity["workflow"] == {
        "provider": "github-actions",
        "repository": "owner/repository",
        "run_id": "1234",
        "run_attempt": "2",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("GITHUB_RUN_ATTEMPT", ""),
        ("GITHUB_RUN_ID", "01"),
        ("GITHUB_SHA", "0" * 40),
        ("GITHUB_ACTIONS", "false"),
        ("GITHUB_REPOSITORY", "../not/a/repository"),
    ],
)
def test_hosted_identity_rejects_partial_or_mismatched_context(
    source, monkeypatch, field, value, load_tool_module
) -> None:
    module = load_tool_module("statistical_provenance")
    _runtime(module, monkeypatch)
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=source, text=True).strip()
    for key, item in zip(WORKFLOW_KEYS, ("true", "owner/repository", "1234", "2", commit), strict=True):
        monkeypatch.setenv(key, item)
    monkeypatch.setenv(field, value)

    with pytest.raises(ValueError, match="workflow"):
        module.capture_identity(source)


def test_statistical_evidence_cannot_disable_scientific_assertions(source, monkeypatch, load_tool_module) -> None:
    from types import SimpleNamespace

    module = load_tool_module("statistical_provenance")
    _runtime(module, monkeypatch)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="darwin", flags=SimpleNamespace(optimize=1)))

    with pytest.raises(ValueError, match="assertions"):
        module.capture_identity(source)
