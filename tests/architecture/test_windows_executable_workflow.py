from __future__ import annotations

from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "windows-executable.yml"


def _workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _step(name: str) -> dict[str, object]:
    steps = _workflow()["jobs"]["build"]["steps"]
    matches = tuple(step for step in steps if step.get("name") == name)
    assert len(matches) == 1
    return matches[0]


def test_patch_release_version_is_0_2_2() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["version"] == "0.2.2"


def test_windows_workflow_supports_release_calls_and_manual_retries() -> None:
    triggers = _workflow()["on"]
    for trigger in ("workflow_call", "workflow_dispatch"):
        inputs = triggers[trigger]["inputs"]
        assert inputs["source_ref"]["required"] is True
        assert inputs["expected_commit"]["required"] is True
        assert "default" not in inputs["source_ref"]
        assert "default" not in inputs["expected_commit"]


def test_windows_workflow_derives_asset_names_from_installed_version() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    install = _step("Install locked build environment")["run"]
    build = _step("Build standalone executable")["run"]
    verify = _step("Verify executable and smoke-test GUI startup")["run"]
    manifest = _step("Write executable manifest")["run"]

    assert (
        'python -c "import importlib.metadata; '
        "print(importlib.metadata.version('xrr-fitter'))\""
    ) in install
    assert "XRR_PACKAGE_VERSION=$packageVersion" in install
    assert (
        '"$release/xrr-fitter-$env:XRR_PACKAGE_VERSION-windows-x86_64.exe"'
        in build
    )
    assert (
        '"$env:RUNNER_TEMP/windows-release/'
        'xrr-fitter-$env:XRR_PACKAGE_VERSION-windows-x86_64.exe"'
    ) in verify
    assert (
        '$filename = "xrr-fitter-$env:XRR_PACKAGE_VERSION-windows-x86_64.exe"'
        in manifest
    )
    assert (
        '"$release/xrr-fitter-$env:XRR_PACKAGE_VERSION-windows-x86_64.json"'
        in manifest
    )
    assert "xrr-fitter-0.2.0-windows-x86_64" not in workflow_text


def test_windows_artifact_name_is_not_hardcoded_to_r23() -> None:
    upload = _step("Upload Windows release assets")
    assert upload["with"]["name"] == "xrr-windows-executable-${{ inputs.expected_commit }}"
