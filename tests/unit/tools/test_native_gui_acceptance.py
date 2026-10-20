from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("actual", ["offscreen", "minimal", "cocoa"])
def test_windows_acceptance_rejects_other_qt_backends(load_tool_module, actual):
    module = load_tool_module("native_gui_acceptance")
    with pytest.raises(ValueError, match="windows"):
        module.require_native_platform(actual, "windows")


def test_native_acceptance_allows_only_explicitly_selected_backend(load_tool_module):
    module = load_tool_module("native_gui_acceptance")
    module.require_native_platform("windows", "windows")
    module.require_native_platform("cocoa", "cocoa")
    with pytest.raises(ValueError):
        module.require_native_platform("offscreen", "offscreen")


def test_acceptance_failure_is_retained_and_never_written_as_pass(load_tool_module, tmp_path, monkeypatch):
    module = load_tool_module("native_gui_acceptance")

    def failure(*_args):
        raise RuntimeError("window was not exposed")

    monkeypatch.setattr(module, "exercise_gui", failure)
    with pytest.raises(RuntimeError, match="window was not exposed"):
        module.run_acceptance(tmp_path, "windows")
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["state"] == "FAIL"
    assert report["error"] == "RuntimeError: window was not exposed"


def test_existing_acceptance_evidence_is_not_overwritten(load_tool_module, tmp_path):
    path = tmp_path / "acceptance.json"
    path.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        load_tool_module("native_gui_acceptance").run_acceptance(tmp_path, "windows")
    assert path.read_text() == "previous evidence"


def test_windows_native_gui_is_opt_in_and_keeps_automatic_checks_unchanged():
    workflow = yaml.safe_load((ROOT / ".github/workflows/audit-windows.yml").read_text())
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["gui_native"]["type"] == "boolean"
    assert inputs["gui_native"]["default"] is False
    assert workflow["permissions"] == {"contents": "read"}


def test_windows_native_gui_reuses_the_verified_environment_and_checks_exit_code():
    workflow = yaml.safe_load((ROOT / ".github/workflows/audit-windows.yml").read_text())
    steps = workflow["jobs"]["windows-headless"]["steps"]
    gui = next(step for step in steps if "tools/native_gui_acceptance.py" in step.get("run", ""))
    assert gui["if"] == "${{ github.event_name == 'workflow_dispatch' && inputs.gui_native }}"
    assert "--expected-platform windows" in gui["run"]
    assert '$env:QT_QPA_PLATFORM = "windows"' in gui["run"]
    assert "$PSNativeCommandUseErrorActionPreference = $true" in gui["run"]
    assert "pip install" not in gui["run"]
