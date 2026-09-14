from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/audit_reports.py").is_file(), "missing reproducible audit entry point"
    return load_tool_module("audit_reports")


def _selected_paths(registry):
    return tuple(
        token
        for mode in ("unit", "integration", "spawn", "regression")
        for command in registry.MODE_REGISTRY[mode].commands
        for token in command
        if token.startswith("tests/")
    )


def test_coverage_reuses_exact_non_gui_verifier_selection(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    expected = _selected_paths(load_tool_module("verify_registry"))
    command = module.coverage_command(ROOT, tmp_path)
    assert tuple(token for token in command if token.startswith("tests/")) == expected
    assert len(set(expected)) == len(expected)
    assert command[:4] == (sys.executable, "-m", "coverage", "run")


def test_coverage_preserves_pytest_isolation_and_outcome_enforcement(load_tool_module, tmp_path):
    command = _module(load_tool_module).coverage_command(ROOT, tmp_path)
    assert "--import-mode=importlib" in command
    assert "tests.outcome_gate" in command
    assert "no:cacheprovider" in command
    assert str(tmp_path / "pytest-tmp") in command


def test_audit_tools_do_not_become_distribution_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    tools = project["tool"]["xrr"].get("audit")
    assert tools is not None, "missing pinned development-only audit tool declarations"
    assert tools["requires"] == [
        "coverage==7.16.0",
        "mypy==2.3.1",
        "pip-audit==2.10.1",
        "scipy-stubs==1.18.0.0",
        "pip==26.1.2",
    ]
    published = project["project"]["dependencies"] + project["project"]["optional-dependencies"]["test"]
    assert not set(tools["requires"]) & set(published)


def test_type_configuration_analyzes_imports_without_blanket_ignores():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    config = project["tool"].get("mypy")
    assert config is not None, "missing incremental static type configuration"
    assert config["follow_imports"] == "silent"
    assert config["check_untyped_defs"] is True
    assert config["disallow_untyped_defs"] is True
    assert not config.get("ignore_missing_imports", False)
    assert not config.get("disable_error_code")
    assert set(config["files"]) >= {
        "src/xrr_fitter/fit/stages.py",
        "src/xrr_fitter/services/batch_routing.py",
        "src/xrr_fitter/services/batch.py",
        "src/xrr_fitter/services/batch_publication.py",
        "src/xrr_fitter/analysis/profile_selection.py",
        "src/xrr_fitter/analysis/profile_tasks.py",
    }


@pytest.mark.parametrize("code", [0, 1, 2])
def test_logged_command_retains_exit_status_and_output(load_tool_module, tmp_path, code):
    module = _module(load_tool_module)
    command = (
        sys.executable,
        "-c",
        f"import sys; print('evidence'); print('diagnostic', file=sys.stderr); sys.exit({code})",
    )
    observed = module.run_logged(command, root=ROOT, report=tmp_path, label="probe", environment={})
    assert observed == code
    assert (tmp_path / "probe.stdout").read_text() == "evidence\n"
    assert (tmp_path / "probe.stderr").read_text() == "diagnostic\n"


def test_missing_executable_is_not_reported_as_success(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    with pytest.raises(FileNotFoundError):
        module.run_logged(
            (str(tmp_path / "missing-python"),), root=ROOT, report=tmp_path, label="missing", environment={}
        )


def test_existing_audit_output_is_never_overwritten(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    target = tmp_path / "probe.stdout"
    target.write_text("prior evidence\n")
    with pytest.raises(ValueError, match="new regular-file"):
        module.run_logged(
            (sys.executable, "-c", "print('new')"), root=ROOT, report=tmp_path, label="probe", environment={}
        )
    assert target.read_text() == "prior evidence\n"


def test_advisory_input_preserves_vcs_exclusion_and_exact_pins(load_tool_module):
    module = _module(load_tool_module)
    pins, scope = module.advisory_input(ROOT, "macos-arm64-py312")
    assert len(pins) == 44
    assert "pytest==9.0.3" in pins
    assert all("refnx" not in pin for pin in pins)
    assert [item["name"] for item in scope["excluded_vcs_components"]] == ["refnx"]
    assert scope["native_libraries_scanned"] is False
    assert scope["lock_sha256"]


def test_advisory_cache_is_explicit_and_local_to_each_target_report(load_tool_module, monkeypatch, tmp_path):
    module = _module(load_tool_module)
    calls = []

    def logged(command, **kwargs):
        calls.append((command, kwargs["label"]))
        return 2

    monkeypatch.setattr(module, "run_logged", logged)
    module.report_advisories(ROOT, tmp_path, {})
    assert len(calls) == 2
    for command, target in calls:
        assert "--cache-dir" in command
        assert command[command.index("--cache-dir") + 1] == str(tmp_path / "advisory-cache" / target)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"dependencies": []},
        {"dependencies": [{"name": "other", "version": "1", "vulns": []}]},
        {"dependencies": [{"name": "sample", "version": "1", "skip_reason": "unavailable"}]},
        {"dependencies": [{"name": "sample", "version": "1", "vulns": []}] * 2},
    ],
)
def test_incomplete_advisory_results_fail_closed(load_tool_module, tmp_path, payload):
    module = _module(load_tool_module)
    report = tmp_path / "advisories.json"
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="advisory"):
        module.advisory_findings(report, ("sample==1",))


def test_advisory_findings_are_counted_not_ignored(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    report = tmp_path / "advisories.json"
    report.write_text(json.dumps({"dependencies": [{"name": "sample", "version": "1", "vulns": [{"id": "CVE-test"}]}]}))
    assert module.advisory_findings(report, ("sample==1",)) == 1


def test_coverage_failure_does_not_become_green_after_reporting(load_tool_module, monkeypatch, tmp_path):
    module = _module(load_tool_module)
    calls = []

    def logged(command, **kwargs):
        calls.append(command)
        return 1 if "run" in command else 0

    monkeypatch.setattr(module, "run_logged", logged)
    codes = module.report_coverage(ROOT, tmp_path, {})
    assert codes["tests"] == 1
    assert set(codes) == {"tests", "combine", "json", "xml", "html"}
    assert any("--keep" in command for command in calls)


def test_audit_rejects_repo_internal_output_before_running(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    with pytest.raises(ValueError, match="external"):
        module.run_audit("typing", root=ROOT, report=ROOT / "audit-output")


def test_missing_or_wrong_tool_version_is_an_error(load_tool_module, monkeypatch):
    module = _module(load_tool_module)
    monkeypatch.setattr(module.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(ValueError, match="version"):
        module.check_tool_versions(ROOT)


def test_audit_identity_includes_hash_lock_and_non_python_test_inputs(load_tool_module):
    identity = _module(load_tool_module)._input_identity(ROOT)
    assert "tools/audit-requirements.lock" in identity
    data_files = tuple((ROOT / "tests").rglob("*.json"))
    assert data_files
    assert all(path.relative_to(ROOT).as_posix() in identity for path in data_files)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_audit_summary_preserves_tool_failure(load_tool_module, monkeypatch, tmp_path, code):
    module = _module(load_tool_module)
    monkeypatch.setattr(module, "check_tool_versions", lambda root: {"mypy": "2.3.1"})
    monkeypatch.setattr(module, "report_types", lambda *args: {"mypy": code})
    report = tmp_path / "report"
    assert module.run_audit("typing", root=ROOT, report=report) == int(code != 0)
    summary = json.loads((report / "summary.json").read_text())
    assert summary["state"] == ("PASS" if code == 0 else "FAIL")
    assert summary["exit_codes"] == {"mypy": code}


def test_audit_rejects_changed_inputs_before_success_publication(load_tool_module, monkeypatch, tmp_path):
    module = _module(load_tool_module)
    identities = iter(({"source": "before"}, {"source": "after"}))
    monkeypatch.setattr(module, "_input_identity", lambda root: next(identities))
    monkeypatch.setattr(module, "check_tool_versions", lambda root: {"mypy": "2.3.1"})
    monkeypatch.setattr(module, "report_types", lambda *args: {"mypy": 0})
    report = tmp_path / "report"
    with pytest.raises(ValueError, match="inputs changed"):
        module.run_audit("typing", root=ROOT, report=report)
    assert not (report / "summary.json").exists()
