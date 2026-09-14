"""CLI preflight, exclusive report ownership, and strict JSON output.

The runner stubs test only output-control ordering and never simulate scientific
estimates. Physical generation/fits and seed accounting remain in their own
suites. Replaced files belong to their external writer, even after our run fails.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("--output", "/tmp/report.json", "--seed", "17"),
        ("--output", "/tmp/report.json", "--seed-count", "10"),
        ("--output", "/tmp/report.json", "--group", "not-a-group"),
        ("--output", "/tmp/report.json", "--workers", "0"),
    ),
)
def test_cli_has_no_seed_cherry_picking_controls(load_tool_module, arguments):
    module = load_tool_module("check_interval_coverage")
    with pytest.raises(SystemExit):
        module.parse_args(arguments)


def test_strict_json_rejects_nonfinite_unmapped_values(load_tool_module):
    module = load_tool_module("check_interval_coverage")
    with pytest.raises(ValueError, match="JSON compliant"):
        module.canonical_json({"bad": float("nan")})


def test_cli_writes_external_strict_report(load_tool_module, monkeypatch, tmp_path):
    module = load_tool_module("check_interval_coverage")
    expected = {"schema": "xrr-interval-coverage-v1", "nominal_coverage": 0.95}
    monkeypatch.setattr(module, "run_experiment", lambda _args: expected)
    target = tmp_path / "coverage.json"

    assert module.main(("--output", str(target))) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == expected


def test_cli_refuses_existing_report_before_running_experiment(load_tool_module, monkeypatch, tmp_path):
    module = load_tool_module("check_interval_coverage")
    path = tmp_path / "previous.json"
    path.write_text("previous evidence", encoding="utf-8")
    monkeypatch.setattr(module, "run_experiment", lambda _args: pytest.fail("experiment should not run"))

    with pytest.raises(FileExistsError):
        module.main(("--output", str(path)))
    assert path.read_text(encoding="utf-8") == "previous evidence"


@pytest.mark.parametrize("parent_kind", ("missing", "file"))
def test_cli_rejects_invalid_parent_before_any_expensive_work(load_tool_module, monkeypatch, tmp_path, parent_kind):
    module = load_tool_module("check_interval_coverage")
    parent = tmp_path / "parent"
    if parent_kind == "file":
        parent.write_text("not a directory", encoding="utf-8")
    calls = []
    monkeypatch.setattr(module, "run_experiment", lambda _args: calls.append(True) or {})

    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        module.main(("--output", str(parent / "report.json")))
    assert calls == []


def test_cli_rejects_uncreatable_target_before_any_expensive_work(load_tool_module, monkeypatch, tmp_path):
    module = load_tool_module("check_interval_coverage")
    calls = []

    def denied(*_args, **_kwargs):
        raise PermissionError("target cannot be created")

    monkeypatch.setattr(module.Path, "open", denied)
    monkeypatch.setattr(module, "run_experiment", lambda _args: calls.append(True) or {})

    with pytest.raises(PermissionError, match="target cannot be created"):
        module.main(("--output", str(tmp_path / "report.json")))
    assert calls == []


@pytest.mark.parametrize("failure_stage", ("experiment", "serialization"))
def test_cli_removes_only_its_unfinished_report_on_failure(load_tool_module, monkeypatch, tmp_path, failure_stage):
    module = load_tool_module("check_interval_coverage")
    path = tmp_path / "report.json"

    def fail(_args):
        if failure_stage == "experiment":
            raise ValueError("experiment failure")
        return {"invalid": float("nan")}

    monkeypatch.setattr(module, "run_experiment", fail)

    with pytest.raises(ValueError):
        module.main(("--output", str(path)))
    assert not path.exists()


def test_cli_preserves_an_external_replacement_and_original_exception(load_tool_module, monkeypatch, tmp_path):
    module = load_tool_module("check_interval_coverage")
    path = tmp_path / "report.json"

    def replace_and_fail(_args):
        assert path.exists(), "output must be reserved before the experiment"
        replacement = tmp_path / "replacement.json"
        replacement.write_text("external evidence", encoding="utf-8")
        replacement.replace(path)
        raise RuntimeError("original experiment failure")

    monkeypatch.setattr(module, "run_experiment", replace_and_fail)

    with pytest.raises(RuntimeError, match="original experiment failure"):
        module.main(("--output", str(path)))
    assert path.read_text(encoding="utf-8") == "external evidence"


def test_cli_rejects_removed_output_after_experiment_returns(load_tool_module, monkeypatch, tmp_path):
    module = load_tool_module("check_interval_coverage")
    path = tmp_path / "report.json"

    def remove_and_return(_args):
        assert path.exists(), "output must be reserved before the experiment"
        path.unlink()
        return {"completed": True}

    monkeypatch.setattr(module, "run_experiment", remove_and_return)

    with pytest.raises(FileNotFoundError, match="report.json"):
        module.main(("--output", str(path)))
    assert not path.exists()


def test_cli_rejects_replaced_output_after_experiment_returns(load_tool_module, monkeypatch, tmp_path):
    module = load_tool_module("check_interval_coverage")
    path = tmp_path / "report.json"

    def replace_and_return(_args):
        assert path.exists(), "output must be reserved before the experiment"
        replacement = tmp_path / "replacement.json"
        replacement.write_text("external evidence", encoding="utf-8")
        replacement.replace(path)
        return {"completed": True}

    monkeypatch.setattr(module, "run_experiment", replace_and_return)

    with pytest.raises(RuntimeError, match="coverage report output was replaced"):
        module.main(("--output", str(path)))
    assert path.read_text(encoding="utf-8") == "external evidence"
