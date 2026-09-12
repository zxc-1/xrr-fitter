from __future__ import annotations

import subprocess

import pytest


def test_statistical_input_is_explicit_and_keeps_the_original_pytest_selection(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    root, inputs, report = tmp_path / "repo", tmp_path / "inputs", tmp_path / "report"
    (root / "src").mkdir(parents=True)
    inputs.mkdir()
    calls = []

    module.run_mode(
        "statistical",
        module.MODE_REGISTRY["statistical"],
        repo_root=root,
        report_dir=report,
        statistical_results=inputs,
        runner=lambda args, **kwargs: calls.append((args, kwargs)),
    )

    pytest_calls = [(args, kwargs) for args, kwargs in calls if "pytest" in args]
    assert len(pytest_calls) == 1
    args, kwargs = pytest_calls[0]
    assert "tests/acceptance/test_synthetic_recovery_corpus.py" in args
    assert "tests.outcome_gate" in args
    assert args[-6:] == (
        "-p",
        "tests.statistical_gate",
        "--statistical-results",
        str(inputs),
        "--statistical-report",
        str(report / "statistical-evidence.json"),
    )
    assert all(command[0][1] == "tools/check_hygiene.py" for command in (calls[0], calls[-1]))
    assert "XRR_STATISTICAL_RESULTS" not in kwargs["env"]


def test_default_statistical_command_does_not_load_external_outcomes(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    calls = []
    module.run_mode(
        "statistical",
        module.MODE_REGISTRY["statistical"],
        repo_root=root,
        report_dir=tmp_path / "report",
        runner=lambda args, **kwargs: calls.append(args),
    )

    command = next(args for args in calls if "pytest" in args)
    assert command[-2:] == ("tests/acceptance/test_synthetic_recovery_corpus.py", "-q")
    assert "--statistical-results" not in command


@pytest.mark.parametrize("mode", ["unit", "distribution", "approved-data"])
def test_programmatic_statistical_input_cannot_bypass_other_modes(mode, tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    root, inputs = tmp_path / "repo", tmp_path / "inputs"
    root.mkdir()
    inputs.mkdir()
    calls = []
    with pytest.raises(ValueError, match="statistical results"):
        module.run_mode(
            mode,
            module.MODE_REGISTRY[mode],
            repo_root=root,
            report_dir=tmp_path / "report",
            statistical_results=inputs,
            runner=lambda *_args, **_kwargs: calls.append(True),
        )
    assert not calls


@pytest.mark.parametrize("kind", ["internal", "symlink", "missing"])
def test_statistical_input_path_is_regular_external_and_present(kind, tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    root, inputs = tmp_path / "repo", tmp_path / "inputs"
    root.mkdir()
    if kind == "internal":
        inputs = root / "inputs"
        inputs.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        inputs.symlink_to(target, target_is_directory=True)
    calls = []
    with pytest.raises(ValueError, match="statistical"):
        module.run_mode(
            "statistical",
            module.MODE_REGISTRY["statistical"],
            repo_root=root,
            report_dir=tmp_path / "report",
            statistical_results=inputs,
            runner=lambda *_args, **_kwargs: calls.append(True),
        )
    assert not calls


def test_release_revalidates_shards_at_the_original_statistical_position(
    tmp_path, monkeypatch, load_tool_module
) -> None:
    module = load_tool_module("verify")
    root, inputs, report = tmp_path / "repo", tmp_path / "inputs", tmp_path / "release"
    root.mkdir()
    inputs.mkdir()
    calls = []

    def run(name, mode, **kwargs):
        calls.append((name, mode, kwargs))
        if name == "distribution":
            report.mkdir()

    monkeypatch.setattr(module, "run_mode", run)

    module.run_release(root, report, report / "artifacts", statistical_results=inputs)

    assert tuple(name for name, _mode, _kwargs in calls) == module.RELEASE_ORDER
    statistical = next(kwargs for name, _mode, kwargs in calls if name == "statistical")
    assert statistical["statistical_results"] == inputs
    assert all("statistical_results" not in kwargs for name, _mode, kwargs in calls if name != "statistical")


def test_failed_shard_aggregation_prevents_distribution_and_identity(tmp_path, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("verify")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    calls = []

    def run(name, _mode, **kwargs):
        calls.append(name)
        if name == "statistical":
            assert kwargs["statistical_results"] == inputs
            raise subprocess.CalledProcessError(1, ("pytest",))

    monkeypatch.setattr(module, "run_mode", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.run_release(
            tmp_path / "repo",
            tmp_path / "report",
            tmp_path / "report/artifacts",
            statistical_results=inputs,
        )
    assert tuple(calls) == module.RELEASE_ORDER[:8]


@pytest.mark.parametrize("explicit_report", [False, True])
def test_cli_passes_statistical_results_with_explicit_or_temporary_reports(
    explicit_report, tmp_path, monkeypatch, load_tool_module
) -> None:
    module = load_tool_module("verify")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    calls = []
    monkeypatch.setattr(module, "_run_with_report", lambda *args: calls.append(args))
    args = ["statistical", "--statistical-results", str(inputs)]
    if explicit_report:
        args.extend(("--report-dir", str(tmp_path / "report")))

    assert module.main(args) == 0
    assert len(calls) == 1
    assert calls[0][-1] == inputs


def test_cli_rejects_statistical_results_for_other_modes(tmp_path, monkeypatch, load_tool_module, capsys) -> None:
    module = load_tool_module("verify")
    calls = []
    monkeypatch.setattr(module, "_run_with_report", lambda *args: calls.append(args))
    with pytest.raises(SystemExit) as raised:
        module.main(["unit", "--statistical-results", str(tmp_path / "inputs")])
    assert raised.value.code == 2
    assert "--statistical-results is only valid" in capsys.readouterr().err
    assert not calls
