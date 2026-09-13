from __future__ import annotations

import pytest


def test_statistical_without_inputs_refuses_before_invoking_commands(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    calls = []
    report = tmp_path / "report"

    with pytest.raises(ValueError, match="explicit"):
        module.run_mode(
            "statistical",
            module.MODE_REGISTRY["statistical"],
            repo_root=tmp_path / "repo",
            report_dir=report,
            runner=lambda *args, **kwargs: calls.append(args),
        )

    assert calls == []
    assert not report.exists()


def test_release_without_inputs_refuses_before_running_any_gate(tmp_path, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("verify")
    calls = []
    monkeypatch.setattr(module, "run_mode", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(ValueError, match="explicit"):
        module.run_release(tmp_path / "repo", tmp_path / "report", tmp_path / "report/artifacts")

    assert calls == []


def test_explicit_compute_keeps_the_original_full_corpus_command(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    calls = []
    module.run_mode(
        "statistical",
        module.MODE_REGISTRY["statistical"],
        repo_root=tmp_path / "repo",
        report_dir=tmp_path / "report",
        compute_statistical=True,
        runner=lambda args, **kwargs: calls.append(args),
    )

    pytest_call = next(args for args in calls if "pytest" in args)
    assert "tests/acceptance/test_synthetic_recovery_corpus.py" in pytest_call
    assert "--statistical-results" not in pytest_call
    assert "--compute-statistical" in pytest_call
    assert "tests.statistical_gate" in pytest_call


def test_compute_and_result_replay_are_mutually_exclusive(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    with pytest.raises(ValueError, match="exclusive"):
        module.run_mode(
            "statistical",
            module.MODE_REGISTRY["statistical"],
            repo_root=tmp_path / "repo",
            report_dir=tmp_path / "report",
            compute_statistical=True,
            statistical_results=inputs,
        )


def test_preflight_runs_all_fast_release_gates_without_statistical(tmp_path, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("verify")
    assert "preflight" in module.MODE_REGISTRY
    calls = []
    report = tmp_path / "report"

    def record(name, mode, **kwargs):
        calls.append(name)
        if name == "distribution":
            report.mkdir()

    monkeypatch.setattr(module, "run_mode", record)
    module.run_preflight(tmp_path / "repo", report, report / "artifacts")

    assert tuple(calls) == tuple(name for name in module.RELEASE_ORDER if name != "statistical")


def test_shard_without_compute_permission_cannot_create_output_or_worker(
    tmp_path, monkeypatch, load_tool_module
) -> None:
    module = load_tool_module("statistical_shards")
    calls = []
    monkeypatch.setattr(module, "_compute_cases", lambda *args: calls.append(args))
    monkeypatch.setattr(module, "capture_identity", lambda root: {})

    with pytest.raises(ValueError, match="explicit"):
        module.run_shard(tmp_path / "repo", tmp_path / "report", 0)

    assert calls == []
    assert not (tmp_path / "report").exists()


def test_cli_explicit_compute_is_forwarded_without_loading_results(tmp_path, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("verify")
    calls = []
    monkeypatch.setattr(module, "run_mode", lambda *args, **kwargs: calls.append(kwargs))

    assert module.main(["statistical", "--compute-statistical", "--report-dir", str(tmp_path / "report")]) == 0
    assert calls[0]["compute_statistical"] is True
    assert calls[0]["statistical_results"] is None


def test_cli_missing_permission_refuses_before_allocating_report(tmp_path, monkeypatch, load_tool_module) -> None:
    module = load_tool_module("verify")
    calls = []
    monkeypatch.setattr(module, "_run_with_report", lambda *args: calls.append(args))

    with pytest.raises(SystemExit) as raised:
        module.main(["statistical"])

    assert raised.value.code == 2
    assert not calls


def test_programmatic_compute_cannot_bypass_other_modes(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    with pytest.raises(ValueError, match="only valid"):
        module.run_mode(
            "unit",
            module.MODE_REGISTRY["unit"],
            repo_root=tmp_path / "repo",
            report_dir=tmp_path / "report",
            compute_statistical=True,
        )
    assert not (tmp_path / "report").exists()


def test_producer_descriptor_is_forwarded_only_with_replay(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    producer = tmp_path / "producer.json"
    producer.write_text("{}")
    calls = []
    module.run_mode(
        "statistical",
        module.MODE_REGISTRY["statistical"],
        repo_root=tmp_path / "repo",
        report_dir=tmp_path / "report",
        statistical_results=inputs,
        statistical_producer=producer,
        runner=lambda args, **kwargs: calls.append(args),
    )
    command = next(args for args in calls if "pytest" in args)
    assert command[-2:] == ("--statistical-producer", str(producer))


def test_descriptor_without_results_is_not_compute_permission(tmp_path, load_tool_module) -> None:
    module = load_tool_module("verify")
    with pytest.raises(ValueError, match="producer"):
        module.run_mode(
            "statistical",
            module.MODE_REGISTRY["statistical"],
            repo_root=tmp_path / "repo",
            report_dir=tmp_path / "report",
            statistical_producer=tmp_path / "producer.json",
        )
    assert not (tmp_path / "report").exists()
