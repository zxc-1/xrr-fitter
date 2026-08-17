"""Prove parser wiring, freeze-support ordering, and api-only dispatch."""

from __future__ import annotations

import pytest

from xrr_fitter.cli import main as cli_main


def test_subcommands_are_exactly_the_designed_four() -> None:
    parser = cli_main.build_parser()
    actions = [
        action
        for action in parser._subparsers._group_actions  # noqa: SLF001
        if action.choices
    ]

    assert len(actions) == 1
    assert sorted(actions[0].choices) == ["export", "fit", "mcmc", "validate"]


def test_no_subcommand_is_an_input_error_not_a_crash(capsys) -> None:
    assert cli_main.main([]) == 2

    assert "usage: xrr-fitter-cli" in capsys.readouterr().err


def test_freeze_support_runs_before_any_command(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli_main, "freeze_support", lambda: events.append("freeze"))
    monkeypatch.setattr(
        cli_main,
        "_dispatch",
        lambda arguments: (events.append("dispatch"), 0)[1],
    )

    assert cli_main.main(["fit", "project.json"]) == 0
    assert events == ["freeze", "dispatch"]


def test_missing_project_file_is_an_input_error(tmp_path, capsys) -> None:
    missing = tmp_path / "absent.json"

    assert cli_main.main(["validate", str(missing)]) == 2

    assert str(missing) in capsys.readouterr().err


def test_stale_source_maps_to_its_own_exit_code(monkeypatch, tmp_path) -> None:
    import xrr_fitter.api as api
    from xrr_fitter.cli import commands

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")
    stale = api.ProjectValidation(
        datasets=(),
        issues=(api.ValidationIssue(code="source", message="源文件已变化"),),
    )
    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(commands.api, "inspect_sources", lambda project: stale)

    assert cli_main.main(["validate", str(project_path)]) == 3


def test_cli_package_never_imports_pyside6() -> None:
    import ast
    from pathlib import Path

    root = Path(cli_main.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            assert not any(name.startswith("PySide6") for name in names), path


def test_export_ort_flag_defaults_off_and_opts_in() -> None:
    parser = cli_main.build_parser()

    assert parser.parse_args(["export", "p.json", "out"]).ort is False
    assert parser.parse_args(["export", "p.json", "out", "--ort"]).ort is True


def test_export_checks_sources_before_writing(monkeypatch, tmp_path) -> None:
    import xrr_fitter.api as api
    from xrr_fitter.cli import commands, exit_codes

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")
    stale = api.ProjectValidation(
        datasets=(),
        issues=(api.ValidationIssue(code="source", message="源文件已变化"),),
    )
    calls: list[str] = []

    class _Manifest:
        run_directory = tmp_path / "run"

    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(commands.api, "inspect_sources", lambda project: stale)
    monkeypatch.setattr(
        commands.api,
        "export_result",
        lambda project, output_dir, *, include_ort: (
            calls.append("export"),
            _Manifest(),
        )[1],
    )

    assert cli_main.main(["export", str(project_path), str(tmp_path / "out")]) == exit_codes.STALE_SOURCE
    assert calls == []


@pytest.mark.parametrize(
    "export_error",
    [
        ValueError("missing fitted result"),
        OSError("cannot write manifest"),
        TypeError("unsupported export value"),
        KeyError("selected_candidate"),
    ],
)
def test_run_export_maps_expected_export_errors_to_input_error(export_error, monkeypatch, tmp_path) -> None:
    from argparse import Namespace

    from xrr_fitter.cli import commands, exit_codes

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(
        commands.api,
        "inspect_sources",
        lambda project: commands.api.ProjectValidation(datasets=(), issues=()),
    )
    monkeypatch.setattr(
        commands.api,
        "export_result",
        lambda project, output_dir, *, include_ort: (_ for _ in ()).throw(export_error),
    )

    with pytest.raises(Exception) as excinfo:
        commands.run_export(Namespace(project=str(project_path), output_dir=tmp_path / "out", ort=False))

    assert isinstance(excinfo.value, commands.CommandError)
    assert excinfo.value.code == exit_codes.INVALID_INPUT
    assert excinfo.value.__cause__ is export_error


def test_export_errors_do_not_leak_tracebacks(monkeypatch, tmp_path, capsys) -> None:
    from xrr_fitter.cli import commands, exit_codes

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(
        commands.api,
        "inspect_sources",
        lambda project: commands.api.ProjectValidation(datasets=(), issues=()),
    )
    monkeypatch.setattr(
        commands.api,
        "export_result",
        lambda project, output_dir, *, include_ort: (_ for _ in ()).throw(ValueError("missing fitted result")),
    )

    assert cli_main.main(["export", str(project_path), str(tmp_path / "out")]) == exit_codes.INVALID_INPUT
    output = capsys.readouterr()
    assert "missing fitted result" in output.err
    assert "Traceback" not in output.err


def test_run_export_forwards_include_ort(monkeypatch, tmp_path, capsys) -> None:
    from xrr_fitter.cli import commands, exit_codes

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Manifest:
        run_directory = tmp_path / "run"

    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(
        commands.api,
        "inspect_sources",
        lambda project: commands.api.ProjectValidation(datasets=(), issues=()),
    )
    monkeypatch.setattr(
        commands.api,
        "export_result",
        lambda project, output_dir, *, include_ort: (
            captured.update(include_ort=include_ort),
            _Manifest(),
        )[1],
    )

    assert cli_main.main(["export", str(project_path), str(tmp_path / "out")]) == exit_codes.SUCCESS
    assert captured["include_ort"] is False

    assert cli_main.main(["export", str(project_path), str(tmp_path / "out"), "--ort"]) == exit_codes.SUCCESS
    assert captured["include_ort"] is True
    assert str(_Manifest.run_directory) in capsys.readouterr().out


def test_mcmc_invalid_config_maps_to_input_error_without_traceback(monkeypatch, tmp_path, capsys) -> None:
    from xrr_fitter.cli import commands, exit_codes

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(
        commands.api,
        "inspect_sources",
        lambda project: commands.api.ProjectValidation(datasets=(), issues=()),
    )

    assert (
        cli_main.main(
            [
                "mcmc",
                str(project_path),
                "--dataset",
                "d1",
                "--candidate",
                "c1",
                "--walkers",
                "3",
                "--burn-in",
                "0",
                "--steps",
                "10",
            ]
        )
        == exit_codes.INVALID_INPUT
    )
    output = capsys.readouterr()
    assert "walkers must be even" in output.err
    assert "Traceback" not in output.err


def test_mcmc_invalid_candidate_maps_to_input_error_without_traceback(monkeypatch, tmp_path, capsys) -> None:
    from xrr_fitter.cli import commands, exit_codes

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(
        commands.api,
        "inspect_sources",
        lambda project: commands.api.ProjectValidation(datasets=(), issues=()),
    )
    monkeypatch.setattr(
        commands.api,
        "run_mcmc",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid MCMC candidate: d1/c1")),
    )

    assert (
        cli_main.main(
            [
                "mcmc",
                str(project_path),
                "--dataset",
                "d1",
                "--candidate",
                "c1",
                "--walkers",
                "4",
                "--burn-in",
                "0",
                "--steps",
                "10",
            ]
        )
        == exit_codes.INVALID_INPUT
    )
    output = capsys.readouterr()
    assert "invalid MCMC candidate: d1/c1" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    "fit_error",
    [
        ValueError("automatic fit requires a measurement preset"),
        KeyError("invalid fit domain"),
    ],
)
def test_fit_auto_readiness_or_domain_failure_maps_to_input_error_without_traceback(
    fit_error, monkeypatch, tmp_path, capsys
) -> None:
    from xrr_fitter.cli import commands, exit_codes

    project_path = tmp_path / "p.json"
    project_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(commands.api, "load_project", lambda path: object())
    monkeypatch.setattr(
        commands.api,
        "inspect_sources",
        lambda project: commands.api.ProjectValidation(datasets=(), issues=()),
    )
    monkeypatch.setattr(
        commands.api,
        "fit_automatically",
        lambda project, *, progress_callback: (_ for _ in ()).throw(fit_error),
    )

    assert cli_main.main(["fit", str(project_path), "--auto"]) == exit_codes.INVALID_INPUT
    output = capsys.readouterr()
    assert str(fit_error).strip("'") in output.err
    assert "Traceback" not in output.err
