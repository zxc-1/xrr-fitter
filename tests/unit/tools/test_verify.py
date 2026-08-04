from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _commands(module) -> tuple[tuple[str, ...], ...]:
    return tuple(
        command
        for mode in module.MODE_REGISTRY.values()
        for command in mode.commands
    )


def _registry_names() -> tuple[str, ...]:
    return (
        "quality",
        "tools",
        "unit",
        "gui",
        "integration",
        "spawn",
        "regression",
        "statistical",
        "approved-data",
        "distribution",
        "identity",
        "release",
    )


def test_registry_names_are_exact_for_completed_suites(load_tool_module) -> None:
    module = load_tool_module("verify")
    assert tuple(module.MODE_REGISTRY) == _registry_names()


def test_registry_commands_are_exact_for_completed_suites(load_tool_module) -> None:
    module = load_tool_module("verify")
    expected_quality = (
        (module.PYTHON, "tools/check_radon.py", "--output", f"{module.REPORT}/radon.json"),
        module.PYTEST_PREFIX
        + (
            "tests/architecture/test_dependency_rules.py",
            "tests/architecture/test_naming_rules.py",
            "tests/architecture/test_public_api.py",
            "tests/architecture/test_distribution.py",
            "tests/architecture/test_windows_executable_workflow.py",
            "tests/architecture/test_quality_gate.py",
            "tests/architecture/test_removed_legacy_modules.py",
            "-q",
        ),
    )
    expected_tools = (module.PYTEST_PREFIX + ("tests/unit/tools", "-q"),)
    expected_unit = (
        module.PYTEST_PREFIX
        + (
            "tests/unit/model",
            "tests/unit/io",
            "tests/unit/physics",
            "tests/unit/test_evaluation.py",
            "tests/unit/fit",
            "tests/unit/analysis",
            "tests/unit/services",
            "-q",
        ),
    )
    expected_regression = (
        module.PYTEST_PREFIX
        + (
            "tests/regression/test_numerical_reference.py",
            "tests/regression/test_recovery_metrics.py",
            "tests/regression/test_profile_basin_regressions.py",
            "tests/regression/test_automatic_recovery.py",
            "-q",
        ),
    )
    expected_gui = (module.PYTEST_PREFIX + ("tests/gui", "-q"),)
    expected_integration = (
        module.PYTEST_PREFIX
        + (
            "tests/integration/test_entrypoints.py",
            "tests/integration/test_project_roundtrip.py",
            "tests/integration/test_single_fit_workflow.py",
            "tests/integration/test_joint_fit_workflow.py",
            "tests/integration/test_batch_resume.py",
            "tests/integration/test_export_workflow.py",
            "tests/integration/test_gui_project_workflow.py",
            "-q",
        ),
    )
    expected_spawn = (
        module.PYTEST_PREFIX
        + ("tests/integration/test_process_workers.py", "-q"),
    )
    expected_distribution = (
        (
            module.PYTHON,
            "tools/verify_distribution.py",
            "--repo-root",
            module.ROOT,
            "--report-dir",
            module.REPORT,
            "--artifact-dir",
            module.ARTIFACT,
        ),
    )
    assert module.MODE_REGISTRY["quality"].commands == expected_quality
    assert module.MODE_REGISTRY["tools"].commands == expected_tools
    assert module.MODE_REGISTRY["unit"].commands == expected_unit
    assert module.MODE_REGISTRY["gui"].commands == expected_gui
    assert module.MODE_REGISTRY["integration"].commands == expected_integration
    assert module.MODE_REGISTRY["spawn"].commands == expected_spawn
    assert module.MODE_REGISTRY["regression"].commands == expected_regression
    assert module.MODE_REGISTRY["distribution"].commands == expected_distribution


def test_registry_uses_nonempty_argument_vectors(load_tool_module) -> None:
    module = load_tool_module("verify")
    for mode in module.MODE_REGISTRY.values():
        assert all(isinstance(command, tuple) and command for command in mode.commands)
        assert all(not isinstance(command, str) for command in mode.commands)


def test_every_pytest_command_uses_outcome_plugin(load_tool_module) -> None:
    module = load_tool_module("verify")
    pytest_commands = [command for command in _commands(module) if "pytest" in command]
    assert pytest_commands
    assert all("tests.outcome_gate" in command for command in pytest_commands)
    assert all("--import-mode=importlib" in command for command in pytest_commands)


def test_subprocess_environment_is_root_relative_and_drops_caller_pythonpath(
    tmp_path: Path, load_tool_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", "/private/caller")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p no:terminal -k fixture")
    monkeypatch.setenv("PYTEST_PLUGINS", "caller_plugin")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    env = module.build_environment(root, tmp_path / "report")
    assert env["PYTHONPATH"] == str(root / "src")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "/private/caller" not in env.values()
    assert not any(name.startswith("PYTEST_") for name in env)
    assert "PYTHONOPTIMIZE" not in env
    assert Path(env["MPLCONFIGDIR"]).is_relative_to(tmp_path / "report")


def test_runner_propagates_nonzero_and_checks_hygiene_before_and_after(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("verify")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(args, **kwargs):
        calls.append((tuple(args), kwargs))
        if "fail" in args:
            raise module.subprocess.CalledProcessError(7, args)
        return None

    mode = module.Mode(commands=(("ok",), ("fail",)))
    with pytest.raises(module.subprocess.CalledProcessError):
        module.run_mode("fixture", mode, repo_root=tmp_path, report_dir=tmp_path / "report", runner=runner)
    commands = [command for command, _kwargs in calls]
    assert commands[0][-1] == "tools/check_hygiene.py"
    assert ("ok",) in commands and ("fail",) in commands
    assert commands[-1][-1] == "tools/check_hygiene.py"
    for _command, kwargs in calls:
        assert set(kwargs) == {"cwd", "env", "check"}
        assert kwargs["cwd"] == tmp_path.resolve()
        assert kwargs["check"] is True
        assert "shell" not in kwargs


def test_distribution_keeps_runtime_caches_outside_bundle_until_publication(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    root.mkdir()
    report = tmp_path / "bundle"
    artifact = report / "artifacts"
    verifier_seen = False

    def runner(args, **kwargs):
        nonlocal verifier_seen
        if "tools/verify_distribution.py" not in args:
            return None
        verifier_seen = True
        assert report.is_dir()
        assert tuple(report.iterdir()) == ()
        assert not Path(kwargs["env"]["MPLCONFIGDIR"]).is_relative_to(report)
        assert not Path(kwargs["env"]["XDG_CACHE_HOME"]).is_relative_to(report)
        return None

    module.run_mode(
        "distribution",
        module.MODE_REGISTRY["distribution"],
        repo_root=root,
        report_dir=report,
        artifact_dir=artifact,
        runner=runner,
    )

    assert verifier_seen


def _write_verifier_fixture(root: Path, verifier: Path, outcome_gate: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tests/architecture").mkdir(parents=True)
    (root / "tests/unit/tools").mkdir(parents=True)
    (root / "tests/__init__.py").write_bytes(b"")
    shutil.copy2(verifier, root / "tools/verify.py")
    shutil.copy2(verifier.parent / "verify_registry.py", root / "tools/verify_registry.py")
    shutil.copy2(outcome_gate, root / "tests/outcome_gate.py")
    checker = (
        "from pathlib import Path\n"
        "import os\n"
        "import sys\n\n"
        "root = Path(os.environ['EXPECTED_ROOT'])\n"
        "assert Path.cwd() == root\n"
        "assert os.environ['PYTHONPATH'] == str(root / 'src')\n"
        "assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n"
        "if '--output' in sys.argv:\n"
        "    Path(sys.argv[sys.argv.index('--output') + 1]).write_text('{}\\n', encoding='utf-8')\n"
    )
    (root / "tools/check_hygiene.py").write_text(checker, encoding="utf-8")
    (root / "tools/check_radon.py").write_text(checker, encoding="utf-8")
    test_source = (
        "from pathlib import Path\n"
        "import os\n\n"
        "def test_verifier_root_and_pythonpath():\n"
        "    root = Path(os.environ['EXPECTED_ROOT'])\n"
        "    assert Path.cwd() == root\n"
        "    assert os.environ['PYTHONPATH'] == str(root / 'src')\n"
        "    with Path(os.environ['EXECUTION_LOG']).open('a', encoding='utf-8') as handle:\n"
        "        handle.write(__file__ + '\\n')\n"
    )
    for name in (
        "test_dependency_rules.py",
        "test_naming_rules.py",
        "test_public_api.py",
        "test_distribution.py",
        "test_windows_executable_workflow.py",
        "test_quality_gate.py",
        "test_removed_legacy_modules.py",
    ):
        (root / "tests/architecture" / name).write_text(test_source, encoding="utf-8")


def test_copied_verifier_derives_each_repository_root_from_its_own_file(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[3]
    roots = (tmp_path / "first/repo", tmp_path / "second/repo")
    for root in roots:
        _write_verifier_fixture(
            root,
            source_root / "tools/verify.py",
            source_root / "tests/outcome_gate.py",
        )
    for root, other in zip(roots, reversed(roots), strict=True):
        execution_log = tmp_path / f"executed-{root.parent.name}.log"
        environment = os.environ.copy()
        environment["EXPECTED_ROOT"] = str(root.resolve())
        environment["EXECUTION_LOG"] = str(execution_log)
        environment["PYTHONPATH"] = "/polluted/caller"
        environment["PYTEST_ADDOPTS"] = "-p no:terminal -k fixture"
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
        report = tmp_path / f"report-{root.parent.name}"
        result = subprocess.run(
            (sys.executable, str(root / "tools/verify.py"), "quality", "--report-dir", str(report)),
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(execution_log.read_text(encoding="utf-8").splitlines()) == 7
        assert str(other) not in result.stdout + result.stderr
        assert not (root / "tools/__pycache__").exists()


@pytest.mark.parametrize(
    ("source", "extra_args"),
    [
        ("import pytest\n\n@pytest.mark.skip(reason='no')\ndef test_case(): pass\n", ()),
        ("import pytest\n\n@pytest.mark.xfail\ndef test_case(): assert False\n", ()),
        ("import pytest\n\n@pytest.mark.xfail\ndef test_case(): pass\n", ()),
        ("def test_selected(): pass\ndef test_other(): pass\n", ("-k", "selected")),
        (
            "def test_selected(): pass\ndef test_other(): pass\n",
            ("-p", "no:terminal", "-k", "selected"),
        ),
        ("VALUE = 1\n", ()),
    ],
)
def test_outcome_gate_rejects_nonpassing_or_incomplete_outcomes(
    tmp_path: Path, source: str, extra_args: tuple[str, ...]
) -> None:
    root = Path(__file__).resolve().parents[3]
    sample = tmp_path / "test_sample.py"
    sample.write_text(source, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    trailing_options = () if "no:terminal" in extra_args else ("-q",)
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--strict-config",
            "--strict-markers",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.outcome_gate",
            "--basetemp",
            str(tmp_path / "pytest-tmp"),
            *extra_args,
            str(sample),
            *trailing_options,
        ),
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "XRR outcome gate rejected" in output
