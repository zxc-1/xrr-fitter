from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "verify.yml"
SETUP_ACTION = ROOT / ".github" / "actions" / "setup-macos-python" / "action.yml"


def _payload() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _named_step(job: dict[str, object], name: str) -> dict[str, object]:
    steps = [step for step in job["steps"] if step.get("name") == name]
    assert len(steps) == 1, f"expected one step named {name!r}"
    return steps[0]


def _setup_action() -> dict[str, object]:
    assert SETUP_ACTION.is_file(), "missing shared macOS environment action"
    return yaml.safe_load(SETUP_ACTION.read_text(encoding="utf-8"))


def _setup_run() -> str:
    return "\n".join(
        (
            "set -euo pipefail",
            'python3.12 -c \'import platform, sys; assert sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12)\'',
            'python3.12 -m venv "$RUNNER_TEMP/venv"',
            'PYTHON="$RUNNER_TEMP/venv/bin/python"',
            '"$PYTHON" -m pip install pip==26.1.2',
            '"$PYTHON" -m pip install -r requirements-macos-arm64-py312.lock',
            '"$PYTHON" -m pip check',
            '"$PYTHON" tools/check_hygiene.py --require-git-clean',
            'printf \'python=%s\\n\' "$PYTHON" >> "$GITHUB_OUTPUT"',
            "",
        )
    )


def test_shared_setup_action_preserves_exact_environment_contract() -> None:
    assert _setup_action() == {
        "name": "Set up locked macOS Python",
        "description": "Create and validate the macOS ARM64 Python 3.12 verification environment",
        "outputs": {
            "python": {"description": "Validated Python executable", "value": "${{ steps.environment.outputs.python }}"}
        },
        "runs": {"using": "composite", "steps": [{"id": "environment", "shell": "bash", "run": _setup_run()}]},
    }


def _fake_setup_environment(tmp_path: Path) -> dict[str, str]:
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python3.12"
    python.write_text(
        '#!/bin/bash\nset -euo pipefail\nprintf "%s\\n" "$*" >> "$CALLS"\n'
        'if test "$*" = "$FAIL_ARGS"; then exit 19; fi\n'
        'if test "$1" = -m && test "$2" = venv; then\n'
        '  mkdir -p "$3/bin"\n  cp "$0" "$3/bin/python"\nfi\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path / "runner with spaces"),
        "GITHUB_OUTPUT": str(tmp_path / "github-output"),
        "CALLS": str(tmp_path / "calls"),
        "FAIL_ARGS": "",
    }


def test_shared_setup_executes_in_order_and_exports_validated_python(tmp_path: Path) -> None:
    environment = _fake_setup_environment(tmp_path)
    run = _setup_action()["runs"]["steps"][0]["run"]
    result = subprocess.run(("bash", "-c", run), cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(environment["CALLS"]).read_text().splitlines() == [
        '-c import platform, sys; assert sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12)',
        f"-m venv {environment['RUNNER_TEMP']}/venv",
        "-m pip install pip==26.1.2",
        "-m pip install -r requirements-macos-arm64-py312.lock",
        "-m pip check",
        "tools/check_hygiene.py --require-git-clean",
    ]
    assert Path(environment["GITHUB_OUTPUT"]).read_text() == f"python={environment['RUNNER_TEMP']}/venv/bin/python\n"


@pytest.mark.parametrize(
    "failure", ["-m pip install pip==26.1.2", "-m pip check", "tools/check_hygiene.py --require-git-clean"]
)
def test_shared_setup_failure_stops_without_exporting_python(tmp_path: Path, failure: str) -> None:
    environment = {**_fake_setup_environment(tmp_path), "FAIL_ARGS": failure}
    run = _setup_action()["runs"]["steps"][0]["run"]
    result = subprocess.run(("bash", "-c", run), cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 19
    assert Path(environment["CALLS"]).read_text().splitlines()[-1] == failure
    assert not Path(environment["GITHUB_OUTPUT"]).exists()


def _readiness_repository(tmp_path: Path, *, has_manifest: bool) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True, capture_output=True)
    if has_manifest:
        manifest = repository / "verification/r23/tests.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        subprocess.run(("git", "add", "verification/r23/tests.json"), cwd=repository, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--no-gpg-sign",
            "--allow-empty",
            "-qm",
            "baseline",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return repository


@pytest.mark.parametrize("has_manifest", [False, True])
def test_readiness_inputs_select_setup_without_installing(tmp_path: Path, has_manifest: bool) -> None:
    environment = _fake_setup_environment(tmp_path)
    repository = _readiness_repository(tmp_path, has_manifest=has_manifest)
    job = _payload()["jobs"]["candidate-readiness"]
    run = _named_step(job, "Check candidate readiness inputs")["run"]
    result = subprocess.run(("bash", "-c", run), cwd=repository, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = "has_tests_manifest=true\n" if has_manifest else "ready=false\nhas_tests_manifest=false\n"
    assert Path(environment["GITHUB_OUTPUT"]).read_text() == expected
    assert not Path(environment["RUNNER_TEMP"]).exists()
    assert len(Path(environment["CALLS"]).read_text().splitlines()) == 1
    condition = "steps.readiness_inputs.outputs.has_tests_manifest == 'true'"
    assert _named_step(job, "Set up locked macOS Python")["if"] == condition
    assert _named_step(job, "Evaluate candidate readiness")["if"] == condition


def test_readiness_inputs_reject_dirty_checkout_without_success_output(tmp_path: Path) -> None:
    environment = _fake_setup_environment(tmp_path)
    repository = _readiness_repository(tmp_path, has_manifest=False)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    job = _payload()["jobs"]["candidate-readiness"]
    run = _named_step(job, "Check candidate readiness inputs")["run"]
    result = subprocess.run(("bash", "-c", run), cwd=repository, env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert not Path(environment["GITHUB_OUTPUT"]).exists()
    assert not Path(environment["RUNNER_TEMP"]).exists()
