from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from tests.support.macos_action_contract import cleanup_step, expected_action

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


def _action_step(identifier: str) -> dict:
    return next(step for step in _setup_action()["runs"]["steps"] if step.get("id") == identifier)


def test_shared_setup_action_preserves_exact_environment_contract() -> None:
    assert _setup_action() == expected_action()


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
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_JOB": "test",
        "CALLS": str(tmp_path / "calls"),
        "FAIL_ARGS": "",
    }


def _bootstrap(tmp_path: Path, environment: dict) -> Path:
    Path(environment["RUNNER_TEMP"]).mkdir(exist_ok=True)
    result = subprocess.run(
        ("bash", "-c", _action_step("bootstrap")["run"]), cwd=tmp_path, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in Path(environment["GITHUB_OUTPUT"]).read_text().splitlines())
    return Path(outputs["root"])


def test_shared_setup_executes_in_order_and_exports_validated_python(tmp_path: Path) -> None:
    environment = _fake_setup_environment(tmp_path)
    job = _bootstrap(tmp_path, environment)
    environment.update(
        JOB_ROOT=str(job),
        CACHE_DIR=str(tmp_path / "cache"),
        CACHE_TRUST_DOMAIN="pr-26",
        GITHUB_OUTPUT=str(tmp_path / "validated-output"),
    )
    run = _action_step("environment")["run"]
    result = subprocess.run(("bash", "-c", run), cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(environment["CALLS"]).read_text().splitlines() == [
        '-c import platform, sys; assert sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12)',
        f"-m venv {job}/venv",
        f"tools/macos_environment.py own --job-root {job}",
        "-m pip --isolated install --no-cache-dir --require-hashes --no-deps --only-binary=:all: -r tools/bootstrap-requirements.lock",
        f"tools/macos_environment.py setup --job-root {job} --cache-dir {tmp_path}/cache --trust-domain pr-26",
        "-m pip check",
        "tools/check_hygiene.py --require-git-clean",
    ]
    assert Path(environment["GITHUB_OUTPUT"]).read_text() == f"python={job}/venv/bin/python\n"
    assert job.parent == Path(environment["RUNNER_TEMP"])


@pytest.mark.parametrize("failure", ["setup", "-m pip check", "tools/check_hygiene.py --require-git-clean"])
def test_shared_setup_failure_stops_without_exporting_python(tmp_path: Path, failure: str) -> None:
    environment = _fake_setup_environment(tmp_path)
    job = _bootstrap(tmp_path, environment)
    if failure == "setup":
        failure = f"tools/macos_environment.py setup --job-root {job} --cache-dir {tmp_path}/cache --trust-domain pr-26"
    environment.update(
        JOB_ROOT=str(job),
        CACHE_DIR=str(tmp_path / "cache"),
        CACHE_TRUST_DOMAIN="pr-26",
        FAIL_ARGS=failure,
        GITHUB_OUTPUT=str(tmp_path / "validated-output"),
    )
    run = _action_step("environment")["run"]
    result = subprocess.run(("bash", "-c", run), cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 19
    assert Path(environment["CALLS"]).read_text().splitlines()[-1] == failure
    assert not Path(environment["GITHUB_OUTPUT"]).exists()


def test_environment_paths_are_unique_while_exact_cache_paths_are_stable(tmp_path: Path) -> None:
    environment = _fake_setup_environment(tmp_path)
    first = _bootstrap(tmp_path, environment)
    second = _bootstrap(tmp_path, environment)
    assert first != second
    cache = _action_step("input-cache")
    assert set(cache["with"]) == {"key", "path"}
    assert "steps.cache-key.outputs.digest" in cache["with"]["path"]
    assert all(value not in cache["with"]["path"] for value in ("run_id", "run_attempt", "bootstrap.outputs.root"))


@pytest.mark.parametrize("workflow", ["verify.yml", "pr-verify.yml", "audit.yml"])
def test_setup_consumers_always_clean_only_the_owned_environment(workflow: str) -> None:
    payload = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    for job in payload["jobs"].values():
        steps = job.get("steps", [])
        setups = [
            index for index, step in enumerate(steps) if step.get("uses") == "./.github/actions/setup-macos-python"
        ]
        if setups:
            assert steps.count(cleanup_step()) == 1
            cleanup_index = steps.index(cleanup_step())
            assert cleanup_index > setups[0]
            consumers = [
                index
                for index, step in enumerate(steps)
                if step.get("env", {}).get("PYTHON") == "${{ steps.python.outputs.python }}"
            ]
            assert all(index < cleanup_index for index in consumers)


def test_cleanup_action_rechecks_ownership_with_system_python() -> None:
    path = ROOT / ".github/actions/cleanup-macos-python/action.yml"
    assert path.is_file(), "missing bounded macOS cleanup action"
    assert yaml.safe_load(path.read_text()) == {
        "name": "Clean owned macOS Python",
        "description": "Remove only the identity-checked job virtual environment; retain evidence",
        "inputs": {"job-root": {"description": "Owned directory returned by setup-macos-python", "required": True}},
        "runs": {
            "using": "composite",
            "steps": [
                {
                    "shell": "bash",
                    "env": {"JOB_ROOT": "${{ inputs.job-root }}", "PYTHONDONTWRITEBYTECODE": "1"},
                    "run": 'set -euo pipefail\npython3.12 tools/macos_environment.py cleanup --job-root "$JOB_ROOT"\n',
                }
            ],
        },
    }


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
