from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "verify.yml"
RUNNER = ["self-hosted", "macOS", "ARM64", "xrr-ci"]
CHECKOUT = "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683"
UPLOAD_ARTIFACT = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
STANDARD_MODES = (
    "quality",
    "tools",
    "unit",
    "gui",
    "integration",
    "spawn",
    "regression",
    "distribution",
)


def _payload() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _standard_run(mode: str) -> str:
    verify = f'"$PYTHON" tools/verify.py {mode}'
    if mode == "gui":
        verify = f"QT_QPA_PLATFORM=offscreen {verify}"
    return "\n".join(
        (
            "set -euo pipefail",
            "python3.12 -c 'import platform, sys; assert sys.platform == \"darwin\" and platform.machine() == \"arm64\" and sys.version_info[:2] == (3, 12)'",
            'python3.12 -m venv "$RUNNER_TEMP/venv"',
            'PYTHON="$RUNNER_TEMP/venv/bin/python"',
            '"$PYTHON" -m pip install pip==26.1.2',
            '"$PYTHON" -m pip install -r requirements-macos-arm64-py312.lock',
            '"$PYTHON" tools/check_hygiene.py --require-git-clean',
            verify,
            "",
        )
    )


def _standard_job(mode: str) -> dict[str, object]:
    return {
        "runs-on": RUNNER,
        "timeout-minutes": 60,
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"persist-credentials": False, "fetch-depth": 0},
            },
            {
                "name": f"Verify {mode}",
                "shell": "bash",
                "run": _standard_run(mode),
            },
        ],
    }


def _distribution_job() -> dict[str, object]:
    job = _standard_job("distribution")
    job["steps"][1]["run"] = _standard_run("distribution").replace(
        '"$PYTHON" tools/verify.py distribution',
        '"$PYTHON" tools/verify.py distribution '
        '--report-dir "$RUNNER_TEMP/distribution-bundle" '
        '--artifact-dir "$RUNNER_TEMP/distribution-bundle/artifacts"',
    )
    job["steps"].append(
        {
            "name": "Upload distribution bundle",
            "uses": UPLOAD_ARTIFACT,
            "with": {
                "name": "r23-distribution-${{ github.sha }}",
                "path": "${{ runner.temp }}/distribution-bundle",
                "if-no-files-found": "error",
                "retention-days": 1,
                "compression-level": 0,
            },
        }
    )
    return job


def _checkpoint_job() -> dict[str, object]:
    return {
        "needs": list(STANDARD_MODES),
        "if": "always()",
        "runs-on": RUNNER,
        "timeout-minutes": 10,
        "steps": [
            {
                "name": "Require all gates",
                "env": {
                    "QUALITY_RESULT": "${{ needs.quality.result }}",
                    "TOOLS_RESULT": "${{ needs.tools.result }}",
                    "UNIT_RESULT": "${{ needs.unit.result }}",
                    "REGRESSION_RESULT": "${{ needs.regression.result }}",
                    "GUI_RESULT": "${{ needs.gui.result }}",
                    "INTEGRATION_RESULT": "${{ needs.integration.result }}",
                    "SPAWN_RESULT": "${{ needs.spawn.result }}",
                    "DISTRIBUTION_RESULT": "${{ needs.distribution.result }}",
                },
                "shell": "bash",
                "run": (
                    "set -euo pipefail\n"
                    'test "$QUALITY_RESULT" = success\n'
                    'test "$TOOLS_RESULT" = success\n'
                    'test "$UNIT_RESULT" = success\n'
                    'test "$REGRESSION_RESULT" = success\n'
                    'test "$GUI_RESULT" = success\n'
                    'test "$INTEGRATION_RESULT" = success\n'
                    'test "$SPAWN_RESULT" = success\n'
                    'test "$DISTRIBUTION_RESULT" = success\n'
                ),
            }
        ],
    }


def _expected_workflow() -> dict[str, object]:
    return {
        "name": "verify",
        "on": {"push": {"branches": ["r23-clean-architecture"]}},
        "permissions": {"contents": "read"},
        "concurrency": {
            "group": "verify-${{ github.ref }}-${{ github.sha }}",
            "cancel-in-progress": False,
        },
        "jobs": {
            "quality": _standard_job("quality"),
            "tools": _standard_job("tools"),
            "unit": _standard_job("unit"),
            "gui": _standard_job("gui"),
            "integration": _standard_job("integration"),
            "spawn": _standard_job("spawn"),
            "regression": _standard_job("regression"),
            "distribution": _distribution_job(),
            "checkpoint": _checkpoint_job(),
        },
    }


def _assert_exact_workflow(payload: dict[str, object]) -> None:
    assert payload == _expected_workflow()


def test_initial_workflow_has_exact_jobs_permissions_and_trigger() -> None:
    payload = _payload()
    assert payload["permissions"] == {"contents": "read"}
    assert payload["on"] == {"push": {"branches": ["r23-clean-architecture"]}}
    assert set(payload["jobs"]) == {*STANDARD_MODES, "checkpoint"}


def test_checkpoint_job_contract() -> None:
    payload = _payload()
    checkpoint = payload["jobs"]["checkpoint"]
    assert checkpoint["needs"] == list(STANDARD_MODES)
    assert checkpoint["if"] == "always()"
    assert checkpoint["runs-on"] == RUNNER
    assert checkpoint["timeout-minutes"] == 10
    assert len(checkpoint["steps"]) == 1


def test_checkpoint_step_requires_every_gate() -> None:
    checkpoint = _payload()["jobs"]["checkpoint"]
    step = checkpoint["steps"][0]
    assert step["env"] == {
        "QUALITY_RESULT": "${{ needs.quality.result }}",
        "TOOLS_RESULT": "${{ needs.tools.result }}",
        "UNIT_RESULT": "${{ needs.unit.result }}",
        "REGRESSION_RESULT": "${{ needs.regression.result }}",
        "GUI_RESULT": "${{ needs.gui.result }}",
        "INTEGRATION_RESULT": "${{ needs.integration.result }}",
        "SPAWN_RESULT": "${{ needs.spawn.result }}",
        "DISTRIBUTION_RESULT": "${{ needs.distribution.result }}",
    }
    commands = step["run"].splitlines()
    assert 'test "$QUALITY_RESULT" = success' in commands
    assert 'test "$TOOLS_RESULT" = success' in commands
    assert 'test "$UNIT_RESULT" = success' in commands
    assert 'test "$GUI_RESULT" = success' in commands
    assert 'test "$INTEGRATION_RESULT" = success' in commands
    assert 'test "$SPAWN_RESULT" = success' in commands
    assert 'test "$REGRESSION_RESULT" = success' in commands
    assert 'test "$DISTRIBUTION_RESULT" = success' in commands


def test_concurrency_never_cancels_an_exact_sha_run() -> None:
    payload = _payload()
    assert payload["concurrency"]["cancel-in-progress"] is False


def _action_steps(payload: dict[str, object]):
    for job in payload["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                yield step


def test_actions_are_commit_pinned_and_checkout_drops_credentials() -> None:
    steps = tuple(_action_steps(_payload()))
    assert steps
    for step in steps:
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])


def _assert_standard_job(name: str, job: dict[str, object]) -> None:
    assert job["runs-on"] == RUNNER
    assert job["timeout-minutes"] == 60
    checkout = [
        step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkout) == 1
    assert checkout[0]["with"] == {"persist-credentials": False, "fetch-depth": 0}
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert f"tools/verify.py {name}" in commands
    assert "tools/check_hygiene.py --require-git-clean" in commands


def test_standard_jobs_use_required_runner_and_explicit_verifier_modes() -> None:
    jobs = _payload()["jobs"]
    for name in STANDARD_MODES:
        _assert_standard_job(name, jobs[name])


def test_gui_job_uses_offscreen_qt_platform() -> None:
    commands = _payload()["jobs"]["gui"]["steps"][1]["run"].splitlines()
    assert 'QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py gui' in commands


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["jobs"]["checkpoint"]["steps"][0].__setitem__(
            "run", "set -euo pipefail\nexit 0\ntest \"$QUALITY_RESULT\" = success\ntest \"$TOOLS_RESULT\" = success\n"
        ),
        lambda payload: payload["jobs"]["quality"]["steps"][1].__setitem__(
            "run", payload["jobs"]["quality"]["steps"][1]["run"] + " || true"
        ),
        lambda payload: payload["jobs"]["tools"].__setitem__("continue-on-error", True),
        lambda payload: payload["jobs"]["unit"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["quality"].__setitem__(
            "permissions", {"contents": "write"}
        ),
        lambda payload: payload["jobs"]["quality"]["steps"].append(
            {"uses": "actions/cache@" + "a" * 40}
        ),
        lambda payload: payload["jobs"]["tools"].__setitem__("if", "false"),
    ],
)
def test_exact_workflow_contract_rejects_success_bypasses(mutate) -> None:
    payload = copy.deepcopy(_payload())
    mutate(payload)
    with pytest.raises(AssertionError):
        _assert_exact_workflow(payload)


def test_exact_workflow_contract_accepts_committed_workflow() -> None:
    _assert_exact_workflow(_payload())
