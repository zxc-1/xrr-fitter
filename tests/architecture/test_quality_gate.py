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


def _payload() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _standard_run(mode: str) -> str:
    return "\n".join(
        (
            "set -euo pipefail",
            "python3.12 -c 'import platform, sys; assert sys.platform == \"darwin\" and platform.machine() == \"arm64\" and sys.version_info[:2] == (3, 12)'",
            'python3.12 -m venv "$RUNNER_TEMP/venv"',
            'PYTHON="$RUNNER_TEMP/venv/bin/python"',
            '"$PYTHON" -m pip install pip==26.1.2',
            '"$PYTHON" -m pip install -r requirements-macos-arm64-py312.lock',
            '"$PYTHON" tools/check_hygiene.py --require-git-clean',
            f'"$PYTHON" tools/verify.py {mode}',
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


def _checkpoint_job() -> dict[str, object]:
    return {
        "needs": ["quality", "tools"],
        "if": "always()",
        "runs-on": RUNNER,
        "timeout-minutes": 10,
        "steps": [
            {
                "name": "Require all gates",
                "env": {
                    "QUALITY_RESULT": "${{ needs.quality.result }}",
                    "TOOLS_RESULT": "${{ needs.tools.result }}",
                },
                "shell": "bash",
                "run": (
                    "set -euo pipefail\n"
                    'test "$QUALITY_RESULT" = success\n'
                    'test "$TOOLS_RESULT" = success\n'
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
            "checkpoint": _checkpoint_job(),
        },
    }


def _assert_exact_workflow(payload: dict[str, object]) -> None:
    assert payload == _expected_workflow()


def test_initial_workflow_has_exact_jobs_permissions_and_trigger() -> None:
    payload = _payload()
    assert payload["permissions"] == {"contents": "read"}
    assert payload["on"] == {"push": {"branches": ["r23-clean-architecture"]}}
    assert set(payload["jobs"]) == {"quality", "tools", "checkpoint"}


def test_checkpoint_and_concurrency_contract() -> None:
    payload = _payload()
    checkpoint = payload["jobs"]["checkpoint"]
    assert checkpoint["needs"] == ["quality", "tools"]
    assert checkpoint["if"] == "always()"
    assert checkpoint["runs-on"] == RUNNER
    assert checkpoint["timeout-minutes"] == 10
    assert len(checkpoint["steps"]) == 1
    step = checkpoint["steps"][0]
    assert step["env"] == {
        "QUALITY_RESULT": "${{ needs.quality.result }}",
        "TOOLS_RESULT": "${{ needs.tools.result }}",
    }
    commands = step["run"].splitlines()
    assert 'test "$QUALITY_RESULT" = success' in commands
    assert 'test "$TOOLS_RESULT" = success' in commands
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
    for name in ("quality", "tools"):
        _assert_standard_job(name, jobs[name])


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
