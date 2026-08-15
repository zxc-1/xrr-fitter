from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "pr-verify.yml"
CHECKOUT = "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
MODES = ("quality", "tools", "unit", "gui", "integration", "spawn", "regression")


def _payload() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_pr_workflow_is_pull_request_only_and_read_only() -> None:
    payload = _payload()
    assert payload["on"] == {"pull_request": {}}
    assert payload["permissions"] == {"contents": "read"}


def test_pr_workflow_cancels_superseded_runs_for_the_same_pr() -> None:
    assert _payload()["concurrency"] == {
        "group": "pr-verify-${{ github.event.pull_request.number }}",
        "cancel-in-progress": True,
    }


def test_pr_workflow_uses_non_self_hosted_runner() -> None:
    jobs = _payload()["jobs"]
    standard = jobs["standard"]
    assert standard["runs-on"] == "macos-15"
    assert "self-hosted" not in str(standard["runs-on"])
    assert jobs["checkpoint"]["runs-on"] == "macos-15"


def test_pr_workflow_runs_only_standard_modes() -> None:
    payload = _payload()
    standard = payload["jobs"]["standard"]
    assert standard["strategy"]["matrix"]["mode"] == list(MODES)
    commands = "\n".join(step.get("run", "") for step in standard["steps"])
    assert 'tools/verify.py "${{ matrix.mode }}"' in commands
    assert "tools/verify.py gui" in commands
    assert "statistical" not in commands
    assert "release" not in commands


def test_pr_workflow_checks_out_full_history_without_credentials() -> None:
    checkout_steps = [
        step for job in _payload()["jobs"].values() for step in job.get("steps", []) if step.get("uses") == CHECKOUT
    ]
    assert len(checkout_steps) == 1
    assert checkout_steps[0]["with"] == {
        "persist-credentials": False,
        "fetch-depth": 0,
    }


def test_pr_checkpoint_requires_matrix_success() -> None:
    checkpoint = _payload()["jobs"]["checkpoint"]
    assert checkpoint["needs"] == ["standard"]
    assert checkpoint["if"] == "${{ always() && !cancelled() }}"
    run = checkpoint["steps"][0]["run"]
    assert 'test "${{ needs.standard.result }}" = success' in run
