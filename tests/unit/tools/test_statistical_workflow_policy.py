from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _workflow(name):
    return yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())


def test_hosted_workflow_cannot_be_triggered_by_pushing_control_layer_changes():
    assert set(_workflow("hosted-release-verify")["on"]) == {"workflow_dispatch"}


@pytest.mark.parametrize("name", ["hosted-release-verify", "verify"])
def test_every_caller_requires_explicit_compute_with_a_false_default(name):
    inputs = _workflow(name)["on"]["workflow_dispatch"]["inputs"]
    assert inputs["compute"]["type"] == "boolean"
    assert inputs["compute"]["default"] is False
    assert inputs["producer"]["default"] == ""


@pytest.mark.parametrize("name", ["hosted-release-verify", "verify"])
def test_callers_forward_inputs_without_inventing_compute_permission(name):
    assert _workflow(name)["jobs"]["statistical"]["with"] == {
        "compute": "${{ inputs.compute || false }}",
        "producer": "${{ inputs.producer || '' }}",
    }


def test_compute_requires_successful_fast_preflight_and_a_manual_first_attempt():
    workflow = _workflow("statistical")
    job = workflow["jobs"]["shards"]
    assert job["needs"] == ["preflight"]
    assert (
        job["if"]
        == "${{ needs.preflight.outputs.mode == 'compute' && github.event_name == 'workflow_dispatch' && github.run_attempt == 1 }}"
    )
    preflight = workflow["jobs"]["preflight"]
    runs = [step.get("run", "") for step in preflight["steps"]]
    assert any("tools/statistical_policy.py" in value for value in runs)
    assert any("tools/verify.py preflight" in value for value in runs)
    assert not any("--compute-statistical" in value or "statistical_shards.py" in value for value in runs)


def test_aggregate_runs_after_preflight_even_when_compute_is_intentionally_skipped():
    job = _workflow("statistical")["jobs"]["aggregate"]
    assert job["needs"] == ["preflight", "shards"]
    assert job["if"] == "${{ always() && needs.preflight.result == 'success' }}"
    command = next(step for step in job["steps"] if step.get("name") == "Verify complete statistical corpus")
    assert "tools/statistical_handoff.py" in command["run"]
    assert "--statistical-producer" in command["run"]
    assert "--compute-statistical" not in command["run"]


@pytest.mark.parametrize("name", ["statistical", "hosted-release-verify", "verify"])
def test_historical_transfers_have_only_read_permissions(name):
    assert _workflow(name)["permissions"] == {"contents": "read", "actions": "read"}


@pytest.mark.parametrize("name", ["hosted-release-verify", "verify"])
def test_release_revalidates_explicit_producer_without_fitting(name):
    step = next(step for step in _workflow(name)["jobs"]["release"]["steps"] if step.get("name") == "Verify release")
    assert step["env"]["PRODUCER"] == "${{ inputs.producer }}"
    assert 'tools/statistical_handoff.py --producer-json "$PRODUCER"' in step["run"]
    assert '--statistical-producer "$RUNNER_TEMP/statistical-handoff/producer.json"' in step["run"]
    assert "--compute-statistical" not in step["run"]
