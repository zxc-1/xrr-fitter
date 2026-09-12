from __future__ import annotations

import json

import pytest


def _producer():
    return json.dumps(
        {
            "schema": "xrr-statistical-producer-v1",
            "repository": "owner/repository",
            "run_id": "1",
            "run_attempt": "1",
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
        }
    )


@pytest.mark.parametrize(
    "event,attempt",
    [
        ("push", "1"),
        ("pull_request", "1"),
        ("workflow_run", "1"),
        ("workflow_dispatch", "2"),
        ("workflow_dispatch", "3"),
    ],
)
def test_automatic_events_and_retries_cannot_start_fits(load_tool_module, event, attempt):
    module = load_tool_module("statistical_policy")
    with pytest.raises(ValueError, match="NOT_READY"):
        module.select_execution(compute=True, producer_json="", event=event, attempt=attempt)


def test_only_explicit_first_attempt_manual_authorization_selects_compute(load_tool_module):
    module = load_tool_module("statistical_policy")
    assert module.select_execution(compute=True, producer_json="", event="workflow_dispatch", attempt="1") == "compute"


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_missing_evidence_is_not_ready_instead_of_fit_or_skipped_pass(load_tool_module, event):
    module = load_tool_module("statistical_policy")
    with pytest.raises(ValueError, match="NOT_READY"):
        module.select_execution(compute=False, producer_json="", event=event, attempt="1")


def test_retry_can_revalidate_an_explicit_earlier_producer(load_tool_module):
    module = load_tool_module("statistical_policy")
    assert (
        module.select_execution(compute=False, producer_json=_producer(), event="workflow_dispatch", attempt="2")
        == "replay"
    )


def test_compute_and_producer_evidence_cannot_be_combined(load_tool_module):
    module = load_tool_module("statistical_policy")
    with pytest.raises(ValueError, match="exclusive"):
        module.select_execution(compute=True, producer_json=_producer(), event="workflow_dispatch", attempt="1")


def test_policy_cli_defaults_to_refusal(monkeypatch, load_tool_module):
    module = load_tool_module("statistical_policy")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    with pytest.raises(SystemExit) as raised:
        module.main([])
    assert raised.value.code == 2
