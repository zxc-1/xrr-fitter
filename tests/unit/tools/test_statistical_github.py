from __future__ import annotations

import copy
import hashlib
import io
import zipfile

import pytest
from tests.support.statistical_fixtures import IDENTITY, json_bytes


def _selection():
    return {
        "schema": "xrr-statistical-producer-v1",
        "repository": "owner/repository",
        "run_id": "1",
        "run_attempt": "1",
        "source_commit": IDENTITY["source_commit"],
        "source_tree": IDENTITY["source_tree"],
    }


def _zip(index):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("result.json", json_bytes({"identity": IDENTITY, "shard_index": index}))
        archive.writestr(f"case-example-{index}.json", b"{}\n")
    return output.getvalue()


def _job(name, identifier):
    return {
        "name": name,
        "id": identifier,
        "run_id": 1,
        "run_attempt": 1,
        "head_sha": IDENTITY["source_commit"],
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-09-12T10:00:00Z",
        "completed_at": "2026-09-12T10:30:00Z",
        "steps": [
            {
                "name": "Retain statistical shard evidence",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-09-12T10:29:00Z",
                "completed_at": "2026-09-12T10:30:00Z",
            }
        ],
    }


def _upstream():
    selection = _selection()
    base = "repos/owner/repository/actions"
    run = {
        "id": 1,
        "run_attempt": 1,
        "head_sha": selection["source_commit"],
        "repository": {"full_name": "owner/repository", "id": 10},
        "head_repository": {"full_name": "owner/repository", "id": 10},
        "status": "completed",
        "conclusion": "failure",
        "event": "workflow_dispatch",
        "path": ".github/workflows/hosted-release-verify.yml",
        "head_branch": "audit-improvements",
        "referenced_workflows": [
            {
                "path": f"owner/repository/.github/workflows/statistical.yml@{selection['source_commit']}",
                "sha": selection["source_commit"],
            }
        ],
    }
    jobs = [_job(f"statistical / shards ({index})", 100 + index) for index in range(8)]
    jobs.append(_job("statistical / aggregate", 200))
    artifacts, responses = [], {}
    for index in range(8):
        identifier, content = 300 + index, _zip(index)
        url = f"https://api.github.com/{base}/artifacts/{identifier}"
        artifacts.append(
            {
                "id": identifier,
                "name": f"statistical-shard-{selection['source_commit']}-1-1-{index}",
                "expired": False,
                "digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
                "size_in_bytes": len(content),
                "url": url,
                "archive_download_url": url + "/zip",
                "created_at": "2026-09-12T10:29:30Z",
                "workflow_run": {
                    "id": 1,
                    "head_sha": selection["source_commit"],
                    "repository_id": 10,
                    "head_repository_id": 10,
                    "head_branch": "audit-improvements",
                },
            }
        )
        responses[f"{base}/artifacts/{identifier}/zip"] = content
    responses[f"{base}/runs/1/attempts/1"] = run
    responses[f"{base}/runs/1/attempts/1/jobs?per_page=100&page=1"] = {"total_count": len(jobs), "jobs": jobs}
    responses[f"{base}/runs/1/artifacts?per_page=100&page=1"] = {"total_count": len(artifacts), "artifacts": artifacts}
    return selection, responses, run, jobs, artifacts


def _mock_api(module, monkeypatch, responses):
    calls = []

    def get(endpoint):
        calls.append(endpoint)
        value = responses[endpoint]
        return value if isinstance(value, bytes) else json_bytes(value)

    monkeypatch.setattr(module, "api_bytes", get)
    return calls


def test_exact_attempt_with_successful_shards_is_reusable_even_if_release_failed(monkeypatch, load_tool_module):
    module = load_tool_module("statistical_github")
    selection, responses, _, _, artifacts = _upstream()
    calls = _mock_api(module, monkeypatch, responses)
    value = module.fetch_artifacts(selection)
    assert value.identity == IDENTITY
    assert len(value.files) == 16
    assert len(value.provenance["artifacts"]) == 8
    assert value.provenance["selection"] == selection
    assert value.provenance["artifacts"][0]["id"] == artifacts[0]["id"]
    assert len(calls) == 11
    assert not any("latest" in endpoint for endpoint in calls)


@pytest.mark.parametrize(
    "mutation",
    [
        "run",
        "attempt",
        "sha",
        "repository",
        "source-workflow",
        "source-reference",
        "unfinished",
        "job-attempt",
        "failed-job",
        "failed-aggregate",
        "missing-job",
        "duplicate-job",
        "artifact-run",
        "artifact-sha",
        "artifact-repository",
        "artifact-attempt",
        "artifact-time",
        "artifact-expired",
        "missing-artifact",
        "duplicate-artifact",
        "archive-hash",
        "archive-size",
    ],
)
def test_upstream_run_job_and_artifact_association_is_not_trusted_from_json_claims(
    monkeypatch, load_tool_module, mutation
):
    module = load_tool_module("statistical_github")
    selection, responses, run, jobs, artifacts = _upstream()
    _mutate_upstream(mutation, run, jobs, artifacts)
    _mock_api(module, monkeypatch, responses)
    with pytest.raises(ValueError):
        module.fetch_artifacts(selection)


def _mutate_upstream(mutation, run, jobs, artifacts):
    run_fields = {
        "run": ("id", 2),
        "attempt": ("run_attempt", 2),
        "sha": ("head_sha", "f" * 40),
        "source-workflow": ("path", ".github/workflows/untrusted.yml"),
        "unfinished": ("status", "in_progress"),
    }
    if mutation in run_fields:
        key, value = run_fields[mutation]
        run[key] = value
    elif mutation == "repository":
        run["head_repository"]["full_name"] = "other/repository"
    elif mutation == "source-reference":
        run["referenced_workflows"][0]["sha"] = "f" * 40
    elif mutation.startswith("job") or mutation in {"failed-job", "failed-aggregate", "missing-job", "duplicate-job"}:
        _mutate_jobs(mutation, jobs)
    else:
        _mutate_artifacts(mutation, artifacts)


def _mutate_jobs(mutation, jobs):
    if mutation == "job-attempt":
        jobs[0]["run_attempt"] = 2
    elif mutation == "failed-job":
        jobs[0]["conclusion"] = "failure"
    elif mutation == "failed-aggregate":
        jobs[-1]["conclusion"] = "failure"
    elif mutation == "missing-job":
        jobs.pop()
    else:
        jobs.append(copy.deepcopy(jobs[0]))


def _mutate_artifacts(mutation, artifacts):
    fields = {
        "artifact-run": ("id", 2),
        "artifact-sha": ("head_sha", "f" * 40),
        "artifact-repository": ("repository_id", 20),
    }
    if mutation in fields:
        key, value = fields[mutation]
        artifacts[0]["workflow_run"][key] = value
    elif mutation == "missing-artifact":
        artifacts.pop()
    elif mutation == "duplicate-artifact":
        artifacts.append(copy.deepcopy(artifacts[0]))
    else:
        changes = {
            "artifact-attempt": ("name", artifacts[0]["name"].replace("-1-1-0", "-1-2-0")),
            "artifact-time": ("created_at", "2026-09-12T10:31:00Z"),
            "artifact-expired": ("expired", True),
            "archive-hash": ("digest", "sha256:" + "f" * 64),
            "archive-size": ("size_in_bytes", 1),
        }
        key, value = changes[mutation]
        artifacts[0][key] = value


@pytest.mark.parametrize("member", ["../escape.json", "/absolute.json", "folder/result.json", "not-evidence.txt"])
def test_downloaded_archives_reject_paths_outside_direct_evidence_files(load_tool_module, member):
    module = load_tool_module("statistical_github")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(member, b"{}")
    with pytest.raises(ValueError, match="member"):
        module.archive_files(output.getvalue())


def test_api_client_is_explicitly_read_only_and_pinned_to_github(monkeypatch, load_tool_module):
    module = load_tool_module("statistical_github")
    calls = []
    from types import SimpleNamespace

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=b"{}")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.api_bytes("repos/owner/repository/actions/runs/1/attempts/1") == b"{}"
    command, kwargs = calls[0]
    assert command[:2] == ("gh", "api")
    assert command[command.index("--method") + 1] == "GET"
    assert command[command.index("--hostname") + 1] == "github.com"
    assert kwargs["env"]["GH_HOST"] == "github.com"
    assert kwargs["check"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "latest-success"),
        ("run_attempt", "0"),
        ("source_commit", "main"),
        ("source_tree", "bad"),
        ("repository", "owner/repo/../../other"),
        ("schema", "unknown"),
    ],
)
def test_producer_selection_requires_canonical_explicit_identity(load_tool_module, field, value):
    module = load_tool_module("statistical_github")
    selection = _selection()
    selection[field] = value
    with pytest.raises(ValueError):
        module.validate_selection(selection)


def test_artifact_finalization_after_upload_step_still_requires_the_same_successful_job(monkeypatch, load_tool_module):
    module = load_tool_module("statistical_github")
    selection, responses, _, jobs, artifacts = _upstream()
    jobs[0]["completed_at"] = "2026-09-12T10:30:05Z"
    artifacts[0]["created_at"] = "2026-09-12T10:30:01Z"
    _mock_api(module, monkeypatch, responses)
    assert module.fetch_artifacts(selection).identity == IDENTITY


def test_artifact_created_before_the_selected_upload_is_rejected(monkeypatch, load_tool_module):
    module = load_tool_module("statistical_github")
    selection, responses, _, _, artifacts = _upstream()
    artifacts[0]["created_at"] = "2026-09-12T10:28:59Z"
    _mock_api(module, monkeypatch, responses)
    with pytest.raises(ValueError, match="upload"):
        module.fetch_artifacts(selection)
