"""Read-only validation of an explicit GitHub producer and its immutable artifacts."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from distribution_manifest import git_oid  # noqa: E402
from installed_inputs import json_object  # noqa: E402

SELECTION_KEYS = {"schema", "repository", "run_id", "run_attempt", "source_commit", "source_tree"}
WORKFLOWS = {".github/workflows/hosted-release-verify.yml", ".github/workflows/verify.yml"}


def validate_selection(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != SELECTION_KEYS:
        raise ValueError("statistical producer descriptor schema differs")
    if value["schema"] != "xrr-statistical-producer-v1" or not all(isinstance(item, str) for item in value.values()):
        raise ValueError("statistical producer descriptor requires canonical strings")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value["repository"]) is None:
        raise ValueError("invalid statistical producer repository")
    for name in ("run_id", "run_attempt"):
        if re.fullmatch(r"[1-9][0-9]*", value[name]) is None:
            raise ValueError("statistical producer requires an explicit run and attempt")
    for name in ("source_commit", "source_tree"):
        git_oid(value[name], f"statistical producer {name}")
    return value


def api_bytes(endpoint: str) -> bytes:
    result = subprocess.run(
        (
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            "GET",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            endpoint,
        ),
        env={**os.environ, "GH_HOST": "github.com"},
        check=True,
        capture_output=True,
        timeout=90,
    )
    if len(result.stdout) > 16 * 1024**2:
        raise ValueError("statistical producer API response exceeds the size limit")
    return result.stdout


def _collection(endpoint: str, key: str) -> list[dict]:
    first = json_object(api_bytes(f"{endpoint}?per_page=100&page=1"))
    count = first["total_count"]
    if type(count) is not int or not 0 <= count <= 1000:
        raise ValueError("statistical producer API collection exceeds the limit")
    values = list(first[key])
    for page in range(2, (count + 99) // 100 + 1):
        payload = json_object(api_bytes(f"{endpoint}?per_page=100&page={page}"))
        if payload["total_count"] != count:
            raise ValueError("statistical producer API collection changed during pagination")
        values.extend(payload[key])
    if len(values) != count or len({item["id"] for item in values}) != count:
        raise ValueError("statistical producer API collection is incomplete or duplicated")
    return values


def _expected_run(selection: dict) -> dict:
    return {
        "id": int(selection["run_id"]),
        "run_attempt": int(selection["run_attempt"]),
        "head_sha": selection["source_commit"],
    }


def _require_fields(value: dict, expected: dict, label: str) -> None:
    if any(type(value.get(key)) is not type(item) or value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"statistical producer {label} identity differs")


def _validate_run(run: dict, selection: dict) -> None:
    _require_fields(run, {**_expected_run(selection), "status": "completed"}, "run")
    repository = selection["repository"]
    _require_fields(run["repository"], {"full_name": repository}, "repository")
    _require_fields(run["head_repository"], {"full_name": repository, "id": run["repository"]["id"]}, "head repository")
    if run["path"] not in WORKFLOWS or run["event"] not in {"workflow_dispatch", "push"}:
        raise ValueError("statistical producer workflow is not a trusted computation entrypoint")
    path = f"{repository}/.github/workflows/statistical.yml@{selection['source_commit']}"
    matches = [item for item in run["referenced_workflows"] if item["path"] == path]
    if len(matches) != 1 or matches[0]["sha"] != selection["source_commit"]:
        raise ValueError("statistical producer reusable workflow source differs")


def _unique_named(values: list[dict], name: str, label: str) -> dict:
    selected = [item for item in values if item["name"] == name]
    if len(selected) != 1:
        raise ValueError(f"statistical producer requires exactly one {label}: {name}")
    return selected[0]


def _successful_job(jobs: list[dict], name: str, selection: dict) -> dict:
    job = _unique_named(jobs, name, "job")
    expected = _expected_run(selection)
    expected["run_id"] = expected.pop("id")
    _require_fields(job, {**expected, "status": "completed", "conclusion": "success"}, "job")
    return job


def _upload_window(job: dict, artifact: dict) -> None:
    step = _unique_named(job["steps"], "Retain statistical shard evidence", "upload step")
    _require_fields(step, {"status": "completed", "conclusion": "success"}, "upload step")
    created = datetime.fromisoformat(artifact["created_at"])
    start = datetime.fromisoformat(step["started_at"])
    uploaded = datetime.fromisoformat(step["completed_at"])
    job_start = datetime.fromisoformat(job["started_at"])
    finish = datetime.fromisoformat(job["completed_at"])
    # The artifact service can finalize its record after the upload step returns.
    # Bind that finalization to the successful job, without arbitrary clock padding.
    if not job_start <= start <= uploaded <= finish or not start <= created <= finish:
        raise ValueError("statistical producer artifact was not created by the selected attempt's upload")


def _validate_artifact(artifact: dict, job: dict, run: dict, selection: dict) -> None:
    if type(artifact["id"]) is not int or artifact["id"] <= 0:
        raise ValueError("statistical artifact ID is invalid")
    url = f"https://api.github.com/repos/{selection['repository']}/actions/artifacts/{artifact['id']}"
    _require_fields(artifact, {"expired": False, "url": url, "archive_download_url": f"{url}/zip"}, "artifact")
    _require_fields(
        artifact["workflow_run"],
        {
            "id": run["id"],
            "head_sha": selection["source_commit"],
            "head_branch": run["head_branch"],
            "repository_id": run["repository"]["id"],
            "head_repository_id": run["head_repository"]["id"],
        },
        "artifact run",
    )
    if re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["digest"] or "") is None:
        raise ValueError("statistical artifact is missing its authoritative byte digest")
    _upload_window(job, artifact)


def _archive_member(member: zipfile.ZipInfo) -> None:
    if re.fullmatch(r"(?:result|case-[A-Za-z0-9_.-]+)\.json", member.filename) is None:
        raise ValueError("statistical archive member must be direct evidence JSON")
    if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK or member.flag_bits & 1:
        raise ValueError("statistical archive member must be regular and unencrypted")
    if member.file_size > 2 * 1024**2:
        raise ValueError("statistical archive member exceeds the size limit")


def archive_files(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > 40 or sum(member.file_size for member in members) > 8 * 1024**2:
            raise ValueError("statistical archive members exceed the size limit")
        if len({member.filename for member in members}) != len(members):
            raise ValueError("statistical archive contains duplicate members")
        for member in members:
            _archive_member(member)
        result = {member.filename: archive.read(member) for member in members}
    if "result.json" not in result:
        raise ValueError("statistical archive is missing its result member")
    return result


def _artifact_files(selection: dict, artifact: dict) -> dict[str, bytes]:
    content = api_bytes(f"repos/{selection['repository']}/actions/artifacts/{artifact['id']}/zip")
    if (
        len(content) != artifact["size_in_bytes"]
        or f"sha256:{hashlib.sha256(content).hexdigest()}" != artifact["digest"]
    ):
        raise ValueError("statistical producer artifact bytes differ from the GitHub digest")
    return archive_files(content)


def _producer_identity(files: dict[str, bytes], selection: dict) -> dict:
    identities = [json_object(content)["identity"] for name, content in files.items() if name.endswith("/result.json")]
    identity = identities[0]
    if any(value != identity for value in identities):
        raise ValueError("statistical producer artifact set contains mixed identities")
    expected = {"source_commit": selection["source_commit"], "source_tree": selection["source_tree"]}
    _require_fields(identity, expected, "raw source")
    workflow = {
        "provider": "github-actions",
        **{name: selection[name] for name in ("repository", "run_id", "run_attempt")},
    }
    if identity["workflow"] != workflow:
        raise ValueError("statistical producer raw workflow identity differs from the selected attempt")
    return identity


@dataclass(frozen=True)
class ProducerArtifacts:
    identity: dict
    files: dict[str, bytes]
    provenance: dict

    @property
    def hashes(self) -> dict[str, str]:
        return {name: hashlib.sha256(content).hexdigest() for name, content in self.files.items()}


def fetch_artifacts(selection: dict) -> ProducerArtifacts:
    validate_selection(selection)
    base = f"repos/{selection['repository']}/actions/runs/{selection['run_id']}"
    run = json_object(api_bytes(f"{base}/attempts/{selection['run_attempt']}"))
    _validate_run(run, selection)
    jobs = _collection(f"{base}/attempts/{selection['run_attempt']}/jobs", "jobs")
    _successful_job(jobs, "statistical / aggregate", selection)
    artifacts = _collection(f"{base}/artifacts", "artifacts")
    files, receipts = _fetch_shards(selection, run, jobs, artifacts)
    provenance = {
        "selection": selection,
        "verified_at": datetime.now(UTC).isoformat(),
        "verification": "github-api-run-attempt-job-artifact-digest",
        "artifacts": receipts,
    }
    return ProducerArtifacts(_producer_identity(files, selection), files, provenance)


def _fetch_shards(selection: dict, run: dict, jobs: list[dict], artifacts: list[dict]) -> tuple[dict, list]:
    files, receipts = {}, []
    prefix = f"statistical-shard-{selection['source_commit']}-{selection['run_id']}-{selection['run_attempt']}"
    for index in range(8):
        job = _successful_job(jobs, f"statistical / shards ({index})", selection)
        artifact = _unique_named(artifacts, f"{prefix}-{index}", "artifact")
        _validate_artifact(artifact, job, run, selection)
        files.update(
            {f"{artifact['name']}/{name}": content for name, content in _artifact_files(selection, artifact).items()}
        )
        receipts.append({**{key: artifact[key] for key in ("id", "name", "digest")}, "job_id": job["id"]})
    return files, receipts
