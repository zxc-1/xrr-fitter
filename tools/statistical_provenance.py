"""Clean source, package pins and current-run binding for statistical shards."""

from __future__ import annotations

import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from distribution_source import clean_head_identity  # noqa: E402
from installed_inputs import InputBindings, json_object  # noqa: E402

INPUTS = (
    "pyproject.toml",
    "requirements-macos-arm64-py312.lock",
    "tools/bootstrap-requirements.lock",
    "tools/package-manifests/macos-arm64-py312.json",
    "tools/package-manifests/refnx-source.json",
    "tools/package-manifests/refnx-build-macos-arm64-py312.json",
)
WORKFLOW_KEYS = ("GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA")


def _workflow_identity(commit: str) -> dict:
    values = {key: os.environ.get(key) for key in WORKFLOW_KEYS}
    if all(value is None for value in values.values()):
        return {"provider": "local"}
    if values["GITHUB_ACTIONS"] != "true" or values["GITHUB_SHA"] != commit:
        raise ValueError("statistical workflow source context differs")
    return _workflow_run(values)


def _workflow_run(values: dict) -> dict:
    repository = values["GITHUB_REPOSITORY"] or ""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository) is None:
        raise ValueError("statistical workflow repository is invalid")
    for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if re.fullmatch(r"[1-9][0-9]*", values[key] or "") is None:
            raise ValueError("statistical workflow run identity is invalid")
    return {
        "provider": "github-actions",
        "repository": repository,
        "run_id": values["GITHUB_RUN_ID"],
        "run_attempt": values["GITHUB_RUN_ATTEMPT"],
    }


def _runtime_identity(manifest: dict) -> dict:
    if sys.flags.optimize:
        raise ValueError("statistical verification requires enabled scientific assertions")
    python = platform.python_version()
    machine = platform.machine()
    if (sys.platform, machine, python.split(".")[:2]) != ("darwin", "arm64", ["3", "12"]):
        raise ValueError("statistical shards require the locked macOS ARM64 Python 3.12 target")
    return {
        "python": python,
        "platform": sys.platform,
        "machine": machine,
        "packages": _runtime_packages(manifest),
    }


def _runtime_packages(manifest: dict) -> dict[str, str]:
    expected = {wheel["name"]: wheel["version"] for wheel in manifest["wheels"]}
    if len(expected) != len(manifest["wheels"]):
        raise ValueError("statistical runtime manifest contains duplicate packages")
    try:
        observed = {name: metadata.version(name) for name in expected}
        refnx = metadata.version("refnx")
    except metadata.PackageNotFoundError as error:
        raise ValueError("statistical runtime is missing a locked package") from error
    if observed != expected:
        raise ValueError("statistical runtime differs from the locked package versions")
    return {**observed, "refnx": refnx}


def capture_identity(root: Path) -> dict:
    source = clean_head_identity(root)
    inputs = InputBindings({name: root / name for name in INPUTS}, limit=2 * 1024**2)
    hashes = inputs.hashes()
    manifest = json_object(inputs.contents["tools/package-manifests/macos-arm64-py312.json"])
    if manifest["lock_sha256"] != hashes["requirements-macos-arm64-py312.lock"]:
        raise ValueError("statistical runtime manifest lock binding differs")
    value = {
        "source_commit": source.head_commit,
        "source_tree": source.head_tree,
        "input_sha256": hashes,
        "runtime": _runtime_identity(manifest),
        "workflow": _workflow_identity(source.head_commit),
    }
    inputs.guard()
    if clean_head_identity(root) != source:
        raise ValueError("statistical source changed while binding its identity")
    return value
