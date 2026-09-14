"""Conservative committed-input proof for re-evaluating an earlier producer's fits."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from distribution_manifest import git_oid  # noqa: E402
from distribution_source import (  # noqa: E402
    _cat_file_blobs,
    _git,
    _regular_blob_oids,
    clean_head_identity,
    committed_tree_files,
)
from installed_inputs import json_object  # noqa: E402
from statistical_compute_contract import shard_semantics  # noqa: E402
from statistical_provenance import INPUTS  # noqa: E402

TEST_INPUTS = {
    "tests/__init__.py",
    "tests/support/__init__.py",
    "tests/acceptance/__init__.py",
    "tests/conftest.py",
    "tests/outcome_gate.py",
    "tests/acceptance/test_synthetic_recovery_corpus.py",
}
TOOL_INPUTS = ("tools/statistical_partition.py", "tools/statistical_records.py", "tools/statistical_provenance.py")
SHARDS = "tools/statistical_shards.py"


class NeedsNewFit(ValueError):
    """Compatibility was not proven; this is not permission to start fitting."""


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_computational_test(path: str) -> bool:
    return path in TEST_INPUTS or path.startswith("tests/support/synthetic_recovery")


def _committed_files(root: Path, commit: str, names: tuple[str, ...]) -> dict[str, bytes]:
    paths = tuple(PurePosixPath(name) for name in names)
    blobs = _cat_file_blobs(root, _regular_blob_oids(root, commit, paths))
    return {str(path): content for path, content in blobs.items()}


def computation_inputs(root: Path, commit: str) -> dict[str, str]:
    source = committed_tree_files(root, commit, "src")
    if not source:
        raise ValueError("statistical computation source tree is empty")
    files = {str(path): content for path, content in source.items()}
    tests = committed_tree_files(root, commit, "tests")
    files.update({str(path): content for path, content in tests.items() if _is_computational_test(str(path))})
    if not {"tests/support/synthetic_recovery.py", "tests/acceptance/test_synthetic_recovery_corpus.py"} <= set(files):
        raise ValueError("statistical computation corpus or assertions are missing")
    files.update(_committed_files(root, commit, (*TOOL_INPUTS, SHARDS)))
    initializers = committed_tree_files(root, commit, "tools/__init__.py")
    files.update({str(path): content for path, content in initializers.items()})
    files[SHARDS] = shard_semantics(files[SHARDS])
    return {path: _digest(content) for path, content in sorted(files.items())}


def _bound_identity_inputs(root: Path, identity: dict) -> dict[str, bytes]:
    if set(identity) != {"source_commit", "source_tree", "input_sha256", "runtime", "workflow"}:
        raise ValueError("statistical producer/consumer identity schema differs")
    commit = git_oid(identity["source_commit"], "statistical source commit")
    tree = git_oid(identity["source_tree"], "statistical source tree")
    inputs = _committed_files(root, commit, INPUTS)
    if _git(root, "rev-parse", f"{commit}^{{tree}}") != tree:
        raise ValueError("statistical source tree differs from its Git commit")
    if identity["input_sha256"] != {name: _digest(content) for name, content in inputs.items()}:
        raise ValueError("statistical input hashes differ from their own source commit")
    return inputs


def _runtime_target(runtime: dict) -> None:
    if set(runtime) != {"python", "platform", "machine", "packages"}:
        raise NeedsNewFit("NEEDS_NEW_FIT: incomplete statistical runtime identity")
    if (runtime["platform"], runtime["machine"]) != ("darwin", "arm64"):
        raise NeedsNewFit("NEEDS_NEW_FIT: incompatible statistical runtime target")
    if re.fullmatch(r"3\.12\.[0-9]+", runtime["python"]) is None:
        raise NeedsNewFit("NEEDS_NEW_FIT: incompatible statistical runtime Python")


def _locked_runtime(identity: dict, inputs: dict[str, bytes]) -> None:
    runtime = identity["runtime"]
    _runtime_target(runtime)
    manifest = json_object(inputs["tools/package-manifests/macos-arm64-py312.json"])
    expected = {item["name"]: item["version"] for item in manifest["wheels"]}
    packages = runtime["packages"]
    if set(packages) != {*expected, "refnx"} or not isinstance(packages["refnx"], str) or not packages["refnx"]:
        raise NeedsNewFit("NEEDS_NEW_FIT: incomplete locked statistical runtime packages")
    if {name: packages[name] for name in expected} != expected:
        raise NeedsNewFit("NEEDS_NEW_FIT: statistical runtime differs from its lock")


def _consumer_head(root: Path, consumer: dict):
    source = clean_head_identity(root)
    if consumer["source_commit"] != source.head_commit:
        raise ValueError("statistical consumer must identify the current clean HEAD")
    if consumer["source_tree"] != source.head_tree:
        raise ValueError("statistical consumer source tree differs from current HEAD")
    return source


def verify_compatibility(root: Path, producer: dict, consumer: dict) -> dict:
    source = _consumer_head(root, consumer)
    before = _bound_identity_inputs(root, producer)
    after = _bound_identity_inputs(root, consumer)
    if before != after:
        raise NeedsNewFit("NEEDS_NEW_FIT: locked statistical computation inputs changed")
    _locked_runtime(producer, before)
    _locked_runtime(consumer, after)
    if producer["runtime"] != consumer["runtime"]:
        raise NeedsNewFit("NEEDS_NEW_FIT: full statistical runtime identity changed")
    original = computation_inputs(root, producer["source_commit"])
    current = computation_inputs(root, consumer["source_commit"])
    if original != current:
        changed = sorted(path for path in set(original) | set(current) if original.get(path) != current.get(path))
        raise NeedsNewFit(f"NEEDS_NEW_FIT: statistical computation inputs changed: {changed}")
    if clean_head_identity(root) != source:
        raise ValueError("statistical consumer HEAD changed during compatibility verification")
    return {
        "state": "COMPATIBLE",
        "policy": "xrr-statistical-compute-v1",
        "producer_commit": producer["source_commit"],
        "consumer_commit": consumer["source_commit"],
        "input_sha256": producer["input_sha256"],
        "computation_sha256": _digest(json.dumps(original, sort_keys=True).encode("utf-8")),
        "computation_inputs": original,
        "runtime": consumer["runtime"],
    }
