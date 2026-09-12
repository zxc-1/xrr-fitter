from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
INPUTS = (
    "pyproject.toml",
    "requirements-macos-arm64-py312.lock",
    "tools/bootstrap-requirements.lock",
    "tools/package-manifests/macos-arm64-py312.json",
    "tools/package-manifests/refnx-source.json",
    "tools/package-manifests/refnx-build-macos-arm64-py312.json",
)


def _git(root, *args):
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _commit(root):
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Statistical Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")


def _identity(root):
    manifest = json.loads((root / INPUTS[3]).read_bytes())
    packages = {item["name"]: item["version"] for item in manifest["wheels"]}
    return {
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "source_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in INPUTS},
        "runtime": {
            "python": "3.12.10",
            "platform": "darwin",
            "machine": "arm64",
            "packages": {**packages, "refnx": "0.1.65.dev0"},
        },
        "workflow": {"provider": "github-actions", "repository": "owner/repository", "run_id": "1", "run_attempt": "1"},
    }


@pytest.fixture
def source_pair(tmp_path):
    root = tmp_path / "repo"
    paths = (
        *INPUTS,
        "tests/__init__.py",
        "tests/support/__init__.py",
        "tests/conftest.py",
        "tests/outcome_gate.py",
        "tests/acceptance/test_synthetic_recovery_corpus.py",
        "tools/statistical_partition.py",
        "tools/statistical_records.py",
        "tools/statistical_provenance.py",
        "tools/statistical_shards.py",
        *(path.relative_to(ROOT).as_posix() for path in (ROOT / "tests/support").glob("synthetic_recovery*.py")),
    )
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    src = root / "src/xrr_fitter/core.py"
    src.parent.mkdir(parents=True)
    src.write_text("BUDGET = 10\n")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "remote", "add", "origin", "https://github.com/owner/repository")
    _commit(root)
    producer = _identity(root)
    (root / "docs").mkdir()
    (root / "docs/control.md").write_text("control-only change\n")
    (root / "tools/verify.py").write_text("# control-only change\n")
    _commit(root)
    consumer = _identity(root)
    consumer["workflow"]["run_id"] = "2"
    return root, producer, consumer


def test_control_only_commits_can_reuse_all_original_computation_inputs(source_pair, load_tool_module):
    root, producer, consumer = source_pair
    value = load_tool_module("statistical_compatibility").verify_compatibility(root, producer, consumer)
    assert value["state"] == "COMPATIBLE"
    assert value["policy"] == "xrr-statistical-compute-v1"
    assert value["producer_commit"] == producer["source_commit"]
    assert value["consumer_commit"] == consumer["source_commit"]
    assert producer["source_commit"] != consumer["source_commit"]
    assert value["input_sha256"] == producer["input_sha256"]
    assert len(value["computation_sha256"]) == 64


@pytest.mark.parametrize(
    "path",
    [
        "src/xrr_fitter/core.py",
        "src/xrr_fitter/new-data.dat",
        "tests/support/synthetic_recovery.py",
        "tests/support/synthetic_recovery_runs.py",
        "tests/support/synthetic_recovery_new.py",
        "tests/conftest.py",
        "tests/acceptance/__init__.py",
        "tests/acceptance/test_synthetic_recovery_corpus.py",
        "tools/statistical_records.py",
        "tools/statistical_partition.py",
    ],
)
def test_computational_changes_require_new_evidence_without_starting_fits(source_pair, load_tool_module, path):
    root, producer, _ = source_pair
    target = root / path
    target.write_bytes((target.read_bytes() if target.exists() else b"") + b"\n# changed calculation input\n")
    _commit(root)
    with pytest.raises(ValueError, match="NEEDS_NEW_FIT"):
        load_tool_module("statistical_compatibility").verify_compatibility(root, producer, _identity(root))


@pytest.mark.parametrize("mutation", ["python", "tzdata", "pytest", "numpy", "missing-package"])
def test_entire_original_runtime_is_compared_not_only_numerical_libraries(source_pair, load_tool_module, mutation):
    root, producer, consumer = source_pair
    if mutation == "python":
        consumer["runtime"]["python"] = "3.12.11"
    elif mutation == "missing-package":
        consumer["runtime"]["packages"].pop("tzdata")
    else:
        consumer["runtime"]["packages"][mutation] = "999"
    with pytest.raises(ValueError, match="NEEDS_NEW_FIT.*runtime"):
        load_tool_module("statistical_compatibility").verify_compatibility(root, producer, consumer)


@pytest.mark.parametrize("side", [1, 2])
def test_claimed_source_tree_must_match_real_git_commit(source_pair, load_tool_module, side):
    values = copy.deepcopy(source_pair)
    values[side]["source_tree"] = "f" * 40
    with pytest.raises(ValueError, match="source tree"):
        load_tool_module("statistical_compatibility").verify_compatibility(*values)


def test_claimed_input_hashes_must_match_the_producers_own_commit(source_pair, load_tool_module):
    root, producer, consumer = source_pair
    producer["input_sha256"][INPUTS[0]] = "f" * 64
    with pytest.raises(ValueError, match="input.*commit"):
        load_tool_module("statistical_compatibility").verify_compatibility(root, producer, consumer)


def test_locked_inputs_cannot_change_even_when_runtime_versions_match(source_pair, load_tool_module):
    root, producer, _ = source_pair
    path = root / INPUTS[2]
    path.write_bytes(path.read_bytes() + b"\n")
    _commit(root)
    with pytest.raises(ValueError, match="NEEDS_NEW_FIT.*locked"):
        load_tool_module("statistical_compatibility").verify_compatibility(root, producer, _identity(root))


def test_only_the_exact_compute_permission_wrapper_is_normalized(source_pair, load_tool_module):
    root, _, consumer = source_pair
    path = root / "tools/statistical_shards.py"
    guarded = path.read_text()
    unguarded = (
        guarded.replace(", *, allow_compute: bool = False", "")
        .replace(
            '    if allow_compute is not True:\n        raise ValueError("statistical shard fitting requires explicit --compute permission")\n',
            "",
        )
        .replace(
            '    parser.add_argument("--compute", action="store_true", help="explicitly allow new shard fits")\n', ""
        )
        .replace(
            '    if not args.compute:\n        parser.error("statistical shard fitting requires explicit --compute permission")\n',
            "",
        )
        .replace(", allow_compute=args.compute", "")
    )
    assert guarded != unguarded
    path.write_text(unguarded)
    _commit(root)
    old = _identity(root)
    module = load_tool_module("statistical_compatibility")
    assert module.verify_compatibility(root, consumer, old)["state"] == "COMPATIBLE"
    path.write_text(guarded.replace("workers = min(5, cpus)", "workers = min(4, cpus)"))
    _commit(root)
    with pytest.raises(ValueError, match="NEEDS_NEW_FIT"):
        module.verify_compatibility(root, consumer, _identity(root))


def test_unrecognized_shard_entrypoint_changes_are_not_ignored(source_pair, load_tool_module):
    root, producer, _ = source_pair
    path = root / "tools/statistical_shards.py"
    path.write_text(
        path.read_text().replace(
            "    args = parser.parse_args(argv)",
            '    os.environ["FIT_BUDGET"] = "1"\n    args = parser.parse_args(argv)',
        )
    )
    _commit(root)
    with pytest.raises(ValueError, match="NEEDS_NEW_FIT"):
        load_tool_module("statistical_compatibility").verify_compatibility(root, producer, _identity(root))


def test_consumer_must_be_the_current_clean_head_not_a_relabelled_older_commit(source_pair, load_tool_module):
    root, producer, _ = source_pair
    with pytest.raises(ValueError, match="consumer.*HEAD"):
        load_tool_module("statistical_compatibility").verify_compatibility(root, producer, copy.deepcopy(producer))


def test_uncommitted_consumer_changes_cannot_be_hidden_by_matching_git_blobs(source_pair, load_tool_module):
    root, producer, consumer = source_pair
    (root / "src/xrr_fitter/core.py").write_text("BUDGET = 1\n")
    with pytest.raises(ValueError, match="clean"):
        load_tool_module("statistical_compatibility").verify_compatibility(root, producer, consumer)
