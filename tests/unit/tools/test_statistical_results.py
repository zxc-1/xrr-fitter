from __future__ import annotations

import copy
import json

import pytest
from tests.support.statistical_fixtures import IDENTITY, json_bytes, write_fixture_shards
from tests.support.synthetic_recovery import build_corpus


@pytest.fixture
def receipt_inputs(tmp_path, monkeypatch, load_tool_module):
    def prepare():
        module = load_tool_module("statistical_results")
        monkeypatch.setattr(module, "capture_identity", lambda _root: copy.deepcopy(IDENTITY))
        root = tmp_path / "repo"
        root.mkdir()
        directory = tmp_path / "downloaded"
        shards = write_fixture_shards(directory)
        return module, root, directory, shards

    return prepare


def test_loading_shards_restores_exact_full_corpus_order_and_hashes(receipt_inputs) -> None:
    module, root, directory, _shards = receipt_inputs()

    evidence = module.load_results(root, directory)

    assert tuple(outcome.case_id for outcome in evidence.outcomes) == tuple(case.case_id for case in build_corpus())
    assert evidence.identity == IDENTITY
    assert len(evidence.input_hashes) == 228
    assert set(evidence.elapsed_seconds.values()) == {1.0}
    evidence.guard()


@pytest.mark.parametrize(
    "mutation", ["missing-shard", "extra-shard", "extra-file", "symlink-root", "symlink-shard", "symlink-result"]
)
def test_loader_rejects_missing_extra_or_nonregular_inputs(receipt_inputs, mutation, tmp_path) -> None:
    module, root, directory, shards = receipt_inputs()
    if mutation == "missing-shard":
        shards[0].rename(tmp_path / "saved-shard")
    elif mutation == "extra-shard":
        (directory / "unexpected").mkdir()
    elif mutation == "extra-file":
        (shards[0] / "unexpected.txt").write_text("not part of a receipt")
    elif mutation == "symlink-root":
        link = tmp_path / "linked"
        link.symlink_to(directory, target_is_directory=True)
        directory = link
    elif mutation == "symlink-shard":
        saved = tmp_path / "saved-shard"
        shards[0].rename(saved)
        shards[0].symlink_to(saved, target_is_directory=True)
    else:
        saved = tmp_path / "saved-result.json"
        (shards[0] / "result.json").rename(saved)
        (shards[0] / "result.json").symlink_to(saved)

    with pytest.raises(ValueError):
        module.load_results(root, directory)


@pytest.mark.parametrize(
    "keys,value",
    [
        (("identity", "source_commit"), "d" * 40),
        (("identity", "workflow", "run_id"), "2"),
        (("identity", "workflow", "run_attempt"), "2"),
        (("identity", "runtime", "python"), "3.12.99"),
        (("shard_count",), 9),
        (("shard_index",), False),
        (("shard_index",), 1),
        (("ignored",), True),
        (("state",), "PASS"),
        (("worker_budget", "case_workers"), 100),
    ],
)
def test_loader_rejects_wrong_identity_or_schema_receipts(receipt_inputs, keys, value) -> None:
    module, root, directory, shards = receipt_inputs()
    path = shards[0] / "result.json"
    document = json.loads(path.read_bytes())
    target = document
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_bytes(json_bytes(document))

    with pytest.raises(ValueError):
        module.load_results(root, directory)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_loader_requires_exact_case_coverage_within_each_shard(receipt_inputs, mutation) -> None:
    module, root, directory, shards = receipt_inputs()
    path = shards[0] / "result.json"
    document = json.loads(path.read_bytes())
    if mutation == "missing":
        document["cases"].pop()
    elif mutation == "duplicate":
        document["cases"][-1] = document["cases"][0]
    else:
        document["cases"].reverse()
    path.write_bytes(json_bytes(document))

    with pytest.raises(ValueError, match="canonical"):
        module.load_results(root, directory)


def test_loader_rejects_changed_case_bytes_before_deserializing(receipt_inputs) -> None:
    module, root, directory, shards = receipt_inputs()
    case = next(shards[0].glob("case-*.json"))
    case.write_bytes(case.read_bytes() + b" ")

    with pytest.raises(ValueError, match="hash"):
        module.load_results(root, directory)


def test_loader_rejects_duplicate_json_keys(receipt_inputs) -> None:
    module, root, directory, shards = receipt_inputs()
    path = shards[0] / "result.json"
    path.write_bytes(path.read_bytes().replace(b'"shard_count": 8', b'"shard_count": 8, "shard_count": 8'))

    with pytest.raises(ValueError, match="duplicate"):
        module.load_results(root, directory)


def test_loader_rejects_oversized_receipts(receipt_inputs) -> None:
    module, root, directory, shards = receipt_inputs()
    (shards[0] / "result.json").write_bytes(b" " * (2 * 1024**2 + 1))

    with pytest.raises(ValueError, match="limit"):
        module.load_results(root, directory)


def test_loaded_evidence_rechecks_input_bytes_after_threshold_evaluation(receipt_inputs) -> None:
    module, root, directory, shards = receipt_inputs()
    evidence = module.load_results(root, directory)
    path = shards[0] / "result.json"
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="changed"):
        evidence.guard()


def test_loaded_evidence_rechecks_source_identity(receipt_inputs, monkeypatch) -> None:
    module, root, directory, _shards = receipt_inputs()
    evidence = module.load_results(root, directory)
    changed = {**IDENTITY, "source_commit": "e" * 40}
    monkeypatch.setattr(module, "capture_identity", lambda _root: changed)

    with pytest.raises(ValueError, match="source"):
        evidence.guard()


def test_missing_global_metric_evidence_cannot_be_promoted_to_pass(receipt_inputs) -> None:
    module, root, directory, _shards = receipt_inputs()
    evidence = module.load_results(root, directory)

    with pytest.raises(AssertionError, match="no closed"):
        evidence.evaluate()


def test_result_inputs_must_be_external_to_checkout(receipt_inputs) -> None:
    module, root, directory, _shards = receipt_inputs()
    internal = root / "inputs"
    directory.rename(internal)

    with pytest.raises(ValueError, match="external"):
        module.load_results(root, internal)


def _approved_report():
    from collections import Counter

    from tests.support.synthetic_recovery import CorpusReport

    return CorpusReport(
        "xrr-r23-synthetic-recovery-v1",
        "PASS",
        220,
        220,
        tuple(sorted(Counter(case.category for case in build_corpus()).items())),
        (),
    )


def test_summary_is_byte_bound_and_published_only_after_global_thresholds(
    receipt_inputs, monkeypatch, tmp_path
) -> None:
    module, root, directory, _shards = receipt_inputs()
    evidence = module.load_results(root, directory)
    report = _approved_report()
    monkeypatch.setattr(module, "validate_corpus_outcomes", lambda _cases, _outcomes: report)
    output = tmp_path / "statistical-evidence.json"

    evidence.publish(report, output)

    value = json.loads(output.read_bytes())
    assert value["schema"] == "xrr-r23-statistical-evidence-v1"
    assert value["state"] == "PASS"
    assert value["identity"] == IDENTITY
    assert value["input_sha256"] == evidence.input_hashes
    assert value["corpus"]["case_count"] == 220
    assert len(value["case_elapsed_seconds"]) == 220


def test_summary_refuses_a_claim_without_real_threshold_evaluation(receipt_inputs, tmp_path) -> None:
    module, root, directory, _shards = receipt_inputs()
    evidence = module.load_results(root, directory)
    output = tmp_path / "statistical-evidence.json"

    with pytest.raises(AssertionError, match="no closed"):
        evidence.publish(_approved_report(), output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["internal", "existing", "symlink"])
def test_summary_never_writes_inside_checkout_or_over_existing_inputs(
    receipt_inputs, monkeypatch, kind, tmp_path
) -> None:
    module, root, directory, _shards = receipt_inputs()
    evidence = module.load_results(root, directory)
    report = _approved_report()
    monkeypatch.setattr(module, "validate_corpus_outcomes", lambda _cases, _outcomes: report)
    target = root / "summary.json" if kind == "internal" else tmp_path / "summary.json"
    preserved = tmp_path / "preserved.json"
    preserved.write_text("keep")
    if kind == "existing":
        target.write_text("keep")
    elif kind == "symlink":
        target.symlink_to(preserved)

    with pytest.raises((ValueError, FileExistsError)):
        evidence.publish(report, target)
    assert preserved.read_text() == "keep"
