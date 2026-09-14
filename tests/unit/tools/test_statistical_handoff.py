from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from tests.support.statistical_fixtures import IDENTITY, json_bytes, write_fixture_shards


def _selection():
    return {
        "schema": "xrr-statistical-producer-v1",
        **IDENTITY["workflow"],
        "source_commit": IDENTITY["source_commit"],
        "source_tree": IDENTITY["source_tree"],
    }


def _descriptor(tmp_path):
    path = tmp_path / "producer.json"
    value = _selection()
    value.pop("provider")
    path.write_bytes(json_bytes(value))
    return path


def _consumer():
    value = copy.deepcopy(IDENTITY)
    value["source_commit"] = "e" * 40
    value["source_tree"] = "f" * 40
    value["workflow"]["run_id"] = "2"
    return value


def test_handoff_authenticates_the_selected_producer_before_proving_compatibility(
    tmp_path, monkeypatch, load_tool_module
):
    module = load_tool_module("statistical_handoff")
    path = _descriptor(tmp_path)
    calls = []
    artifacts = SimpleNamespace(identity=IDENTITY, hashes={"result": "d" * 64}, provenance={"source": "github"})
    monkeypatch.setattr(module, "_require_repository", lambda *_: None)

    def fetch(selection):
        calls.append("authenticate")
        assert selection == json.loads(path.read_bytes())
        return artifacts

    def compatible(root, producer, consumer):
        calls.append("compare")
        assert producer == IDENTITY and consumer == _consumer()
        return {"state": "COMPATIBLE"}

    monkeypatch.setattr(module, "fetch_artifacts", fetch)
    monkeypatch.setattr(module, "verify_compatibility", compatible)
    handoff = module.load_handoff(tmp_path / "repo", path, _consumer())

    assert calls == ["authenticate", "compare"]
    assert handoff.producer == IDENTITY
    assert handoff.proof["state"] == "COMPATIBLE"
    assert handoff.hashes == artifacts.hashes
    handoff.guard()
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        handoff.guard()


@pytest.mark.parametrize("kind", ["internal", "symlink", "missing"])
def test_handoff_rejects_bad_descriptor_paths_before_network(tmp_path, monkeypatch, load_tool_module, kind):
    module = load_tool_module("statistical_handoff")
    root = tmp_path / "repo"
    root.mkdir()
    path = _descriptor(root if kind == "internal" else tmp_path)
    if kind == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    elif kind == "missing":
        path = tmp_path / "missing.json"
    monkeypatch.setattr(module, "fetch_artifacts", lambda *_: pytest.fail("must not query GitHub"))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.load_handoff(root, path, _consumer())


def test_producer_repository_must_match_raw_fetch_and_push_origin(tmp_path, monkeypatch, load_tool_module):
    module = load_tool_module("statistical_handoff")
    descriptor = json.loads(_descriptor(tmp_path).read_bytes())
    monkeypatch.setattr(module, "_git", lambda *_: "https://github.com/owner/repository.git")
    with pytest.raises(ValueError, match="origin"):
        module._require_repository(tmp_path, descriptor, _consumer())


def _replay_fixture(tmp_path, monkeypatch, load_tool_module):
    module = load_tool_module("statistical_results")
    consumer = _consumer()
    monkeypatch.setattr(module, "capture_identity", lambda _: copy.deepcopy(consumer))
    directory = tmp_path / "shards"
    shards = write_fixture_shards(directory)
    hashes = {
        f"{shard.name}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for shard in shards
        for path in shard.iterdir()
    }
    handoff = SimpleNamespace(
        producer=copy.deepcopy(IDENTITY),
        proof={"state": "COMPATIBLE"},
        provenance={"verification": "fixture"},
        hashes=hashes,
        guard=lambda: None,
    )
    monkeypatch.setattr(module, "load_handoff", lambda *_: handoff)
    return module, directory, shards, handoff


def test_explicit_handoff_preserves_producer_and_binds_all_228_original_files(tmp_path, monkeypatch, load_tool_module):
    module, directory, shards, handoff = _replay_fixture(tmp_path, monkeypatch, load_tool_module)
    before = [path.read_bytes() for shard in shards for path in sorted(shard.iterdir())]
    evidence = module.load_results(tmp_path / "repo", directory, producer_path=_descriptor(tmp_path))
    assert evidence.identity == _consumer()
    assert evidence.handoff.producer == IDENTITY
    assert evidence.input_hashes == handoff.hashes
    assert len(evidence.input_hashes) == 228 and len(evidence.outcomes) == 220
    assert [path.read_bytes() for shard in shards for path in sorted(shard.iterdir())] == before
    with pytest.raises(AssertionError, match="no closed"):
        evidence.evaluate()


def test_self_consistent_local_bytes_cannot_replace_upstream_artifact_bytes(tmp_path, monkeypatch, load_tool_module):
    module, directory, shards, _ = _replay_fixture(tmp_path, monkeypatch, load_tool_module)
    path = shards[0] / "result.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="producer.*bytes"):
        module.load_results(tmp_path / "repo", directory, producer_path=_descriptor(tmp_path))


def test_invalid_producer_never_falls_back_to_same_run_or_fitting(tmp_path, monkeypatch, load_tool_module):
    module, directory, _, _ = _replay_fixture(tmp_path, monkeypatch, load_tool_module)

    def invalid(*_):
        raise ValueError("invalid producer")

    monkeypatch.setattr(module, "load_handoff", invalid)
    with pytest.raises(ValueError, match="invalid producer"):
        module.load_results(tmp_path / "repo", directory, producer_path=_descriptor(tmp_path))


def test_cross_run_summary_names_producer_and_consumer_without_claiming_new_fits(
    tmp_path, monkeypatch, load_tool_module
):
    from tests.support.synthetic_recovery import CorpusReport

    module, directory, _, _ = _replay_fixture(tmp_path, monkeypatch, load_tool_module)
    evidence = module.load_results(tmp_path / "repo", directory, producer_path=_descriptor(tmp_path))
    report = CorpusReport("xrr-r23-synthetic-recovery-v1", "PASS", 220, 220, (), ())
    monkeypatch.setattr(module, "validate_corpus_outcomes", lambda *_: report)
    path = tmp_path / "summary.json"
    evidence.publish(report, path)
    value = json.loads(path.read_bytes())
    assert value["schema"] == "xrr-r23-statistical-evidence-v2"
    assert value["producer_identity"] == IDENTITY
    assert value["consumer_identity"] == _consumer()
    assert value["execution"] == "revalidated-existing-outcomes"
    assert value["new_fit_count"] == 0
    assert "identity" not in value
