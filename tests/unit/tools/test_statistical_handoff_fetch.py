from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tests.support.statistical_fixtures import IDENTITY, json_bytes


def _selection():
    value = {
        "schema": "xrr-statistical-producer-v1",
        **IDENTITY["workflow"],
        "source_commit": IDENTITY["source_commit"],
        "source_tree": IDENTITY["source_tree"],
    }
    value.pop("provider")
    return value


def _fetch_fixture(module, monkeypatch):
    artifacts = SimpleNamespace(
        identity=IDENTITY, files={"shard-0/result.json": b"original bytes\n"}, provenance={"source": "authenticated"}
    )
    monkeypatch.setattr(module, "capture_identity", lambda *_: IDENTITY)
    monkeypatch.setattr(module, "_require_repository", lambda *_: None)
    monkeypatch.setattr(module, "fetch_artifacts", lambda *_: artifacts)
    monkeypatch.setattr(module, "verify_compatibility", lambda *_: {"state": "COMPATIBLE"})
    return artifacts


def test_fetch_preserves_original_bytes_and_only_claims_download_not_new_computation(
    tmp_path, monkeypatch, load_tool_module
):
    module = load_tool_module("statistical_handoff")
    _fetch_fixture(module, monkeypatch)
    report = tmp_path / "handoff"
    module.fetch_handoff(tmp_path / "repo", _selection(), report)
    assert (report / "shards/shard-0/result.json").read_bytes() == b"original bytes\n"
    assert json.loads((report / "producer.json").read_bytes()) == _selection()
    value = json.loads((report / "fetch-evidence.json").read_bytes())
    assert value["state"] == "DOWNLOADED"
    assert value["producer_identity"] == IDENTITY
    assert value["new_fit_count"] == 0


@pytest.mark.parametrize("kind", ["existing", "internal", "symlink"])
def test_fetch_rejects_unsafe_output_before_network(tmp_path, monkeypatch, load_tool_module, kind):
    module = load_tool_module("statistical_handoff")
    root = tmp_path / "repo"
    root.mkdir()
    report = tmp_path / "handoff"
    if kind == "existing":
        report.mkdir()
        (report / "old").write_bytes(b"keep")
    elif kind == "internal":
        report = root / "handoff"
    else:
        report.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(module, "fetch_artifacts", lambda *_: pytest.fail("must not download"))
    with pytest.raises(ValueError):
        module.fetch_handoff(root, _selection(), report)


def test_incompatible_inputs_cannot_publish_a_transfer_bundle(tmp_path, monkeypatch, load_tool_module):
    module = load_tool_module("statistical_handoff")
    _fetch_fixture(module, monkeypatch)

    def reject(*_):
        raise ValueError("NEEDS_NEW_FIT")

    monkeypatch.setattr(module, "verify_compatibility", reject)
    with pytest.raises(ValueError, match="NEEDS_NEW_FIT"):
        module.fetch_handoff(tmp_path / "repo", _selection(), tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_fetch_cli_uses_only_the_explicit_descriptor(monkeypatch, tmp_path, load_tool_module):
    module = load_tool_module("statistical_handoff")
    calls = []
    monkeypatch.setattr(module, "fetch_handoff", lambda *args: calls.append(args))
    assert module.main(["--producer-json", json_bytes(_selection()).decode(), "--report-dir", str(tmp_path)]) == 0
    assert calls[0][1] == _selection()
