from __future__ import annotations

import json
from pathlib import Path

import pytest


MANIFEST = """\
# Ti-Ni multilayer, from BornAgain FitSpecularBasics.py
# dQ/Q = 0.05 FWHM or 0.0212 1-sigma
layers/test1.layers
data/test5.dat
"""


def _suite(root: Path) -> Path:
    unpolarised = root / "unpolarised"
    (unpolarised / "layers").mkdir(parents=True)
    (unpolarised / "data").mkdir(parents=True)
    (unpolarised / "test5.txt").write_text(MANIFEST, encoding="utf-8")
    (unpolarised / "layers/test1.layers").write_text("0 0 0 0\n30 -1.9493 0 0\n", encoding="utf-8")
    (unpolarised / "data/test5.dat").write_text("0.005 0.9995 0.0 1.0617e-04\n", encoding="utf-8")
    (unpolarised / "data/Untitled.ipynb").write_text("{}\n", encoding="utf-8")
    return root


def test_suite_commit_is_pinned_to_forty_hex_characters(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    assert module.SUITE_COMMIT == "6a01b4a4febfc52cd3881d2147c732dd1701bc8e"
    assert len(module.SUITE_COMMIT) == 40
    assert set(module.SUITE_COMMIT) <= set("0123456789abcdef")


def test_manifest_references_skip_comments_and_keep_order(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    assert module.manifest_references(MANIFEST) == ("layers/test1.layers", "data/test5.dat")


def test_manifest_without_exactly_two_references_is_rejected(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    with pytest.raises(ValueError, match="two references"):
        module.manifest_references("# only a comment\nlayers/test1.layers\n")


def test_frozen_index_excludes_files_no_manifest_references(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    index = module.frozen_index(_suite(tmp_path))
    paths = {record["path"] for record in index["files"]}
    assert paths == {
        "unpolarised/test5.txt",
        "unpolarised/layers/test1.layers",
        "unpolarised/data/test5.dat",
    }
    assert not any("Untitled" in path for path in paths)


def test_frozen_index_records_schema_commit_and_sha256(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    index = module.frozen_index(_suite(tmp_path))
    assert index["schema"] == module.INDEX_SCHEMA
    assert index["suite_commit"] == module.SUITE_COMMIT
    assert all(len(record["sha256"]) == 64 and record["size"] > 0 for record in index["files"])
    assert [record["path"] for record in index["files"]] == sorted(
        record["path"] for record in index["files"]
    )


def test_verify_index_rejects_mutated_content(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    root = _suite(tmp_path)
    (root / "index.json").write_bytes(module.canonical_json_bytes(module.frozen_index(root)))
    module.verify_index(root)
    (root / "unpolarised/data/test5.dat").write_text("0.005 0.5 0.0 1e-04\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        module.verify_index(root)


def test_verify_index_rejects_a_missing_referenced_tier(load_tool_module, tmp_path: Path) -> None:
    module = load_tool_module("sync_orso_suite")
    root = _suite(tmp_path)
    (root / "index.json").write_bytes(module.canonical_json_bytes(module.frozen_index(root)))
    (root / "unpolarised/layers/test1.layers").unlink()
    with pytest.raises(ValueError, match="missing"):
        module.verify_index(root)


def test_index_json_in_the_repository_matches_its_recorded_hashes(load_tool_module) -> None:
    module = load_tool_module("sync_orso_suite")
    root = Path(__file__).resolve().parents[2] / "fixtures/orso"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["suite_commit"] == module.SUITE_COMMIT
    assert len(index["files"]) == 22
    module.verify_index(root)
