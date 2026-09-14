from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path("tools/package-manifests/qt-cocoa-macos-arm64-py312.json")


def _module(load_tool_module):
    assert (ROOT / "tools/qt_cocoa_inputs.py").is_file(), "missing product Qt source input owner"
    return load_tool_module("qt_cocoa_inputs")


def _copy_inputs(root, manifest):
    for name in [MANIFEST, "tools/package-manifests/macos-arm64-py312.json", *[r["path"] for r in manifest["recipes"]]]:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)


def _case(tmp_path, load_tool_module):
    module = _module(load_tool_module)
    manifest = module.read_inputs(ROOT)
    root = tmp_path / "repo"
    _copy_inputs(root, manifest)
    contents = {}
    manifest["sources"] = manifest["sources"][:2]
    for record in [manifest["sdk"], *manifest["sources"]]:
        content = record["filename"].encode()
        record.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
        contents[record["filename"]] = content
    (root / MANIFEST).write_text(json.dumps(manifest))
    return module, root, module.read_inputs(root), contents


def test_product_manifest_pins_upstream_qt_and_distinct_arm64_output(load_tool_module):
    module = _module(load_tool_module)
    manifest = module.read_inputs(ROOT)
    assert manifest["qt_version"] == "6.11.2"
    assert manifest["target"] == "macos-arm64-py312"
    assert manifest["wheel_tag"] == "cp310-abi3-macosx_13_0_arm64"
    assert manifest["wheel_build"] == "1xrrcocoa"
    assert len(manifest["sources"]) == 97
    assert manifest["sdk"]["sha256"] == "9592f84f7e26d532c5c56824d1da7c9214a766cb0a17beb5af71022bcfbcd271"
    assert manifest["upstream_wheel"]["name"] == "pyside6-essentials"
    assert len(module.input_records(manifest)) == 98


@pytest.mark.parametrize(
    "mutation",
    ["source_path", "source_url", "source_hash", "source_duplicate", "sdk_url", "recipe", "upstream", "architecture"],
)
def test_input_manifest_refuses_changed_trust_boundaries(load_tool_module, tmp_path, mutation):
    module, root, manifest, _ = _case(tmp_path, load_tool_module)
    changed = deepcopy(manifest)
    if mutation == "source_path":
        changed["sources"][0]["path"] = "../outside"
    elif mutation == "source_url":
        changed["sources"][0]["url"] = "https://example.invalid/source"
    elif mutation == "source_hash":
        changed["sources"][0]["sha256"] = "bad"
    elif mutation == "source_duplicate":
        changed["sources"].append(changed["sources"][0])
    elif mutation == "sdk_url":
        changed["sdk"]["url"] = "file:///tmp/untrusted-sdk"
    elif mutation == "recipe":
        (root / changed["recipes"][0]["path"]).write_bytes(b"changed")
    elif mutation == "upstream":
        changed["upstream_wheel"]["sha256"] = "0" * 64
    else:
        changed["wheel_tag"] = "cp310-abi3-macosx_13_0_universal2"
    (root / MANIFEST).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        module.read_inputs(root)


def _download_case(load_tool_module, tmp_path, monkeypatch):
    module, root, manifest, contents = _case(tmp_path, load_tool_module)
    calls = []

    def fetch(record, destination):
        calls.append(record["filename"])
        destination.write_bytes(contents[record["filename"]])

    monkeypatch.setattr(module, "_fetch_file", fetch)
    return module, root, manifest, calls


def test_qt_input_cache_verifies_cold_and_warm_sources(load_tool_module, tmp_path, monkeypatch):
    module, root, manifest, calls = _download_case(load_tool_module, tmp_path, monkeypatch)
    for name in ("cold", "warm"):
        report = tmp_path / name
        assert module.cached_inputs(root, report, tmp_path / "cache", "pr-26") == 0
        assert json.loads((report / "cache.json").read_text())["hit"] is (name == "warm")
        assert len(module.verify_inputs(root, manifest, report / "inputs")) == 3
    assert len(calls) == 3
    assert module.input_cache_key(manifest, "trusted") != module.input_cache_key(manifest, "pr-26")


@pytest.mark.parametrize("member", ["sdk", "source", "extra"])
def test_poisoned_qt_cache_fails_without_replacing_or_redownloading(load_tool_module, tmp_path, monkeypatch, member):
    module, root, manifest, calls = _download_case(load_tool_module, tmp_path, monkeypatch)
    module.cached_inputs(root, tmp_path / "cold", tmp_path / "cache", "trusted")
    directory = next((tmp_path / "cache").iterdir())
    names = {"sdk": manifest["sdk"]["filename"], "source": manifest["sources"][0]["filename"], "extra": "derived.whl"}
    target = directory / names[member]
    target.write_bytes(b"poisoned")
    with pytest.raises(ValueError):
        module.cached_inputs(root, tmp_path / "warm", tmp_path / "cache", "trusted")
    assert len(calls) == 3
    assert target.read_bytes() == b"poisoned"
    assert not (tmp_path / "warm/inputs").exists()


def test_wrong_download_bytes_publish_no_cache(load_tool_module, tmp_path, monkeypatch):
    module, root, _, _ = _download_case(load_tool_module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_fetch_file", lambda record, path: path.write_bytes(b"incorrect"))
    with pytest.raises(ValueError):
        module.cached_inputs(root, tmp_path / "report", tmp_path / "cache", "trusted")
    assert not list((tmp_path / "cache").iterdir())
    assert not (tmp_path / "report/inputs").exists()


def test_hash_bound_patch_retains_literal_unified_diff_context_bytes():
    attributes = subprocess.check_output(
        ("git", "check-attr", "text", "whitespace", "--", "tools/qt-cocoa/ownership.patch"), cwd=ROOT, text=True
    )
    assert "tools/qt-cocoa/ownership.patch: text: unset" in attributes
    assert "tools/qt-cocoa/ownership.patch: whitespace: unset" in attributes
