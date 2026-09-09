from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/package_sbom.py").is_file(), "missing package-byte SBOM"
    return load_tool_module("package_sbom")


def _inventory(name, version, requirements=()):
    return {"name": name, "version": version, "requirements": list(requirements), "licenses": [], "files": []}


def test_dependency_edges_evaluate_target_platform_not_host(load_tool_module):
    module = _module(load_tool_module)
    inventories = [_inventory("sample", "1", ['dep>=2; sys_platform == "win32"']), _inventory("dep", "2")]
    windows = module.dependency_edges(inventories, "windows-x64-py312")
    macos = module.dependency_edges(inventories, "macos-arm64-py312")
    assert windows["sample"] == ["dep"]
    assert macos["sample"] == []


def test_dependency_edges_follow_required_extras(load_tool_module):
    module = _module(load_tool_module)
    inventories = [
        _inventory("sample", "1", ["dep[fast]>=2"]),
        _inventory("dep", "2", ['extra-lib; extra == "fast"']),
        _inventory("extra-lib", "1"),
    ]
    assert module.dependency_edges(inventories, "windows-x64-py312")["dep"] == ["extra-lib"]


@pytest.mark.parametrize("requirement", ["missing>=1", "dep>=3", "dep @ https://untrusted.invalid/dep.whl"])
def test_dependency_edges_reject_missing_incompatible_or_unbound_requirements(load_tool_module, requirement):
    module = _module(load_tool_module)
    with pytest.raises(ValueError, match="dependency"):
        module.dependency_edges([_inventory("sample", "1", [requirement]), _inventory("dep", "2")], "windows-x64-py312")


def test_sbom_binds_wheel_hash_and_preserves_native_completeness_limit(load_tool_module):
    module = _module(load_tool_module)
    manifest = {
        "target": "windows-x64-py312",
        "lock_sha256": "a" * 64,
        "vcs": [],
        "wheels": [{"name": "sample", "version": "1", "sha256": "b" * 64, "url": "https://example.invalid/sample.whl"}],
    }
    inventory = _inventory("sample", "1")
    inventory["files"] = [{"path": "sample/native", "sha256": "c" * 64, "size": 20, "kind": "native", "format": "pe"}]
    result = module.assemble_sbom(manifest, [inventory])
    assert result["bomFormat"] == "CycloneDX"
    assert result["specVersion"] == "1.6"
    assert result["components"][0]["hashes"] == [{"alg": "SHA-256", "content": "b" * 64}]
    file = result["components"][0]["components"][0]
    assert file["hashes"] == [{"alg": "SHA-256", "content": "c" * 64}]
    assert result["compositions"] == [{"aggregate": "incomplete"}]
    properties = {item["name"]: item["value"] for item in result["metadata"]["properties"]}
    assert properties["xrr:inventory:native-files"] == "1"
    assert properties["xrr:inventory:native-relationships"] == "unresolved"


def test_vendored_component_retains_metadata_evidence_without_a_fake_wheel_hash(load_tool_module):
    module = _module(load_tool_module)
    inventory = _inventory("sample", "1")
    embedded = _inventory("embedded", "2")
    embedded.update(metadata_path="sample/_vendor/embedded-2.dist-info/METADATA", metadata_sha256="c" * 64)
    inventory["vendored"] = [embedded]
    wheel = {"name": "sample", "version": "1", "sha256": "b" * 64, "url": "https://example.invalid/sample.whl"}
    result = module._package_component(wheel, inventory)
    vendor = result["components"][0]
    assert vendor["name"] == "embedded"
    assert vendor["version"] == "2"
    assert "hashes" not in vendor
    properties = {item["name"]: item["value"] for item in vendor["properties"]}
    assert properties["xrr:vendored:metadata-sha256"] == "c" * 64
    assert properties["xrr:vendored:dependency-relationships"] == "unresolved"
