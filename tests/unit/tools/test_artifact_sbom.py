from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/artifact_sbom.py").is_file(), "missing artifact-bound SBOM"
    return load_tool_module("artifact_sbom")


def _manifest(tmp_path):
    artifacts = tmp_path / "bundle" / "artifacts"
    artifacts.mkdir(parents=True)
    wheel = artifacts / "sample-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel bytes")
    sdist = artifacts / "sample-1.0.tar.gz"
    sdist.write_bytes(b"sdist bytes")
    records = [
        {
            "kind": "sdist",
            "path": "artifacts/sample-1.0.tar.gz",
            "filename": sdist.name,
            "size": sdist.stat().st_size,
            "sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
        },
        {
            "kind": "wheel",
            "path": "artifacts/sample-1.0-py3-none-any.whl",
            "filename": wheel.name,
            "size": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
    ]
    manifest = {
        "schema": "xrr-r23-artifact-manifest-v1",
        "status": "PASS",
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "artifacts": records,
    }
    path = tmp_path / "bundle" / "artifact-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    return artifacts, path, manifest


def test_artifact_sbom_binds_manifest_bytes_source_and_archive_files(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    artifacts, manifest_path, manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        module,
        "inspect_wheel",
        lambda path, record: {
            "name": "sample",
            "version": "1.0",
            "requirements": [],
            "licenses": [],
            "files": [{"path": "sample.py", "size": 3, "sha256": "c" * 64, "kind": "python"}],
            "vendored": [],
        },
    )
    monkeypatch.setattr(
        module,
        "_sdist_inventory",
        lambda path: [{"path": "sample-1.0/PKG-INFO", "size": 4, "sha256": "d" * 64, "kind": "metadata"}],
    )
    result = module.build_artifact_sbom(ROOT, manifest_path, artifacts)
    assert result["bomFormat"] == "CycloneDX"
    assert result["specVersion"] == "1.6"
    assert result["compositions"] == [{"aggregate": "incomplete"}]
    properties = {item["name"]: item["value"] for item in result["metadata"]["properties"]}
    assert properties["xrr:artifact-manifest:sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert properties["xrr:source:commit"] == manifest["head_commit"]
    assert len(result["components"]) == 2
    assert {item["name"] for item in result["components"]} == {"sample-1.0.tar.gz", "sample-1.0-py3-none-any.whl"}


def test_artifact_sbom_rejects_changed_artifact_bytes(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    artifacts, manifest_path, _manifest_value = _manifest(tmp_path)
    (artifacts / "sample-1.0.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact manifest or artifact content drift"):
        module.build_artifact_sbom(ROOT, manifest_path, artifacts)


def test_artifact_sbom_require_complete_is_an_explicit_failure(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    artifacts, manifest_path, _manifest_value = _manifest(tmp_path)
    monkeypatch.setattr(
        module,
        "inspect_wheel",
        lambda path, record: {
            "name": "sample",
            "version": "1.0",
            "requirements": [],
            "licenses": [],
            "files": [],
            "vendored": [],
        },
    )
    monkeypatch.setattr(module, "_sdist_inventory", lambda path: [])
    with pytest.raises(ValueError, match="complete"):
        module.write_artifact_sbom(ROOT, manifest_path, artifacts, require_complete=True)


def test_sdist_inventory_rejects_unsafe_members(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("../outside")
        info.size = 1
        archive.addfile(info, __import__("io").BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        module._sdist_inventory(path)
