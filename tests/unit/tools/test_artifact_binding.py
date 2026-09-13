from __future__ import annotations

import hashlib

import pytest


def _component(name, content):
    return {"name": name, "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(content).hexdigest()}]}


def _bom():
    return {
        "components": [
            {"name": "xrr_fitter-1.0-py3-none-any.whl", "components": [_component("xrr_fitter/app.py", b"source")]},
            {
                "name": "xrr_fitter-1.0.tar.gz",
                "components": [
                    _component("xrr_fitter-1.0/src/xrr_fitter/app.py", b"source"),
                    _component("xrr_fitter-1.0/pyproject.toml", b"config"),
                ],
            },
        ]
    }


def test_artifact_sources_are_verified_against_actual_captured_git_bytes(load_tool_module):
    module = load_tool_module("artifact_binding")
    expected = {"src/xrr_fitter/app.py": b"source", "pyproject.toml": b"config"}
    assert module.verify_artifact_sources(_bom(), expected) == {"wheel": 1, "sdist": 2}


@pytest.mark.parametrize("kind", [0, 1])
def test_artifact_sources_reject_rehashed_source_drift(load_tool_module, kind):
    module = load_tool_module("artifact_binding")
    bom = _bom()
    bom["components"][kind]["components"][0]["hashes"][0]["content"] = "a" * 64
    with pytest.raises(ValueError, match="source.*bytes"):
        module.verify_artifact_sources(bom, {"src/xrr_fitter/app.py": b"source", "pyproject.toml": b"config"})


def test_artifact_runtime_closure_excludes_unrelated_installer_and_tests(load_tool_module):
    module = load_tool_module("artifact_binding")
    graph = {"app": ["numpy"], "numpy": [], "pip": [], "pytest": ["pluggy"], "pluggy": []}
    assert module.runtime_closure(graph, "app") == {"numpy"}


def test_artifact_binding_rejects_other_source_identities(load_tool_module):
    module = load_tool_module("artifact_binding")
    from distribution_manifest import ArtifactManifest

    manifest = ArtifactManifest("schema", "PASS", "a" * 40, "b" * 40, ())
    with pytest.raises(ValueError, match="source identity"):
        module.require_source_identity(manifest, {"source_commit": "c" * 40, "source_tree": "b" * 40})
