from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.support.qt_cocoa_fixtures import qt_build_case as _case

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "tools/package-manifests/qt-cocoa-macos-arm64-py312.json"


def _module(load_tool_module):
    assert (ROOT / "tools/qt_cocoa_evidence.py").is_file(), "missing Qt source-build evidence validation"
    return load_tool_module("qt_cocoa_evidence")


def test_build_receipt_binds_source_recipes_builder_and_derived_bytes(load_tool_module, tmp_path):
    module, root, wheels, report, result = _case(load_tool_module, tmp_path)
    loaded = module.read_build(root, report / "build.json", wheels)
    assert loaded["record"] == result["wheel"]
    assert loaded["path"] == report / "wheels" / result["wheel"]["filename"]
    assert result["offline"] is True and result["derived_wheel_reused"] is False
    assert result["builder_sha256"] == module.code_bindings(root)
    assert len(result["native_regressions"]) == 6
    assert "qt-cocoa-build" in loaded["bindings"]


@pytest.mark.parametrize(
    "mutation", ["source", "recipe", "builder", "upstream", "native", "regression", "cached", "schema"]
)
def test_forged_or_stale_qt_build_receipt_is_not_installable(load_tool_module, tmp_path, mutation):
    module, root, wheels, report, result = _case(load_tool_module, tmp_path)
    if mutation == "source":
        result["inputs"]["sources"][0]["sha256"] = "0" * 64
    elif mutation == "recipe":
        (root / result["inputs"]["recipes"][0]["path"]).write_bytes(b"changed")
    elif mutation == "builder":
        path = root / module.BUILD_CODE[0]
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "upstream":
        result["inputs"]["upstream_wheel"]["sha256"] = "0" * 64
    elif mutation == "native":
        result["derivation"]["plugin_sha256"] = "0" * 64
    elif mutation == "regression":
        result["native_regressions"][module.CASES[0]] = 1
    elif mutation == "cached":
        result["derived_wheel_reused"] = True
    else:
        result["schema"] = "unrecognized"
    (report / "build.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        module.read_build(root, report / "build.json", wheels)


def test_source_additions_rejects_mismatched_source_bytes(load_tool_module):
    module = _module(load_tool_module)
    inputs = load_tool_module("qt_cocoa_inputs")
    with pytest.raises(ValueError, match="source"):
        module.source_additions(ROOT, inputs.read_inputs(ROOT), lambda record: b"changed")


def test_build_evidence_paths_cover_every_runtime_derivation_input(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    paths = module.binding_paths(ROOT, tmp_path / "build.json")
    assert paths["qt-cocoa-build"] == tmp_path / "build.json"
    assert paths["qt-cocoa-inputs"] == ROOT / MANIFEST
    assert set(module.BUILD_CODE) <= {str(p.relative_to(ROOT)) for p in paths.values() if p.is_relative_to(ROOT)}
    assert ROOT / "tools/qt-cocoa/ownership.patch" in paths.values()


@pytest.mark.parametrize("content", ["[]", "{}", '{"schema":"bad","schema":"xrr-qt-cocoa-build-v1"}'])
def test_malformed_receipt_is_a_validation_failure(load_tool_module, tmp_path, content):
    module, root, wheels, report, result = _case(load_tool_module, tmp_path)
    (report / "build.json").write_text(content)
    with pytest.raises(ValueError):
        module.read_build(root, report / "build.json", wheels)


def test_receipt_rejects_undeclared_fields(load_tool_module, tmp_path):
    module, root, wheels, report, result = _case(load_tool_module, tmp_path)
    result["unbound"] = "unexpected"
    (report / "build.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        module.read_build(root, report / "build.json", wheels)


def test_builder_binding_includes_execution_and_byte_verification_helpers(load_tool_module):
    module = _module(load_tool_module)
    assert {
        "tools/refnx_build_runtime.py",
        "tools/package_downloads.py",
        "tools/package_manifest.py",
        "tools/package_cache.py",
        "tools/installed_files.py",
        "tools/installed_transforms.py",
        "tools/wheel_inventory.py",
        "tools/native_binary.py",
        "tools/native_bytes.py",
        "tools/native_pe.py",
        "tools/audit_reports.py",
        "tools/verify_report.py",
        "tools/verify_publish.py",
        "tools/distribution_manifest.py",
    } <= set(module.BUILD_CODE)
