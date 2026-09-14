from __future__ import annotations

import hashlib
import json
import types

import pytest
from tests.support.qt_cocoa_fixtures import qt_build_case

SOURCE = {"source_commit": "a" * 40, "source_tree": "b" * 40}


def _installed_bom(hashes):
    properties = {
        "xrr:source:commit": SOURCE["source_commit"],
        "xrr:source:tree": SOURCE["source_tree"],
        "xrr:inputs:sha256": json.dumps(hashes),
    }
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {"properties": [{"name": name, "value": value} for name, value in properties.items()]},
    }


def _qt_scope_case(load_tool_module, tmp_path):
    evidence, root, wheels, report, receipt = qt_build_case(load_tool_module, tmp_path)
    module = load_tool_module("audit_bundle")
    sample = {"name": "sample", "version": "1.0", "sha256": "f" * 64}
    inventory = {
        "target": "macos-arm64-py312",
        "packages": [
            {"wheel": sample, "provenance": "ordinary-manifest"},
            {"wheel": receipt["wheel"], "provenance": "qt-cocoa-source-build"},
        ],
    }
    manifest = {"target": inventory["target"], "wheels": [sample, receipt["inputs"]["upstream_wheel"]]}
    pins = ("sample==1.0", "pyside6-essentials==6.11.2")
    return module, root, wheels, report, receipt, inventory, manifest, pins


def test_verified_qt_derivative_is_not_misrepresented_as_scanned_pypi_bytes(load_tool_module, tmp_path):
    module, root, wheels, report, receipt, inventory, manifest, pins = _qt_scope_case(load_tool_module, tmp_path)
    scope = module.bind_ordinary_scope(inventory, manifest, pins, receipt)
    assert scope["scanned"] == [manifest["wheels"][0]]
    assert scope["not_scanned"] == [inventory["packages"][1]]
    assert scope["upstream_advisory_only"] == [
        {"upstream": receipt["inputs"]["upstream_wheel"], "derived": receipt["wheel"]}
    ]


@pytest.mark.parametrize("mutation", ["installed", "upstream", "target", "provenance", "pins", "missing"])
def test_qt_advisory_scope_rejects_unbound_replacements(load_tool_module, tmp_path, mutation):
    module, root, wheels, report, receipt, inventory, manifest, pins = _qt_scope_case(load_tool_module, tmp_path)
    if mutation == "installed":
        inventory["packages"][1]["wheel"] = {**receipt["wheel"], "sha256": "0" * 64}
    elif mutation == "upstream":
        manifest["wheels"][1] = {**manifest["wheels"][1], "sha256": "0" * 64}
    elif mutation == "target":
        inventory["target"] = "windows-x64-py312"
    elif mutation == "provenance":
        inventory["packages"][1]["provenance"] = "ordinary-manifest"
    elif mutation == "pins":
        pins = ("sample==1.0",)
    else:
        receipt = None
    with pytest.raises(ValueError, match="Qt|wheel|bytes|pin"):
        module.bind_ordinary_scope(inventory, manifest, pins, receipt)


def _qt_bound_case(load_tool_module, tmp_path):
    module, root, wheels, report, receipt, inventory, manifest, pins = _qt_scope_case(load_tool_module, tmp_path)
    evidence = load_tool_module("qt_cocoa_evidence")
    hashes = module.InputBindings(evidence.binding_paths(root, report / "build.json")).hashes()
    documents = {
        "inventory.json": inventory,
        "installed.cdx.json": _installed_bom(hashes),
    }
    args = types.SimpleNamespace(repo_root=root, qt_build=report / "build.json", wheel_dir=wheels)
    bound = module.InputBindings({})
    return module, args, documents, bound, receipt


def test_qt_advisory_receipt_reverifies_actual_derivative_and_binds_installed_input(load_tool_module, tmp_path):
    module, args, documents, bound, receipt = _qt_bound_case(load_tool_module, tmp_path)
    actual, guard = module._qt_scope_receipt(args, documents, SOURCE, bound)
    assert actual == receipt
    assert bound.hashes()["qt-cocoa-build"] == hashlib.sha256(args.qt_build.read_bytes()).hexdigest()
    guard()


@pytest.mark.parametrize("mutation", ["receipt", "wheel", "upstream", "recipe"])
def test_qt_advisory_guard_rejects_build_inputs_changed_after_binding(load_tool_module, tmp_path, mutation):
    module, args, documents, bound, receipt = _qt_bound_case(load_tool_module, tmp_path)
    actual, guard = module._qt_scope_receipt(args, documents, SOURCE, bound)
    paths = {
        "receipt": args.qt_build,
        "wheel": args.qt_build.parent / "wheels" / receipt["wheel"]["filename"],
        "upstream": args.wheel_dir / receipt["inputs"]["upstream_wheel"]["filename"],
        "recipe": args.repo_root / "tools/qt-cocoa/ownership.patch",
    }
    paths[mutation].write_bytes(b"changed after binding")
    with pytest.raises(ValueError):
        guard()


@pytest.mark.parametrize("mutation", ["hash", "missing-build", "missing-wheels", "target"])
def test_qt_advisory_receipt_rejects_wrong_installed_input_or_incomplete_arguments(load_tool_module, tmp_path, mutation):
    module, args, documents, bound, receipt = _qt_bound_case(load_tool_module, tmp_path)
    if mutation == "hash":
        rows = documents["installed.cdx.json"]["metadata"]["properties"]
        next(row for row in rows if row["name"] == "xrr:inputs:sha256")["value"] = "{}"
    elif mutation == "missing-build":
        args.qt_build = None
    elif mutation == "missing-wheels":
        args.wheel_dir = None
    else:
        documents["inventory.json"]["target"] = "windows-x64-py312"
    with pytest.raises(ValueError, match="Qt|macOS"):
        module._qt_scope_receipt(args, documents, SOURCE, bound)
