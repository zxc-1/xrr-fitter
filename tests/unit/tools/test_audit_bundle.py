from __future__ import annotations

import hashlib
import json
import tomllib
import types
from pathlib import Path

import pytest

SOURCE = {"source_commit": "a" * 40, "source_tree": "b" * 40}
ROOT = Path(__file__).resolve().parents[3]


def _json(path, value):
    path.write_text(json.dumps(value))


def _replace_report(root, name, value):
    _json(root / name, value)
    summary = json.loads((root / "summary.json").read_bytes())
    summary["reports"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    summary["inventory_sha256"] = summary["reports"]["inventory.json"]
    summary["sbom_sha256"] = summary["reports"]["installed.cdx.json"]
    _json(root / "summary.json", summary)


def _bom_properties(inventory_hash, **extra):
    properties = {
        "xrr:source:commit": SOURCE["source_commit"],
        "xrr:source:tree": SOURCE["source_tree"],
        "xrr:inventory:sha256": inventory_hash,
        **extra,
    }
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {"properties": [{"name": key, "value": value} for key, value in properties.items()]},
        "compositions": [{"aggregate": "incomplete"}],
    }


def _installed_report(tmp_path, packages=()):
    root = tmp_path / "installed"
    root.mkdir()
    inventory = {
        "schema": "xrr-installed-inventory-v1",
        "aggregate": "incomplete",
        "target": "windows-x64-py312",
        "packages": list(packages),
    }
    executable = {
        "schema": "xrr-executable-inventory-v1",
        "aggregate": "incomplete",
        "target": inventory["target"],
        "executable": {"sha256": "c" * 64, "size": 123},
    }
    inventory_hash = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    executable_hash = hashlib.sha256(json.dumps(executable).encode()).hexdigest()
    documents = {
        "inventory.json": inventory,
        "installed.cdx.json": _bom_properties(inventory_hash),
        "executable-inventory.json": executable,
        "executable.cdx.json": _bom_properties(
            executable_hash, **{"xrr:inputs:sha256": json.dumps({"installed-inventory": inventory_hash})}
        ),
    }
    hashes = {}
    for name, value in documents.items():
        content = json.dumps(value).encode()
        (root / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    summary = {
        "state": "PASS",
        "aggregate": "incomplete",
        **SOURCE,
        "reports": hashes,
        "inventory_sha256": hashes["inventory.json"],
        "sbom_sha256": hashes["installed.cdx.json"],
    }
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def test_audit_bundle_reads_byte_bound_reports_without_claiming_complete_composition(tmp_path, load_tool_module):
    module = load_tool_module("audit_bundle")
    root = _installed_report(tmp_path)
    values, bound = module.read_installed_report(root, SOURCE)
    assert values["executable-inventory.json"]["executable"]["sha256"] == "c" * 64
    assert "summary.json" in bound.hashes()
    bound.guard()


@pytest.mark.parametrize("change", ["bytes", "source", "status", "missing", "escape"])
def test_audit_bundle_rejects_drifted_incomplete_or_unsafe_report_bindings(tmp_path, load_tool_module, change):
    module = load_tool_module("audit_bundle")
    root = _installed_report(tmp_path)
    summary = json.loads((root / "summary.json").read_text())
    if change == "bytes":
        (root / "inventory.json").write_text("{}")
    elif change == "source":
        summary["source_commit"] = "d" * 40
    elif change == "status":
        summary["state"] = "FAIL"
    elif change == "missing":
        summary["reports"].pop("executable.cdx.json")
    else:
        summary["reports"]["../unrelated.json"] = "e" * 64
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        module.read_installed_report(root, SOURCE)


def test_audit_scope_requires_actual_ordinary_wheel_hashes_not_only_versions(load_tool_module):
    module = load_tool_module("audit_bundle")
    wheel = {"name": "sample", "version": "1.0", "sha256": "a" * 64}
    inventory = {
        "packages": [
            {"wheel": wheel, "provenance": "ordinary-manifest"},
            {"wheel": {"name": "pip", "version": "26.1.2", "sha256": "b" * 64}, "provenance": "bootstrap-lock"},
        ]
    }
    report = module.bind_ordinary_scope(inventory, {"wheels": [wheel]}, ("sample==1.0",))
    assert [item["name"] for item in report["scanned"]] == ["sample"]
    assert report["not_scanned"][0]["wheel"]["name"] == "pip"
    changed = {"wheels": [{**wheel, "sha256": "c" * 64}]}
    with pytest.raises(ValueError, match="wheel|bytes"):
        module.bind_ordinary_scope(inventory, changed, ("sample==1.0",))


def test_audit_scope_rejects_a_scan_with_another_pin_set(load_tool_module):
    module = load_tool_module("audit_bundle")
    wheel = {"name": "sample", "version": "1.0", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="pin"):
        module.bind_ordinary_scope(
            {"packages": [{"wheel": wheel, "provenance": "ordinary-manifest"}]}, {"wheels": [wheel]}, ("sample==2.0",)
        )


def test_installed_only_report_is_not_an_artifact_audit_binding(tmp_path, load_tool_module):
    module = load_tool_module("audit_bundle")
    root = _installed_report(tmp_path)
    summary = json.loads((root / "summary.json").read_bytes())
    for name in ("executable-inventory.json", "executable.cdx.json"):
        summary["reports"].pop(name)
    _json(root / "summary.json", summary)
    with pytest.raises(ValueError, match="artifact|executable"):
        module.read_installed_report(root, SOURCE)


def test_rehashed_complete_sbom_is_not_accepted_as_incomplete(tmp_path, load_tool_module):
    module = load_tool_module("audit_bundle")
    root = _installed_report(tmp_path)
    _replace_report(root, "executable.cdx.json", {"compositions": [{"aggregate": "complete"}]})
    with pytest.raises(ValueError, match="composition|incomplete"):
        module.read_installed_report(root, SOURCE)


def _bundle_fixture(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("audit_bundle")
    manifest = json.loads((ROOT / "tools/package-manifests/windows-x64-py312.json").read_bytes())
    packages = [{"wheel": wheel, "provenance": "ordinary-manifest"} for wheel in manifest["wheels"]]
    installed = _installed_report(tmp_path, packages)
    advisory = tmp_path / "advisory"
    advisory.mkdir()
    inputs = module._input_identity(ROOT)
    _json(advisory / "inputs.json", inputs)
    for target in module.TARGETS:
        pins, scope = module.advisory_input(ROOT, target)
        (advisory / f"{target}.requirements").write_text("\n".join(pins) + "\n")
        _json(advisory / f"{target}.scope.json", scope)
        findings = [{"name": pin.split("==")[0], "version": pin.split("==")[1], "vulns": []} for pin in pins]
        _json(advisory / f"{target}.advisories.json", {"dependencies": findings})
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["xrr"]["audit"]["requires"]
    summary = {
        "schema": "xrr-audit-report-v1",
        "kind": "advisories",
        "state": "PASS",
        "exit_codes": {target: 0 for target in module.TARGETS},
        "inputs_sha256": hashlib.sha256((advisory / "inputs.json").read_bytes()).hexdigest(),
        "tool_versions": dict(pin.split("==") for pin in requires),
    }
    _json(advisory / "summary.json", summary)
    identity = types.SimpleNamespace(head_commit=SOURCE["source_commit"], head_tree=SOURCE["source_tree"])
    monkeypatch.setattr(module, "clean_head_identity", lambda root: identity)
    argv = [
        "--repo-root",
        str(ROOT),
        "--installed-report",
        str(installed),
        "--advisory-report",
        str(advisory),
        "--report-dir",
        str(tmp_path / "binding"),
    ]
    return module, argv, advisory


def test_audit_bundle_cli_binds_actual_executable_and_exact_ordinary_scan(tmp_path, load_tool_module, monkeypatch):
    module, argv, advisory = _bundle_fixture(tmp_path, load_tool_module, monkeypatch)
    assert module.main(argv) == 0
    receipt = json.loads((tmp_path / "binding/summary.json").read_bytes())
    assert receipt["state"] == "PASS"
    assert receipt["aggregate"] == "incomplete"
    assert receipt["artifacts"][0]["sha256"] == "c" * 64
    assert len(receipt["ordinary_advisory_scope"]["scanned"]) == 38
    assert receipt["native_libraries_scanned"] is False
    assert (
        receipt["advisory_reports"]["summary.json"]
        == hashlib.sha256((advisory / "summary.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("change", ["status", "exit", "tool", "scope", "pins", "findings", "inputs", "input-hash"])
def test_audit_bundle_cli_preserves_failure_for_unbound_scan(tmp_path, load_tool_module, monkeypatch, change):
    module, argv, advisory = _bundle_fixture(tmp_path, load_tool_module, monkeypatch)
    summary = json.loads((advisory / "summary.json").read_bytes())
    target = "windows-x64-py312"
    if change in {"status", "exit", "tool", "input-hash"}:
        mutations = {
            "status": {"state": "FAIL"},
            "exit": {"exit_codes": {target: 0}},
            "tool": {"tool_versions": {}},
            "input-hash": {"inputs_sha256": "f" * 64},
        }
        summary.update(mutations[change])
        _json(advisory / "summary.json", summary)
    elif change == "pins":
        (advisory / f"{target}.requirements").write_text("sample==1.0\n")
    elif change == "inputs":
        _json(advisory / "inputs.json", {})
        summary["inputs_sha256"] = hashlib.sha256((advisory / "inputs.json").read_bytes()).hexdigest()
        _json(advisory / "summary.json", summary)
    else:
        suffix = "scope" if change == "scope" else "advisories"
        path = advisory / f"{target}.{suffix}.json"
        value = json.loads(path.read_bytes())
        if change == "scope":
            value["native_libraries_scanned"] = True
        else:
            value["dependencies"][0]["vulns"].append({"id": "TEST-ADVISORY"})
        _json(path, value)
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2
    assert not (tmp_path / "binding/summary.json").exists()
    assert json.loads((tmp_path / "binding/failure.json").read_bytes())["state"] == "FAIL"


def test_scan_bytes_are_guarded_after_scope_binding(tmp_path, load_tool_module, monkeypatch):
    module, argv, advisory = _bundle_fixture(tmp_path, load_tool_module, monkeypatch)
    original = module.bind_ordinary_scope

    def changed(*args):
        value = original(*args)
        (advisory / "windows-x64-py312.advisories.json").write_text("{}")
        return value

    monkeypatch.setattr(module, "bind_ordinary_scope", changed)
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2
    assert not (tmp_path / "binding/summary.json").exists()


@pytest.mark.parametrize("change", ["source", "inventory", "installed-input", "schema", "target"])
def test_inner_executable_bindings_must_agree_even_if_outer_hashes_are_rewritten(tmp_path, load_tool_module, change):
    module = load_tool_module("audit_bundle")
    root = _installed_report(tmp_path)
    name = "executable-inventory.json" if change in {"schema", "target"} else "executable.cdx.json"
    value = json.loads((root / name).read_bytes())
    if change in {"schema", "target"}:
        value[change] = "wrong"
    else:
        mutations = {
            "source": ("xrr:source:commit", "f" * 40),
            "inventory": ("xrr:inventory:sha256", "f" * 64),
            "installed-input": ("xrr:inputs:sha256", json.dumps({"installed-inventory": "f" * 64})),
        }
        key, replacement = mutations[change]
        next(row for row in value["metadata"]["properties"] if row["name"] == key)["value"] = replacement
    _replace_report(root, name, value)
    with pytest.raises(ValueError, match="source|inventory|schema|target|input|binding"):
        module.read_installed_report(root, SOURCE)
