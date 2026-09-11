from __future__ import annotations

import json
import types

import pytest
from tests.support.installed_wheels import sample_installation


def _inventory(load_tool_module, tmp_path):
    module = load_tool_module("installed_inventory")
    layout, values = sample_installation(tmp_path)
    inputs = [module.WheelInput(**value) for value in values]
    return module.inspect_installation(module.InstallLayout(**layout), inputs, inputs[0])


def _flatten(components):
    for component in components:
        yield component
        yield from _flatten(component.get("components", []))


def _assert_bound_graph(bom):
    components = list(_flatten(bom["components"]))
    refs = {component["bom-ref"] for component in components}
    assert len(refs) == len(components)
    assert all(row["ref"] in refs and set(row["dependsOn"]) <= refs for row in bom["dependencies"])
    return components


def test_installed_sbom_binds_actual_files_and_derivations_to_wheel_bytes(tmp_path, load_tool_module):
    module = load_tool_module("installed_sbom")
    inventory = _inventory(load_tool_module, tmp_path)
    source = {"source_commit": "a" * 40, "source_tree": "b" * 40}
    bom = module.assemble_installed_sbom(inventory, {"example.lock": "c" * 64}, source)
    assert bom["compositions"] == [{"aggregate": "incomplete"}]
    components = _assert_bound_graph(bom)
    installed = [c for c in components if c["bom-ref"].startswith("urn:xrr:installed:")]
    assert len(installed) == len(inventory["files"])
    actual = {item["path"]: item["sha256"] for item in inventory["files"]}
    assert {item["name"]: item["hashes"][0]["content"] for item in installed} == actual
    assert str(tmp_path) not in json.dumps(bom)


def test_refnx_observed_wheel_version_is_not_presented_as_a_pypi_release(tmp_path, load_tool_module):
    module = load_tool_module("installed_sbom")
    inventory = _inventory(load_tool_module, tmp_path)
    package = inventory["packages"][1]
    package["wheel"]["name"] = package["inventory"]["name"] = "refnx"
    package["provenance"] = "refnx-source-build"
    for item in inventory["files"]:
        for claim in item["claims"]:
            if claim["package"] == "sample":
                claim["package"] = "refnx"
    bom = module.assemble_installed_sbom(inventory, {}, {"source_commit": "a" * 40, "source_tree": "b" * 40})
    component = next(c for c in bom["components"] if c["name"] == "refnx")
    assert "purl" not in component
    assert component["version"] == "1.0"


def test_installed_sbom_refuses_missing_archive_source_claims(tmp_path, load_tool_module):
    module = load_tool_module("installed_sbom")
    inventory = _inventory(load_tool_module, tmp_path)
    inventory["files"][0]["claims"][0]["wheel_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="claim|wheel"):
        module.assemble_installed_sbom(inventory, {}, {"source_commit": "a" * 40, "source_tree": "b" * 40})


def _cli_fixture(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_sbom")
    inventory = _inventory(load_tool_module, tmp_path)
    identity = types.SimpleNamespace(head_commit="a" * 40, head_tree="b" * 40)
    monkeypatch.setattr(module, "clean_head_identity", lambda root: identity)
    monkeypatch.setattr(
        module,
        "load_installation_inputs",
        lambda *args, **kwargs: ({"target": "macos-arm64-py312"}, [1, 2], None, {}, lambda: None),
    )
    monkeypatch.setattr(module, "current_layout", lambda target: None)
    monkeypatch.setattr(module, "inspect_installation", lambda *args, **kwargs: inventory)
    argv = [
        "--repo-root",
        str(tmp_path / "repo"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--wheel-dir",
        str(tmp_path / "wheels"),
        "--pip-wheel",
        str(tmp_path / "pip.whl"),
        "--report-dir",
        str(tmp_path / "report"),
    ]
    return module, argv


def test_installed_cli_strict_incomplete_fails_but_retains_byte_evidence(tmp_path, load_tool_module, monkeypatch):
    module, argv = _cli_fixture(tmp_path, load_tool_module, monkeypatch)
    with pytest.raises(SystemExit) as error:
        module.main([*argv, "--require-complete"])
    assert error.value.code == 2
    assert (tmp_path / "report/inventory.json").is_file()
    assert json.loads((tmp_path / "report/summary.json").read_bytes())["aggregate"] == "incomplete"


def test_installed_cli_retains_real_verification_failures_without_a_success_summary(
    tmp_path, load_tool_module, monkeypatch
):
    module, argv = _cli_fixture(tmp_path, load_tool_module, monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("installed bytes changed")

    monkeypatch.setattr(module, "inspect_installation", fail)
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2
    assert not (tmp_path / "report/summary.json").exists()
    assert json.loads((tmp_path / "report/failure.json").read_bytes())["error"] == "installed bytes changed"


def test_installed_cli_binds_all_emitted_reports_by_hash(tmp_path, load_tool_module, monkeypatch):
    module, argv = _cli_fixture(tmp_path, load_tool_module, monkeypatch)
    assert module.main(argv) == 0
    summary = json.loads((tmp_path / "report/summary.json").read_bytes())
    assert set(summary["reports"]) == {"inventory.json", "installed.cdx.json"}
    for name, checksum in summary["reports"].items():
        assert __import__("hashlib").sha256((tmp_path / "report" / name).read_bytes()).hexdigest() == checksum


def test_installed_cli_failure_retains_bounded_byte_diagnostics(tmp_path, load_tool_module, monkeypatch):
    module, argv = _cli_fixture(tmp_path, load_tool_module, monkeypatch)

    def fail(*args, **kwargs):
        raise module.InstalledMismatch("changed installed bytes", {"path": "sample/file.py"})

    monkeypatch.setattr(module, "inspect_installation", fail)
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2
    failure = json.loads((tmp_path / "report/failure.json").read_bytes())
    assert failure["verification"] == {"path": "sample/file.py"}
    assert not (tmp_path / "report/summary.json").exists()
