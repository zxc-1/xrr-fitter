from __future__ import annotations

import json

import pytest
from tests.support.installed_wheels import make_wheel
from tests.support.qt_cocoa_fixtures import qt_build_case


def _installer(load_tool_module, tmp_path, monkeypatch):
    evidence, root, wheels, build, receipt = qt_build_case(load_tool_module, tmp_path)
    module = load_tool_module("package_downloads")
    manifest = root / "tools/package-manifests/macos-arm64-py312.json"
    monkeypatch.setattr(module, "read_manifest", lambda root, path: json.loads(path.read_bytes()))
    monkeypatch.setattr(module, "require_install_target", lambda target: None)
    monkeypatch.setattr(module, "_installed_versions", lambda manifest: None)
    return module, root, wheels, build / "build.json", receipt, manifest


def test_installer_selects_only_the_derived_essentials_with_its_hash(load_tool_module, tmp_path, monkeypatch):
    module, root, wheels, build, receipt, manifest = _installer(load_tool_module, tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(module, "run_logged", lambda command, **kwargs: commands.append(command) or 0)
    report = tmp_path / "installation"
    assert module.install_packages(root, manifest, wheels, report, qt_build=build) == 0
    requirements = (report / "install.requirements").read_text()
    assert receipt["wheel"]["filename"] in requirements
    assert receipt["wheel"]["sha256"] in requirements
    assert receipt["inputs"]["upstream_wheel"]["filename"] not in requirements
    assert "--no-index" in commands[0] and "--require-hashes" in commands[0]
    assert json.loads((report / "qt-cocoa-build.json").read_text()) == receipt


def test_invalid_qt_receipt_never_reaches_pip(load_tool_module, tmp_path, monkeypatch):
    module, root, wheels, build, receipt, manifest = _installer(load_tool_module, tmp_path, monkeypatch)
    receipt["native_regressions"]["borrowed-destructor"] = 3
    build.write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: calls.append(args) or 0)
    with pytest.raises(ValueError):
        module.install_packages(root, manifest, wheels, tmp_path / "install", qt_build=build)
    assert calls == []


def test_receipt_changed_during_installation_cannot_report_success(load_tool_module, tmp_path, monkeypatch):
    module, root, wheels, build, receipt, manifest = _installer(load_tool_module, tmp_path, monkeypatch)

    def run(*args, **kwargs):
        receipt["derived_wheel_reused"] = True
        build.write_text(json.dumps(receipt))
        return 0

    monkeypatch.setattr(module, "run_logged", run)
    report = tmp_path / "install"
    with pytest.raises(ValueError):
        module.install_packages(root, manifest, wheels, report, qt_build=build)
    assert not (report / "summary.json").exists()


def test_windows_install_rejects_macos_qt_evidence(load_tool_module, tmp_path, monkeypatch):
    module, root, wheels, build, receipt, manifest = _installer(load_tool_module, tmp_path, monkeypatch)
    value = json.loads(manifest.read_text())
    value["target"] = "windows-x64-py312"
    manifest.write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: calls.append(args) or 0)
    with pytest.raises(ValueError, match="macOS|Qt"):
        module.install_packages(root, manifest, wheels, tmp_path / "install", qt_build=build)
    assert not calls


def _loader(load_tool_module, tmp_path, monkeypatch):
    evidence, root, wheels, build, receipt = qt_build_case(load_tool_module, tmp_path)
    module = load_tool_module("installed_inputs")
    packaging = make_wheel(wheels, name="packaging")
    pip = make_wheel(tmp_path / "bootstrap", name="pip", version="26.1.2")
    (root / "tools/bootstrap-requirements.lock").write_text(
        "".join(
            f"{item['record']['name']}=={item['record']['version']} --hash=sha256:{item['record']['sha256']}\n"
            for item in (packaging, pip)
        )
    )
    manifest = root / "tools/package-manifests/macos-arm64-py312.json"
    value = json.loads(manifest.read_text())
    value["wheels"].append(packaging["record"])
    value["vcs"] = [{"name": "refnx"}]
    manifest.write_text(json.dumps(value))
    for name in ("refnx-source.json", "refnx-build-macos-arm64-py312.json"):
        (root / "tools/package-manifests" / name).write_text("{}")
    refnx = make_wheel(tmp_path / "refnx/wheels", name="refnx")
    refnx_report = tmp_path / "refnx/build.json"
    refnx_report.write_text("{}")
    monkeypatch.setattr(module, "read_manifest", lambda root, path: json.loads(path.read_bytes()))
    monkeypatch.setattr(
        module,
        "_refnx_input",
        lambda *args: module.WheelInput(refnx["path"], refnx["record"], "refnx-source-build", True),
    )
    args = (root, manifest, wheels, pip["path"])
    return module, args, build / "build.json", refnx_report, receipt


def test_installed_macos_closure_requires_qt_build_evidence(load_tool_module, tmp_path, monkeypatch):
    module, args, qt, refnx, receipt = _loader(load_tool_module, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Qt|qt"):
        module.load_installation_inputs(*args, refnx_build=refnx)


def test_installed_closure_replaces_not_duplicates_upstream_qt(load_tool_module, tmp_path, monkeypatch):
    module, args, qt, refnx, receipt = _loader(load_tool_module, tmp_path, monkeypatch)
    manifest, inputs, pip, bindings, guard = module.load_installation_inputs(*args, refnx_build=refnx, qt_build=qt)
    selected = [item for item in inputs if item.record["name"] == "pyside6-essentials"]
    assert len(selected) == 1
    assert selected[0].record == receipt["wheel"]
    assert selected[0].provenance == "qt-cocoa-source-build" and selected[0].direct_url is True
    assert "qt-cocoa-build" in bindings and "qt-cocoa-inputs" in bindings
    guard()


@pytest.mark.parametrize("mutation", ["receipt", "wheel", "recipe", "builder", "original"])
def test_complete_installed_inspection_guards_qt_sources_and_outputs(load_tool_module, tmp_path, monkeypatch, mutation):
    module, args, qt, refnx, receipt = _loader(load_tool_module, tmp_path, monkeypatch)
    *_, guard = module.load_installation_inputs(*args, refnx_build=refnx, qt_build=qt)
    targets = {
        "receipt": qt,
        "wheel": qt.parent / "wheels" / receipt["wheel"]["filename"],
        "recipe": args[0] / "tools/qt-cocoa/ownership.patch",
        "builder": args[0] / "tools/qt_cocoa_build.py",
        "original": args[2] / receipt["inputs"]["upstream_wheel"]["filename"],
    }
    targets[mutation].write_bytes(b"changed during installed inspection")
    with pytest.raises(ValueError):
        guard()
