from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/qt_cocoa_build.py").is_file(), "missing production Qt Cocoa source builder"
    return load_tool_module("qt_cocoa_build")


def test_build_commands_pin_architecture_version_and_relative_rpath(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    commands = module.compile_commands(tmp_path / "sdk", tmp_path / "stage")
    assert [label for label, _, _ in commands] == ["qmake", "make", "signature", "regression-qmake", "regression-make"]
    qmake = commands[0][1]
    assert qmake[0] == str(tmp_path / "sdk/bin/qmake")
    assert "CONFIG+=no_qt_rpath no_default_rpath" in qmake
    assert "QMAKE_RPATHDIR=@loader_path/../../lib" in qmake
    assert commands[1][1] == ("/usr/bin/make", "-j4")
    assert commands[2][1][:2] == ("/usr/bin/codesign", "--verify")


@pytest.mark.parametrize("name", ["../outside", "/outside", "lib/../../outside", "lib\\escape"])
def test_sdk_member_validation_rejects_escape_before_extraction(load_tool_module, name):
    module = _module(load_tool_module)
    with pytest.raises(ValueError):
        module.sdk_member_names(name + "\n")


def test_sdk_member_validation_accepts_only_canonical_relative_paths(load_tool_module):
    module = _module(load_tool_module)
    assert module.sdk_member_names("bin/qmake\nlib/QtCore.framework/Versions/A/QtCore\ninclude/\n") == [
        "bin/qmake",
        "lib/QtCore.framework/Versions/A/QtCore",
        "include",
    ]


def test_native_regression_verifies_the_actually_loaded_plugin(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    plugin = tmp_path / "build/platforms/libqcocoa.dylib"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b"fixture")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return f"PLUGIN: {plugin}\nPASS: {command[-1]}"

    monkeypatch.setattr(module, "run_checked", run)
    result = module.run_regressions(tmp_path, tmp_path, tmp_path, {})
    assert len(result) == len(calls) == 6
    assert set(result.values()) == {0}
    assert all(item[1]["environment"]["QT_PLUGIN_PATH"] == str(tmp_path / "build") for item in calls)
    monkeypatch.setattr(
        module, "run_checked", lambda *args, **kwargs: "PLUGIN: /unpatched/libqcocoa.dylib\nPASS: borrowed-destructor"
    )
    with pytest.raises(ValueError, match="plugin"):
        module.run_regressions(tmp_path, tmp_path, tmp_path, {})


def test_native_regression_failure_never_becomes_pass(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(19, ["regression"])

    monkeypatch.setattr(module, "run_checked", fail)
    with pytest.raises(subprocess.CalledProcessError) as result:
        module.run_regressions(tmp_path, tmp_path, tmp_path, {})
    assert result.value.returncode == 19


def test_sdk_extraction_checks_names_before_any_writes(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return "../outside\n"

    monkeypatch.setattr(module, "run_checked", run)
    with pytest.raises(ValueError):
        module.extract_sdk(tmp_path, tmp_path / "input.7z", tmp_path / "sdk", tmp_path, {})
    assert len(calls) == 1 and "-tf" in calls[0]
    assert not (tmp_path / "sdk").exists()


def test_sdk_extraction_refuses_a_different_qt_version(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "-tf" in command:
            return "bin/qmake\n"
        if "-query" in command:
            return "6.11.1"
        return ""

    monkeypatch.setattr(module, "run_checked", run)
    with pytest.raises(ValueError, match="version"):
        module.extract_sdk(tmp_path, tmp_path / "input.7z", tmp_path / "sdk", tmp_path, {})
    assert not any("make" == Path(command[0]).name for command in commands)


def _build_driver(module, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    report = tmp_path / "report"
    monkeypatch.setattr(module, "require_install_target", lambda target: None)
    monkeypatch.setattr(module, "read_inputs", lambda root: {"sources": []})
    monkeypatch.setattr(module, "verify_inputs", lambda *args: None)
    monkeypatch.setattr(module, "code_bindings", lambda root: {"builder": "unchanged"})
    return root, report


def test_failed_build_has_no_success_receipt_and_removes_owned_stage(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    root, report = _build_driver(module, tmp_path, monkeypatch)
    stages = []

    def fail(root, inputs, wheels, report, stage, manifest, environment):
        stages.append(stage)
        (stage / "temporary").write_bytes(b"build")
        raise subprocess.CalledProcessError(9, ["make"])

    monkeypatch.setattr(module, "_build_stage", fail)
    assert module.build_cocoa(root, tmp_path / "inputs", tmp_path / "wheels", report) == 9
    assert stages and not stages[0].exists()
    assert not (report / "build.json").exists()
    assert json.loads((report / "summary.json").read_text())["state"] == "FAIL"


def test_changed_source_inputs_cannot_publish_success(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    root, report = _build_driver(module, tmp_path, monkeypatch)
    calls = []

    def verify(*args):
        calls.append(args)
        if len(calls) > 1:
            raise ValueError("source changed")

    monkeypatch.setattr(module, "verify_inputs", verify)
    monkeypatch.setattr(module, "_build_stage", lambda *args: {"state": "PASS"})
    with pytest.raises(ValueError, match="source changed"):
        module.build_cocoa(root, tmp_path / "inputs", tmp_path / "wheels", report)
    assert not (report / "build.json").exists()


def test_changed_builder_cannot_publish_success(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    root, report = _build_driver(module, tmp_path, monkeypatch)
    identities = iter(({"builder": "original"}, {"builder": "changed"}))
    monkeypatch.setattr(module, "code_bindings", lambda root: next(identities))
    monkeypatch.setattr(module, "_build_stage", lambda *args: {"state": "PASS"})
    with pytest.raises(ValueError, match="builder"):
        module.build_cocoa(root, tmp_path / "inputs", tmp_path / "wheels", report)
    assert not (report / "build.json").exists()


def test_successful_build_publishes_receipt_after_final_input_guard(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    root, report = _build_driver(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_build_stage", lambda *args: {"state": "PASS", "offline": True})
    assert module.build_cocoa(root, tmp_path / "inputs", tmp_path / "wheels", report) == 0
    assert json.loads((report / "build.json").read_text())["offline"] is True


def test_build_environment_does_not_inherit_runtime_or_compiler_overrides(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    inherited = {
        "QT_PLUGIN_PATH": "/wrong",
        "QT_QPA_PLATFORM": "offscreen",
        "DYLD_LIBRARY_PATH": "/wrong",
        "QMAKEFEATURES": "/wrong",
        "CXXFLAGS": "wrong",
        "PYTHONPATH": "/wrong",
        "HOME": "/wrong",
        "PATH": "/wrong",
    }
    environment = module.build_environment(
        inherited, tmp_path, {"cc": "/usr/bin/clang", "cxx": "/usr/bin/clang++", "sdk_path": "/sdk"}
    )
    assert not set(inherited).difference({"HOME", "PATH"}).intersection(environment)
    assert environment["HOME"] == str(tmp_path / "home")


def test_resource_collection_name_matches_qt_cocoa_init_symbol():
    recipe = (ROOT / "tools/qt-cocoa/cocoa.pro").read_text()
    assert "RESOURCES += qcocoaresources.qrc" in recipe
    assert (ROOT / "tools/qt-cocoa/qcocoaresources.qrc").is_file()
