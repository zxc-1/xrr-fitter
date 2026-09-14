from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from tests.support.qt_cocoa_fixtures import INFO, PLUGIN, _plugin, _records, _upstream, _write_zip

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/qt_cocoa_wheel.py").is_file(), "missing auditable Qt wheel derivation"
    return load_tool_module("qt_cocoa_wheel")


def _case(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    source, record, original = _upstream(tmp_path)
    plugin = tmp_path / "libqcocoa.dylib"
    plugin.write_bytes(_plugin())
    additions = {f"{INFO}/licenses/xrr-qt-cocoa/NOTICE.txt": b"Patched Qt Cocoa sources and build receipt.\n"}
    return module, source, record, original, plugin, additions


def test_derivation_preserves_original_bytes_and_licenses_with_arm64_build_identity(load_tool_module, tmp_path):
    module, source, record, original, plugin, additions = _case(load_tool_module, tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    derived = module.derive_wheel(source, record, plugin, output, additions)
    assert derived["filename"] == "pyside6_essentials-6.11.2-1xrrcocoa-cp310-abi3-macosx_13_0_arm64.whl"
    path = output / derived["filename"]
    evidence = module.verify_derivation(source, record, path, derived, additions)
    assert evidence["plugin_sha256"] == hashlib.sha256(plugin.read_bytes()).hexdigest()
    assert evidence["unchanged_members"] == len(original) - 3
    with zipfile.ZipFile(path) as archive:
        for name in set(original) - {PLUGIN, f"{INFO}/WHEEL", f"{INFO}/RECORD"}:
            assert archive.read(name) == original[name]
        assert b"Build: 1xrrcocoa" in archive.read(f"{INFO}/WHEEL")
        assert b"macosx_13_0_arm64" in archive.read(f"{INFO}/WHEEL")
        assert "xrr-qt-cocoa" in next(name for name in additions)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record["sha256"]


def test_wheel_derivation_is_deterministic_for_identical_inputs(load_tool_module, tmp_path):
    module, source, record, _, plugin, additions = _case(load_tool_module, tmp_path)
    results = []
    for name in ("first", "second"):
        output = tmp_path / name
        output.mkdir()
        results.append(module.derive_wheel(source, record, plugin, output, additions))
    assert results[0] == results[1]


@pytest.mark.parametrize("case", ["x86", "new_os", "absolute_rpath", "not_macho"])
def test_wrong_native_target_cannot_be_shipped_as_arm64_macos13(load_tool_module, tmp_path, case):
    module, source, record, _, plugin, additions = _case(load_tool_module, tmp_path)
    choices = {
        "x86": _plugin(cpu=0x1000007),
        "new_os": _plugin(minos=15 << 16),
        "absolute_rpath": _plugin(rpath="/tmp/sdk/lib"),
        "not_macho": b"invalid",
    }
    plugin.write_bytes(choices[case])
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError):
        module.derive_wheel(source, record, plugin, output, additions)
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    "changed", ["PySide6/module.py", f"{INFO}/METADATA", f"{INFO}/RECORD", "PySide6/unexpected.py"]
)
def test_derivation_audit_rejects_tampering_even_with_a_new_archive_hash(load_tool_module, tmp_path, changed):
    module, source, record, _, plugin, additions = _case(load_tool_module, tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    derived = module.derive_wheel(source, record, plugin, output, additions)
    path = output / derived["filename"]
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files[changed] = b"tampered\n"
    _write_zip(path, files)
    derived.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
    with pytest.raises(ValueError):
        module.verify_derivation(source, record, path, derived, additions)


def test_additions_cannot_overwrite_runtime_members(load_tool_module, tmp_path):
    module, source, record, _, plugin, _ = _case(load_tool_module, tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError):
        module.derive_wheel(source, record, plugin, output, {"PySide6/module.py": b"backdoor"})
    assert not list(output.iterdir())


def test_large_unchanged_qt_modules_are_streamed_without_relaxing_read_limits(load_tool_module, tmp_path):
    module, source, record, original, plugin, additions = _case(load_tool_module, tmp_path)
    name = "PySide6/QtWidgets.abi3.so"
    content = b"unchanged vendor bytes" * 900_000
    assert len(content) > 16 * 1024**2
    original[name] = content
    original.pop(f"{INFO}/RECORD")
    original[f"{INFO}/RECORD"] = _records(original)
    _write_zip(source, original)
    record["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "out"
    output.mkdir()
    derived = module.derive_wheel(source, record, plugin, output, additions)
    with zipfile.ZipFile(output / derived["filename"]) as archive, archive.open(name) as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == hashlib.sha256(content).hexdigest()
    assert (
        module.verify_derivation(source, record, output / derived["filename"], derived, additions)["unchanged_members"]
        == len(original) - 3
    )
