from __future__ import annotations

import hashlib
import stat
import zipfile
from pathlib import Path

import pytest
from tests.support.native_fixtures import macho, macho_command, pe_image, universal

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/wheel_inventory.py").is_file(), "missing actual wheel inventory"
    return load_tool_module("wheel_inventory")


def _wheel(tmp_path, metadata=None, entries=()):
    path = tmp_path / "sample-1.0-py3-none-any.whl"
    if metadata is None:
        metadata = (
            "Metadata-Version: 2.4\nName: sample\nVersion: 1.0\nLicense-Expression: MIT\nRequires-Dist: dep>=2\n\n"
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample-1.0.dist-info/METADATA", metadata)
        archive.writestr("sample-1.0.dist-info/licenses/LICENSE", "actual license bytes")
        archive.writestr("sample/__init__.py", "raise RuntimeError('must never import')\n")
        for name, content in entries:
            archive.writestr(name, content)
    return path, {"name": "sample", "version": "1.0", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_inventory_reads_real_metadata_and_hashes_files_without_importing(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    path, record = _wheel(tmp_path)
    result = module.inspect_wheel(path, record)
    assert result["requirements"] == ["dep>=2"]
    assert result["licenses"] == [{"expression": "MIT"}]
    files = {item["path"]: item for item in result["files"]}
    license_file = files["sample-1.0.dist-info/licenses/LICENSE"]
    assert license_file["sha256"] == hashlib.sha256(b"actual license bytes").hexdigest()
    assert license_file["kind"] == "license"
    assert files["sample/__init__.py"]["kind"] == "python"


@pytest.mark.parametrize("magic,kind", [(b"\x7fELF", "elf"), (b"\xcf\xfa\xed\xfe", "mach-o"), (b"MZ", "pe")])
def test_inventory_identifies_extensionless_native_bytes(load_tool_module, tmp_path, magic, kind):
    module = _module(load_tool_module)
    path, record = _wheel(tmp_path, entries=[("sample/native/library", magic + b"\0" * 128)])
    result = module.inspect_wheel(path, record)
    native = [item for item in result["files"] if item["kind"] == "native"]
    assert len(native) == 1
    assert native[0]["format"] == kind


@pytest.mark.parametrize("name", ["../outside", "/absolute", "sample/../outside", "sample\\outside"])
def test_inventory_rejects_unsafe_zip_names(load_tool_module, tmp_path, name):
    module = _module(load_tool_module)
    path, record = _wheel(tmp_path, entries=[(name, b"outside")])
    with pytest.raises(ValueError, match="wheel"):
        module.inspect_wheel(path, record)


def test_inventory_rejects_symlink_members(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    link = zipfile.ZipInfo("sample/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    path, record = _wheel(tmp_path, entries=[(link, b"outside")])
    with pytest.raises(ValueError, match="symlink"):
        module.inspect_wheel(path, record)


def test_inventory_rejects_metadata_identity_drift(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    path, record = _wheel(tmp_path, "Metadata-Version: 2.4\nName: other\nVersion: 1.0\n\n")
    with pytest.raises(ValueError, match="identity"):
        module.inspect_wheel(path, record)


def test_inventory_rejects_wrong_archive_hash_before_parsing(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    path, record = _wheel(tmp_path)
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes"):
        module.inspect_wheel(path, record)


def test_inventory_separates_vendored_metadata_from_top_level_distribution(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    metadata = b"Metadata-Version: 2.4\nName: embedded\nVersion: 2.0\nLicense-Expression: BSD-3-Clause\n\n"
    location = "sample/_vendor/embedded-2.0.dist-info/METADATA"
    path, record = _wheel(tmp_path, entries=[(location, metadata)])
    result = module.inspect_wheel(path, record)
    assert result["name"] == "sample"
    assert result["vendored"] == [
        {
            "name": "embedded",
            "version": "2.0",
            "requirements": [],
            "licenses": [{"expression": "BSD-3-Clause"}],
            "metadata_path": location,
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        }
    ]


def test_native_loader_inventory_binds_declarations_to_the_actual_member_bytes(load_tool_module, tmp_path):
    content = macho([macho_command(12, "@loader_path/libdependency.dylib")])
    path, record = _wheel(tmp_path, entries=[("sample/native/library", content)])
    result = _module(load_tool_module).inspect_wheel(path, record, native_loaders=True)
    native = next(item for item in result["files"] if item["kind"] == "native")
    image = native["native"]["images"][0]
    assert native["sha256"] == image["sha256"] == hashlib.sha256(content).hexdigest()
    assert image["imports"][0]["name"] == "@loader_path/libdependency.dylib"
    assert native["native"]["unparsed"] == []


@pytest.mark.parametrize("endian,wide", [("<", False), ("<", True), (">", False), (">", True)])
def test_extensionless_universal_formats_are_not_silently_classified_as_data(load_tool_module, tmp_path, endian, wide):
    content = universal([(0x100000C, macho())], endian=endian, wide=wide)
    path, record = _wheel(tmp_path, entries=[("sample/native/library", content)])
    result = _module(load_tool_module).inspect_wheel(path, record)
    file = next(item for item in result["files"] if item["path"] == "sample/native/library")
    assert file["kind"] == "native" and file["format"] == "mach-o-universal"


def test_native_loader_inventory_rejects_malformed_recognized_images(load_tool_module, tmp_path):
    path, record = _wheel(tmp_path, entries=[("sample/broken.dll", b"MZ")])
    with pytest.raises(ValueError, match="native|PE") as failure:
        _module(load_tool_module).inspect_wheel(path, record, native_loaders=True)
    assert "sample/broken.dll" in str(failure.value)


def test_unsupported_native_format_retains_its_incompleteness(load_tool_module, tmp_path):
    path, record = _wheel(tmp_path, entries=[("sample/library.so", b"\x7fELF" + b"\0" * 128)])
    result = _module(load_tool_module).inspect_wheel(path, record, native_loaders=True)
    file = next(item for item in result["files"] if item["kind"] == "native")
    assert file["format"] == "elf"
    assert file["native"]["images"] == [] and file["native"]["unparsed"]


def test_native_loader_buffer_has_an_explicit_size_bound(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    path, record = _wheel(tmp_path, entries=[("sample/library.dll", pe_image())])
    monkeypatch.setattr(module, "NATIVE_READ_LIMIT", 1024, raising=False)
    with pytest.raises(ValueError, match="native.*limit"):
        module.inspect_wheel(path, record, native_loaders=True)


def test_inventory_rejects_a_wheel_rewritten_then_restored_mid_read(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    path, record = _wheel(tmp_path)
    content = path.read_bytes()
    original = module._inventory_file

    def rewrite_and_restore(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_bytes(content + b"changed")
        path.write_bytes(content)
        return result

    monkeypatch.setattr(module, "_inventory_file", rewrite_and_restore)
    with pytest.raises(ValueError, match="changed|identity"):
        module.inspect_wheel(path, record)
