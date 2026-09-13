from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from tests.support.native_fixtures import (
    fat_archive,
    fat_macho,
    macho,
    macho_command,
    pe_image,
    static_archive,
    universal,
)

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/native_binary.py").is_file(), "missing byte-derived native loader inspection"
    return load_tool_module("native_binary")


def test_macho_records_loader_kinds_rpaths_and_install_identity(load_tool_module):
    module = _module(load_tool_module)
    commands = [
        macho_command(12, "@rpath/libsample.dylib"),
        macho_command(0x80000018, "/usr/lib/libweak.dylib"),
        macho_command(0x8000001F, "libreexport.dylib"),
        macho_command(0x80000023, "libupward.dylib"),
        macho_command(0x20, "liblazy.dylib"),
        macho_command(13, "@rpath/owner.dylib"),
        macho_command(0x8000001C, "@loader_path/..", dylib=False),
        macho_command(14, "/usr/lib/dyld", dylib=False),
    ]
    result = module.inspect_native(macho(commands))
    assert result["format"] == "mach-o" and result["unparsed"] == []
    image = result["images"][0]
    assert image["cpu_type"] == 0x100000C and image["file_type"] == 6
    assert image["install_name"] == "@rpath/owner.dylib"
    assert image["rpaths"] == ["@loader_path/.."]
    assert {(item["name"], item["kind"]) for item in image["imports"]} == {
        ("@rpath/libsample.dylib", "load"),
        ("/usr/lib/libweak.dylib", "weak"),
        ("libreexport.dylib", "reexport"),
        ("libupward.dylib", "upward"),
        ("liblazy.dylib", "lazy"),
        ("/usr/lib/dyld", "dylinker"),
    }
    linked = next(item for item in image["imports"] if item["kind"] == "load")
    assert linked["current_version"] == "1.2.3" and linked["compatibility_version"] == "1.0.0"


def test_universal_macho_inspects_every_architecture_not_just_the_host(load_tool_module):
    result = _module(load_tool_module).inspect_native(fat_macho())
    assert result["format"] == "mach-o-universal"
    assert [(item["cpu_type"], item["imports"][0]["name"]) for item in result["images"]] == [
        (0x1000007, "x86.dylib"),
        (0x100000C, "arm.dylib"),
    ]


def test_universal_static_archives_inspect_each_architecture_and_member(load_tool_module):
    result = _module(load_tool_module).inspect_native(fat_archive())
    assert result["unparsed"] == []
    assert [(image["cpu_type"], image["member"], image["file_type"]) for image in result["images"]] == [
        (0x1000007, "object.o", 1),
        (0x100000C, "object.o", 1),
    ]


def test_universal_non_native_payload_is_a_controlled_parse_failure(load_tool_module):
    content = bytearray(fat_macho())
    content[0x100:0x104] = b"bad!"
    with pytest.raises(ValueError, match="native|Mach-O"):
        _module(load_tool_module).inspect_native(bytes(content))


def test_nested_static_archive_depth_is_bounded(load_tool_module):
    content = macho()
    for _index in range(20):
        content = static_archive([("nested.a", content)])
    with pytest.raises(ValueError, match="nesting"):
        _module(load_tool_module).inspect_native(content)


def test_nested_archive_keeps_leaf_bytes_and_complete_member_path(load_tool_module):
    leaf = macho(file_type=1)
    content = static_archive([("nested.a", static_archive([("object.o", leaf)]))])
    image = _module(load_tool_module).inspect_native(content)["images"][0]
    assert image["member"] == "object.o"
    assert image["member_path"] == ["nested.a", "object.o"]
    assert image["sha256"] == hashlib.sha256(leaf).hexdigest()
    assert image["size"] == len(leaf)


def test_universal_rejects_duplicate_architectures(load_tool_module):
    content = universal([(0x100000C, macho()), (0x100000C, macho())])
    with pytest.raises(ValueError, match="duplicate.*architecture"):
        _module(load_tool_module).inspect_native(content)


@pytest.mark.parametrize("kind", [6, 0x10, 0x27, 0x2D, 0x80000035, 0x8000FFFF])
def test_unhandled_macho_loader_commands_remain_explicitly_unparsed(load_tool_module, kind):
    command = struct.pack("<2I", kind, 8)
    result = _module(load_tool_module).inspect_native(macho([command]))
    assert result["unparsed"], "unhandled loader commands cannot become an empty successful dependency list"
    assert result["unparsed"][0]["command"] == kind
    assert result["unparsed"][0]["sha256"] == hashlib.sha256(command).hexdigest()


@pytest.mark.parametrize("endian,wide", [("<", False), ("<", True), (">", False), (">", True)])
def test_macho_byte_order_and_word_size_are_derived_from_magic(load_tool_module, endian, wide):
    content = macho([macho_command(12, "ordered.dylib", endian=endian)], endian=endian, wide=wide)
    image = _module(load_tool_module).inspect_native(content)["images"][0]
    assert image["imports"][0]["name"] == "ordered.dylib"
    assert image["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("endian,wide", [("<", False), ("<", True), (">", False), (">", True)])
def test_universal_byte_order_and_64bit_offsets(load_tool_module, endian, wide):
    content = universal([(0x100000C, macho())], endian=endian, wide=wide)
    result = _module(load_tool_module).inspect_native(content)
    assert result["format"] == "mach-o-universal"
    assert len(result["images"]) == 1 and result["images"][0]["cpu_type"] == 0x100000C


@pytest.mark.parametrize(
    "mutation", ["truncated", "command-size", "name-offset", "unterminated", "fat-overlap", "fat-cpu"]
)
def test_malformed_macho_fails_instead_of_claiming_no_dependencies(load_tool_module, mutation):
    content = bytearray(macho([macho_command(12, "library.dylib")]))
    if mutation == "truncated":
        content = content[:12]
    elif mutation == "command-size":
        struct.pack_into("<I", content, 36, 4)
    elif mutation == "name-offset":
        struct.pack_into("<I", content, 40, 500)
    elif mutation == "unterminated":
        content[56:] = b"x" * (len(content) - 56)
    else:
        content = bytearray(fat_macho())
        struct.pack_into(
            ">I", content, 36 if mutation == "fat-overlap" else 8, 0x100 if mutation == "fat-overlap" else 7
        )
    with pytest.raises(ValueError):
        _module(load_tool_module).inspect_native(bytes(content))


def test_pe_includes_normal_delay_imports_and_export_forwarders(load_tool_module):
    result = _module(load_tool_module).inspect_native(pe_image())
    assert result["format"] == "pe" and result["unparsed"] == []
    image = result["images"][0]
    assert image["machine"] == 0x8664
    assert {(item["name"], item["kind"]) for item in image["imports"]} == {
        ("KERNEL32.dll", "load"),
        ("USER32.dll", "delay-load"),
        ("python312.dll", "forwarder"),
    }
    assert image["forwarded_exports"] == ["python312.PyObject_Call"]


def test_pe_legacy_delay_load_virtual_addresses(load_tool_module):
    image = _module(load_tool_module).inspect_native(pe_image(plus=False, delay_rva=False))["images"][0]
    assert image["machine"] == 0x14C
    assert {"name": "USER32.dll", "kind": "delay-load"} in image["imports"]


def test_pe_export_forwarder_cannot_read_past_its_directory(load_tool_module):
    content = bytearray(pe_image())
    content[0x360:0x400] = b"x" * 0xA0
    content[0x400:0x406] = b".name\0"
    with pytest.raises(ValueError, match="forwarder|unterminated"):
        _module(load_tool_module).inspect_native(bytes(content))


@pytest.mark.parametrize(
    "mutation", ["signature", "optional", "section", "unbacked-rva", "unterminated", "export-table"]
)
def test_pe_rejects_unmapped_or_malformed_loader_tables(load_tool_module, mutation):
    content = bytearray(pe_image())
    offsets = {"optional": 0x94, "section": 0x188 + 20, "unbacked-rva": 0x220 + 12, "export-table": 0x300 + 28}
    if mutation == "signature":
        content[0x80:0x84] = b"bad!"
    elif mutation == "unterminated":
        content[0x500:] = b"x" * 0x100
    elif mutation == "optional":
        struct.pack_into("<H", content, offsets[mutation], 2)
    else:
        struct.pack_into("<I", content, offsets[mutation], 0x7000)
    with pytest.raises(ValueError):
        _module(load_tool_module).inspect_native(bytes(content))


def test_static_archive_inspects_objects_and_preserves_unknown_members(load_tool_module):
    content = static_archive([("object.o", macho(file_type=1)), ("unknown.o", b"opaque native object")])
    result = _module(load_tool_module).inspect_native(content)
    assert result["format"] == "ar" and result["images"][0]["file_type"] == 1
    assert result["images"][0]["member"] == "object.o"
    assert result["unparsed"][0]["member"] == "unknown.o"
    assert result["unparsed"][0]["reason"] == "unrecognized native header"


@pytest.mark.parametrize("content", [b"!<arch>\nshort", b"MZ", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"])
def test_recognized_but_truncated_native_formats_fail_closed(load_tool_module, content):
    with pytest.raises(ValueError):
        _module(load_tool_module).inspect_native(content)


def test_unknown_native_header_is_explicitly_unparsed(load_tool_module):
    result = _module(load_tool_module).inspect_native(b"not a recognized native image")
    assert result["images"] == [] and result["unparsed"] == [{"reason": "unrecognized native header"}]


@pytest.mark.parametrize("terminator", [b"/\n", b"\0"])
def test_archive_long_names_support_gnu_and_microsoft_terminators(load_tool_module, terminator):
    name = "a-long-native-object-name.o"
    names = b"another-object.o" + terminator + name.encode() + terminator
    offset = len(b"another-object.o" + terminator)
    content = static_archive([("//", names), (f"/{offset}", macho(file_type=1))], raw_names=True)
    image = _module(load_tool_module).inspect_native(content)["images"][0]
    assert image["member"] == name


def test_archive_long_name_reference_cannot_start_inside_a_name(load_tool_module):
    content = static_archive([("//", b"library-object.o/\n"), ("/2", macho(file_type=1))], raw_names=True)
    with pytest.raises(ValueError, match="long-name"):
        _module(load_tool_module).inspect_native(content)


def test_duplicate_unknown_archive_members_retain_distinct_byte_identities(load_tool_module):
    members = [("object.o", b"unknown first"), ("object.o", b"unknown second")]
    result = _module(load_tool_module).inspect_native(static_archive(members))
    assert [item["member_indices"] for item in result["unparsed"]] == [[0], [1]]
    assert [item["member_sha256"] for item in result["unparsed"]] == [
        hashlib.sha256(content).hexdigest() for _name, content in members
    ]


def test_universal_archive_slice_inventory_keeps_container_bytes(load_tool_module):
    content = fat_archive()
    result = _module(load_tool_module).inspect_native(content)
    slices = result["slices"]
    assert [item["format"] for item in slices] == ["ar", "ar"]
    for item in slices:
        assert item["sha256"] == hashlib.sha256(content[item["offset"] : item["offset"] + item["size"]]).hexdigest()
        assert item["members"][0]["name"] == "object.o"


def test_symbol_table_prefix_does_not_hide_an_ordinary_archive_member(load_tool_module):
    content = static_archive([("__.SYMDEFevil.o", macho(file_type=1))])
    result = _module(load_tool_module).inspect_native(content)
    assert len(result["images"]) == 1
    assert result["images"][0]["member"] == "__.SYMDEFevil.o"
