from __future__ import annotations

import hashlib
import struct

import pytest
from tests.support.native_fixtures import static_archive
from tests.unit.tools.test_native_inventory import _dependencies, _file, _properties, _report, _requests


def _short_import(*, machine=0x8664, kind=0, name_type=1, hint=17, symbol="PyInit_sample"):
    strings = symbol.encode() + b"\0sample.pyd\0"
    return struct.pack("<4H2I2H", 0, 0xFFFF, 0, machine, 0, len(strings), hint, kind | (name_type << 2)) + strings


@pytest.mark.parametrize("machine", [0x14C, 0x8664, 0xAA64])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_short_coff_import_retains_symbol_library_and_byte_identity(load_tool_module, machine, kind):
    content = _short_import(machine=machine, kind=kind)
    result = load_tool_module("native_binary").inspect_native(static_archive([("sample.pyd", content)]))
    assert result["unparsed"] == []
    image = result["images"][0]
    assert image["format"] == "coff-import"
    assert image["machine"] == machine
    assert image["sha256"] == hashlib.sha256(content).hexdigest()
    assert image["member_path"] == ["sample.pyd"]
    assert image["imports"] == [
        {"name": "sample.pyd", "kind": "static-import", "symbol": "PyInit_sample", "hint": 17, "import_type": kind}
    ]


@pytest.mark.parametrize(
    "name_type,symbol,expected",
    [
        (0, "named_symbol", {"ordinal": 17}),
        (2, "_symbol@8", {"symbol": "symbol@8", "hint": 17}),
        (3, "_symbol@8", {"symbol": "symbol", "hint": 17}),
    ],
)
def test_short_import_name_mapping_preserves_ordinal_and_decoration(load_tool_module, name_type, symbol, expected):
    result = load_tool_module("native_binary").inspect_native(_short_import(name_type=name_type, symbol=symbol))
    request = result["images"][0]["imports"][0]
    assert request == {"name": "sample.pyd", "kind": "static-import", "import_type": 0, **expected}


@pytest.mark.parametrize("mutation", ["truncated", "size", "unterminated", "reserved", "type", "name-type", "tail"])
def test_short_import_rejects_malformed_payloads(load_tool_module, mutation):
    content = bytearray(_short_import())
    if mutation == "truncated":
        content = content[:12]
    elif mutation == "size":
        struct.pack_into("<I", content, 12, len(content))
    elif mutation == "unterminated":
        content[-1] = ord("x")
    elif mutation == "tail":
        content += b"unclaimed"
    else:
        struct.pack_into("<H", content, 18, {"reserved": 32, "type": 7, "name-type": 28}[mutation])
    with pytest.raises(ValueError):
        load_tool_module("native_binary").inspect_native(bytes(content))


def test_coff_import_is_static_evidence_and_never_a_runtime_dependency(load_tool_module):
    content = static_archive([("sample.pyd", _short_import())])
    report = _report(load_tool_module, [("sample", "sample/import.lib", content)], target="windows-x64-py312")
    image = _file(report, "sample/import.lib")["components"][0]
    assert _properties(image)["xrr:native:target-family"] == "target"
    assert _requests(image)[0]["reason"] == "static-archive-link-context"
    assert _dependencies(report, image) == []
    assert report["compositions"] == [{"aggregate": "incomplete"}]


def test_anonymous_coff_object_is_not_misread_as_short_import(load_tool_module):
    content = bytearray(_short_import())
    struct.pack_into("<H", content, 4, 2)
    result = load_tool_module("native_binary").inspect_native(bytes(content))
    assert result["images"] == []
    assert result["unparsed"]


def test_short_import_export_alias_uses_the_explicit_link_name(load_tool_module):
    content = bytearray(_short_import(name_type=4))
    content.extend(b"exported_symbol\0")
    struct.pack_into("<I", content, 12, len(content) - 20)

    result = load_tool_module("native_binary").inspect_native(bytes(content))

    assert result["unparsed"] == []
    assert result["images"][0]["imports"] == [
        {"name": "sample.pyd", "kind": "static-import", "import_type": 0, "symbol": "exported_symbol", "hint": 17}
    ]
