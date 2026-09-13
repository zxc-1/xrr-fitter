from __future__ import annotations

import hashlib
import marshal
import struct

import pytest
from tests.support.executable_archives import alter_cookie, alter_toc, carchive, pyz


def test_carchive_inventories_encoded_and_actual_bytes(load_tool_module):
    module = load_tool_module("executable_archive")
    content = carchive([("pkg\\data.bin", b"actual data", "x", True), ("option value", b"", "o", False)])
    archive = module.CArchive(content)
    assert archive.python_library == "python312.dll"
    assert [row["path"] for row in archive.entries] == ["pkg/data.bin", "option value"]
    assert archive.extract(archive.entries[0]) == b"actual data"
    assert (
        archive.entries[0]["stored_sha256"]
        == hashlib.sha256(content[archive.entries[0]["offset"] :][: archive.entries[0]["stored_size"]]).hexdigest()
    )


@pytest.mark.parametrize("field,value", [(1, 2**32 - 1), (2, 2**32 - 1), (3, 2**32 - 1), (4, 311)])
def test_carchive_rejects_bad_cookie_bounds_or_python(load_tool_module, field, value):
    module = load_tool_module("executable_archive")
    content = carchive([("data", b"value", "x", False)])
    with pytest.raises(ValueError):
        module.CArchive(alter_cookie(content, field, value))


@pytest.mark.parametrize("field,value", [(0, 1), (1, 2**32 - 1), (2, 2**32 - 1), (3, 2**32 - 1), (4, 2), (5, b"!")])
def test_carchive_rejects_unsafe_toc_fields(load_tool_module, field, value):
    module = load_tool_module("executable_archive")
    content = carchive([("data", b"value", "x", False)])
    with pytest.raises(ValueError):
        archive = module.CArchive(alter_toc(content, field, value))
        archive.extract(archive.entries[0])


@pytest.mark.parametrize(
    "name", ["../escape", r"C:\escape", r"\absolute", "a//b", "a/./b", "a/../b", "a:stream", "a. "]
)
def test_carchive_rejects_noncanonical_extraction_paths(load_tool_module, name):
    module = load_tool_module("executable_archive")
    with pytest.raises(ValueError, match="path"):
        module.CArchive(carchive([(name, b"value", "x", False)]))


@pytest.mark.parametrize("second", ["data", "DATA", r"folder\data"])
def test_carchive_rejects_duplicate_windows_paths(load_tool_module, second):
    module = load_tool_module("executable_archive")
    first = "folder/data" if "folder" in second else "data"
    with pytest.raises(ValueError, match="duplicate|ambiguous"):
        module.CArchive(carchive([(first, b"a", "x", False), (second, b"b", "x", False)]))


def test_carchive_rejects_uninventoried_trailer_and_overlapping_payloads(load_tool_module):
    module = load_tool_module("executable_archive")
    content = carchive([("first", b"a", "x", False), ("second", b"b", "x", False)])
    with pytest.raises(ValueError):
        module.CArchive(content + b"hidden trailer")
    with pytest.raises(ValueError, match="overlap|coverage"):
        module.CArchive(alter_toc(content, 1, 1))


def test_carchive_bounded_zlib_requires_exact_declared_size(load_tool_module):
    module = load_tool_module("executable_archive")
    content = alter_toc(carchive([("data", b"a" * 10000, "x", True)]), 3, 1)
    archive = module.CArchive(content)
    with pytest.raises(ValueError, match="decompress|size"):
        archive.extract(archive.entries[0])


def test_pyz_inventories_payloads_without_loading_or_executing_code(load_tool_module, monkeypatch):
    module = load_tool_module("executable_archive")
    code = marshal.dumps(compile("raise RuntimeError('never execute')", "sample.py", "exec"))
    data = pyz([("sample", code, 0), ("namespace", b"", 3)])
    monkeypatch.setattr(marshal, "loads", lambda *args: pytest.fail("untrusted marshal.loads is forbidden"))
    monkeypatch.setattr(marshal, "load", lambda *args: pytest.fail("untrusted marshal.load is forbidden"))
    archive = module.PyzArchive(data)
    assert archive.extract(archive.entries[0]) == code
    assert archive.extract(archive.entries[1]) == b""


def test_pyz_rejects_duplicate_modules_and_trailing_toc_bytes(load_tool_module):
    module = load_tool_module("executable_archive")
    with pytest.raises(ValueError, match="duplicate"):
        module.PyzArchive(pyz([("sample", b"one", 0), ("sample", b"two", 0)]))
    with pytest.raises(ValueError, match="trailing|boundary"):
        module.PyzArchive(pyz([("sample", b"one", 0)]) + b"extra")


@pytest.mark.parametrize("toc", [b"[\xff\xff\xff\x7f", b"r\x00\x00\x00\x00", b"c", b"N", b"[\x01\x00\x00\x00" * 12])
def test_pyz_toc_rejects_unbounded_or_executable_marshal_types(load_tool_module, toc):
    module = load_tool_module("executable_archive")
    content = bytearray(pyz([]))
    offset = struct.unpack("!I", content[8:12])[0]
    with pytest.raises(ValueError):
        module.PyzArchive(bytes(content[:offset]) + toc)
