"""Bounded, non-executing readers for unsigned CPython 3.12 PyInstaller archives."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
import zlib
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from executable_toc import ITEM_LIMIT, TABLE_LIMIT, read_table  # noqa: E402
from native_bytes import NativeBytes  # noqa: E402

COOKIE = struct.Struct("!8sIIII64s")
TOC = struct.Struct("!IIIIBc")
MAGIC = b"MEI\014\013\012\013\016"
FILE_LIMIT = 512 * 1024**2
PAYLOAD_LIMIT = 128 * 1024**2
TOTAL_LIMIT = 2 * 1024**3


def digest(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def extraction_path(value: str) -> str:
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or path.as_posix() != value or ":" in value:
        raise ValueError("executable member has an unsafe or noncanonical path")
    _windows_path(value, path)
    return value


def _windows_path(value: str, path: PurePosixPath) -> None:
    if any(part in {".", ".."} or part.rstrip(". ") != part for part in path.parts):
        raise ValueError("executable member path is ambiguous on Windows")
    if any(ord(character) < 32 for character in value):
        raise ValueError("executable member path contains a control character")


def decompress(data: bytes | memoryview, expected: int | None = None) -> bytes:
    limit = PAYLOAD_LIMIT if expected is None else expected
    if not 0 <= limit <= PAYLOAD_LIMIT:
        raise ValueError("executable decompression size exceeds its limit")
    decoder = zlib.decompressobj()
    try:
        result = decoder.decompress(data, limit + 1)
    except zlib.error as error:
        raise ValueError("invalid executable compressed payload") from error
    _decompressed_stream(decoder, len(result), limit, expected)
    return result


def _decompressed_stream(decoder, size: int, limit: int, expected: int | None) -> None:
    if size > limit or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("executable decompression exceeds size or stream boundary")
    if expected is not None and size != expected:
        raise ValueError("executable decompressed size differs from its table")


def contiguous(entries: list[dict], start: int, end: int) -> None:
    position = start
    for entry in sorted((row for row in entries if row["stored_size"]), key=lambda row: row["offset"]):
        if entry["offset"] != position:
            raise ValueError("executable payload overlap or incomplete byte coverage")
        position += entry["stored_size"]
    if position != end:
        raise ValueError("executable payload byte coverage differs from its table")


def _name(data: memoryview) -> str:
    value, separator, tail = bytes(data).partition(b"\0")
    if not value or not separator or any(tail) or len(value) > 4096:
        raise ValueError("executable table name has invalid termination or padding")
    return value.decode("utf-8")


def _unique(entries: list[dict]) -> None:
    names = [entry["path"].casefold() for entry in entries if entry["kind"] != "o"]
    if len(set(names)) != len(names):
        raise ValueError("duplicate or ambiguous executable member path")


def _carchive_header(reader: NativeBytes, total: int) -> tuple[int, int, int, str]:
    magic, size, offset, length, python, library = COOKIE.unpack(reader.span(total - COOKIE.size, COOKIE.size))
    if magic != MAGIC or python != 312:
        raise ValueError("executable cookie or Python version differs from the pinned format")
    start = total - size
    if start < 2 or offset + length + COOKIE.size != size or length > TABLE_LIMIT:
        raise ValueError("executable cookie has invalid archive/table bounds")
    python_library = extraction_path(_name(memoryview(library)))
    if "/" in python_library:
        raise ValueError("executable Python library must be a filename")
    return start, start + offset, length, python_library


def _carchive_payload(kind: str, stored: int, size: int, compressed: int) -> None:
    if kind == "o" and (stored or size or compressed):
        raise ValueError("executable table payload size differs from its kind")
    if not compressed and stored != size:
        raise ValueError("executable table payload size differs from its kind")


class CArchive:
    def __init__(self, data: bytes | memoryview) -> None:
        if not COOKIE.size < len(data) <= FILE_LIMIT or bytes(data[:2]) != b"MZ":
            raise ValueError("executable is not a bounded PE CArchive")
        self.reader = NativeBytes(data)
        self.start, self.table, length, self.python_library = _carchive_header(self.reader, len(data))
        self.entries = self._entries(length)
        _unique(self.entries)
        contiguous(self.entries, self.start, self.table)
        if sum(row["size"] for row in self.entries) > TOTAL_LIMIT:
            raise ValueError("executable aggregate payload size exceeds its limit")
        self.framing = {
            "prefix": {"offset": 0, "size": self.start, "sha256": digest(self.reader.span(0, self.start))},
            "table": {"offset": self.table, "size": length, "sha256": digest(self.reader.span(self.table, length))},
            "cookie": {"offset": len(data) - COOKIE.size, "size": COOKIE.size, "sha256": digest(data[-COOKIE.size :])},
        }

    def _entries(self, length: int) -> list[dict]:
        table = NativeBytes(self.reader.span(self.table, length))
        position, entries = 0, []
        while position < length:
            size, offset, stored, unpacked, compressed, code = table.unpack(TOC.format, position)
            if size <= TOC.size or size % 16 or len(entries) >= ITEM_LIMIT:
                raise ValueError("invalid executable table entry length or count")
            name = _name(table.span(position + TOC.size, size - TOC.size))
            entries.append(self._entry(name, offset, stored, unpacked, compressed, code))
            position += size
        return entries

    def _entry(self, name, offset, stored, size, compressed, code) -> dict:
        kind = code.decode("ascii")
        if kind not in "bmMszZxo" or compressed not in (0, 1):
            raise ValueError("unsupported executable member type or compression flag")
        if size > PAYLOAD_LIMIT or self.start + offset + stored > self.table:
            raise ValueError("executable payload exceeds its size or archive boundary")
        _carchive_payload(kind, stored, size, compressed)
        return {
            "path": name if kind == "o" else extraction_path(name),
            "kind": kind,
            "offset": self.start + offset,
            "stored_size": stored,
            "size": size,
            "compressed": bool(compressed),
            "stored_sha256": digest(self.reader.span(self.start + offset, stored)),
        }

    def extract(self, entry: dict) -> bytes:
        data = self.reader.span(entry["offset"], entry["stored_size"])
        return decompress(data, entry["size"]) if entry["compressed"] else bytes(data)


def _pyz_values(row) -> tuple:
    if not isinstance(row, tuple) or len(row) != 2 or not isinstance(row[0], str):
        raise ValueError("PYZ module entry schema differs")
    name, values = row
    if not isinstance(values, tuple) or len(values) != 3 or any(type(value) is not int for value in values):
        raise ValueError("PYZ payload entry schema differs")
    return name, *values


def _pyz_bounds(kind: int, offset: int, size: int, boundary: int) -> None:
    if kind not in (0, 1, 3) or offset < 17 or size < 0 or offset + size > boundary:
        raise ValueError("PYZ payload exceeds its boundary or has an unsupported type")


def _pyz_entry(row, data: NativeBytes, boundary: int) -> dict:
    name, kind, offset, size = _pyz_values(row)
    _pyz_bounds(kind, offset, size, boundary)
    if not all(part.isidentifier() for part in name.split(".")) or (kind == 3 and size):
        raise ValueError("PYZ module path or namespace payload differs")
    return {
        "path": name,
        "kind": kind,
        "offset": offset,
        "stored_size": size,
        "stored_sha256": digest(data.span(offset, size)),
    }


class PyzArchive:
    def __init__(self, data: bytes | memoryview) -> None:
        self.reader = NativeBytes(data)
        header = self.reader.span(0, 17)
        if bytes(header[:8]) != b"PYZ\0" + importlib.util.MAGIC_NUMBER or any(header[12:]):
            raise ValueError("PYZ header or CPython bytecode version differs")
        boundary = self.reader.unpack("!I", 8)[0]
        table = read_table(self.reader.span(boundary, len(data) - boundary))
        self.entries = [_pyz_entry(row, self.reader, boundary) for row in table]
        _unique(self.entries)
        contiguous(self.entries, 17, boundary)
        self.framing = {
            "header_sha256": digest(header),
            "table_offset": boundary,
            "table_sha256": digest(data[boundary:]),
        }

    def extract(self, entry: dict) -> bytes:
        if entry["kind"] == 3:
            return b""
        return decompress(self.reader.span(entry["offset"], entry["stored_size"]))
