from __future__ import annotations

import importlib.util
import marshal
import struct
import zlib

COOKIE = struct.Struct("!8sIIII64s")
TOC = struct.Struct("!IIIIBc")
MAGIC = b"MEI\014\013\012\013\016"


def carchive(entries, *, prefix=b"MZ synthetic bootloader", python=312):
    payload, table = bytearray(), bytearray()
    for name, data, kind, compressed in entries:
        raw = zlib.compress(data) if compressed else data
        name_bytes = name.encode() + b"\0"
        size = (TOC.size + len(name_bytes) + 15) // 16 * 16
        table.extend(TOC.pack(size, len(payload), len(raw), len(data), compressed, kind.encode()))
        table.extend(name_bytes.ljust(size - TOC.size, b"\0"))
        payload.extend(raw)
    cookie = COOKIE.pack(
        MAGIC, len(payload) + len(table) + COOKIE.size, len(payload), len(table), python, b"python312.dll"
    )
    return prefix + payload + table + cookie


def pyz(entries):
    payload, table = bytearray(b"\0" * 17), []
    for name, data, kind in entries:
        compressed = zlib.compress(data) if kind != 3 else b""
        table.append((name, (kind, len(payload), len(compressed))))
        payload.extend(compressed)
    offset = len(payload)
    payload.extend(marshal.dumps(table))
    payload[:12] = b"PYZ\0" + importlib.util.MAGIC_NUMBER + struct.pack("!I", offset)
    return bytes(payload)


def alter_cookie(data, field, value):
    values = list(COOKIE.unpack(data[-COOKIE.size :]))
    values[field] = value
    return data[: -COOKIE.size] + COOKIE.pack(*values)


def alter_toc(data, field, value):
    cookie = COOKIE.unpack(data[-COOKIE.size :])
    start = len(data) - cookie[1] + cookie[2]
    values = list(TOC.unpack_from(data, start))
    values[field] = value
    return data[:start] + TOC.pack(*values) + data[start + TOC.size :]
