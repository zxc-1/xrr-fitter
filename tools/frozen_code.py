"""Fingerprint every CPython 3.12 code field without constructing or executing code.

Marshal interning and reference flags depend on the writer's live object graph.
This bounded grammar retains all code/constant fields, including debug data and
floating-point bits, but gives equivalent valid sharing layouts the same digest.
The caller separately retains the complete, unmodified wire-byte SHA-256.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from executable_toc import ITEM_LIMIT, TABLE_LIMIT, TableReader  # noqa: E402


def _node(kind: str, content: bytes = b"") -> tuple[str, bytes]:
    return kind, hashlib.sha256(kind.encode() + b"\0" + len(content).to_bytes(8, "little") + content).digest()


def _integer(value: int) -> tuple[str, bytes]:
    magnitude = abs(value)
    return _node("int", bytes([value < 0]) + magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "little"))


class CodeReader(TableReader):
    depth_limit = 64

    def _long(self) -> tuple[str, bytes]:
        count = self.integer()
        if abs(count) > 4096:
            raise ValueError("code integer exceeds its bounded constant size")
        value = 0
        for index in range(abs(count)):
            digit = int.from_bytes(self.read(2), "little")
            if digit >= 32768:
                raise ValueError("code integer has invalid marshal digits")
            value |= digit << (15 * index)
        return _integer(-value if count < 0 else value)

    def _string(self, kind: str) -> tuple[str, bytes]:
        data = self.read(self.size(kind in "zZ", TABLE_LIMIT))
        if kind == "s":
            return _node("bytes", data)
        text = data.decode("utf-8" if kind in "ut" else "ascii")
        return _node("str", text.encode("utf-8"))

    def _sequence(self, kind: str, depth: int) -> tuple[str, bytes]:
        length = self.size(kind == ")", ITEM_LIMIT)
        values = [self.value(depth + 1)[1] for _ in range(length)]
        if kind == ">":
            if len(set(values)) != len(values):
                raise ValueError("code frozenset contains duplicate constants")
            values.sort()
        return _node("frozenset" if kind == ">" else "tuple", b"".join(values))

    def _code(self, depth: int) -> tuple[str, bytes]:
        content = self.read(20)
        expected = ("bytes", "tuple", "tuple", "tuple", "bytes", "str", "str", "str")
        for kind in expected:
            value = self.value(depth + 1)
            if value[0] != kind:
                raise ValueError("CPython code field type differs from its marshal schema")
            content += value[1]
        content += self.read(4)
        for _ in range(2):
            value = self.value(depth + 1)
            if value[0] != "bytes":
                raise ValueError("CPython code debug field must be bytes")
            content += value[1]
        return _node("code", content)

    def _value(self, kind: str, depth: int):
        readers = {
            "c": lambda: self._code(depth),
            "i": lambda: _integer(self.integer()),
            "l": self._long,
            "g": lambda: _node("float-bits", self.read(8)),
            "y": lambda: _node("complex-bits", self.read(16)),
        }
        if kind in readers:
            return readers[kind]()
        if kind in "sutaAzZ":
            return self._string(kind)
        if kind in "()>":
            return self._sequence(kind, depth)
        if kind in "NFTS.":
            return _node("singleton:" + kind)
        if kind == "r":
            return super()._value(kind, depth)
        raise ValueError("unsupported or non-code marshal type in frozen code")


def code_fields_digest(data: bytes) -> str:
    reader = CodeReader(data)
    kind, checksum = reader.value()
    if reader.position != len(data):
        raise ValueError("frozen code has trailing marshal bytes")
    if kind != "code":
        raise ValueError("frozen payload is not a CPython code object")
    return checksum.hex()
