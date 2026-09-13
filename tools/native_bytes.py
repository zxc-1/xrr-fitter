"""Bounded reads shared by the native loader-table inspectors."""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeBytes:
    data: bytes | memoryview

    def span(self, offset: int, size: int) -> memoryview:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("native structure exceeds the file boundary")
        return memoryview(self.data)[offset : offset + size]

    def unpack(self, format: str, offset: int) -> tuple:
        return struct.unpack(format, self.span(offset, struct.calcsize(format)))

    def string(self, offset: int, size: int) -> str:
        value, terminator, _tail = bytes(self.span(offset, min(size, 4096))).partition(b"\0")
        if not value or not terminator:
            raise ValueError("native loader string is empty or unterminated")
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("native loader string is not UTF-8") from error
