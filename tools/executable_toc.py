"""Read the data-only marshal subset used by pinned PyInstaller PYZ tables."""

from __future__ import annotations

from native_bytes import NativeBytes

TABLE_LIMIT = 16 * 1024**2
ITEM_LIMIT = 100_000


class TableReader:
    depth_limit = 8

    def __init__(self, data: bytes | memoryview) -> None:
        if len(data) > TABLE_LIMIT:
            raise ValueError("PYZ table exceeds its byte limit")
        self.reader = NativeBytes(data)
        self.position = 0
        self.references = []
        self.count = 0

    def read(self, size: int) -> bytes:
        result = self.reader.span(self.position, size)
        self.position += size
        return bytes(result)

    def integer(self) -> int:
        result = int.from_bytes(self.read(4), "little", signed=True)
        return result

    def size(self, short: bool, limit: int) -> int:
        result = self.read(1)[0] if short else self.integer()
        if not 0 <= result <= limit:
            raise ValueError("PYZ table item exceeds its size limit")
        return result

    def value(self, depth: int = 0):
        self.count += 1
        if depth > self.depth_limit or self.count > ITEM_LIMIT * 8:
            raise ValueError("PYZ table exceeds its nesting or item limit")
        marker = self.read(1)[0]
        kind, reference = chr(marker & 0x7F), None
        if marker & 0x80:
            reference = len(self.references)
            self.references.append(None)
        result = self._value(kind, depth)
        if reference is not None:
            self.references[reference] = result
        return result

    def _value(self, kind: str, depth: int):
        if kind == "i":
            return self.integer()
        if kind in "aAuzZ":
            return self.read(self.size(kind in "zZ", 4096)).decode("utf-8")
        if kind in "([)":
            result = [self.value(depth + 1) for _ in range(self.size(kind == ")", ITEM_LIMIT))]
            return result if kind == "[" else tuple(result)
        if kind == "r":
            index = self.integer()
            if not 0 <= index < len(self.references) or self.references[index] is None:
                raise ValueError("PYZ table has an invalid or recursive marshal reference")
            return self.references[index]
        raise ValueError("PYZ table contains an unsupported executable or non-data marshal type")


def read_table(data: bytes | memoryview) -> list:
    reader = TableReader(data)
    result = reader.value()
    if reader.position != len(data):
        raise ValueError("PYZ table has trailing bytes")
    if not isinstance(result, list):
        raise ValueError("PYZ table must be the pinned writer's list")
    return result
