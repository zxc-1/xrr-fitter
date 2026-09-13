"""Read PE imports, delay imports and forwarded exports without loading code."""

from __future__ import annotations

from native_bytes import NativeBytes


class PEImage:
    def __init__(self, data: bytes | memoryview) -> None:
        self.reader = NativeBytes(data)
        position = self.reader.unpack("<I", 0x3C)[0]
        if bytes(self.reader.span(position, 4)) != b"PE\0\0":
            raise ValueError("invalid PE signature")
        self.machine, count, _time, _symbols, _number, size, _flags = self.reader.unpack("<HHIIIHH", position + 4)
        optional = NativeBytes(self.reader.span(position + 24, size))
        magic = optional.unpack("<H", 0)[0]
        if magic not in {0x10B, 0x20B}:
            raise ValueError("unsupported PE optional header")
        plus = magic == 0x20B
        self.image_base = optional.unpack("<Q" if plus else "<I", 24 if plus else 28)[0]
        self.header_size = optional.unpack("<I", 60)[0]
        directories = optional.unpack("<I", 108 if plus else 92)[0]
        if not 0 < count <= 96 or directories > 16:
            raise ValueError("PE section or directory count exceeds the limit")
        self.directories = [optional.unpack("<II", (112 if plus else 96) + index * 8) for index in range(directories)]
        self.sections = self._sections(position + 24 + size, count)

    def _sections(self, offset: int, count: int) -> list[tuple[int, int, int]]:
        result = []
        self.reader.span(0, self.header_size)
        for index in range(count):
            _name, _virtual_size, address, size, pointer, *_rest = self.reader.unpack(
                "<8sIIIIIIHHI", offset + index * 40
            )
            self.reader.span(pointer, size)
            result.append((address, size, pointer))
        return result

    def mapping(self, rva: int, size: int) -> tuple[int, int]:
        candidates = []
        if 0 <= rva and rva + size <= self.header_size:
            candidates.append((rva, self.header_size - rva))
        for address, length, pointer in self.sections:
            if address <= rva and rva + size <= address + length:
                candidates.append((pointer + rva - address, address + length - rva))
        if len(candidates) != 1:
            raise ValueError("PE RVA is unmapped, unbacked or ambiguous")
        return candidates[0]

    def unpack(self, format: str, rva: int) -> tuple:
        import struct

        offset, _remaining = self.mapping(rva, struct.calcsize(format))
        return self.reader.unpack(format, offset)

    def string(self, rva: int, limit: int | None = None) -> str:
        offset, remaining = self.mapping(rva, 1)
        return self.reader.string(offset, remaining if limit is None else min(limit, remaining))

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)


def _import_name(image: PEImage, entry: tuple, delayed: bool) -> str:
    if not delayed:
        return image.string(entry[3])
    if entry[0] & ~1:
        raise ValueError("unsupported PE delay import attributes")
    address = entry[1] if entry[0] & 1 else entry[1] - image.image_base
    return image.string(address)


def _imports(image: PEImage, *, delayed: bool) -> list[dict]:
    address, size = image.directory(13 if delayed else 1)
    if address == size == 0:
        return []
    width = 32 if delayed else 20
    if size < width:
        raise ValueError("PE import directory is truncated")
    result = []
    for offset in range(0, min(size - width + 1, 4096 * width), width):
        entry = image.unpack("<8I" if delayed else "<5I", address + offset)
        if not any(entry):
            return result
        result.append({"name": _import_name(image, entry, delayed), "kind": "delay-load" if delayed else "load"})
    raise ValueError("PE import directory has no bounded terminator")


def _forwarders(image: PEImage) -> list[str]:
    address, size = image.directory(0)
    if address == size == 0:
        return []
    if size < 40:
        raise ValueError("PE export directory is truncated")
    entry = image.unpack("<IIHH7I", address)
    count, functions = entry[6], entry[8]
    if count > 65536:
        raise ValueError("PE export function count exceeds the limit")
    values = image.unpack(f"<{count}I", functions) if count else ()
    return sorted(image.string(value, address + size - value) for value in values if address <= value < address + size)


def _forwarded_library(value: str) -> str:
    name, separator, symbol = value.rpartition(".")
    if not name or not separator or not symbol:
        raise ValueError("invalid PE export forwarder")
    return name if name.lower().endswith((".dll", ".exe")) else name + ".dll"


def inspect_pe(data: bytes | memoryview) -> dict:
    image = PEImage(data)
    forwards = _forwarders(image)
    requests = _imports(image, delayed=False) + _imports(image, delayed=True)
    requests.extend(
        {"name": name, "kind": "forwarder"} for name in sorted({_forwarded_library(value) for value in forwards})
    )
    return {"format": "pe", "machine": image.machine, "imports": requests, "forwarded_exports": forwards}
