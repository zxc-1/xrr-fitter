"""Small, structurally valid native images for byte-level loader tests."""

from __future__ import annotations

import struct


def macho_command(kind: int, name: str, *, dylib: bool = True, endian: str = "<") -> bytes:
    offset = 24 if dylib else 12
    value = name.encode() + b"\0"
    size = (offset + len(value) + 7) & ~7
    header = (
        struct.pack(endian + "6I", kind, size, offset, 0, 0x10203, 0x10000)
        if dylib
        else struct.pack(endian + "3I", kind, size, offset)
    )
    return (header + value).ljust(size, b"\0")


def macho(commands=(), *, cpu: int = 0x100000C, file_type: int = 6, endian: str = "<", wide: bool = True) -> bytes:
    table = b"".join(commands)
    fields = [0xFEEDFACF if wide else 0xFEEDFACE, cpu, 0, file_type, len(commands), len(table), 0]
    if wide:
        fields.append(0)
    return struct.pack(endian + f"{len(fields)}I", *fields) + table


def universal(images, *, endian: str = ">", wide: bool = False) -> bytes:
    table = struct.pack(endian + "2I", 0xCAFEBABF if wide else 0xCAFEBABE, len(images))
    payloads = []
    for index, (cpu, image) in enumerate(images, start=1):
        fields = [cpu, 0, index * 0x100, len(image), 8]
        if wide:
            fields.append(0)
        table += struct.pack(endian + ("IIQQII" if wide else "5I"), *fields)
        payloads.append(image.ljust(0x100, b"\0"))
    return table.ljust(0x100, b"\0") + b"".join(payloads)


def fat_macho() -> bytes:
    images = [
        macho([macho_command(12, name)], cpu=cpu) for cpu, name in [(0x1000007, "x86.dylib"), (0x100000C, "arm.dylib")]
    ]
    table = struct.pack(">2I", 0xCAFEBABE, 2)
    for cpu, offset, image in zip([0x1000007, 0x100000C], [0x100, 0x200], images, strict=True):
        table += struct.pack(">5I", cpu, 0, offset, len(image), 8)
    return table.ljust(0x100, b"\0") + images[0].ljust(0x100, b"\0") + images[1]


def fat_archive() -> bytes:
    images = [static_archive([("object.o", macho(cpu=cpu, file_type=1))]) for cpu in [0x1000007, 0x100000C]]
    table = struct.pack(">2I", 0xCAFEBABE, 2)
    for cpu, offset, image in zip([0x1000007, 0x100000C], [0x100, 0x200], images, strict=True):
        table += struct.pack(">5I", cpu, 0, offset, len(image), 8)
    return table.ljust(0x100, b"\0") + images[0].ljust(0x100, b"\0") + images[1]


def _pe_headers(content: bytearray, plus: bool) -> tuple[int, int]:
    content[:2] = b"MZ"
    struct.pack_into("<I", content, 0x3C, 0x80)
    content[0x80:0x84] = b"PE\0\0"
    optional_size = 240 if plus else 224
    struct.pack_into("<HHIIIHH", content, 0x84, 0x8664 if plus else 0x14C, 1, 0, 0, 0, optional_size, 0x2022)
    optional = 0x98
    image_base = 0x180000000 if plus else 0x400000
    struct.pack_into("<H", content, optional, 0x20B if plus else 0x10B)
    struct.pack_into("<Q" if plus else "<I", content, optional + (24 if plus else 28), image_base)
    struct.pack_into("<I", content, optional + 60, 0x200)
    directory_offset = optional + (112 if plus else 96)
    struct.pack_into("<I", content, directory_offset - 4, 16)
    for index, rva, size in [(0, 0x1100, 0x100), (1, 0x1020, 40), (13, 0x1080, 64)]:
        struct.pack_into("<II", content, directory_offset + 8 * index, rva, size)
    return optional + optional_size, image_base


def pe_image(*, plus: bool = True, delay_rva: bool = True) -> bytes:
    content = bytearray(0x600)
    section, image_base = _pe_headers(content, plus)
    struct.pack_into(
        "<8sIIIIIIHHI", content, section, b".rdata\0\0", 0x400, 0x1000, 0x400, 0x200, 0, 0, 0, 0, 0x40000040
    )
    struct.pack_into("<5I", content, 0x220, 0, 0, 0, 0x1300, 0)
    struct.pack_into(
        "<8I", content, 0x280, int(delay_rva), 0x1320 if delay_rva else image_base + 0x1320, 0, 0, 0, 0, 0, 0
    )
    struct.pack_into("<IIHH7I", content, 0x300, 0, 0, 0, 0, 0, 1, 1, 0, 0x1140, 0, 0)
    struct.pack_into("<I", content, 0x340, 0x1160)
    for offset, value in [(0x360, b"python312.PyObject_Call\0"), (0x500, b"KERNEL32.dll\0"), (0x520, b"USER32.dll\0")]:
        content[offset : offset + len(value)] = value
    return bytes(content)


def static_archive(members, *, raw_names: bool = False) -> bytes:
    result = b"!<arch>\n"
    for name, content in members:
        field = name if raw_names else name + "/"
        header = f"{field:16}{0:<12}{0:<6}{0:<6}{'100644':8}{len(content):<10}`\n".encode()
        assert len(header) == 60
        result += header + content + (b"\n" if len(content) % 2 else b"")
    return result
