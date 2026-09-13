"""Inspect native loader declarations in PE, Mach-O and static archives."""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from native_bytes import NativeBytes  # noqa: E402
from native_pe import inspect_pe  # noqa: E402

MACHO = {
    b"\xce\xfa\xed\xfe": ("<", False),
    b"\xfe\xed\xfa\xce": (">", False),
    b"\xcf\xfa\xed\xfe": ("<", True),
    b"\xfe\xed\xfa\xcf": (">", True),
}
FAT = {
    b"\xca\xfe\xba\xbe": (">", False),
    b"\xbe\xba\xfe\xca": ("<", False),
    b"\xca\xfe\xba\xbf": (">", True),
    b"\xbf\xba\xfe\xca": ("<", True),
}
DYLIB_COMMANDS = {12: "load", 0x80000018: "weak", 0x8000001F: "reexport", 0x80000023: "upward", 0x20: "lazy"}
# These structural commands do not introduce external library requests. Other
# unhandled commands retain explicit evidence instead of disappearing from the inventory.
STRUCTURAL_COMMANDS = {
    0x1,
    0x2,
    0x3,
    0x4,
    0x5,
    0x8,
    0xA,
    0xB,
    0xF,
    0x11,
    0x16,
    0x17,
    0x19,
    0x1A,
    0x1B,
    0x1D,
    0x1E,
    0x21,
    0x22,
    0x24,
    0x25,
    0x26,
    0x29,
    0x2A,
    0x2B,
    0x2C,
    0x2E,
    0x2F,
    0x30,
    0x31,
    0x32,
    0x36,
    0x37,
    0x38,
    0x39,
    0x80000022,
    0x80000028,
    0x80000033,
    0x80000034,
}


def _version(value: int) -> str:
    return f"{value >> 16}.{value >> 8 & 255}.{value & 255}"


def _dylib_command(reader: NativeBytes, endian: str, kind: int, size: int, image: dict) -> None:
    offset, _timestamp, current, compatible = reader.unpack(endian + "4I", 8)
    if offset < 24:
        raise ValueError("Mach-O dylib name overlaps its header")
    name = reader.string(offset, size - offset)
    if kind == 13:
        if image["install_name"] is not None:
            raise ValueError("Mach-O has duplicate install identities")
        image["install_name"] = name
    else:
        image["imports"].append(
            {
                "name": name,
                "kind": DYLIB_COMMANDS[kind],
                "current_version": _version(current),
                "compatibility_version": _version(compatible),
            }
        )


def _path_command(reader: NativeBytes, endian: str, kind: int, size: int, image: dict) -> None:
    offset = reader.unpack(endian + "I", 8)[0]
    if offset < 12:
        raise ValueError("Mach-O path overlaps its header")
    name = reader.string(offset, size - offset)
    if kind == 14:
        image["imports"].append({"name": name, "kind": "dylinker"})
    else:
        image["rpaths"].append(name)


def _load_command(reader: NativeBytes, endian: str, image: dict) -> None:
    kind, size = reader.unpack(endian + "2I", 0)
    if kind in DYLIB_COMMANDS or kind == 13:
        _dylib_command(reader, endian, kind, size, image)
    elif kind in {0x8000001C, 14}:
        _path_command(reader, endian, kind, size, image)
    elif kind not in STRUCTURAL_COMMANDS:
        image["unparsed_commands"].append(
            {
                "command": kind,
                "sha256": hashlib.sha256(reader.data).hexdigest(),
                "reason": "unsupported Mach-O loader command",
            }
        )


def _macho(data: bytes | memoryview) -> dict:
    reader = NativeBytes(data)
    endian, wide = MACHO[bytes(reader.span(0, 4))]
    _magic, cpu, subtype, kind, count, size, _flags, *_reserved = reader.unpack(endian + ("8I" if wide else "7I"), 0)
    if count > 16384 or size > 16 * 1024**2:
        raise ValueError("Mach-O load command table exceeds the limit")
    table = NativeBytes(reader.span(32 if wide else 28, size))
    image = {
        "format": "mach-o",
        "cpu_type": cpu,
        "cpu_subtype": subtype,
        "file_type": kind,
        "imports": [],
        "rpaths": [],
        "install_name": None,
        "unparsed_commands": [],
    }
    position = 0
    for _index in range(count):
        _command, length = table.unpack(endian + "2I", position)
        if length < 8 or length % 4:
            raise ValueError("invalid Mach-O load command size")
        _load_command(NativeBytes(table.span(position, length)), endian, image)
        position += length
    if position != size:
        raise ValueError("Mach-O load command count does not cover its table")
    return image


def _slice_bounds(offset: int, size: int, alignment: int, table_end: int, spans: list) -> None:
    if offset < table_end or alignment > 31 or offset % (1 << alignment):
        raise ValueError("invalid Mach-O universal slice alignment")
    if any(offset < end and start < offset + size for start, end in spans):
        raise ValueError("overlapping Mach-O universal slices")


def _fat_slices(reader: NativeBytes):
    endian, wide = FAT[bytes(reader.span(0, 4))]
    count = reader.unpack(endian + "I", 4)[0]
    if not 0 < count < 64:
        raise ValueError("invalid Mach-O universal architecture count")
    format = endian + ("IIQQII" if wide else "5I")
    width = struct.calcsize(format)
    reader.span(8, count * width)
    spans = []
    architectures = set()
    for index in range(count):
        cpu, subtype, offset, size, alignment, *_reserved = reader.unpack(format, 8 + index * width)
        if (cpu, subtype) in architectures:
            raise ValueError("duplicate Mach-O universal architecture")
        architectures.add((cpu, subtype))
        _slice_bounds(offset, size, alignment, 8 + count * width, spans)
        spans.append((offset, offset + size))
        yield cpu, subtype, offset, reader.span(offset, size)


def _fat_payload(data: memoryview, cpu: int, subtype: int, depth: int) -> dict:
    child = _inspect_native(data, depth + 1)
    if child["format"] not in {"mach-o", "ar"}:
        raise ValueError("Mach-O universal slice is not a supported native payload")
    for image in child["images"]:
        if image["format"] != "mach-o" or (image["cpu_type"], image["cpu_subtype"]) != (cpu, subtype):
            raise ValueError("Mach-O universal architecture differs from its slice")
    return child


def _fat(data: bytes | memoryview, depth: int) -> dict:
    result = {"format": "mach-o-universal", "images": [], "unparsed": [], "slices": []}
    for cpu, subtype, offset, payload in _fat_slices(NativeBytes(data)):
        child = _fat_payload(payload, cpu, subtype, depth)
        result["slices"].append(
            {
                "cpu_type": cpu,
                "cpu_subtype": subtype,
                "offset": offset,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "format": child["format"],
                "members": child.get("members", []),
            }
        )
        result["images"].extend(child["images"])
        result["unparsed"].extend({**item, "cpu_type": cpu, "cpu_subtype": subtype} for item in child["unparsed"])
    return result


def _long_archive_name(strings: bytes, offset: int) -> str:
    if not 0 <= offset < len(strings):
        raise ValueError("invalid static archive long-name reference")
    if offset and not strings[max(0, offset - 2) : offset].endswith((b"\0", b"/\n")):
        raise ValueError("static archive long-name reference is not at a name boundary")
    # GNU ar terminates long names with /\n; Microsoft COFF libraries use NUL.
    endings = [strings.find(separator, offset) for separator in (b"/\n", b"\0")]
    endings = [end for end in endings if end >= 0]
    if not endings or min(endings) == offset:
        raise ValueError("invalid static archive long-name terminator")
    return strings[offset : min(endings)].decode("utf-8")


def _archive_name(name: str, content: memoryview, strings: bytes) -> tuple[str, memoryview]:
    if name.startswith("#1/"):
        size = int(name[3:])
        reader = NativeBytes(content)
        return bytes(reader.span(0, size)).rstrip(b"\0").decode("utf-8"), reader.span(size, len(content) - size)
    if name.startswith("/") and name[1:].isdigit():
        return _long_archive_name(strings, int(name[1:])), content
    return name.rstrip("/"), content


def _member_records(records: list[dict], name: str, index: int, content: memoryview) -> list[dict]:
    identity = {"member_sha256": hashlib.sha256(content).hexdigest(), "member_size": len(content)}
    return [
        {
            **identity,
            **item,
            "member": item.get("member", name),
            "member_path": [name, *item.get("member_path", [])],
            "member_indices": [index, *item.get("member_indices", [])],
        }
        for item in records
    ]


def _archive(data: bytes | memoryview, depth: int) -> dict:
    reader = NativeBytes(data)
    result = {"format": "ar", "images": [], "unparsed": [], "members": []}
    position, strings = 8, b""
    while position < len(data):
        if len(result["members"]) >= 10000:
            raise ValueError("static archive member count exceeds the limit")
        header = bytes(reader.span(position, 60))
        if header[-2:] != b"`\n":
            raise ValueError("invalid static archive member header")
        size = int(header[48:58])
        content = reader.span(position + 60, size)
        name = header[:16].decode("ascii").strip()
        if name == "//":
            strings = bytes(content)
        elif name not in {"/", "/SYM64/"}:
            name, content = _archive_name(name, content, strings)
            if name not in {"__.SYMDEF", "__.SYMDEF SORTED", "__.SYMDEF_64", "__.SYMDEF_64 SORTED"}:
                child = _inspect_native(content, depth + 1)
                index = len(result["members"])
                result["images"].extend(_member_records(child["images"], name, index, content))
                result["unparsed"].extend(_member_records(child["unparsed"], name, index, content))
        result["members"].append({"name": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        position += 60 + size + size % 2
    if position != len(data):
        raise ValueError("static archive padding is truncated")
    return result


def _native_image(data: bytes | memoryview, image: dict) -> dict:
    unparsed = image.pop("unparsed_commands", [])
    image.update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    return {"format": image["format"], "images": [image], "unparsed": unparsed}


def _inspect_native(data: bytes | memoryview, depth: int) -> dict:
    if depth > 16:
        raise ValueError("native container nesting exceeds the limit")
    prefix = bytes(data[:8])
    if prefix[:4] in MACHO:
        return _native_image(data, _macho(data))
    if prefix[:4] in FAT:
        return _fat(data, depth)
    if prefix.startswith(b"MZ"):
        return _native_image(data, inspect_pe(data))
    if prefix == b"!<arch>\n":
        return _archive(data, depth)
    return {"format": "unclassified", "images": [], "unparsed": [{"reason": "unrecognized native header"}]}


def inspect_native(data: bytes | memoryview) -> dict:
    return _inspect_native(data, 0)
