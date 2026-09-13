"""Inventory every frozen payload, its framing and exact verified-input candidates."""

from __future__ import annotations

import io
import stat
import sys
import zipfile
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from executable_archive import (  # noqa: E402
    ITEM_LIMIT,
    PAYLOAD_LIMIT,
    TOTAL_LIMIT,
    CArchive,
    PyzArchive,
    digest,
    extraction_path,
)
from native_binary import inspect_native  # noqa: E402


def _zip_member(member, names: set[str]) -> str:
    path = extraction_path(member.filename)
    if path.casefold() in names or stat.S_ISLNK(member.external_attr >> 16):
        raise ValueError("executable nested ZIP has duplicate paths or links")
    names.add(path.casefold())
    if member.file_size > PAYLOAD_LIMIT or member.flag_bits & 1 or member.is_dir():
        raise ValueError("executable nested ZIP member exceeds the supported file boundary")
    return path


def _zip_entries(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > ITEM_LIMIT or sum(member.file_size for member in members) > TOTAL_LIMIT:
            raise ValueError("executable nested ZIP exceeds its inventory limit")
        names = set()
        for member in members:
            path = _zip_member(member, names)
            content = archive.read(member)
            if len(content) != member.file_size:
                raise ValueError("executable nested ZIP decompressed size differs")
            yield (
                {"path": path, "kind": "zip-file", "offset": member.header_offset, "stored_size": member.compress_size},
                content,
            )


class ExecutableInspection:
    def __init__(self, sources, native_loaders: bool) -> None:
        self.sources, self.native_loaders = sources, native_loaders
        self.files, self.containers = [], []
        self.size = 0

    def file(self, entry: dict, data: bytes, container: str, parent: str, index: int) -> dict:
        self.size += len(data)
        if self.size > TOTAL_LIMIT or len(self.files) >= ITEM_LIMIT:
            raise ValueError("executable expanded inventory exceeds its limit")
        row = {
            **entry,
            "id": f"{parent}/{container}/{index}",
            "parent": parent,
            "container": container,
            "sha256": digest(data),
            "size": len(data),
        }
        row["claims"] = self.sources.match(row, data)
        row["unresolved"] = "" if row["claims"] else "no verified input-byte or code-transform provenance"
        if entry["kind"] == "b" and self.native_loaders:
            row["native"] = inspect_native(data)
        self.files.append(row)
        return row

    def nested(self, row: dict, data: bytes) -> None:
        if row["kind"] == "z":
            archive = PyzArchive(data)
            self.containers.append({"parent": row["id"], "kind": "pyz", **archive.framing})
            for index, entry in enumerate(archive.entries):
                self.file(entry, archive.extract(entry), "pyz", row["id"], index)
        elif row["kind"] == "Z" or row["path"] == "base_library.zip":
            for index, (entry, content) in enumerate(_zip_entries(data)):
                self.file(entry, content, "zip", row["id"], index)


def inspect_executable(data: bytes, sources, *, native_loaders: bool = True) -> dict:
    archive = CArchive(data)
    inspection = ExecutableInspection(sources, native_loaders)
    for index, entry in enumerate(archive.entries):
        content = archive.extract(entry)
        row = inspection.file(entry, content, "carchive", "executable", index)
        inspection.nested(row, content)
    archive.framing["prefix"]["unresolved"] = "modified bootloader; no exact wheel-byte ownership established"
    result = {
        "schema": "xrr-executable-inventory-v1",
        "target": "windows-x64-py312",
        "aggregate": "incomplete",
        "executable": {"sha256": digest(data), "size": len(data)},
        "framing": archive.framing,
        "python_library": archive.python_library,
        "files": inspection.files,
        "containers": inspection.containers,
        "provenance": list(sources.components.values()),
        "provenance_dependencies": sources.dependencies,
        "boundary": "verified payload bytes and matching input candidates; bootloader, unmatched OS/CPython files and runtime loader closure remain unresolved",
    }
    if native_loaders:
        result["native"] = inspect_native(data)
    return result
