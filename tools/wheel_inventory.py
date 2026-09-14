"""Read wheel metadata and every member's bytes without extracting or importing."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.licenses import canonicalize_license_expression  # noqa: E402
from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402
from packaging.version import Version  # noqa: E402

from native_binary import FAT, MACHO, inspect_native  # noqa: E402
from package_manifest import _file_identity, _same_file_identity, _verified_file  # noqa: E402

NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".exe", ".a", ".lib"}
NATIVE_READ_LIMIT = 512 * 1024**2


def _native_format(prefix: bytes, path: PurePosixPath) -> str | None:
    signatures = {b"\x7fELF": "elf", b"MZ": "pe", b"!<arch>\n": "ar"}
    for signature, kind in signatures.items():
        if prefix.startswith(signature):
            return kind
    if prefix[:4] in MACHO:
        return "mach-o"
    if prefix[:4] in FAT:
        return "mach-o-universal"
    return "unclassified" if path.suffix.lower() in NATIVE_SUFFIXES else None


def _member_path(member: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(member.filename)
    if path.is_absolute() or ".." in path.parts or "\\" in member.filename or ":" in member.filename:
        raise ValueError("wheel contains an unsafe member path")
    if path.as_posix() != member.filename.rstrip("/") or not path.parts:
        raise ValueError("wheel contains a noncanonical member path")
    return path


def _check_member_limits(members: list[zipfile.ZipInfo]) -> None:
    names = [member.filename.rstrip("/") for member in members]
    if len(names) != len(set(names)):
        raise ValueError("wheel contains duplicate members")
    if sum(member.file_size for member in members) > 4 * 1024**3 or len(members) > 100000:
        raise ValueError("wheel exceeds the inventory limit")


def _members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    _check_member_limits(members)
    for member in members:
        _member_path(member)
        if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
            raise ValueError("wheel contains a symlink member")
        if member.flag_bits & 1:
            raise ValueError("wheel contains an encrypted member")
    return sorted((member for member in members if not member.is_dir()), key=lambda member: member.filename)


def _file_kind(path: PurePosixPath, prefix: bytes) -> dict[str, str]:
    native = _native_format(prefix, path)
    if native is not None:
        return {"kind": "native", "format": native}
    if "licenses" in path.parts or path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
        return {"kind": "license"}
    return {"kind": "python" if path.suffix in {".py", ".pyi", ".pyc"} else "data"}


def _native_buffer(kind: str, prefix: bytes, size: int, enabled: bool) -> bytearray | None:
    if not enabled or kind != "native":
        return None
    if size > NATIVE_READ_LIMIT:
        raise ValueError("native wheel member exceeds the loader inspection limit")
    return bytearray(prefix)


def _inventory_file(archive: zipfile.ZipFile, member: zipfile.ZipInfo, *, native_loaders: bool = False) -> dict:
    with archive.open(member) as handle:
        prefix = handle.read(16)
        kind = _file_kind(PurePosixPath(member.filename), prefix)
        buffer = _native_buffer(kind["kind"], prefix, member.file_size, native_loaders)
        digest = hashlib.sha256(prefix)
        size = len(prefix)
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
            if buffer is not None:
                buffer.extend(block)
    if size != member.file_size:
        raise ValueError("wheel member size changed during reading")
    result = {
        "path": member.filename,
        "sha256": digest.hexdigest(),
        "size": size,
        **kind,
    }
    if buffer is not None:
        try:
            result["native"] = inspect_native(memoryview(buffer))
        except ValueError as error:
            raise ValueError(f"invalid native wheel member {member.filename}: {error}") from error
    return result


def _licenses(message) -> list[dict]:
    expression = message.get("License-Expression")
    if expression:
        return [{"expression": canonicalize_license_expression(expression)}]
    legacy = message.get("License")
    if legacy and legacy != "UNKNOWN":
        return [{"license": {"name": legacy}}]
    return []


def _read_metadata(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> dict:
    if member.file_size > 2 * 1024 * 1024:
        raise ValueError("wheel METADATA exceeds the inventory limit")
    message = BytesParser(policy=policy.default).parsebytes(archive.read(member))
    identity = canonicalize_name(message["Name"], validate=True), str(Version(message["Version"]))
    requirements = sorted(str(Requirement(value)) for value in message.get_all("Requires-Dist", ()))
    return {"name": identity[0], "version": identity[1], "requirements": requirements, "licenses": _licenses(message)}


def _metadata(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], record: dict) -> dict:
    candidates = [item for item in members if len(PurePosixPath(item.filename).parts) == 2]
    if len(candidates) != 1:
        raise ValueError("wheel must contain exactly one top-level METADATA file")
    result = _read_metadata(archive, candidates[0])
    if (result["name"], result["version"]) != (record["name"], record["version"]):
        raise ValueError("wheel metadata identity differs from the verified manifest")
    return result


def _vendored_metadata(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo]) -> list[dict]:
    result = []
    for member in members:
        if len(PurePosixPath(member.filename).parts) <= 2:
            continue
        result.append(
            {
                **_read_metadata(archive, member),
                "metadata_path": member.filename,
                "metadata_sha256": hashlib.sha256(archive.read(member)).hexdigest(),
            }
        )
    return result


def _require_wheel_identity(value: os.stat_result, expected: tuple) -> None:
    if not _same_file_identity(_file_identity(value), expected):
        raise ValueError("wheel identity changed during inventory generation")


def inspect_wheel(path: Path, record: dict, *, native_loaders: bool = False) -> dict:
    identity = _file_identity(path.lstat())
    _verified_file(path, record["sha256"])
    with zipfile.ZipFile(path) as archive:
        _require_wheel_identity(os.fstat(archive.fp.fileno()), identity)
        members = _members(archive)
        metadata_members = [item for item in members if item.filename.endswith(".dist-info/METADATA")]
        metadata = _metadata(archive, metadata_members, record)
        vendored = _vendored_metadata(archive, metadata_members)
        files = [_inventory_file(archive, member, native_loaders=native_loaders) for member in members]
        _require_wheel_identity(os.fstat(archive.fp.fileno()), identity)
    _verified_file(path, record["sha256"])
    _require_wheel_identity(path.lstat(), identity)
    return {**metadata, "files": files, "vendored": vendored}
