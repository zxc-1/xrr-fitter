"""Derive a distinct arm64 Qt wheel while proving all upstream bytes retained."""

from __future__ import annotations

import copy
import hashlib
import struct
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from installed_files import VerifiedWheel  # noqa: E402
from installed_transforms import _archive_digest, record_bytes, record_row, wheel_record  # noqa: E402
from native_binary import inspect_native  # noqa: E402
from package_manifest import _verified_file  # noqa: E402

NAME = "pyside6-essentials"
VERSION = "6.11.2"
INFO = "pyside6_essentials-6.11.2.dist-info"
PLUGIN = "PySide6/Qt/plugins/platforms/libqcocoa.dylib"
ADDITIONS = f"{INFO}/licenses/xrr-qt-cocoa/"
FILENAME = "pyside6_essentials-6.11.2-1xrrcocoa-cp310-abi3-macosx_13_0_arm64.whl"
CHANGED = frozenset((PLUGIN, f"{INFO}/WHEEL", f"{INFO}/RECORD"))


@contextmanager
def _verified_wheel(path: Path, record: dict):
    try:
        wheel = VerifiedWheel(path, record)
    except (KeyError, TypeError, zipfile.BadZipFile) as error:
        raise ValueError("invalid Qt wheel archive or package metadata") from error
    with wheel:
        yield wheel


def _minimum_system(content: bytes) -> int:
    count = struct.unpack_from("<I", content, 16)[0]
    offset = 32
    versions = []
    for _ in range(count):
        kind, size = struct.unpack_from("<II", content, offset)
        if kind == 0x32:
            platform, minimum = struct.unpack_from("<II", content, offset + 8)
            if size < 24 or platform != 1:
                raise ValueError("Qt plugin build target is not macOS")
            versions.append(minimum)
        offset += size
    if len(versions) != 1:
        raise ValueError("Qt plugin requires exactly one macOS build version")
    return versions[0]


def _plugin_linkage(image: dict) -> None:
    if image["rpaths"] != ["@loader_path/../../lib"]:
        raise ValueError("Qt plugin must use only its wheel-relative Qt library rpath")
    for request in image["imports"]:
        if not request["name"].startswith(("@rpath/Qt", "/System/Library/Frameworks/", "/usr/lib/")):
            raise ValueError("Qt plugin links an unapproved runtime library path")


def validate_plugin(content: bytes) -> dict:
    native = inspect_native(memoryview(content))
    images = native["images"]
    if len(images) != 1 or native["unparsed"]:
        raise ValueError("Qt plugin must be one fully inspected arm64 image")
    image = images[0]
    if (image["format"], image["cpu_type"], image["file_type"]) != ("mach-o", 0x100000C, 6):
        raise ValueError("Qt plugin must be an arm64 Mach-O dylib")
    if content[:4] != b"\xcf\xfa\xed\xfe" or _minimum_system(content) != 13 << 16:
        raise ValueError("Qt plugin deployment target must be macOS 13.0")
    _plugin_linkage(image)
    return {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content), "native": native}


def _wheel_metadata(content: bytes) -> bytes:
    lines = content.decode("utf-8").splitlines()
    kept = [line for line in lines if not line.startswith(("Tag:", "Build:", "Generator:")) and line]
    return (
        "\n".join(
            [
                *kept,
                "Generator: xrr-fitter Qt Cocoa source build",
                "Build: 1xrrcocoa",
                "Tag: cp310-abi3-macosx_13_0_arm64",
            ]
        )
        + "\n"
    ).encode()


def _check_additions(additions: dict[str, bytes], upstream: VerifiedWheel) -> None:
    if not additions:
        raise ValueError("Qt derivative must carry its source provenance notice")
    for name, content in additions.items():
        path = PurePosixPath(name)
        if not name.startswith(ADDITIONS) or path.as_posix() != name or ".." in path.parts or "\\" in name:
            raise ValueError("Qt source additions must stay in their license subtree")
        if name in upstream.files or not isinstance(content, bytes):
            raise ValueError("Qt source addition overlaps an upstream member")


def _upstream_identity(path: Path, record: dict) -> None:
    if (record["name"], record["version"]) != (NAME, VERSION):
        raise ValueError("Qt derivation requires the pinned PySide6-Essentials identity")
    expected = "pyside6_essentials-6.11.2-cp310-abi3-macosx_13_0_universal2.whl"
    if record["filename"] != expected or path.name != expected:
        raise ValueError("Qt derivation requires the exact upstream wheel filename")


def _entry(upstream: VerifiedWheel, name: str) -> zipfile.ZipInfo:
    if name in upstream.files:
        entry = copy.copy(upstream.archive.getinfo(name))
    else:
        entry = zipfile.ZipInfo(name)
        entry.external_attr = 0o100644 << 16
        entry.create_system = 3
    entry.date_time = (1980, 1, 1, 0, 0, 0)
    entry.compress_type = zipfile.ZIP_DEFLATED
    return entry


def _copy_upstream(source: VerifiedWheel, archive: zipfile.ZipFile, name: str) -> tuple[str, str, str]:
    source.guard()
    expected = source.files[name]
    digest = hashlib.sha256()
    size = 0
    with source.archive.open(name) as reader, archive.open(_entry(source, name), "w") as writer:
        while chunk := reader.read(1024**2):
            size += len(chunk)
            if size > expected["size"]:
                raise ValueError("Qt upstream member grew while copying")
            digest.update(chunk)
            writer.write(chunk)
    source.guard()
    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("Qt upstream member changed while copying")
    return name, *_archive_digest(expected)


def _write_derivative(source: VerifiedWheel, path: Path, plugin: bytes, additions: dict[str, bytes]) -> None:
    replacements = {PLUGIN: plugin, f"{INFO}/WHEEL": _wheel_metadata(source.read(f"{INFO}/WHEEL")), **additions}
    names = (set(source.files) | set(additions)) - {f"{INFO}/RECORD"}
    rows = []
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(names):
            if name in replacements:
                content = replacements[name]
                archive.writestr(_entry(source, name), content)
                rows.append(record_row(name, content))
            else:
                rows.append(_copy_upstream(source, archive, name))
        rows.append((f"{INFO}/RECORD", "", ""))
        archive.writestr(_entry(source, f"{INFO}/RECORD"), record_bytes(rows))


def derive_wheel(upstream: Path, record: dict, plugin: Path, output: Path, additions: dict[str, bytes]) -> dict:
    _upstream_identity(upstream, record)
    if plugin.is_symlink() or not plugin.is_file():
        raise ValueError("Qt built plugin must be a regular file")
    content = plugin.read_bytes()
    validate_plugin(content)
    if output.is_symlink() or not output.is_dir() or list(output.iterdir()):
        raise ValueError("Qt wheel output must be a fresh empty directory")
    path = output / FILENAME
    with _verified_wheel(upstream, record) as source:
        wheel_record(source, INFO)
        _check_additions(additions, source)
        _write_derivative(source, path, content, additions)
    result = {"name": NAME, "version": VERSION, **_verified_file(path, hashlib.sha256(path.read_bytes()).hexdigest())}
    verify_derivation(upstream, record, path, result, additions)
    return result


def _verify_members(source: VerifiedWheel, target: VerifiedWheel, additions: dict[str, bytes]) -> None:
    if set(target.files) != set(source.files) | set(additions):
        raise ValueError("Qt derivative member set differs from the declared transformation")
    for name in set(source.files) - CHANGED:
        if any(target.files[name][key] != source.files[name][key] for key in ("sha256", "size")):
            raise ValueError(f"Qt derivative changed an unrelated upstream member: {name}")
    if target.read(f"{INFO}/WHEEL") != _wheel_metadata(source.read(f"{INFO}/WHEEL")):
        raise ValueError("Qt derivative wheel metadata differs")
    for name, expected in additions.items():
        if target.read(name) != expected:
            raise ValueError("Qt derivative source additions differ from their bound bytes")


def verify_derivation(
    upstream: Path, record: dict, derived: Path, derived_record: dict, additions: dict[str, bytes]
) -> dict:
    _upstream_identity(upstream, record)
    if derived.name != FILENAME or derived_record["filename"] != FILENAME:
        raise ValueError("Qt derived wheel build tag or architecture differs")
    if (derived_record["name"], derived_record["version"]) != (NAME, VERSION):
        raise ValueError("Qt derived wheel identity differs")
    observed = _verified_file(derived, derived_record["sha256"])
    if observed["size"] != derived_record["size"]:
        raise ValueError("Qt derived wheel size differs")
    with _verified_wheel(upstream, record) as source, _verified_wheel(derived, derived_record) as target:
        wheel_record(source, INFO)
        wheel_record(target, INFO)
        _check_additions(additions, source)
        _verify_members(source, target, additions)
        plugin = validate_plugin(target.read(PLUGIN))
        if plugin["sha256"] == source.files[PLUGIN]["sha256"]:
            raise ValueError("Qt derivative retained the unpatched upstream plugin")
        return {
            "plugin_sha256": plugin["sha256"],
            "plugin": plugin,
            "unchanged_members": len(source.files) - len(CHANGED),
            "inventory": target.inventory,
        }
