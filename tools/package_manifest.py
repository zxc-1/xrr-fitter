"""Target-specific Python wheel identities and offline package-byte checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.tags import Tag, compatible_tags, cpython_tags, mac_platforms  # noqa: E402
from packaging.utils import canonicalize_name, parse_wheel_filename  # noqa: E402
from packaging.version import Version  # noqa: E402

from lock_environment import PIP_VERSION  # noqa: E402
from lock_sbom import TARGETS, build_lock_sbom  # noqa: E402
from verify_publish import _directory_identity, _require_same_directory  # noqa: E402

SCHEMA = "xrr-package-manifest-v1"
WHEEL_FIELDS = {"name", "version", "filename", "url", "sha256"}


def target_platforms(target: str) -> tuple[str, ...]:
    if target not in TARGETS:
        raise ValueError(f"unsupported package target: {target}")
    if target == "windows-x64-py312":
        return ("win_amd64",)
    return tuple(mac_platforms(version=(15, 0), arch="arm64"))


@cache
def target_tags(target: str) -> frozenset[Tag]:
    platforms = target_platforms(target)
    return frozenset(
        (
            *cpython_tags((3, 12), platforms=platforms),
            *compatible_tags((3, 12), interpreter="cp312", platforms=platforms),
        )
    )


def pypi_wheel_filename(value: str) -> str:
    url = urlsplit(value)
    if (url.scheme, url.netloc) != ("https", "files.pythonhosted.org") or url.query or url.fragment:
        raise ValueError("package wheel must use the HTTPS PyPI file host")
    filename = url.path.rsplit("/", 1)[-1]
    if not filename or unquote(filename) != filename or "\\" in filename:
        raise ValueError("package wheel URL has a noncanonical filename")
    return filename


def _sha256(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("package wheel requires canonical SHA-256")
    return value


def _validate_wheel(record: dict, target: str) -> dict[str, str]:
    if set(record) != WHEEL_FIELDS or not all(isinstance(value, str) for value in record.values()):
        raise ValueError("package wheel record schema mismatch")
    filename = pypi_wheel_filename(record["url"])
    name, version, _build, tags = parse_wheel_filename(filename)
    if (record["name"], record["version"], record["filename"]) != (name, str(version), filename):
        raise ValueError("package wheel identity differs from metadata")
    if not tags.intersection(target_tags(target)):
        raise ValueError("package wheel tags do not support the target")
    _sha256(record["sha256"])
    return record


def _report_wheel(item: dict, target: str) -> dict[str, str]:
    if item.get("is_direct") is not False or item.get("is_yanked") is not False:
        raise ValueError("package resolution requires non-yanked index wheels")
    metadata = item["metadata"]
    download = item["download_info"]
    record = {
        "name": canonicalize_name(metadata["name"], validate=True),
        "version": str(Version(metadata["version"])),
        "url": download["url"],
        "filename": pypi_wheel_filename(download["url"]),
        "sha256": download.get("archive_info", {}).get("hashes", {}).get("sha256"),
    }
    return _validate_wheel(record, target)


def locked_inputs(root: Path, target: str) -> tuple[dict[str, str], dict]:
    inventory = build_lock_sbom(root, target=target)
    properties = {item["name"]: item["value"] for item in inventory["metadata"]["properties"]}
    pins = {item["name"]: item["version"] for item in inventory["components"] if "version" in item}
    vcs = [
        {"name": item["name"], "url": item["externalReferences"][0]["url"], "commit": item["properties"][0]["value"]}
        for item in inventory["components"]
        if "version" not in item
    ]
    return pins, {
        "schema": SCHEMA,
        "target": target,
        "python_version": "3.12",
        "pip_version": PIP_VERSION,
        "lock_sha256": properties["xrr:lock:sha256"],
        "pyproject_sha256": properties["xrr:pyproject:sha256"],
        "vcs": vcs,
    }


def _exact_wheels(wheels: list[dict], pins: dict[str, str]) -> None:
    names = [item["name"] for item in wheels]
    if len(set(names)) != len(names):
        raise ValueError("package manifest contains duplicate wheels")
    if {item["name"]: item["version"] for item in wheels} != pins:
        raise ValueError("package wheels do not cover the exact locked pins")
    if names != sorted(names):
        raise ValueError("package wheel records must be canonically sorted")


def build_manifest(root: Path, target: str, report: dict) -> dict:
    if report.get("version") != "1" or report.get("pip_version") != PIP_VERSION:
        raise ValueError("package resolution requires the pinned pip report schema")
    pins, header = locked_inputs(root, target)
    wheels = sorted((_report_wheel(item, target) for item in report["install"]), key=lambda item: item["name"])
    _exact_wheels(wheels, pins)
    return {**header, "wheels": wheels}


def validate_manifest(root: Path, value: dict) -> dict:
    pins, header = locked_inputs(root, value.get("target"))
    if {key: item for key, item in value.items() if key != "wheels"} != header:
        raise ValueError("package manifest input bindings or schema changed")
    wheels = value.get("wheels")
    if not isinstance(wheels, list):
        raise ValueError("package manifest wheels must be an array")
    for record in wheels:
        _validate_wheel(record, header["target"])
    _exact_wheels(wheels, pins)
    return value


def manifest_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")


def download_requirements(manifest: dict) -> bytes:
    return "".join(
        f"{item['name']} @ {item['url']} --hash=sha256:{item['sha256']}\n" for item in manifest["wheels"]
    ).encode("ascii")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("package wheel must be a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _verified_file(path: Path, expected: str) -> dict:
    before = _file_identity(path.lstat())
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        if _file_identity(os.fstat(handle.fileno())) != before:
            raise ValueError("package wheel changed before reading")
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if _file_identity(os.fstat(handle.fileno())) != before:
            raise ValueError("package wheel changed during reading")
    if digest != expected or _file_identity(path.lstat()) != before:
        raise ValueError("package wheel bytes do not match the manifest")
    return {"filename": path.name, "sha256": digest, "size": before[2]}


def _exact_directory(directory: Path, names: set[str]) -> None:
    if {path.name for path in directory.iterdir()} != names:
        raise ValueError("package directory must contain exactly the recorded wheels")


def verify_wheels(directory: Path, wheels: Sequence[dict]) -> list[dict]:
    identity = _directory_identity(directory, "package directory")
    names = {item["filename"] for item in wheels}
    if len(names) != len(wheels) or any(Path(name).name != name for name in names):
        raise ValueError("package filenames must be unique direct children")
    _exact_directory(directory, names)
    before = {name: _file_identity((directory / name).lstat()) for name in names}
    records = [_verified_file(directory / item["filename"], _sha256(item["sha256"])) for item in wheels]
    _require_same_directory(directory, identity, "package directory")
    _exact_directory(directory, names)
    if {name: _file_identity((directory / name).lstat()) for name in names} != before:
        raise ValueError("package files changed during directory verification")
    return records
