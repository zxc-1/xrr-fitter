"""Pinned Qt Cocoa source/SDK inputs; cache originals, never derived binaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from audit_reports import _write  # noqa: E402
from package_cache import _cache_directory, _cache_members, cache_key, copy_verified  # noqa: E402
from package_downloads import _summary, prepare_report  # noqa: E402
from package_manifest import _verified_file, manifest_bytes, verify_wheels  # noqa: E402
from verify_report import _directory_identity, _require_same_directory  # noqa: E402

MANIFEST = Path("tools/package-manifests/qt-cocoa-macos-arm64-py312.json")
SOURCE_BASE = "https://code.qt.io/cgit/qt/qtbase.git/plain/src/plugins/platforms/cocoa/"
SDK_URL = (
    "https://download.qt.io/online/qtsdkrepository/mac_x64/desktop/qt6_6112/qt6_6112/"
    "qt.qt6.6112.clang_64/6.11.2-0-202608131016qtbase-MacOS-MacOS_15-Clang-MacOS-MacOS_15-X86_64-ARM64.7z"
)
RECIPES = (
    "tools/qt-cocoa/ownership.patch",
    "tools/qt-cocoa/cocoa.pro",
    "tools/qt-cocoa/qcocoaresources.qrc",
    "tools/qt-cocoa/ownership_regression.mm",
    "tools/qt-cocoa/ownership_regression.pro",
    "tools/qt-cocoa/NOTICE.txt",
)
IDENTITY = {
    "schema": "xrr-qt-cocoa-inputs-v1",
    "target": "macos-arm64-py312",
    "python_version": "3.12",
    "pip_version": "26.1.2",
    "qt_version": "6.11.2",
    "wheel_tag": "cp310-abi3-macosx_13_0_arm64",
    "wheel_build": "1xrrcocoa",
}


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Qt source path is not canonical and relative")
    if re.fullmatch(r"[A-Za-z0-9_.\-/]+", value) is None or not path.parts:
        raise ValueError("Qt source path contains unsupported characters")
    return value


def _record(record: dict, fields: set[str]) -> None:
    if set(record) != fields:
        raise ValueError("Qt input record schema differs")
    filename = record["filename"]
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Qt input filename must be a direct child")
    if re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
        raise ValueError("Qt input digest is not canonical SHA-256")
    if type(record["size"]) is not int or not 0 < record["size"] <= 128 * 1024**2:
        raise ValueError("Qt input size is outside the bounded download limit")


def _source_records(sources: list) -> None:
    if not isinstance(sources, list) or not sources:
        raise ValueError("Qt source manifest must not be empty")
    paths = []
    for source in sources:
        _record(source, {"path", "filename", "url", "sha256", "size"})
        path = _safe_path(source["path"])
        paths.append(path)
        if source["url"] != f"{SOURCE_BASE}{path}?h=v6.11.2":
            raise ValueError("Qt source must use the pinned upstream tag and subtree")
        if source["filename"] != "source-" + path.replace("/", "__"):
            raise ValueError("Qt source cache filename differs from its path")
    if len(set(paths)) != len(paths):
        raise ValueError("Qt source manifest contains duplicate paths")


def _recipes(root: Path, records: list) -> None:
    if {item["path"] for item in records} != set(RECIPES) or len(records) != len(RECIPES):
        raise ValueError("Qt build recipe set differs")
    for record in records:
        if set(record) != {"path", "sha256"}:
            raise ValueError("Qt recipe identity fields differ")
        _verified_file(root / record["path"], record["sha256"])


def _upstream_binding(root: Path, record: dict) -> None:
    ordinary = json.loads((root / "tools/package-manifests/macos-arm64-py312.json").read_text(encoding="utf-8"))
    upstream = [wheel for wheel in ordinary["wheels"] if wheel["name"] == "pyside6-essentials"]
    if upstream != [record] or record["version"] != "6.11.2":
        raise ValueError("Qt derivation upstream wheel differs from the ordinary manifest")


def read_inputs(root: Path) -> dict:
    path = root / MANIFEST
    if path.is_symlink() or not path.is_file():
        raise ValueError("Qt input manifest must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {*IDENTITY, "sdk", "sources", "recipes", "upstream_wheel"}:
        raise ValueError("Qt input manifest schema differs")
    if any(value[key] != expected for key, expected in IDENTITY.items()):
        raise ValueError("Qt input target, version or derived wheel identity differs")
    _record(value["sdk"], {"filename", "url", "sha256", "size"})
    if value["sdk"]["url"] != SDK_URL or value["sdk"]["filename"] != "qtbase-sdk-6.11.2.7z":
        raise ValueError("Qt SDK must use the pinned official build")
    _source_records(value["sources"])
    _recipes(root, value["recipes"])
    _upstream_binding(root, value["upstream_wheel"])
    return value


def input_records(manifest: dict) -> list[dict]:
    return sorted([manifest["sdk"], *manifest["sources"]], key=lambda item: item["filename"])


def verify_inputs(root: Path, manifest: dict, directory: Path) -> list[dict]:
    if read_inputs(root) != manifest:
        raise ValueError("Qt source input bindings changed")
    expected = input_records(manifest)
    observed = verify_wheels(directory, expected)
    if [item["size"] for item in observed] != [item["size"] for item in expected]:
        raise ValueError("Qt input sizes differ from the manifest")
    return observed


def input_cache_key(manifest: dict, trust_domain: str) -> str:
    header = {**manifest, "lock_sha256": hashlib.sha256(manifest_bytes(manifest)).hexdigest()}
    return cache_key(header, trust_domain).replace("xrr-wheels-v1-", "xrr-qt-cocoa-inputs-v1-", 1)


def _fetch_file(record: dict, destination: Path) -> None:
    request = urllib.request.Request(record["url"], headers={"User-Agent": "xrr-qt-cocoa-inputs/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read(record["size"] + 1)
    if len(content) != record["size"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError("downloaded Qt input differs from the pinned bytes")
    with destination.open("xb") as handle:
        handle.write(content)


def _download(root: Path, manifest: dict, report: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="qt-input-stage-", dir=report) as temporary:
        stage = Path(temporary)
        records = input_records(manifest)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_fetch_file, record, stage / record["filename"]) for record in records]
            for future in futures:
                future.result()
        verify_inputs(root, manifest, stage)
        copy_verified(stage, report / "inputs", records)


def cached_inputs(root: Path, report: Path, cache: Path, trust_domain: str) -> int:
    manifest = read_inputs(root)
    key = input_cache_key(manifest, trust_domain)
    cache = _cache_directory(root, cache, report)
    identity = _directory_identity(cache, "Qt input cache")
    cached = cache / hashlib.sha256(key.encode("ascii")).hexdigest()
    _cache_members(cache, cached.name)
    hit = os.path.lexists(cached)
    if hit:
        verify_inputs(root, manifest, cached)
    report, guard, _environment = prepare_report(root, report)
    _write(report, "build-inputs.json", manifest_bytes(manifest).decode("ascii"))
    if hit:
        copy_verified(cached, report / "inputs", input_records(manifest))
    else:
        _download(root, manifest, report)
        guard()
        _require_same_directory(cache, identity, "Qt input cache")
        copy_verified(report / "inputs", cached, input_records(manifest))
    guard()
    _require_same_directory(cache, identity, "Qt input cache")
    _cache_members(cache, cached.name)
    verify_inputs(root, manifest, cached)
    verify_inputs(root, manifest, report / "inputs")
    _write(
        report,
        "cache.json",
        manifest_bytes({"state": "PASS", "key": key, "hit": hit, "trust_domain": trust_domain}).decode("ascii"),
    )
    _summary(report, "qt-cocoa-inputs", 0, derived_wheel_reused=False)
    return 0
