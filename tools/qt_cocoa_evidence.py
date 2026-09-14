"""Bind the Qt derivative to original sources, the recipe and native verification."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from distribution_manifest import _json_object  # noqa: E402
from installed_files import InstalledSnapshot, VerifiedWheel  # noqa: E402
from package_manifest import _verified_file, manifest_bytes, verify_wheels  # noqa: E402
from qt_cocoa_inputs import MANIFEST, RECIPES, read_inputs  # noqa: E402
from qt_cocoa_wheel import ADDITIONS, verify_derivation  # noqa: E402

BUILD_CODE = (
    "tools/qt_cocoa_inputs.py",
    "tools/qt_cocoa_wheel.py",
    "tools/qt_cocoa_build.py",
    "tools/qt_cocoa_evidence.py",
    "tools/refnx_build_runtime.py",
    "tools/package_downloads.py",
    "tools/package_manifest.py",
    "tools/package_cache.py",
    "tools/installed_files.py",
    "tools/installed_transforms.py",
    "tools/wheel_inventory.py",
    "tools/native_binary.py",
    "tools/native_bytes.py",
    "tools/native_pe.py",
    "tools/audit_reports.py",
    "tools/verify_report.py",
    "tools/verify_publish.py",
    "tools/distribution_manifest.py",
)
CASES = (
    "borrowed-destructor",
    "snapshot-invalidation",
    "placeholder-invalidation",
    "materialized-cell",
    "native-invalidation",
    "repeated-lifecycle",
)


def _read(path: Path) -> bytes:
    snapshot = InstalledSnapshot(path.absolute().parent)
    content = snapshot.read(path, limit=32 * 1024**2)
    snapshot.finish()
    return content


def code_bindings(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256(_read(root / name)).hexdigest() for name in BUILD_CODE}


def binding_paths(root: Path, receipt_path: Path) -> dict[str, Path]:
    return {
        "qt-cocoa-build": receipt_path,
        "qt-cocoa-inputs": root / MANIFEST,
        **{name: root / name for name in (*BUILD_CODE, *RECIPES)},
    }


def source_additions(root: Path, manifest: dict, read_source) -> dict[str, bytes]:
    additions = {ADDITIONS + "inputs.json": manifest_bytes(manifest)}
    for record in manifest["sources"]:
        content = read_source(record)
        if len(content) != record["size"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Qt source bytes differ from the pinned input")
        additions[ADDITIONS + "source/" + record["path"]] = content
    for record in manifest["recipes"]:
        path = root / record["path"]
        content = _read(path)
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Qt recipe bytes differ from the pinned input")
        additions[ADDITIONS + "recipe/" + path.name] = content
    return additions


def _native_results(regressions: dict) -> None:
    if not isinstance(regressions, dict) or set(regressions) != set(CASES):
        raise ValueError("Qt native regression coverage differs")
    if any(type(code) is not int or code != 0 for code in regressions.values()):
        raise ValueError("Qt source build requires all native regressions to pass")


def build_receipt(
    root: Path, manifest: dict, derived_record: dict, derivation: dict, toolchain: dict, regressions: dict
) -> dict:
    _native_results(regressions)
    if toolchain.get("machine") != "arm64":
        raise ValueError("Qt source build requires the native arm64 toolchain")
    return copy.deepcopy(
        {
            "schema": "xrr-qt-cocoa-build-v1",
            "state": "PASS",
            "inputs": manifest,
            "wheel": derived_record,
            "derivation": derivation,
            "builder_sha256": code_bindings(root),
            "toolchain": toolchain,
            "native_regressions": regressions,
            "offline": True,
            "derived_wheel_reused": False,
        }
    )


def _receipt_contract(receipt: dict) -> None:
    fields = {
        "schema",
        "state",
        "inputs",
        "wheel",
        "derivation",
        "builder_sha256",
        "toolchain",
        "native_regressions",
        "offline",
        "derived_wheel_reused",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise ValueError("Qt source build receipt fields differ")
    if receipt["schema"] != "xrr-qt-cocoa-build-v1" or receipt["state"] != "PASS":
        raise ValueError("Qt source build report is not a successful supported receipt")
    if receipt["offline"] is not True or receipt["derived_wheel_reused"] is not False:
        raise ValueError("Qt installation requires a fresh offline source build")
    _native_results(receipt["native_regressions"])
    if not isinstance(receipt["toolchain"], dict) or receipt["toolchain"].get("machine") != "arm64":
        raise ValueError("Qt build toolchain target differs")


def _receipt_identity(root: Path, receipt: dict) -> dict:
    _receipt_contract(receipt)
    manifest = read_inputs(root)
    if receipt["inputs"] != manifest or receipt["builder_sha256"] != code_bindings(root):
        raise ValueError("Qt source or builder bindings changed")
    return manifest


def read_build(root: Path, report: Path, upstream_directory: Path) -> dict:
    receipt_bytes = _read(report)
    receipt = json.loads(receipt_bytes, object_pairs_hook=_json_object)
    manifest = _receipt_identity(root, receipt)
    record = receipt["wheel"]
    directory = report.parent / "wheels"
    verify_wheels(directory, [record])
    path = directory / record["filename"]
    with VerifiedWheel(path, record) as target:
        additions = source_additions(root, manifest, lambda item: target.read(ADDITIONS + "source/" + item["path"]))
    upstream = upstream_directory / manifest["upstream_wheel"]["filename"]
    _verified_file(upstream, manifest["upstream_wheel"]["sha256"])
    derivation = verify_derivation(upstream, manifest["upstream_wheel"], path, record, additions)
    if receipt["derivation"] != derivation:
        raise ValueError("Qt build derivation differs from the actual wheel bytes")
    _receipt_identity(root, receipt)
    if _read(report) != receipt_bytes:
        raise ValueError("Qt build receipt changed during verification")
    return {"record": record, "path": path, "bindings": binding_paths(root, report), "receipt": receipt}
