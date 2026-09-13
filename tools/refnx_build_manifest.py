"""Dedicated refnx builder closure, separate from the application wheel lock."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402

from lock_audit_tools import _validate_closure, _validate_constraints, render_lock  # noqa: E402
from lock_environment import PIP_VERSION  # noqa: E402
from lock_sbom import _read_input  # noqa: E402
from package_manifest import _report_wheel, _validate_wheel, verify_wheels  # noqa: E402
from package_sbom import _marker_environment  # noqa: E402
from vcs_source import _validate_manifest, verify_source  # noqa: E402
from wheel_inventory import inspect_wheel  # noqa: E402

MANIFEST = Path("tools/package-manifests/refnx-build-macos-arm64-py312.json")
SOURCE = Path("tools/package-manifests/refnx-source.json")
REQUIREMENTS = Path("tools/refnx-build-requirements.in")
LOCK = Path("tools/refnx-build-requirements.lock")
TARGET = "macos-arm64-py312"


def source_manifest(root: Path) -> dict:
    value = json.loads(_read_input(root / SOURCE))
    _validate_manifest(root, value)
    return value


def build_requirements(root: Path) -> list[str]:
    values = _read_input(root / REQUIREMENTS).decode("ascii").splitlines()
    names = []
    for value in values:
        requirement = Requirement(value)
        pins = tuple(requirement.specifier)
        if requirement.url or requirement.marker or requirement.extras or len(pins) != 1:
            raise ValueError("refnx build requirements must have exact unconditional pins")
        if pins[0].operator != "==" or "*" in pins[0].version:
            raise ValueError("refnx build requirements must have exact version pins")
        names.append(canonicalize_name(requirement.name))
    if not names or names != sorted(set(names)):
        raise ValueError("refnx build requirements must be uniquely sorted")
    return values


def _constraints(root: Path) -> list[str]:
    return _read_input(root / f"requirements-{TARGET}.lock").decode("utf-8").splitlines()


def _header(root: Path, lock: bytes) -> dict:
    source_manifest(root)
    return {
        "schema": "xrr-refnx-build-inputs-v1",
        "target": TARGET,
        "python_version": "3.12",
        "pip_version": PIP_VERSION,
        "lock_sha256": hashlib.sha256(lock).hexdigest(),
        "requirements_sha256": hashlib.sha256(_read_input(root / REQUIREMENTS)).hexdigest(),
        "application_lock_sha256": hashlib.sha256(_read_input(root / f"requirements-{TARGET}.lock")).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(_read_input(root / SOURCE)).hexdigest(),
    }


def build_inputs(root: Path, report: dict) -> tuple[dict, bytes]:
    lock = render_lock(report, build_requirements(root), _constraints(root))
    wheels = sorted((_report_wheel(item, TARGET) for item in report["install"]), key=lambda item: item["name"])
    return {**_header(root, lock), "wheels": wheels}, lock


def _locked_wheels(root: Path, wheels: list[dict], lock: bytes) -> None:
    for record in wheels:
        _validate_wheel(record, TARGET)
    names = [item["name"] for item in wheels]
    if names != sorted(set(names)):
        raise ValueError("refnx builder wheels must be uniquely sorted")
    expected = "".join(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n" for item in wheels)
    if expected.encode("ascii") != lock:
        raise ValueError("refnx builder wheel hashes differ from the lock")
    entries = {item["name"]: (item["version"], item["sha256"]) for item in wheels}
    _validate_constraints(entries, _constraints(root))
    for value in build_requirements(root):
        requirement = Requirement(value)
        name = canonicalize_name(requirement.name)
        if name not in entries or not requirement.specifier.contains(entries[name][0]):
            raise ValueError("refnx builder lock is missing a direct build requirement")


def read_inputs(root: Path) -> dict:
    value = json.loads(_read_input(root / MANIFEST))
    lock = _read_input(root / LOCK)
    if {key: item for key, item in value.items() if key != "wheels"} != _header(root, lock):
        raise ValueError("refnx build input bindings or schema changed")
    if not isinstance(value.get("wheels"), list):
        raise ValueError("refnx builder wheels must be an array")
    _locked_wheels(root, value["wheels"], lock)
    return value


def input_records(root: Path, manifest: dict) -> list[dict]:
    source = source_manifest(root)
    return [*manifest["wheels"], {"filename": "refnx-source.tar.gz", "sha256": source["archive"]["sha256"]}]


def verify_inputs(root: Path, manifest: dict, directory: Path) -> list[dict]:
    records = input_records(root, manifest)
    observed = verify_wheels(directory, records)
    verify_source(root, source_manifest(root), directory / "refnx-source.tar.gz")
    inventories = [inspect_wheel(directory / item["filename"], item) for item in manifest["wheels"]]
    entries = {item["name"]: (item["version"], item["sha256"]) for item in manifest["wheels"]}
    metadata = {item["name"]: {"requires_dist": item["requirements"]} for item in inventories}
    _validate_closure(entries, metadata, build_requirements(root), _marker_environment(TARGET))
    if verify_wheels(directory, records) != observed or read_inputs(root) != manifest:
        raise ValueError("refnx inputs changed during verification")
    return observed
