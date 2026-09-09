#!/usr/bin/env python3
"""Emit byte-bound CycloneDX evidence for verified target wheel archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402

from lock_sbom import TARGETS, canonical_sbom_bytes  # noqa: E402
from package_downloads import read_manifest  # noqa: E402
from package_manifest import manifest_bytes, verify_wheels  # noqa: E402
from wheel_inventory import inspect_wheel  # noqa: E402


def _marker_environment(target: str) -> dict[str, str]:
    if target not in TARGETS:
        raise ValueError("unsupported SBOM target")
    windows = target == "windows-x64-py312"
    return {
        "os_name": "nt" if windows else "posix",
        "sys_platform": "win32" if windows else "darwin",
        "platform_system": "Windows" if windows else "Darwin",
        "platform_machine": "AMD64" if windows else "arm64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "python_version": "3.12",
        "python_full_version": "3.12.0",
        "implementation_version": "3.12.0",
        "platform_release": "unspecified",
        "platform_version": "unspecified",
    }


def _active_requirements(inventory: dict, extra: str, environment: dict) -> list[Requirement]:
    result = []
    for value in inventory["requirements"]:
        requirement = Requirement(value)
        if requirement.marker is None or requirement.marker.evaluate({**environment, "extra": extra}):
            result.append(requirement)
    return result


def _dependency_name(requirement: Requirement, inventories: dict) -> str:
    name = canonicalize_name(requirement.name)
    if requirement.url or name not in inventories:
        raise ValueError(f"dependency is missing or not bound to a wheel: {requirement}")
    if not requirement.specifier.contains(inventories[name]["version"], prereleases=True):
        raise ValueError(f"dependency version is incompatible: {requirement}")
    return name


def dependency_edges(inventories: list[dict], target: str) -> dict[str, list[str]]:
    by_name = {item["name"]: item for item in inventories}
    if len(by_name) != len(inventories):
        raise ValueError("duplicate dependency metadata")
    environment = _marker_environment(target)
    edges = {name: set() for name in by_name}
    pending = [(name, "") for name in by_name]
    seen = set()
    while pending:
        name, extra = pending.pop()
        if (name, extra) in seen:
            continue
        seen.add((name, extra))
        for requirement in _active_requirements(by_name[name], extra, environment):
            dependency = _dependency_name(requirement, by_name)
            edges[name].add(dependency)
            pending.extend((dependency, value) for value in requirement.extras)
    return {name: sorted(values) for name, values in sorted(edges.items())}


def _purl(item: dict) -> str:
    return f"pkg:pypi/{item['name']}@{quote(item['version'], safe='')}"


def _properties(values: dict) -> list[dict[str, str]]:
    return [{"name": name, "value": str(value)} for name, value in sorted(values.items())]


def _file_component(wheel: dict, file: dict) -> dict:
    properties = {"xrr:archive:path": file["path"], "xrr:file:kind": file["kind"], "xrr:file:size": file["size"]}
    if "format" in file:
        properties["xrr:native:header-format"] = file["format"]
    return {
        "type": "file",
        "bom-ref": f"urn:xrr:wheel:{wheel['sha256']}:{quote(file['path'], safe='')}",
        "name": file["path"],
        "hashes": [{"alg": "SHA-256", "content": file["sha256"]}],
        "properties": _properties(properties),
    }


def _vendored_component(wheel: dict, inventory: dict) -> dict:
    result = {
        "type": "library",
        "bom-ref": f"urn:xrr:vendor:{wheel['sha256']}:{quote(inventory['metadata_path'], safe='')}",
        "name": inventory["name"],
        "version": inventory["version"],
        "purl": _purl(inventory),
        "properties": _properties(
            {
                "xrr:vendored:metadata-path": inventory["metadata_path"],
                "xrr:vendored:metadata-sha256": inventory["metadata_sha256"],
                "xrr:vendored:dependency-relationships": "unresolved",
                "xrr:metadata:requires-dist": json.dumps(inventory["requirements"]),
            }
        ),
    }
    if inventory["licenses"]:
        result["licenses"] = inventory["licenses"]
    return result


def _package_component(wheel: dict, inventory: dict) -> dict:
    result = {
        "type": "library",
        "bom-ref": _purl(wheel),
        "purl": _purl(wheel),
        "name": wheel["name"],
        "version": wheel["version"],
        "hashes": [{"alg": "SHA-256", "content": wheel["sha256"]}],
        "externalReferences": [{"type": "distribution", "url": wheel["url"]}],
        "properties": _properties({"xrr:metadata:requires-dist": json.dumps(inventory["requirements"])}),
        "components": [_file_component(wheel, file) for file in inventory["files"]],
    }
    if inventory["licenses"]:
        result["licenses"] = inventory["licenses"]
    result["components"].extend(_vendored_component(wheel, item) for item in inventory.get("vendored", ()))
    return result


def _inventory_properties(manifest: dict, inventories: list[dict]) -> list[dict]:
    native_count = sum(file["kind"] == "native" for item in inventories for file in item["files"])
    properties = {
        "xrr:inventory:scope": "verified-wheel-archives",
        "xrr:inventory:target": manifest["target"],
        "xrr:inventory:files": "complete",
        "xrr:inventory:native-files": native_count,
        "xrr:inventory:native-relationships": "unresolved" if native_count else "not-observed",
        "xrr:inventory:installed-distribution": "not-inspected",
        "xrr:inventory:vcs-excluded": json.dumps(manifest["vcs"], sort_keys=True),
        "xrr:markers:environment": json.dumps(_marker_environment(manifest["target"]), sort_keys=True),
        "xrr:lock:sha256": manifest["lock_sha256"],
        "xrr:package-manifest:sha256": hashlib.sha256(manifest_bytes(manifest)).hexdigest(),
    }
    return _properties(properties)


def assemble_sbom(manifest: dict, inventories: list[dict]) -> dict:
    by_name = {item["name"]: item for item in inventories}
    wheels = {item["name"]: item for item in manifest["wheels"]}
    if {name: item["version"] for name, item in by_name.items()} != {
        name: item["version"] for name, item in wheels.items()
    }:
        raise ValueError("SBOM metadata does not cover the exact wheel manifest")
    edges = dependency_edges(inventories, manifest["target"])
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {"properties": _inventory_properties(manifest, inventories)},
        "components": [_package_component(wheels[name], by_name[name]) for name in sorted(wheels)],
        "dependencies": [
            {"ref": _purl(wheels[name]), "dependsOn": [_purl(wheels[dependency]) for dependency in dependencies]}
            for name, dependencies in edges.items()
        ],
        "compositions": [{"aggregate": "incomplete"}],
    }


def build_package_sbom(root: Path, manifest_path: Path, wheel_directory: Path) -> dict:
    manifest = read_manifest(root, manifest_path)
    before = verify_wheels(wheel_directory, manifest["wheels"])
    inventories = [inspect_wheel(wheel_directory / item["filename"], item) for item in manifest["wheels"]]
    result = assemble_sbom(manifest, inventories)
    if read_manifest(root, manifest_path) != manifest or verify_wheels(wheel_directory, manifest["wheels"]) != before:
        raise ValueError("SBOM inputs changed during inventory generation")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_package_sbom(args.repo_root, args.manifest, args.wheel_dir)
        sys.stdout.buffer.write(canonical_sbom_bytes(report))
        if args.require_complete and report["compositions"] != [{"aggregate": "complete"}]:
            raise ValueError("package archive SBOM cannot establish complete installed/native composition")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
