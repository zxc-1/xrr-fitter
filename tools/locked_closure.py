#!/usr/bin/env python3
"""Verify the frozen Windows dependency graph and wheel bytes, without resolution."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from installed_inputs import InputBindings, json_object  # noqa: E402
from lock_windows_environment import read_windows_dependencies  # noqa: E402
from package_downloads import download_packages  # noqa: E402
from package_manifest import validate_manifest, verify_wheels  # noqa: E402
from package_sbom import dependency_edges  # noqa: E402
from wheel_inventory import inspect_wheel  # noqa: E402

TARGET = "windows-x64-py312"
MANIFEST = f"tools/package-manifests/{TARGET}.json"
INPUTS = ("pyproject.toml", f"requirements-{TARGET}.lock", MANIFEST)
ROOT_NODE = "<project-declarations>"


def _reachable(edges: dict[str, list[str]]) -> set[str]:
    pending = list(edges[ROOT_NODE])
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(edges[name])
    return seen


def _metadata_coverage(manifest: dict, inventories: list[dict]) -> None:
    expected = {item["name"]: item["version"] for item in manifest["wheels"]}
    observed = {item["name"]: item["version"] for item in inventories}
    if observed != expected or len(observed) != len(inventories):
        raise ValueError("closure metadata must cover exactly the frozen wheel manifest")


def validate_closure(root: Path, manifest: dict, inventories: list[dict]) -> dict:
    validate_manifest(root, manifest)
    if manifest["target"] != TARGET or manifest["vcs"]:
        raise ValueError("frozen closure requires the Windows wheel target")
    _metadata_coverage(manifest, inventories)
    groups = read_windows_dependencies(root / "pyproject.toml")
    requirements = [requirement for group in groups for requirement in group]
    roots = {"name": ROOT_NODE, "version": "0", "requirements": requirements}
    edges = dependency_edges([*inventories, roots], TARGET)
    if _reachable(edges) != set(edges) - {ROOT_NODE}:
        raise ValueError("frozen closure contains packages unreachable from project declarations")
    declared = edges.pop(ROOT_NODE)
    return {
        "state": "PASS",
        "validation": "frozen-wheel-closure",
        "target": TARGET,
        "wheel_count": len(inventories),
        "roots": declared,
        "dependencies": edges,
    }


def _verify_directory(root: Path, manifest: dict, directory: Path) -> dict:
    before = verify_wheels(directory, manifest["wheels"])
    inventories = [inspect_wheel(directory / item["filename"], item) for item in manifest["wheels"]]
    result = validate_closure(root, manifest, inventories)
    if verify_wheels(directory, manifest["wheels"]) != before:
        raise ValueError("frozen wheel inputs changed during closure validation")
    return result


def verify_locked_closure(root: Path, wheel_dir: Path | None = None) -> dict:
    root = root.resolve()
    inputs = InputBindings({name: root / name for name in INPUTS}, limit=2 * 1024**2)
    manifest = validate_manifest(root, json_object(inputs.contents[MANIFEST]))
    if wheel_dir is None:
        with tempfile.TemporaryDirectory(prefix="xrr-frozen-closure-") as temporary:
            report = Path(temporary) / "download"
            if download_packages(root, root / MANIFEST, report) != 0:
                raise ValueError("frozen wheel download failed; no dependency resolution fallback")
            inputs.guard()
            result = _verify_directory(root, manifest, report / "wheels")
    else:
        result = _verify_directory(root, manifest, wheel_dir)
    inputs.guard()
    return {"schema": "xrr-frozen-closure-v1", **result, "input_sha256": inputs.hashes()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--wheel-dir", type=Path, help="verify existing frozen wheels without network access")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_locked_closure(args.repo_root, args.wheel_dir), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
