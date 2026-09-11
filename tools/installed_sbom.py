#!/usr/bin/env python3
"""Emit source-bound SBOM evidence from verified installed bytes, not RECORD claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from artifact_binding import bind_distribution  # noqa: E402
from audit_reports import _write  # noqa: E402
from distribution_source import clean_head_identity  # noqa: E402
from executable_binding import bind_executable  # noqa: E402
from installed_inputs import load_installation_inputs  # noqa: E402
from installed_inventory import current_layout, inspect_installation  # noqa: E402
from lock_sbom import canonical_sbom_bytes  # noqa: E402
from native_inventory import build_native_inventory, wheel_file_reference  # noqa: E402
from package_downloads import prepare_report  # noqa: E402
from package_sbom import _file_component, dependency_edges  # noqa: E402


def _properties(values: dict) -> list[dict]:
    return [{"name": key, "value": str(value)} for key, value in sorted(values.items())]


def _wheel_ref(wheel: dict) -> str:
    return f"urn:xrr:archive:{wheel['sha256']}"


def _package_component(package: dict, native: dict) -> dict:
    wheel, inventory = package["wheel"], package["inventory"]
    result = {
        "type": "library",
        "bom-ref": _wheel_ref(wheel),
        "name": wheel["name"],
        "version": wheel["version"],
        "hashes": [{"alg": "SHA-256", "content": wheel["sha256"]}],
        "properties": _properties(
            {
                "xrr:installed:provenance": package["provenance"],
                "xrr:metadata:requires-dist": json.dumps(inventory["requirements"]),
            }
        ),
        "components": [_file_component(wheel, file, native["files"]) for file in inventory["files"]],
    }
    if package["provenance"] != "refnx-source-build":
        result["purl"] = f"pkg:pypi/{wheel['name']}@{quote(wheel['version'], safe='')}"
    if "url" in wheel:
        result["externalReferences"] = [{"type": "distribution", "url": wheel["url"]}]
    if inventory["licenses"]:
        result["licenses"] = inventory["licenses"]
    return result


def _installed_component(file: dict) -> dict:
    return {
        "type": "file",
        "bom-ref": f"urn:xrr:installed:{quote(file['path'], safe='')}",
        "name": file["path"],
        "hashes": [{"alg": "SHA-256", "content": file["sha256"]}],
        "properties": _properties(
            {
                "xrr:installed:size": file["size"],
                "xrr:installed:claims": json.dumps(file["claims"], sort_keys=True),
                "xrr:installed:ownership": "shared" if len(file["claims"]) > 1 else "single",
            }
        ),
    }


def _derivation(file: dict, packages: dict, installer: str) -> dict:
    targets = {f"urn:xrr:archive:{installer}"}
    for claim in file["claims"]:
        package = packages.get(claim["wheel_sha256"])
        if package is None or package["wheel"]["name"] != claim["package"]:
            raise ValueError("installed claim refers to an unbound wheel")
        targets.add(_wheel_ref(package["wheel"]))
        if claim["source"] in {item["path"] for item in package["inventory"]["files"]}:
            targets.add(wheel_file_reference(package["wheel"], claim["source"]))
    return {"ref": _installed_component(file)["bom-ref"], "dependsOn": sorted(targets)}


def _installed_metadata(inventory: dict, bindings: dict, source: dict) -> dict:
    return {
        "xrr:inventory:scope": "verified-installed-environment-not-minimal-artifact-runtime",
        "xrr:inventory:sha256": hashlib.sha256(canonical_sbom_bytes(inventory)).hexdigest(),
        "xrr:inventory:target": inventory["target"],
        "xrr:inventory:python": inventory["python_version"],
        "xrr:inputs:sha256": json.dumps(bindings, sort_keys=True),
        "xrr:source:commit": source["source_commit"],
        "xrr:source:tree": source["source_tree"],
        "xrr:composition:boundary": inventory["boundary"],
        "xrr:installed:unowned": json.dumps(inventory["unowned"], sort_keys=True),
        "xrr:installed:compile-failures": json.dumps(inventory["compile_failures"], sort_keys=True),
        "xrr:installed:compile-warnings": json.dumps(inventory["compile_warnings"], sort_keys=True),
        "xrr:native:resolution-scope": "archive-path-declarations-not-runtime-loads",
    }


def _installed_dependencies(inventory, archives, native, by_name, by_hash) -> list[dict]:
    edges = dependency_edges(archives, inventory["target"])
    return (
        [
            {"ref": _wheel_ref(by_name[name]), "dependsOn": [_wheel_ref(by_name[dep]) for dep in deps]}
            for name, deps in edges.items()
        ]
        + native["dependencies"]
        + [_derivation(file, by_hash, inventory["installer_wheel_sha256"]) for file in inventory["files"]]
    )


def assemble_installed_sbom(inventory: dict, bindings: dict, source: dict) -> dict:
    packages = inventory["packages"]
    wheels = [package["wheel"] for package in packages]
    archives = [package["inventory"] for package in packages]
    by_hash = {package["wheel"]["sha256"]: package for package in packages}
    by_name = {wheel["name"]: wheel for wheel in wheels}
    if len(by_hash) != len(packages) or inventory["installer_wheel_sha256"] not in by_hash:
        raise ValueError("installed SBOM has ambiguous wheel or installer identities")
    native = build_native_inventory(wheels, archives, inventory["target"])
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {"properties": _properties(_installed_metadata(inventory, bindings, source))},
        "components": [_package_component(package, native) for package in packages]
        + [_installed_component(file) for file in inventory["files"]],
        "dependencies": _installed_dependencies(inventory, archives, native, by_name, by_hash),
        "compositions": [{"aggregate": "incomplete"}],
    }


def _artifact_reports(args, root, inputs, inventory, bom, source):
    reports, guards = {}, []
    if bool(args.artifact_manifest) != bool(args.artifact_dir):
        raise ValueError("artifact binding requires both its manifest and directory")
    if bool(args.executable) != bool(args.executable_evidence):
        raise ValueError("executable binding requires both actual bytes and execution evidence")
    if args.artifact_manifest:
        artifact_bom, guard = bind_distribution(root, args.artifact_manifest, args.artifact_dir, inventory, bom, source)
        reports["artifact.cdx.json"] = artifact_bom
        guards.append(guard)
    if args.executable:
        executable_inventory, executable_bom, guard = bind_executable(
            root, args.executable, args.executable_evidence, inputs, inventory, source
        )
        reports.update({"executable-inventory.json": executable_inventory, "executable.cdx.json": executable_bom})
        guards.append(guard)
    return reports, guards


def report_installation(args) -> dict:
    root = args.repo_root.resolve()
    report, guard, _environment = prepare_report(root, args.report_dir)
    try:
        identity = clean_head_identity(root)
        source = {"source_commit": identity.head_commit, "source_tree": identity.head_tree}
        manifest, inputs, pip, bindings, input_guard = load_installation_inputs(
            root,
            args.manifest,
            args.wheel_dir,
            args.pip_wheel,
            refnx_build=args.refnx_build,
            auxiliary_directory=args.auxiliary_dir,
        )
        layout = current_layout(manifest["target"])
        inventory = inspect_installation(layout, inputs, pip, native_loaders=True)
        bom = assemble_installed_sbom(inventory, bindings, source)
        reports, artifact_guards = _artifact_reports(args, root, inputs, inventory, bom, source)
        reports.update({"inventory.json": inventory, "installed.cdx.json": bom})
        input_guard()
        for artifact_guard in artifact_guards:
            artifact_guard()
        if clean_head_identity(root) != identity:
            raise ValueError("source identity changed during installation inspection")
        guard()
        hashes = {}
        for name, value in reports.items():
            content = canonical_sbom_bytes(value)
            _write(report, name, content.decode())
            hashes[name] = hashlib.sha256(content).hexdigest()
        summary = {
            "state": "PASS",
            **source,
            "files": len(inventory["files"]),
            "packages": len(inputs),
            "inventory_sha256": hashlib.sha256(canonical_sbom_bytes(inventory)).hexdigest(),
            "sbom_sha256": hashlib.sha256(canonical_sbom_bytes(bom)).hexdigest(),
            "aggregate": "incomplete",
            "reports": hashes,
        }
        _write(report, "summary.json", json.dumps(summary, sort_keys=True, indent=2) + "\n")
        return summary
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        guard()
        _write(report, "failure.json", json.dumps({"state": "FAIL", "error": str(error)}, sort_keys=True) + "\n")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("manifest", "wheel-dir", "pip-wheel", "report-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--refnx-build", type=Path)
    parser.add_argument("--auxiliary-dir", type=Path)
    for name in ("artifact-manifest", "artifact-dir", "executable", "executable-evidence"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = report_installation(args)
        print(json.dumps(summary, sort_keys=True))
        if args.require_complete:
            raise ValueError("installed SBOM retains incomplete interpreter/native/runtime composition")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
