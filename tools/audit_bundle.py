#!/usr/bin/env python3
"""Bind exact-pin advisory results to verified installed and artifact byte evidence."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402

from audit_reports import _input_identity, _write, advisory_findings, advisory_input  # noqa: E402
from distribution_source import clean_head_identity  # noqa: E402
from installed_inputs import InputBindings, json_object  # noqa: E402
from lock_sbom import TARGETS, canonical_sbom_bytes  # noqa: E402
from package_downloads import prepare_report  # noqa: E402
from package_manifest import validate_manifest  # noqa: E402

CORE_REPORTS = {"inventory.json", "installed.cdx.json"}
EXECUTABLE_REPORTS = {"executable-inventory.json", "executable.cdx.json"}
ALLOWED_REPORTS = CORE_REPORTS | EXECUTABLE_REPORTS | {"artifact.cdx.json"}


def _installed_status(summary: dict, source: dict) -> None:
    expected = {"state": "PASS", "aggregate": "incomplete", **source}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("installed report status, composition or source binding differs")


def _require_artifact_pair(names: set[str]) -> None:
    executable = names & EXECUTABLE_REPORTS
    if executable and executable != EXECUTABLE_REPORTS:
        raise ValueError("executable inventory and SBOM must be bound together")
    if not executable and "artifact.cdx.json" not in names:
        raise ValueError("installed report lacks an artifact or executable binding")


def _installed_report_names(summary: dict, source: dict) -> set[str]:
    _installed_status(summary, source)
    reports = summary["reports"]
    if not isinstance(reports, dict) or not CORE_REPORTS <= reports.keys() <= ALLOWED_REPORTS:
        raise ValueError("installed report filenames are missing or unsafe")
    names = set(reports)
    _require_artifact_pair(names)
    return names


def _installed_hashes(bound: InputBindings, summary: dict, names: set[str]) -> None:
    hashes = bound.hashes()
    if {name: hashes[name] for name in names} != summary["reports"]:
        raise ValueError("installed report bytes differ from the summary hashes")
    if (hashes["inventory.json"], hashes["installed.cdx.json"]) != (
        summary["inventory_sha256"],
        summary["sbom_sha256"],
    ):
        raise ValueError("installed inventory/SBOM summary hashes differ")


def _incomplete_compositions(documents: dict) -> None:
    for name, value in documents.items():
        if name.endswith(".cdx.json") and value.get("compositions") != [{"aggregate": "incomplete"}]:
            raise ValueError("artifact composition must remain explicitly incomplete")


def _require_fields(value: dict, expected: dict, label: str) -> None:
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"{label} binding differs")


def _bom_properties(value: dict, source: dict) -> dict:
    _require_fields(value, {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1}, "SBOM schema")
    rows = value["metadata"]["properties"]
    properties = {item["name"]: item["value"] for item in rows}
    if len(properties) != len(rows):
        raise ValueError("duplicate SBOM source/input properties")
    _require_fields(
        properties,
        {"xrr:source:commit": source["source_commit"], "xrr:source:tree": source["source_tree"]},
        "SBOM source",
    )
    return properties


def _executable_bindings(documents: dict, properties: dict, hashes: dict) -> None:
    _require_fields(
        documents["executable-inventory.json"],
        {
            "schema": "xrr-executable-inventory-v1",
            "aggregate": "incomplete",
            "target": documents["inventory.json"]["target"],
        },
        "executable inventory schema/target",
    )
    _require_fields(properties, {"xrr:inventory:sha256": hashes["executable-inventory.json"]}, "executable inventory")
    inputs = json_object(properties["xrr:inputs:sha256"].encode("utf-8"))
    _require_fields(inputs, {"installed-inventory": hashes["inventory.json"]}, "executable installed input")


def _inner_bindings(documents: dict, hashes: dict, source: dict) -> None:
    _require_fields(
        documents["inventory.json"],
        {"schema": "xrr-installed-inventory-v1", "aggregate": "incomplete"},
        "installed inventory schema",
    )
    properties = {
        name: _bom_properties(value, source) for name, value in documents.items() if name.endswith(".cdx.json")
    }
    _require_fields(
        properties["installed.cdx.json"], {"xrr:inventory:sha256": hashes["inventory.json"]}, "installed inventory"
    )
    if "artifact.cdx.json" in properties:
        _require_fields(
            properties["artifact.cdx.json"],
            {
                "xrr:installed:inventory-sha256": hashes["inventory.json"],
                "xrr:installed:sbom-sha256": hashes["installed.cdx.json"],
            },
            "artifact installed input",
        )
    if "executable.cdx.json" in properties:
        _executable_bindings(documents, properties["executable.cdx.json"], hashes)


def read_installed_report(directory: Path, source: dict) -> tuple[dict, InputBindings]:
    bound = InputBindings({"summary.json": directory / "summary.json"}, limit=128 * 1024**2)
    summary = json_object(bound.contents["summary.json"])
    names = _installed_report_names(summary, source)
    bound.add({name: directory / name for name in sorted(names)})
    _installed_hashes(bound, summary, names)
    documents = {name: json_object(bound.contents[name]) for name in names}
    _incomplete_compositions(documents)
    _inner_bindings(documents, bound.hashes(), source)
    bound.guard()
    return documents, bound


def _ordinary_wheels(inventory: dict) -> list[dict]:
    packages = inventory["packages"]
    names = [item["wheel"]["name"] for item in packages]
    if len(names) != len(set(names)):
        raise ValueError("installed wheel identities are duplicated")
    return [item["wheel"] for item in packages if item["provenance"] == "ordinary-manifest"]


def bind_ordinary_scope(inventory: dict, manifest: dict, pins: tuple[str, ...]) -> dict:
    ordinary = _ordinary_wheels(inventory)
    expected = sorted(manifest["wheels"], key=lambda item: item["name"])
    if sorted(ordinary, key=lambda item: item["name"]) != expected:
        raise ValueError("installed ordinary wheel bytes differ from the strict manifest")
    if sorted(pins) != sorted(f"{item['name']}=={item['version']}" for item in expected):
        raise ValueError("advisory pins differ from the installed ordinary wheel scope")
    return {
        "scanned": expected,
        "not_scanned": [
            {"wheel": item["wheel"], "provenance": item["provenance"]}
            for item in inventory["packages"]
            if item["provenance"] != "ordinary-manifest"
        ],
    }


def _audit_tool_pins(root: Path) -> dict[str, str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    result = {}
    for value in project["tool"]["xrr"]["audit"]["requires"]:
        requirement = Requirement(value)
        pins = tuple(requirement.specifier)
        if len(pins) != 1 or pins[0].operator != "==":
            raise ValueError("advisory tool requires an exact version pin")
        name = canonicalize_name(requirement.name)
        if name in result:
            raise ValueError("advisory tool pins are duplicated")
        result[name] = pins[0].version
    return result


def _advisory_summary(root: Path, bound: InputBindings) -> dict:
    summary = json_object(bound.contents["summary.json"])
    expected = {"schema": "xrr-audit-report-v1", "kind": "advisories", "state": "PASS"}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("advisory summary schema, kind or status differs")
    if summary["exit_codes"] != dict.fromkeys(TARGETS, 0):
        raise ValueError("both ordinary advisory targets must pass")
    if summary["tool_versions"] != _audit_tool_pins(root):
        raise ValueError("advisory tool versions differ from the pinned configuration")
    inputs = json_object(bound.contents["inputs.json"])
    if bound.hashes()["inputs.json"] != summary["inputs_sha256"] or inputs != _input_identity(root):
        raise ValueError("advisory input bytes differ from the current source")
    return inputs


def _advisory_target(root: Path, directory: Path, bound: InputBindings, target: str) -> tuple[str, ...]:
    pins, scope = advisory_input(root, target)
    if bound.contents[f"{target}.requirements"] != ("\n".join(pins) + "\n").encode("utf-8"):
        raise ValueError("advisory requirements differ from the exact pins")
    if json_object(bound.contents[f"{target}.scope.json"]) != scope:
        raise ValueError("advisory scope differs from the verified lock")
    filename = f"{target}.advisories.json"
    json_object(bound.contents[filename])
    if advisory_findings(directory / filename, pins) != 0:
        raise ValueError("advisory report contains vulnerability findings")
    return pins


def read_advisory_report(root: Path, directory: Path) -> tuple[dict, dict, InputBindings]:
    names = ["summary.json", "inputs.json"]
    names.extend(
        f"{target}.{suffix}" for target in TARGETS for suffix in ("requirements", "scope.json", "advisories.json")
    )
    bound = InputBindings({name: directory / name for name in names})
    inputs = _advisory_summary(root, bound)
    pins = {target: _advisory_target(root, directory, bound, target) for target in TARGETS}
    bound.guard()
    return pins, inputs, bound


def _distribution_identities(bom: dict) -> list[dict]:
    result = []
    for component in bom["components"]:
        if component["type"] != "application":
            continue
        properties = {item["name"]: item["value"] for item in component["properties"]}
        result.append(
            {
                "kind": properties["xrr:artifact:kind"],
                "filename": component["name"],
                "sha256": component["hashes"][0]["content"],
                "size": int(properties["xrr:artifact:size"]),
            }
        )
    if sorted(item["kind"] for item in result) != ["sdist", "wheel"]:
        raise ValueError("distribution report must bind its wheel and sdist")
    return result


def _artifact_identities(documents: dict) -> list[dict]:
    result = []
    if "artifact.cdx.json" in documents:
        result.extend(_distribution_identities(documents["artifact.cdx.json"]))
    if "executable-inventory.json" in documents:
        executable = documents["executable-inventory.json"]["executable"]
        result.append({"kind": "headless-executable", "sha256": executable["sha256"], "size": executable["size"]})
    return result


def bind_audit(args) -> dict:
    root = args.repo_root.resolve()
    report, guard, _environment = prepare_report(root, args.report_dir)
    try:
        identity = clean_head_identity(root)
        source = {"source_commit": identity.head_commit, "source_tree": identity.head_tree}
        documents, installed_bound = read_installed_report(args.installed_report, source)
        pins, inputs, advisory_bound = read_advisory_report(root, args.advisory_report)
        inventory = documents["inventory.json"]
        target = inventory["target"]
        if target not in TARGETS:
            raise ValueError("unsupported installed advisory target")
        manifest_bound = InputBindings({"ordinary-manifest": root / f"tools/package-manifests/{target}.json"})
        manifest = validate_manifest(root, json_object(manifest_bound.contents["ordinary-manifest"]))
        scope = bind_ordinary_scope(inventory, manifest, pins[target])
        receipt = {
            "schema": "xrr-artifact-audit-binding-v1",
            "state": "PASS",
            **source,
            "target": target,
            "aggregate": "incomplete",
            "artifacts": _artifact_identities(documents),
            "ordinary_advisory_scope": scope,
            "native_libraries_scanned": False,
            "unowned_installed_members": inventory.get("unowned", []),
            "advisory_execution_platform": "not-attested-by-advisory-report",
            "boundary": "ordinary-PyPI-advisories-only; not VCS, bootstrap, test-only, native, OS or loader completeness",
            "installed_reports": installed_bound.hashes(),
            "advisory_reports": advisory_bound.hashes(),
            "manifest_inputs": manifest_bound.hashes(),
        }
        for bound in (installed_bound, advisory_bound, manifest_bound):
            bound.guard()
        if clean_head_identity(root) != identity or _input_identity(root) != inputs:
            raise ValueError("source inputs changed during artifact/advisory binding")
        guard()
        _write(report, "summary.json", canonical_sbom_bytes(receipt).decode("utf-8"))
        return receipt
    except (OSError, ValueError, KeyError, TypeError) as error:
        guard()
        _write(report, "failure.json", json.dumps({"state": "FAIL", "error": str(error)}, sort_keys=True) + "\n")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("installed-report", "advisory-report", "report-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = bind_audit(args)
        print(
            json.dumps({key: receipt[key] for key in ("schema", "state", "source_commit", "aggregate")}, sort_keys=True)
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
