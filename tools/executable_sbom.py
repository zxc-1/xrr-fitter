"""CycloneDX view over actual frozen payloads and only their observed input witnesses."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import quote

from lock_sbom import canonical_sbom_bytes


def _properties(values: dict) -> list[dict]:
    return [{"name": key, "value": str(value)} for key, value in sorted(values.items())]


def _reference(inventory: dict, identifier: str) -> str:
    return f"urn:xrr:executable:{inventory['executable']['sha256']}:{quote(identifier, safe='')}"


def _file(inventory: dict, row: dict) -> dict:
    return {
        "type": "file",
        "bom-ref": _reference(inventory, row["id"]),
        "name": row["path"],
        "hashes": [{"alg": "SHA-256", "content": row["sha256"]}],
        "properties": _properties(
            {
                "xrr:frozen:container": row["container"],
                "xrr:frozen:size": row["size"],
                "xrr:frozen:claims": json.dumps(row["claims"], sort_keys=True),
                "xrr:frozen:unresolved": row["unresolved"],
                "xrr:frozen:native-declarations": json.dumps(row.get("native", {}), sort_keys=True),
            }
        ),
    }


def _dependencies(inventory: dict) -> list[dict]:
    result = {_reference(inventory, "executable"): set()}
    for file in inventory["files"]:
        ref, parent = _reference(inventory, file["id"]), _reference(inventory, file["parent"])
        result.setdefault(parent, set()).add(ref)
        result.setdefault(ref, set()).update(value for claim in file["claims"] for value in claim["inputs"])
    for name in inventory["framing"]:
        result[_reference(inventory, "executable")].add(_reference(inventory, "framing/" + name))
    result.update({ref: set(values) for ref, values in inventory["provenance_dependencies"].items()})
    return [{"ref": ref, "dependsOn": sorted(values)} for ref, values in sorted(result.items())]


def assemble_executable_sbom(inventory: dict, source: dict, bindings: dict) -> dict:
    executable = {
        "type": "application",
        "bom-ref": _reference(inventory, "executable"),
        "name": "xrr-fitter-cli.exe",
        "hashes": [{"alg": "SHA-256", "content": inventory["executable"]["sha256"]}],
    }
    framing = [
        {
            "type": "file",
            "bom-ref": _reference(inventory, "framing/" + name),
            "name": name,
            "hashes": [{"alg": "SHA-256", "content": row["sha256"]}],
            "properties": _properties({key: value for key, value in row.items() if key != "sha256"}),
        }
        for name, row in inventory["framing"].items()
    ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "properties": _properties(
                {
                    "xrr:inventory:scope": "actual-frozen-payloads-and-input-byte-witnesses-not-full-dev-runtime",
                    "xrr:inventory:sha256": hashlib.sha256(canonical_sbom_bytes(inventory)).hexdigest(),
                    "xrr:source:commit": source["source_commit"],
                    "xrr:source:tree": source["source_tree"],
                    "xrr:inputs:sha256": json.dumps(bindings, sort_keys=True),
                    "xrr:composition:boundary": inventory["boundary"],
                    "xrr:native:resolution-scope": "observed-declarations-not-runtime-loads",
                }
            )
        },
        "components": [
            executable,
            *framing,
            *(_file(inventory, row) for row in inventory["files"]),
            *inventory["provenance"],
        ],
        "dependencies": _dependencies(inventory),
        "compositions": [{"aggregate": "incomplete"}],
    }
