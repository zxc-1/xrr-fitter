"""Bind a successful headless execution receipt to actual frozen bytes and Git source."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from distribution_source import committed_tree_files  # noqa: E402
from executable_archive import FILE_LIMIT  # noqa: E402
from executable_inventory import inspect_executable  # noqa: E402
from executable_sbom import assemble_executable_sbom  # noqa: E402
from executable_sources import ExecutableSources  # noqa: E402
from installed_files import InstalledSnapshot  # noqa: E402
from installed_inputs import InputBindings, json_object  # noqa: E402
from lock_sbom import canonical_sbom_bytes  # noqa: E402


def verify_executable_evidence(evidence: dict, data: bytes, source: dict) -> None:
    expected = {
        "schema": "xrr-headless-executable-evidence-v1",
        "source_commit": source["source_commit"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "help": "PASS",
        "validate": "PASS",
        "export": "PASS",
        "gui_tested": False,
    }
    if evidence != expected or type(evidence.get("gui_tested")) is not bool or type(evidence.get("size")) is not int:
        raise ValueError("executable evidence differs from successful source-bound bytes")


def bind_executable(root, path, evidence_path, inputs, installed, source):
    if installed["target"] != "windows-x64-py312":
        raise ValueError("executable binding requires the verified Windows installation")
    bound = InputBindings({"executable-evidence": evidence_path})
    snapshot = InstalledSnapshot(path.absolute().parent)
    data = snapshot.read(path, limit=FILE_LIMIT)
    verify_executable_evidence(json_object(bound.contents["executable-evidence"]), data, source)
    files = {
        relative.as_posix(): content
        for relative, content in committed_tree_files(root, source["source_commit"], "src").items()
    }
    with ExecutableSources(inputs, installed, files, source, entry_point="src/xrr_fitter/cli/main.py") as sources:
        inventory = inspect_executable(data, sources)
    bindings = {**bound.hashes(), "installed-inventory": hashlib.sha256(canonical_sbom_bytes(installed)).hexdigest()}
    bom = assemble_executable_sbom(inventory, source, bindings)

    def guard():
        bound.guard()
        snapshot.finish()

    guard()
    return inventory, bom, guard
