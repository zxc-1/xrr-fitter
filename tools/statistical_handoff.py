"""Bind an explicit earlier producer without rewriting its source or run identity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __name__ == "__main__":
    sys.dont_write_bytecode = True

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from distribution_source import _git  # noqa: E402
from installed_inputs import InputBindings, json_object  # noqa: E402
from statistical_compatibility import verify_compatibility  # noqa: E402
from statistical_github import fetch_artifacts, validate_selection  # noqa: E402
from statistical_provenance import capture_identity  # noqa: E402
from verify_publish import _write_new_file_in_anchored_directory  # noqa: E402
from verify_report import (  # noqa: E402
    _make_report_anchor,
    _prepare_report_directory,
    _require_same_directory,
    _resolve_output_path,
)


@dataclass(frozen=True)
class StatisticalHandoff:
    inputs: InputBindings
    producer: dict
    proof: dict
    hashes: dict[str, str]
    provenance: dict

    def guard(self) -> None:
        self.inputs.guard()


def _require_repository(root: Path, selection: dict, consumer: dict) -> None:
    expected = f"https://github.com/{selection['repository']}"
    commands = (
        ("config", "--get-all", "remote.origin.url"),
        ("remote", "get-url", "--all", "origin"),
        ("remote", "get-url", "--push", "--all", "origin"),
    )
    if any(_git(root, *command) != expected for command in commands):
        raise ValueError("statistical producer repository differs from the exact raw/fetch/push origin")
    workflow = consumer["workflow"]
    if workflow["provider"] == "github-actions" and workflow["repository"] != selection["repository"]:
        raise ValueError("statistical producer repository differs from the consumer workflow")


def load_handoff(root: Path, path: Path, consumer: dict) -> StatisticalHandoff:
    path = _resolve_output_path(path, "statistical producer descriptor")
    if path.is_relative_to(root.resolve()):
        raise ValueError("statistical producer descriptor must be external")
    inputs = InputBindings({"producer": path}, limit=2 * 1024**2)
    selection = validate_selection(json_object(inputs.contents["producer"]))
    _require_repository(root, selection, consumer)
    artifacts = fetch_artifacts(selection)
    proof = verify_compatibility(root, artifacts.identity, consumer)
    inputs.guard()
    provenance = {**artifacts.provenance, "descriptor_sha256": inputs.hashes()["producer"]}
    return StatisticalHandoff(inputs, artifacts.identity, proof, artifacts.hashes, provenance)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write(directory: Path, identity: tuple[int, int], name: str, content: bytes) -> None:
    _write_new_file_in_anchored_directory(
        directory,
        identity,
        name,
        content,
        directory_label="statistical handoff",
        file_label="statistical handoff evidence",
    )


def _publish_shards(report: Path, report_identity: tuple[int, int], files: dict[str, bytes]) -> None:
    shards = report / "shards"
    shards_identity = _prepare_report_directory(shards)
    directories = {}
    for name, content in files.items():
        parts = PurePosixPath(name).parts
        if len(parts) != 2 or any(part in {".", ".."} for part in parts):
            raise ValueError("statistical transfer path must name a shard and direct file")
        _require_same_directory(report, report_identity, "statistical handoff")
        _require_same_directory(shards, shards_identity, "statistical handoff shards")
        directory = shards / parts[0]
        if directory not in directories:
            directories[directory] = _prepare_report_directory(directory)
        _write(directory, directories[directory], parts[1], content)


def fetch_handoff(root: Path, selection: dict, report: Path) -> Path:
    root = root.resolve()
    report = _resolve_output_path(report, "statistical handoff")
    if report.is_relative_to(root) or os.path.lexists(report):
        raise ValueError("statistical handoff requires a new external report directory")
    anchor = _make_report_anchor(report)
    validate_selection(selection)
    consumer = capture_identity(root)
    _require_repository(root, selection, consumer)
    artifacts = fetch_artifacts(selection)
    proof = verify_compatibility(root, artifacts.identity, consumer)
    if capture_identity(root) != consumer:
        raise ValueError("statistical consumer changed during artifact acquisition")
    identity = _prepare_report_directory(report, anchor)
    _publish_shards(report, identity, artifacts.files)
    _write(report, identity, "producer.json", _json_bytes(selection))
    evidence = {
        "state": "DOWNLOADED",
        "new_fit_count": 0,
        "producer_identity": artifacts.identity,
        "consumer_identity": consumer,
        "compatibility": proof,
        "provenance": artifacts.provenance,
    }
    _write(report, identity, "fetch-evidence.json", _json_bytes(evidence))
    if capture_identity(root) != consumer:
        raise ValueError("statistical consumer changed during artifact publication")
    return report / "producer.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        fetch_handoff(args.repo_root, json_object(args.producer_json.encode("utf-8")), args.report_dir)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
