#!/usr/bin/env python3
"""Validate the immutable mapping from R22 tests to R23 test contracts.

The source-draft phase proves that every source node is represented exactly
once and that every row is structurally canonical.  The final phase repeats
those checks and binds each declared target to the frozen R23 test manifest.
Both phases are deliberately path-explicit and never inspect a default tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, Sequence


MANIFEST_SCHEMA = "xrr-test-manifest-v1"
HEADER = (
    "source_tree",
    "source_nodeid",
    "contract_id",
    "action",
    "target_nodeids",
    "reason",
)
MANIFEST_FIELDS = {
    "schema",
    "source_commit",
    "suite",
    "test_tree",
    "node_count",
    "nodes",
    "python_version",
    "platform",
    "lock_sha256",
    "collection_sha256",
}
ACTIONS = {"port", "rewrite", "merge", "delete_layout_only"}
SOURCE_TREES = {"tests", "tests_r21"}
CONTRACT_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _canonical(value: object) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.read_bytes()


def _decode_manifest(content: bytes, path: Path) -> dict[str, object]:
    try:
        text = content.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return value


def _load_json(path: Path) -> dict[str, object]:
    content = _regular_bytes(path, "manifest")
    value = _decode_manifest(content, path)
    if _canonical(value) != content:
        raise ValueError(f"manifest is not canonical: {path}")
    return value


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _canonical_path(value: object, suite: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label} path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError(f"invalid {label} path")
    if len(path.parts) < 2 or path.parts[0] != suite or ".." in path.parts:
        raise ValueError(f"invalid {label} path")
    return value


def _tree_record(item: object, suite: str) -> tuple[str, str]:
    if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
        raise ValueError("invalid test_tree record")
    path = _canonical_path(item["path"], suite, "test_tree")
    size = item["size"]
    if type(size) is not int or size < 0:
        raise ValueError("invalid test_tree size")
    digest = _require_hash(item["sha256"], "test_tree hash")
    return path, digest


def _require_sorted_unique(values: list[str], label: str) -> None:
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError(f"{label} are not canonical")


def _validate_tree(tree: object, suite: str) -> dict[str, str]:
    if not isinstance(tree, list):
        raise ValueError("manifest test_tree must be a list")
    records = [_tree_record(item, suite) for item in tree]
    paths = [path for path, _digest in records]
    _require_sorted_unique(paths, "test_tree paths")
    return dict(records)


def _node_path(nodeid: object, suite: str) -> str:
    if not isinstance(nodeid, str) or not nodeid:
        raise ValueError("invalid source nodeid")
    return _canonical_path(nodeid.split("::", 1)[0], suite, "node")


def _validate_markers(markers: object) -> None:
    if not isinstance(markers, list):
        raise ValueError("invalid marker metadata")
    if not all(isinstance(marker, str) and marker for marker in markers):
        raise ValueError("invalid marker metadata")
    if markers != sorted(markers) or len(markers) != len(set(markers)):
        raise ValueError("invalid marker metadata")


def _node_record(item: object, suite: str, tree_paths: set[str]) -> str:
    if not isinstance(item, dict) or set(item) != {"nodeid", "markers"}:
        raise ValueError("invalid node record")
    nodeid = item["nodeid"]
    path = _node_path(nodeid, suite)
    _validate_markers(item["markers"])
    if path not in tree_paths:
        raise ValueError(f"manifest node file is absent from test_tree: {nodeid}")
    return str(nodeid)


def _validate_nodes(nodes: object, suite: str, tree_paths: set[str]) -> list[dict[str, object]]:
    if not isinstance(nodes, list):
        raise ValueError("manifest nodes must be a list")
    nodeids = [_node_record(item, suite, tree_paths) for item in nodes]
    _require_sorted_unique(nodeids, "manifest nodeids")
    return nodes


def _validate_manifest_fields(payload: dict[str, object], expected_suite: str) -> None:
    if set(payload) != MANIFEST_FIELDS or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("invalid manifest schema")
    if payload.get("suite") != expected_suite:
        raise ValueError("manifest suite mismatch")


def _validate_manifest_identity(payload: dict[str, object]) -> None:
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("invalid source commit")
    for field in ("python_version", "platform"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"invalid manifest {field}")
    _require_hash(payload.get("lock_sha256"), "lock hash")


def _validate_node_count(value: object, nodes: list[dict[str, object]]) -> None:
    if type(value) is not int or value != len(nodes):
        raise ValueError("manifest node count mismatch")


def _validate_collection_hash(payload: dict[str, object]) -> None:
    observed = _require_hash(payload.get("collection_sha256"), "collection hash")
    base = {key: value for key, value in payload.items() if key != "collection_sha256"}
    expected = hashlib.sha256(_canonical(base)).hexdigest()
    if observed != expected:
        raise ValueError("manifest collection hash mismatch")


def _manifest(path: Path, expected_suite: str) -> tuple[dict[str, object], dict[str, str]]:
    payload = _load_json(path)
    _validate_manifest_fields(payload, expected_suite)
    _validate_manifest_identity(payload)
    tree_hashes = _validate_tree(payload["test_tree"], expected_suite)
    nodes = _validate_nodes(payload["nodes"], expected_suite, set(tree_hashes))
    _validate_node_count(payload.get("node_count"), nodes)
    _validate_collection_hash(payload)
    return payload, tree_hashes


def _decode_ledger(content: bytes) -> str:
    if b"\r" in content or not content.endswith(b"\n"):
        raise ValueError("ledger must use UTF-8 with LF line endings")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("ledger must be valid UTF-8") from error


def _ledger_rows(text: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        rows = list(reader)
    except csv.Error as error:
        raise ValueError("invalid ledger CSV") from error
    return tuple(reader.fieldnames or ()), rows


def _row_has_exact_width(row: dict[str | None, str | None]) -> bool:
    return set(row) == set(HEADER) and all(value is not None for value in row.values())


def _read_ledger(path: Path) -> list[dict[str, str]]:
    content = _regular_bytes(path, "ledger")
    fieldnames, rows = _ledger_rows(_decode_ledger(content))
    if fieldnames != HEADER:
        raise ValueError("ledger header mismatch")
    if not all(_row_has_exact_width(row) for row in rows):
        raise ValueError("ledger row width mismatch")
    return rows


def _target_nodeid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid target nodeid")
    try:
        _node_path(value, "tests")
    except ValueError as error:
        raise ValueError("invalid target nodeid") from error
    return value


def _targets(raw: str) -> list[str]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("target_nodeids must be canonical JSON") from error
    if not isinstance(value, list) or not value:
        raise ValueError("target_nodeids must be a non-empty array")
    targets = [_target_nodeid(item) for item in value]
    if targets != sorted(targets) or len(targets) != len(set(targets)):
        raise ValueError("target_nodeids must be sorted and unique")
    canonical = json.dumps(targets, ensure_ascii=False, separators=(",", ":"))
    if canonical != raw:
        raise ValueError("target_nodeids must use canonical JSON")
    return targets


def _validate_row_text(row: dict[str, str]) -> None:
    contract = row["contract_id"]
    if CONTRACT_PATTERN.fullmatch(contract) is None:
        raise ValueError("invalid contract_id")
    if row["action"] not in ACTIONS:
        raise ValueError("invalid action")
    reason = row["reason"]
    if not reason or reason != reason.strip():
        raise ValueError("reason must be trimmed non-empty text")


def _validate_row(row: dict[str, str]) -> dict[str, object]:
    tree = row["source_tree"]
    if tree not in SOURCE_TREES:
        raise ValueError("invalid source_tree")
    _validate_row_text(row)
    return {**row, "target_nodeids": _targets(row["target_nodeids"])}


def _source_keys(payload: dict[str, object]) -> set[tuple[str, str]]:
    suite = str(payload["suite"])
    return {(suite, str(item["nodeid"])) for item in payload["nodes"]}


def _validate_coverage(rows: list[dict[str, object]], expected: set[tuple[str, str]]) -> None:
    observed = [(str(row["source_tree"]), str(row["source_nodeid"])) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate source coverage in ledger")
    if set(observed) != expected:
        raise ValueError("ledger source coverage mismatch")


def validate_source_draft(
    active_manifest: str | Path,
    r21_manifest: str | Path,
    ledger: str | Path,
) -> dict[str, object]:
    active, _active_hashes = _manifest(Path(active_manifest), "tests")
    r21, _r21_hashes = _manifest(Path(r21_manifest), "tests_r21")
    rows = [_validate_row(row) for row in _read_ledger(Path(ledger))]
    expected = _source_keys(active) | _source_keys(r21)
    _validate_coverage(rows, expected)
    return {"phase": "source-draft", "source_count": len(expected), "rows": rows}


def validate_final_targets(rows: Iterable[dict[str, object]], target_nodes: set[str]) -> None:
    for row in rows:
        targets = row["target_nodeids"]
        if not isinstance(targets, list) or not all(target in target_nodes for target in targets):
            raise ValueError("target nodeid does not exist")
        if row["action"] == "delete_layout_only" and not all(
            target.startswith("tests/architecture/") for target in targets
        ):
            raise ValueError("delete_layout_only targets must be architecture tests")


def validate_final(
    active_manifest: str | Path,
    r21_manifest: str | Path,
    ledger: str | Path,
    target_manifest: str | Path,
) -> dict[str, object]:
    draft = validate_source_draft(active_manifest, r21_manifest, ledger)
    target, _hashes = _manifest(Path(target_manifest), "tests")
    targets = {str(item["nodeid"]) for item in target["nodes"]}
    validate_final_targets(draft["rows"], targets)
    return {**draft, "phase": "final", "target_count": len(targets)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("source-draft", "final"), required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--r21-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path)
    return parser


def _run_phase(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, object]:
    if args.phase == "source-draft":
        if args.target_manifest is not None:
            parser.error("--target-manifest is only valid for final phase")
        return validate_source_draft(args.active_manifest, args.r21_manifest, args.ledger)
    if args.target_manifest is None:
        parser.error("final phase requires --target-manifest")
    return validate_final(args.active_manifest, args.r21_manifest, args.ledger, args.target_manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    result = _run_phase(parser.parse_args(argv), parser)
    public_result = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(public_result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
