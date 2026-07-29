"""Bind the frozen R22 tree and publish an external aggregate comparison report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence


REPORT_NAME = "r22-reference-report.json"
REPORT_SCHEMA = "xrr-r22-reference-comparison-v1"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular {label}")
    content = path.read_bytes()
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if content != _canonical(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("invalid R22 oracle SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("invalid R22 oracle SHA-256")
    return value


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _oracle_record(path: Path, root: Path) -> tuple[str, int, str] | None:
    if path.is_symlink():
        raise ValueError("R22 oracle tree contains a symlink")
    if path.is_dir():
        return None
    if not path.is_file():
        raise ValueError("R22 oracle tree must contain only regular files and directories")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError("unable to read R22 oracle file") from error
    return (
        path.relative_to(root).as_posix(),
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def _oracle_tree(root: Path) -> tuple[str, int]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R22 oracle root must be a regular directory")
    records = [record for path in root.rglob("*") if (record := _oracle_record(path, root))]
    records.sort()
    if not records:
        raise ValueError("R22 oracle tree is empty")
    digest = hashlib.sha256()
    for path, size, sha256 in records:
        for value in (path.encode("utf-8"), str(size).encode("ascii"), sha256.encode("ascii")):
            digest.update(_frame(value))
    return digest.hexdigest(), len(records)


def _oracle_root(manifest_path: Path) -> Path:
    path = manifest_path.resolve()
    root = path.parent.parent
    if path != root / "reference/manifest.json":
        raise ValueError("reference manifest must be verification/r22/reference/manifest.json")
    return root


def _validate_binding_paths(root: Path, collections_root: Path, release_spec: Path) -> None:
    if collections_root.is_symlink() or collections_root.resolve() != root / "collections":
        raise ValueError("collections root must be verification/r22/collections")
    expected_spec = root.parent / "release-spec.json"
    if release_spec.is_symlink() or release_spec.resolve() != expected_spec:
        raise ValueError("release spec must be verification/release-spec.json")


def _release_expectations(path: Path) -> tuple[str, int]:
    spec = _json(path, "release spec")
    if spec.get("schema") != "xrr-r23-release-spec-v1":
        raise ValueError("unexpected release spec schema")
    tree_sha256 = _sha(spec.get("r22_oracle_tree_sha256"))
    file_count = spec.get("r22_oracle_file_count")
    if not isinstance(file_count, int) or isinstance(file_count, bool) or file_count <= 0:
        raise ValueError("invalid R22 oracle file count")
    return tree_sha256, file_count


def bind_oracle(
    manifest_path: str | Path,
    collections_root: str | Path,
    release_spec: str | Path,
) -> dict[str, object]:
    root = _oracle_root(Path(manifest_path))
    collections = Path(collections_root)
    spec_path = Path(release_spec)
    _validate_binding_paths(root, collections, spec_path)
    expected = _release_expectations(spec_path)
    observed = _oracle_tree(root)
    if observed != expected:
        raise ValueError("R22 oracle tree digest drift")
    return {
        "r22_oracle_tree_sha256": observed[0],
        "r22_oracle_file_count": observed[1],
    }


def _external_report_dir(report_dir: str | Path, repo_root: str | Path) -> Path:
    repository = Path(repo_root).resolve()
    source = Path(report_dir)
    report = source.resolve()
    if report == repository or report.is_relative_to(repository):
        raise ValueError("R22 reference report must be outside the repository")
    if source.is_symlink() or (report.exists() and not report.is_dir()):
        raise ValueError("R22 reference report path must be a regular directory")
    return report


def _atomic_write(path: Path, content: bytes) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("R22 reference report parent must be a regular directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("R22 reference report must be a regular file path")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run_all_groups(
    manifest_path: str | Path,
    report_dir: str | Path,
    *,
    group_names: Sequence[str],
    registry: Mapping[str, object],
    repo_root: str | Path,
    self_check: Callable[[], Mapping[str, object]],
    compare_group: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    groups = tuple(group_names)
    if tuple(registry) != groups:
        raise ValueError("R22 reference registry must contain the exact eight groups in order")
    checked = self_check()
    report = _external_report_dir(report_dir, repo_root)
    results = [compare_group(group) for group in groups]
    manifest = Path(manifest_path)
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "source_commit": checked["source_commit"],
        "reference_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "group_count": len(results),
        "groups": results,
    }
    _atomic_write(report / REPORT_NAME, _canonical(payload))
    return payload
