#!/usr/bin/env python3
"""Validate normalized R22 data and compare only explicitly registered groups.

Self-check parses reference artifacts without importing or executing R22 code.
Group adapters are supplied by a closed registry; filesystem discovery is forbidden.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Callable, Mapping, Sequence, cast
import zipfile


sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_comparison import compare_value  # noqa: E402
from reference_groups.registry import GROUP_REGISTRY  # noqa: E402
from reference_manifest import GROUPS, validate_groups  # noqa: E402
from reference_provenance import validate_provenance  # noqa: E402


MANIFEST_FIELDS = {
    "schema",
    "source_commit",
    "source_tree",
    "archive_sha256",
    "freeze_receipt_sha256",
    "release_identity_sha256",
    "product_manifest_sha256",
    "builder_sha256",
    "reference_sidecar_manifest_sha256",
    "reference_sidecar_tree_sha256",
    "reference_sidecar_lock_sha256",
    "real_data_acceptance_status",
    "real_data_acceptance_reason",
    "provenance",
    "groups",
    "artifacts",
}
HASH_FIELDS = {
    "archive_sha256",
    "freeze_receipt_sha256",
    "release_identity_sha256",
    "product_manifest_sha256",
    "builder_sha256",
    "reference_sidecar_manifest_sha256",
    "reference_sidecar_tree_sha256",
    "reference_sidecar_lock_sha256",
}


@dataclass(frozen=True)
class ReplayInput:
    input_id: str
    input_class: str
    path: str
    size: int
    sha256: str
    content: bytes


@dataclass(frozen=True)
class ReplayContext:
    group: str
    artifacts: tuple[str, ...]
    inputs: tuple[ReplayInput, ...]
    configuration: object
    seeds: tuple[int, ...]


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


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid artifact path: {value}")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("invalid artifact SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("invalid artifact SHA-256")
    return value


def _git_oid(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"invalid {label}")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid {label}")
    return value


def _current_builder_hash() -> str:
    return hashlib.sha256(Path(__file__).with_name("build_r22_reference.py").read_bytes()).hexdigest()


def _record(value: object) -> tuple[str, dict[str, object]]:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise ValueError("invalid artifact record")
    path = _relative(value.get("path"))
    size = value.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"invalid artifact size: {path}")
    _sha(value.get("sha256"))
    return path, value


def _artifact_index(artifacts: object) -> dict[str, dict[str, object]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("reference artifacts must be non-empty")
    by_path: dict[str, dict[str, object]] = {}
    for value in artifacts:
        path, record = _record(value)
        if path in by_path:
            raise ValueError(f"duplicate artifact path: {path}")
        by_path[path] = record
    return by_path


def _input_index(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("inputs"), list):
        raise ValueError("invalid reference provenance inputs")
    by_id: dict[str, dict[str, object]] = {}
    for record in provenance["inputs"]:
        if not isinstance(record, dict):
            raise ValueError("invalid reference provenance input")
        input_id = str(record.get("input_id"))
        path = _relative(record.get("path"))
        if input_id in by_id or any(item["path"] == path for item in by_id.values()):
            raise ValueError("duplicate reference provenance input")
        by_id[input_id] = record
    return by_id


def _records(
    manifest: dict[str, object],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    by_path = _artifact_index(manifest.get("artifacts"))
    inputs = _input_index(manifest)
    normalized_groups = validate_groups(
        manifest.get("groups"),
        artifact_paths=set(by_path),
        known_input_ids=set(inputs),
    )
    return normalized_groups, by_path, inputs


def _valid_zip_metadata(info: zipfile.ZipInfo) -> bool:
    return (
        info.date_time == (1980, 1, 1, 0, 0, 0)
        and info.compress_type == zipfile.ZIP_STORED
        and info.create_system == 3
        and info.external_attr == 0o600 << 16
        and info.flag_bits == 0
        and info.extra == b""
        and info.comment == b""
    )


def _npz_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
    names = [member.filename for member in members]
    valid_names = all(
        PurePosixPath(name).name == name and name.endswith(".npy")
        for name in names
    )
    if not names or len(names) != len(set(names)) or names != sorted(names) or not valid_names:
        raise ValueError(f"invalid NPZ members: {path.name}")
    if any(not _valid_zip_metadata(member) for member in members):
        raise ValueError(f"invalid NPZ metadata: {path.name}")
    return names


def _npz_arrays(path: Path, names: list[str]) -> None:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if list(archive.files) != [PurePosixPath(name).stem for name in names]:
            raise ValueError(f"NPZ key order drift: {path.name}")
        for key in archive.files:
            if archive[key].dtype.hasobject:
                raise ValueError(f"object array is forbidden: {path.name}:{key}")


def _validate_npz(path: Path) -> None:
    try:
        names = _npz_names(path)
        _npz_arrays(path, names)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid NPZ artifact: {path.name}: {error}") from error


def _validate_artifact(root: Path, record: dict[str, object]) -> None:
    relative = str(record["path"])
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular artifact: {relative}")
    content = path.read_bytes()
    if len(content) != record["size"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError(f"artifact size or hash drift: {relative}")
    if path.suffix == ".json":
        _json(path, f"artifact {relative}")
    elif path.suffix == ".npz":
        _validate_npz(path)
    else:
        raise ValueError(f"unsupported reference artifact: {relative}")


def _manifest_identity(manifest: dict[str, object]) -> str:
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("reference manifest field set drift")
    if manifest.get("schema") != "xrr-r22-reference-v1":
        raise ValueError("unexpected reference manifest schema")
    source_commit = _git_oid(manifest.get("source_commit"), "R22 source commit")
    source_tree = _git_oid(manifest.get("source_tree"), "R22 source tree")
    for field in HASH_FIELDS:
        _sha(manifest.get(field))
    if _current_builder_hash() != manifest["builder_sha256"]:
        raise ValueError("reference builder hash drift")
    expected_status = ("NOT_RUN", "owner post-delivery acceptance")
    observed_status = (
        manifest.get("real_data_acceptance_status"),
        manifest.get("real_data_acceptance_reason"),
    )
    if observed_status != expected_status:
        raise ValueError("real-data acceptance status drift")
    validate_provenance(manifest.get("provenance"), source_commit, source_tree)
    return source_commit


def _tree_paths(root: Path, expected_files: set[str]) -> set[str]:
    expected = {"manifest.json", *expected_files}
    observed: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("reference tree contains a symlink")
        if candidate.is_file():
            observed.add(candidate.relative_to(root).as_posix())
    if observed != expected:
        raise ValueError("reference tree contains missing or undeclared files")
    return observed


def _replay_input_content(root: Path, record: dict[str, object]) -> bytes:
    relative = str(record["path"])
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular replay input: {relative}")
    content = path.read_bytes()
    expected_size = record.get("size")
    expected_hash = record.get("sha256")
    if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"replay input size or hash drift: {relative}")
    return content


def _validate_group_policies(
    root: Path,
    groups: dict[str, dict[str, object]],
) -> None:
    for group in GROUPS:
        entry = groups[group]
        reference = {
            relative: _artifact_value(root, relative)
            for relative in entry["artifacts"]
        }
        compare_value(reference, reference, entry["comparison_policy"])


def self_check(manifest_path: str | Path) -> dict[str, object]:
    path = Path(manifest_path)
    manifest = _json(path, "reference manifest")
    source_commit = _manifest_identity(manifest)
    groups, records, inputs = _records(manifest)
    root = path.parent
    for record in records.values():
        _validate_artifact(root, record)
    for record in inputs.values():
        _replay_input_content(root, record)
    expected_files = {*records, *(str(record["path"]) for record in inputs.values())}
    _tree_paths(root, expected_files)
    _validate_group_policies(root, groups)
    return {
        "schema": manifest["schema"],
        "source_commit": source_commit,
        "group_count": len(groups),
        "artifact_count": len(records),
        "input_count": len(inputs),
    }


def _artifact_value(root: Path, relative: str) -> object:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.suffix == ".json":
        return _json(path, f"artifact {relative}")
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _replay_input(root: Path, record: dict[str, object]) -> ReplayInput:
    return ReplayInput(
        input_id=str(record["input_id"]),
        input_class=str(record["input_class"]),
        path=str(record["path"]),
        size=int(record["size"]),
        sha256=str(record["sha256"]),
        content=_replay_input_content(root, record),
    )


def _replay_context(
    root: Path,
    manifest: dict[str, object],
    group: str,
    entry: dict[str, object],
) -> ReplayContext:
    provenance = cast(dict[str, object], manifest["provenance"])
    input_records = cast(list[dict[str, object]], provenance["inputs"])
    configurations = cast(dict[str, dict[str, object]], provenance["configurations"])
    seeds = cast(dict[str, list[int]], provenance["seeds"])
    by_id = {
        str(record["input_id"]): record
        for record in input_records
    }
    replay_inputs = tuple(
        _replay_input(root, by_id[input_id])
        for input_id in entry["input_ids"]
    )
    configuration = configurations[group]
    group_seeds = seeds[group]
    return ReplayContext(
        group=group,
        artifacts=tuple(str(path) for path in entry["artifacts"]),
        inputs=replay_inputs,
        configuration=copy.deepcopy(configuration["value"]),
        seeds=tuple(group_seeds),
    )


def compare_group(
    manifest_path: str | Path,
    group: str,
    registry: Mapping[str, Callable[[ReplayContext], object]] | None = None,
) -> dict[str, object]:
    self_check(manifest_path)
    selected = GROUP_REGISTRY if registry is None else registry
    if group not in selected:
        raise ValueError(f"reference group is not registered: {group}")
    manifest = _json(Path(manifest_path), "reference manifest")
    root = Path(manifest_path).parent
    groups, _records_by_path, _inputs_by_id = _records(manifest)
    entry = groups[group]
    reference = {
        relative: _artifact_value(root, relative)
        for relative in entry["artifacts"]
    }
    actual = selected[group](_replay_context(root, manifest, group, entry))
    compare_value(reference, actual, entry.get("comparison_policy", "exact"))
    return {"group": group, "status": "PASS", "artifact_count": len(reference)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--group", choices=GROUPS)
    args = parser.parse_args(argv)
    if args.self_check is not None:
        if args.manifest is not None or args.group is not None:
            parser.error("--self-check cannot be combined with --manifest or --group")
        print(json.dumps(self_check(args.self_check), sort_keys=True))
        return 0
    if args.manifest is None or args.group is None:
        parser.error("--manifest and --group are required together")
    print(json.dumps(compare_group(args.manifest, args.group), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
