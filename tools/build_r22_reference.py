#!/usr/bin/env python3
"""Normalize an externally supplied, independently approved R22 sidecar.

The archive and freeze receipt establish the immutable R22 software identity.
The sidecar supplies only data artifacts that are absent from that archive.
An independently reviewed, checked-in lock binds its manifest and complete tree.
Every sidecar file is also declared by relative path, size, and SHA-256.
The normalized manifest binds both source identities without recording host paths.
Golden path normalization is injective, so no artifact can overwrite another.
No R22 module is imported or executed, and no missing golden field is synthesized.
Publication owns one output directory and replaces it only after full validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Sequence


# Tool scripts are executed by path, so import the audited sibling validator directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_manifest import GROUPS, validate_groups  # noqa: E402
from reference_provenance import validate_provenance  # noqa: E402


SHA256_LENGTH = 64
ARCHIVE_FILES = (
    ".integration/release/release-identity.json",
    ".integration/release/product-manifest.tsv",
)
SIDECAR_LOCK_FIELDS = {
    "schema",
    "status",
    "reference_sidecar_manifest_sha256",
    "reference_sidecar_tree_sha256",
}


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


def _json(content: bytes, label: str, *, canonical: bool = True) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if canonical and content != _canonical(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise ValueError(f"invalid {label} SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _git_oid(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"invalid {label}")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid {label}")
    return value


def _relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label} path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid {label} path: {value}")
    return value


def _record(path: str, content: bytes) -> dict[str, object]:
    return {"path": path, "size": len(content), "sha256": _sha256(content)}


def _tree_hash(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        for value in (str(record["path"]), str(record["size"]), str(record["sha256"])):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _receipt(path: Path) -> tuple[dict[str, object], bytes]:
    content = _regular_file(path, "freeze receipt")
    receipt = _json(content, "freeze receipt")
    if receipt.get("schema") != "xrr-r22-delivery-freeze-v1" or receipt.get("status") != "PASS":
        raise ValueError("freeze receipt is not a passing R22 delivery receipt")
    _git_oid(receipt.get("head_commit"), "head_commit")
    _git_oid(receipt.get("head_tree"), "head_tree")
    for field in ("archive_sha256", "release_identity_sha256", "product_manifest_sha256"):
        _sha(receipt.get(field), field)
    status = receipt.get("post_delivery_real_data_acceptance_status")
    if status != "NOT_RUN":
        raise ValueError("real-data acceptance status must be NOT_RUN for this software delivery")
    return receipt, content


def _member_bytes(handle: tarfile.TarFile, member: tarfile.TarInfo, label: str) -> bytes:
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError(f"archive member is not regular: {label}")
    stream = handle.extractfile(member)
    if stream is None:
        raise ValueError(f"cannot read archive member: {label}")
    return stream.read()


def _archive_path(name: str) -> tuple[str, str] | None:
    relative = _relative(name, "archive member")
    parts = PurePosixPath(relative).parts
    if len(parts) < 2:
        return None
    return parts[0], PurePosixPath(*parts[1:]).as_posix()


def _read_archive(archive: Path, requested: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    roots: set[str] = set()
    try:
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                parsed = _archive_path(member.name)
                if parsed is None or parsed[1] not in requested:
                    continue
                root, relative = parsed
                if relative in found:
                    raise ValueError(f"duplicate archive member: {relative}")
                roots.add(root)
                found[relative] = _member_bytes(handle, member, relative)
    except (tarfile.TarError, OSError) as error:
        raise ValueError("invalid R22 archive") from error
    if set(found) != requested:
        raise ValueError("R22 archive is missing release metadata or declared input")
    if len(roots) != 1:
        raise ValueError("R22 archive inputs do not share one root")
    return found


def _validate_identity(identity: bytes) -> None:
    parsed = _json(identity, "release identity")
    if parsed.get("schema") != "xrr-r22-release-identity-v1":
        raise ValueError("unexpected R22 release identity schema")
    if parsed.get("gui_task_10_status") != "blocked: missing approved dataset":
        raise ValueError("R22 real-data history is not the frozen blocked state")
    acceptance = parsed.get("canonical_acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("status") != "PASS":
        raise ValueError("R22 canonical acceptance is not PASS")


def _validate_archive_inputs(
    found: dict[str, bytes], records: list[tuple[str, int, str]]
) -> None:
    for path, size, expected_hash in records:
        content = found[path]
        if len(content) != size or _sha256(content) != expected_hash:
            raise ValueError(f"R22 archive input size or hash drift: {path}")


def _archive_payload(
    archive: Path,
    receipt: dict[str, object],
    input_records: list[tuple[str, int, str]],
) -> tuple[bytes, bytes]:
    archive_content = _regular_file(archive, "R22 archive")
    if _sha256(archive_content) != receipt["archive_sha256"]:
        raise ValueError("R22 archive hash does not match freeze receipt")
    requested = {*ARCHIVE_FILES, *(record[0] for record in input_records)}
    found = _read_archive(archive, requested)
    identity, product = (found[path] for path in ARCHIVE_FILES)
    if _sha256(identity) != receipt["release_identity_sha256"]:
        raise ValueError("release identity hash does not match freeze receipt")
    if _sha256(product) != receipt["product_manifest_sha256"]:
        raise ValueError("product manifest hash does not match freeze receipt")
    _validate_identity(identity)
    _validate_archive_inputs(found, input_records)
    return identity, product


def _artifact_record(value: object) -> tuple[str, int, str]:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise ValueError("invalid sidecar artifact record")
    path = _relative(value.get("path"), "sidecar artifact")
    size = value.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"invalid artifact size: {path}")
    return path, size, _sha(value.get("sha256"), f"artifact {path}")


def _artifact_records(artifacts: object) -> list[tuple[str, int, str]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("reference sidecar artifacts must be non-empty")
    records = [_artifact_record(value) for value in artifacts]
    paths = [record[0] for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate sidecar artifact path")
    return records


def _read_payload(root: Path, records: list[tuple[str, int, str]]) -> list[tuple[str, bytes]]:
    payload: list[tuple[str, bytes]] = []
    for path, size, expected_hash in records:
        source = root.joinpath(*PurePosixPath(path).parts)
        content = _regular_file(source, f"sidecar artifact {path}")
        if len(content) != size or _sha256(content) != expected_hash:
            raise ValueError(f"sidecar artifact size or hash drift: {path}")
        payload.append((path, content))
    return payload


def _check_sidecar_tree(root: Path, expected: set[str]) -> None:
    paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("reference sidecar contains a symlink")
    observed = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if observed != expected:
        raise ValueError("reference sidecar contains missing or undeclared files")


def _sidecar(root: Path) -> tuple[dict[str, object], list[tuple[str, bytes]], bytes, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reference sidecar must be a regular directory")
    manifest_content = _regular_file(root / "manifest.json", "reference sidecar manifest")
    manifest = _json(manifest_content, "reference sidecar manifest")
    if manifest.get("schema") != "xrr-r22-reference-sidecar-v1":
        raise ValueError("unexpected reference sidecar schema")
    records = _artifact_records(manifest.get("artifacts"))
    payload = _read_payload(root, records)
    source_records = [_record("manifest.json", manifest_content)]
    source_records.extend(_record(path, content) for path, content in payload)
    _check_sidecar_tree(root, {str(record["path"]) for record in source_records})
    return manifest, payload, manifest_content, _tree_hash(source_records)


def _sidecar_lock(path: Path, manifest_content: bytes, tree_hash: str) -> bytes:
    content = _regular_file(path, "reference sidecar lock")
    lock = _json(content, "reference sidecar lock")
    if set(lock) != SIDECAR_LOCK_FIELDS:
        raise ValueError("invalid reference sidecar lock fields")
    if lock.get("schema") != "xrr-r22-reference-sidecar-lock-v1":
        raise ValueError("unexpected reference sidecar lock schema")
    if lock.get("status") != "APPROVED":
        raise ValueError("reference sidecar lock is not approved")
    expected_manifest = _sha(
        lock.get("reference_sidecar_manifest_sha256"),
        "reference sidecar manifest",
    )
    expected_tree = _sha(
        lock.get("reference_sidecar_tree_sha256"),
        "reference sidecar tree",
    )
    if expected_manifest != _sha256(manifest_content) or expected_tree != tree_hash:
        raise ValueError("reference sidecar does not match approved sidecar lock")
    return content


def _golden_path(path: str) -> str:
    relative = PurePosixPath(path)
    return relative.as_posix() if relative.parts[0] == "golden" else (PurePosixPath("golden") / relative).as_posix()


def _converted_payload(payload: list[tuple[str, bytes]]) -> list[tuple[str, str, bytes]]:
    converted = [(path, _golden_path(path), content) for path, content in payload]
    output_paths = [output for _, output, _ in converted]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("sidecar artifacts collide after golden path normalization")
    return converted


def _normalized_groups(
    sidecar_manifest: dict[str, object], source_to_output: dict[str, str]
) -> dict[str, object]:
    source_groups = sidecar_manifest["groups"]
    assert isinstance(source_groups, dict)
    groups: dict[str, object] = {}
    for group in GROUPS:
        entry = source_groups[group]
        assert isinstance(entry, dict)
        policy = entry["comparison_policy"]
        assert isinstance(policy, dict)
        policy_fields = policy["fields"]
        assert isinstance(policy_fields, dict)
        groups[group] = {
            "artifacts": [source_to_output[path] for path in entry["artifacts"]],
            "comparison_policy": {
                "kind": "mapping",
                "fields": {
                    source_to_output[path]: policy_fields[path]
                    for path in entry["artifacts"]
                },
            },
            "input_ids": list(entry["input_ids"]),
        }
    return groups


def _source_identity(
    receipt: dict[str, object],
    receipt_content: bytes,
    identity: bytes,
    product: bytes,
    archive_hash: str,
) -> dict[str, object]:
    return {
        "source_commit": receipt["head_commit"],
        "source_tree": receipt["head_tree"],
        "archive_sha256": archive_hash,
        "freeze_receipt_sha256": _sha256(receipt_content),
        "release_identity_sha256": _sha256(identity),
        "product_manifest_sha256": _sha256(product),
    }


def _normalized_manifest(
    receipt: dict[str, object],
    receipt_content: bytes,
    identity: bytes,
    product: bytes,
    sidecar_manifest: dict[str, object],
    sidecar_manifest_content: bytes,
    sidecar_tree_hash: str,
    sidecar_lock_content: bytes,
    payload: list[tuple[str, bytes]],
    archive_hash: str,
    provenance: dict[str, object],
) -> tuple[dict[str, object], list[tuple[str, bytes]]]:
    converted = _converted_payload(payload)
    artifacts = [_record(output_path, content) for _, output_path, content in converted]
    source_to_output = {source: output for source, output, _ in converted}
    acceptance = provenance["real_data_acceptance"]
    assert isinstance(acceptance, dict)
    manifest = {
        "schema": "xrr-r22-reference-v1",
        **_source_identity(receipt, receipt_content, identity, product, archive_hash),
        "builder_sha256": _sha256(Path(__file__).read_bytes()),
        "reference_sidecar_manifest_sha256": _sha256(sidecar_manifest_content),
        "reference_sidecar_tree_sha256": sidecar_tree_hash,
        "reference_sidecar_lock_sha256": _sha256(sidecar_lock_content),
        "real_data_acceptance_status": acceptance["status"],
        "real_data_acceptance_reason": acceptance["reason"],
        "provenance": provenance,
        "groups": _normalized_groups(sidecar_manifest, source_to_output),
        "artifacts": artifacts,
    }
    return manifest, [(output, content) for _, output, content in converted]


def _validate_output(output: Path) -> None:
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("reference output parent must be an existing regular directory")
    if not output.exists() and not output.is_symlink():
        return
    if output.is_symlink() or not output.is_dir():
        raise ValueError("reference output must be a directory")
    if any(output.iterdir()):
        raise ValueError("reference output directory is non-empty")


def _write_staging(staging: Path, manifest: dict[str, object], payload: list[tuple[str, bytes]]) -> None:
    for relative, content in payload:
        target = staging.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (staging / "manifest.json").write_bytes(_canonical(manifest))


def _publish(output: Path, manifest: dict[str, object], payload: list[tuple[str, bytes]]) -> None:
    _validate_output(output)
    parent = output.parent
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        _write_staging(staging, manifest, payload)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_reference(
    r22_archive: str | Path,
    freeze_receipt: str | Path,
    reference_sidecar: str | Path,
    sidecar_lock: str | Path,
    output: str | Path,
) -> dict[str, object]:
    archive_path = Path(r22_archive)
    receipt, receipt_content = _receipt(Path(freeze_receipt))
    sidecar_manifest, payload, sidecar_content, sidecar_tree_hash = _sidecar(Path(reference_sidecar))
    provenance, input_records = validate_provenance(
        sidecar_manifest.get("provenance"),
        receipt["head_commit"],
        receipt["head_tree"],
    )
    known_input_ids = {str(record["input_id"]) for record in provenance["inputs"]}
    validate_groups(
        sidecar_manifest.get("groups"),
        artifact_paths={path for path, _content in payload},
        known_input_ids=known_input_ids,
    )
    identity, product = _archive_payload(archive_path, receipt, input_records)
    sidecar_lock_content = _sidecar_lock(
        Path(sidecar_lock),
        sidecar_content,
        sidecar_tree_hash,
    )
    manifest, normalized = _normalized_manifest(
        receipt,
        receipt_content,
        identity,
        product,
        sidecar_manifest,
        sidecar_content,
        sidecar_tree_hash,
        sidecar_lock_content,
        payload,
        _sha256(archive_path.read_bytes()),
        provenance,
    )
    _publish(Path(output), manifest, normalized)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r22-archive", required=True, type=Path)
    parser.add_argument("--freeze-receipt", required=True, type=Path)
    parser.add_argument("--reference-sidecar", required=True, type=Path)
    parser.add_argument("--sidecar-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    build_reference(
        args.r22_archive,
        args.freeze_receipt,
        args.reference_sidecar,
        args.sidecar_lock,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
