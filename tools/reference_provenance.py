"""Validate self-contained provenance for the frozen R22 reference sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath


GROUPS = (
    "model_project",
    "io",
    "physics",
    "fit_compile",
    "fit_search",
    "analysis",
    "services",
    "gui",
)
PROVENANCE_KEYS = {
    "schema",
    "source_commit",
    "source_tree",
    "generator",
    "collection_lock",
    "python",
    "platform",
    "inputs",
    "seeds",
    "configurations",
    "real_data_acceptance",
}


def _canonical(value: object) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label}")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {label} SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label} path")
    path = PurePosixPath(value)
    invalid_part = any(part in {"", ".", ".."} for part in path.parts)
    if path.is_absolute() or value != path.as_posix() or invalid_part:
        raise ValueError(f"invalid {label} path: {value}")
    return value


def _embedded_file(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"content", "size", "sha256"}:
        raise ValueError(f"invalid embedded {label}")
    content = _nonempty_text(value.get("content"), f"embedded {label} content")
    size = value.get("size")
    encoded = content.encode("utf-8")
    expected = _sha(value.get("sha256"), f"embedded {label}")
    if not isinstance(size, int) or isinstance(size, bool) or size != len(encoded):
        raise ValueError(f"embedded {label} size drift")
    if _sha256(encoded) != expected:
        raise ValueError(f"embedded {label} hash drift")


def _environment(value: object, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"invalid {label} identity")
    for field in fields:
        _nonempty_text(value[field], f"{label} {field}")


def _input_record(value: object) -> tuple[str, int, str]:
    fields = {"input_id", "input_class", "path", "size", "sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid provenance input record")
    _nonempty_text(value["input_id"], "provenance input ID")
    _nonempty_text(value["input_class"], "provenance input class")
    path = _relative(value.get("path"), "provenance input")
    size = value.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"invalid provenance input size: {path}")
    return path, size, _sha(value.get("sha256"), f"provenance input {path}")


def _inputs(value: object) -> list[tuple[str, int, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("provenance inputs must be non-empty")
    records = [_input_record(item) for item in value]
    paths = [record[0] for record in records]
    identifiers = [str(item["input_id"]) for item in value]
    if len(paths) != len(set(paths)) or len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate provenance input")
    return records


def _seed_list(value: object, group: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"invalid provenance seeds: {group}")
    invalid = any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in value)
    if invalid:
        raise ValueError(f"invalid provenance seeds: {group}")
    return value


def _seeds(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(GROUPS):
        raise ValueError("provenance seeds must contain the exact eight groups")
    normalized = {group: _seed_list(value[group], group) for group in GROUPS}
    if not normalized["fit_search"] or not normalized["analysis"]:
        raise ValueError("fit_search and analysis provenance seeds must be non-empty")


def _configuration(value: object, group: str) -> None:
    if not isinstance(value, dict) or set(value) != {"value", "sha256"}:
        raise ValueError(f"invalid provenance configuration: {group}")
    expected = _sha(value.get("sha256"), f"configuration {group}")
    if _sha256(_canonical(value.get("value"))) != expected:
        raise ValueError(f"provenance configuration hash drift: {group}")


def _configurations(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(GROUPS):
        raise ValueError("provenance configurations must contain the exact eight groups")
    for group in GROUPS:
        _configuration(value[group], group)


def _real_data_status(value: object) -> None:
    expected = {"status": "NOT_RUN", "reason": "owner post-delivery acceptance"}
    if value != expected:
        raise ValueError("real-data acceptance must remain NOT_RUN for owner post-delivery acceptance")


def validate_provenance(
    value: object,
    source_commit: object,
    source_tree: object,
) -> tuple[dict[str, object], list[tuple[str, int, str]]]:
    if not isinstance(value, dict) or set(value) != PROVENANCE_KEYS:
        raise ValueError("invalid reference provenance")
    if value.get("schema") != "xrr-r22-reference-provenance-v1":
        raise ValueError("unexpected reference provenance schema")
    if value.get("source_commit") != source_commit:
        raise ValueError("reference provenance source commit drift")
    if value.get("source_tree") != source_tree:
        raise ValueError("reference provenance source tree drift")
    _embedded_file(value.get("generator"), "generator")
    _embedded_file(value.get("collection_lock"), "collection lock")
    _environment(value.get("python"), {"implementation", "version"}, "Python")
    _environment(value.get("platform"), {"system", "machine"}, "platform")
    records = _inputs(value.get("inputs"))
    _seeds(value.get("seeds"))
    _configurations(value.get("configurations"))
    _real_data_status(value.get("real_data_acceptance"))
    return value, records
