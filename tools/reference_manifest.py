"""Validate the closed group schema shared by R22 reference tools."""

from __future__ import annotations

from collections.abc import Collection
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
GROUP_FIELDS = {"artifacts", "comparison_policy", "input_ids"}


def _invalid_path(path: PurePosixPath, value: str) -> bool:
    invalid_parts = any(part in {"", ".", ".."} for part in path.parts)
    return path.is_absolute() or value != path.as_posix() or invalid_parts


def _relative(value: object, group: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid reference group artifact path: {group}")
    path = PurePosixPath(value)
    if _invalid_path(path, value):
        raise ValueError(f"invalid reference group artifact path: {group}")
    return value


def _artifacts(entry: dict[str, object], group: str) -> list[str]:
    value = entry["artifacts"]
    if not isinstance(value, list):
        raise ValueError(f"invalid reference group artifacts: {group}")
    paths = [_relative(path, group) for path in value]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError(f"empty or duplicate reference group: {group}")
    return paths


def _validate_policy(entry: dict[str, object], group: str, paths: list[str]) -> None:
    policy = entry["comparison_policy"]
    if not isinstance(policy, dict) or set(policy) != {"kind", "fields"}:
        raise ValueError(f"invalid reference group policy: {group}")
    fields = policy.get("fields")
    if policy.get("kind") != "mapping" or not isinstance(fields, dict):
        raise ValueError(f"invalid reference group policy: {group}")
    if set(fields) != set(paths):
        raise ValueError(f"reference group policy artifacts drift: {group}")


def _input_ids(entry: dict[str, object], group: str) -> list[str]:
    input_ids = entry["input_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError(f"invalid reference group input binding: {group}")
    if any(not isinstance(value, str) or not value for value in input_ids):
        raise ValueError(f"invalid reference group input binding: {group}")
    if len(input_ids) != len(set(input_ids)):
        raise ValueError(f"duplicate reference group input binding: {group}")
    return input_ids


def _validate_inputs(
    entry: dict[str, object],
    group: str,
    known_input_ids: Collection[str],
) -> None:
    input_ids = _input_ids(entry, group)
    if not set(input_ids).issubset(known_input_ids):
        raise ValueError(f"unknown reference group input binding: {group}")


def _entry(
    value: object,
    group: str,
    known_input_ids: Collection[str],
) -> tuple[dict[str, object], list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid reference group metadata: {group}")
    if "comparison_policy" not in value:
        raise ValueError(f"missing reference group comparison policy: {group}")
    if "input_ids" not in value:
        raise ValueError(f"missing reference group input IDs: {group}")
    if set(value) != GROUP_FIELDS:
        raise ValueError(f"invalid reference group metadata: {group}")
    paths = _artifacts(value, group)
    _validate_policy(value, group, paths)
    _validate_inputs(value, group, known_input_ids)
    return value, paths


def validate_groups(
    groups: object,
    *,
    artifact_paths: Collection[str],
    known_input_ids: Collection[str],
) -> dict[str, dict[str, object]]:
    """Return exact groups after validating artifact and replay-input coverage."""
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError("reference manifest must contain the exact eight groups")
    normalized: dict[str, dict[str, object]] = {}
    assigned: list[str] = []
    for group in GROUPS:
        entry, paths = _entry(groups[group], group, known_input_ids)
        normalized[group] = entry
        assigned.extend(paths)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(artifact_paths):
        raise ValueError("reference group artifacts do not match artifact records")
    return normalized
