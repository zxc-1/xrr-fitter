"""Strict cross-version tolerance policies for frozen reference replay."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math


Exact = Callable[[object, object], None]
ArrayPair = Callable[[object, object, str], tuple[object, object]]


def _failure(detail: str) -> None:
    raise ValueError(f"reference comparison failed: {detail}")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _failure(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _failure(f"{label} must be finite")
    return result


def _tolerances(policy: dict[str, object], kind: str) -> tuple[float, float]:
    if set(policy) != {"kind", "absolute", "relative"}:
        _failure(f"invalid {kind} policy")
    absolute = _number(policy["absolute"], "absolute tolerance")
    relative = _number(policy["relative"], "relative tolerance")
    if min(absolute, relative) < 0.0:
        _failure(f"{kind} tolerances must be nonnegative")
    return absolute, relative


def compare_array_tolerance(
    reference: object,
    actual: object,
    policy: dict[str, object],
    array_pair: ArrayPair,
) -> None:
    """Require identical numeric axes and bounded finite element drift."""
    import numpy as np

    absolute, relative = _tolerances(policy, "array")
    reference_array, actual_array = array_pair(
        reference,
        actual,
        "array dtype or shape drift",
    )
    if not np.issubdtype(reference_array.dtype, np.number):
        _failure("array tolerance requires numeric arrays")
    finite = np.isfinite(reference_array) & np.isfinite(actual_array)
    matching_nonfinite = (~finite) & (
        (np.isnan(reference_array) & np.isnan(actual_array))
        | (reference_array == actual_array)
    )
    if not np.all(finite | matching_nonfinite):
        _failure("array nonfinite pattern drift")
    difference = np.abs(actual_array[finite] - reference_array[finite])
    allowed = absolute + relative * np.abs(reference_array[finite])
    if np.any(difference > allowed):
        _failure("array tolerance exceeded")


def _float_mapping(
    reference: Mapping,
    actual: Mapping,
    policy: dict[str, object],
    exact: Exact,
) -> None:
    if set(reference) != set(actual):
        _failure("float tree mapping field set drift")
    for key in reference:
        compare_float_tree_tolerance(reference[key], actual[key], policy, exact)


def _float_sequence(
    reference: list | tuple,
    actual: list | tuple,
    policy: dict[str, object],
    exact: Exact,
) -> None:
    if type(reference) is not type(actual) or len(reference) != len(actual):
        _failure("float tree sequence drift")
    for reference_item, actual_item in zip(reference, actual, strict=True):
        compare_float_tree_tolerance(reference_item, actual_item, policy, exact)


def compare_float_tree_tolerance(
    reference: object,
    actual: object,
    policy: dict[str, object],
    exact: Exact,
) -> None:
    """Allow bounded float leaves while keeping every other value exact."""
    absolute, relative = _tolerances(policy, "float tree")
    if type(reference) is float and type(actual) is float:
        allowed = absolute + relative * abs(reference)
        if not all(map(math.isfinite, (reference, actual))):
            _failure("float tree values must be finite")
        if abs(actual - reference) > allowed:
            _failure("float tree tolerance exceeded")
    elif isinstance(reference, Mapping) and isinstance(actual, Mapping):
        _float_mapping(reference, actual, policy, exact)
    elif isinstance(reference, (list, tuple)) and isinstance(actual, (list, tuple)):
        _float_sequence(reference, actual, policy, exact)
    else:
        exact(reference, actual)


def _mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _failure(f"normalized export {label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _failure(f"normalized export {label} must be a sequence")
    return value


def _sha256(value: object) -> None:
    if not isinstance(value, str) or len(value) != 64:
        _failure("normalized export SHA-256 is invalid")
    if any(character not in "0123456789abcdef" for character in value):
        _failure("normalized export SHA-256 is invalid")


def _size(reference: object, actual: object, maximum_delta: int) -> None:
    if type(reference) is not int or type(actual) is not int:
        _failure("normalized export size must be an integer")
    if min(reference, actual) < 0 or abs(actual - reference) > maximum_delta:
        _failure("normalized export size drift exceeded")


def _members(
    reference: object,
    actual: object,
    maximum_delta: int,
    exact: Exact,
) -> None:
    reference_items = _sequence(reference, "members")
    actual_items = _sequence(actual, "members")
    if len(reference_items) != len(actual_items):
        _failure("normalized export member count drift")
    for reference_item, actual_item in zip(reference_items, actual_items, strict=True):
        _mutable_record(reference_item, actual_item, maximum_delta, exact)


def _record_value(
    key: object,
    reference: object,
    actual: object,
    maximum_delta: int,
    exact: Exact,
) -> None:
    if key == "sha256":
        _sha256(reference)
        _sha256(actual)
    elif key == "size":
        _size(reference, actual, maximum_delta)
    elif key == "members":
        _members(reference, actual, maximum_delta, exact)
    else:
        exact(reference, actual)


def _mutable_record(
    reference: object,
    actual: object,
    maximum_delta: int,
    exact: Exact,
) -> None:
    reference_record = _mapping(reference, "artifact")
    actual_record = _mapping(actual, "artifact")
    if set(reference_record) != set(actual_record) or "path" not in reference_record:
        _failure("normalized export artifact fields drift")
    for key in reference_record:
        _record_value(
            key,
            reference_record[key],
            actual_record[key],
            maximum_delta,
            exact,
        )


def _suffixes(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _failure("invalid normalized export suffixes")
    if any(not isinstance(item, str) or not item for item in value):
        _failure("invalid normalized export suffixes")
    if len(value) != len(set(value)):
        _failure("invalid normalized export suffixes")
    return tuple(value)


def _maximum_delta(value: object) -> int:
    if type(value) is not int or value < 0:
        _failure("invalid normalized export maximum size delta")
    return value


def _export_pair(reference: object, actual: object) -> tuple[Mapping, Mapping]:
    reference_export = _mapping(reference, "value")
    actual_export = _mapping(actual, "value")
    fields = {"artifacts", "manifest", "timestamp", "token"}
    if set(reference_export) != fields or set(actual_export) != fields:
        _failure("normalized export field set drift")
    return reference_export, actual_export


def _is_mutable(record: object, suffixes: tuple[str, ...]) -> bool:
    value = _mapping(record, "artifact").get("path")
    return isinstance(value, str) and any(value.endswith(suffix) for suffix in suffixes)


def compare_normalized_export(
    reference: object,
    actual: object,
    policy: dict[str, object],
    exact: Exact,
) -> None:
    """Relax hashes only for declared result artifacts with stable structure."""
    fields = {"kind", "mutable_suffixes", "maximum_size_delta"}
    if set(policy) != fields:
        _failure("invalid normalized export policy")
    suffixes = _suffixes(policy["mutable_suffixes"])
    maximum_delta = _maximum_delta(policy["maximum_size_delta"])
    reference_export, actual_export = _export_pair(reference, actual)
    for key in ("manifest", "timestamp", "token"):
        exact(reference_export[key], actual_export[key])
    reference_items = _sequence(reference_export["artifacts"], "artifacts")
    actual_items = _sequence(actual_export["artifacts"], "artifacts")
    if len(reference_items) != len(actual_items):
        _failure("normalized export artifact count drift")
    for reference_item, actual_item in zip(reference_items, actual_items, strict=True):
        if _is_mutable(reference_item, suffixes):
            _mutable_record(reference_item, actual_item, maximum_delta, exact)
        else:
            exact(reference_item, actual_item)
