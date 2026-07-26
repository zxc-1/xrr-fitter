"""Closed comparison-policy engine for normalized R22 reference values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def _failure(detail: str) -> None:
    raise ValueError(f"reference comparison failed: {detail}")


def _array_pair(reference: object, actual: object, mismatch: str):
    import numpy as np

    reference_array = np.asarray(reference)
    actual_array = np.asarray(actual)
    if reference_array.dtype != actual_array.dtype or reference_array.shape != actual_array.shape:
        _failure(mismatch)
    return reference_array, actual_array


def _array_equal(reference: object, actual: object) -> None:
    import numpy as np

    reference_array, actual_array = _array_pair(
        reference,
        actual,
        "array dtype or shape drift",
    )
    if not np.array_equal(actual_array, reference_array, equal_nan=True):
        _failure("array value drift")


def _exact(reference: object, actual: object) -> None:
    import numpy as np

    if type(reference) is not type(actual):
        _failure("exact value type drift")
    if isinstance(reference, np.ndarray):
        _array_equal(reference, actual)
    elif isinstance(reference, Mapping):
        _mapping(reference, actual, {key: "exact" for key in reference})
    elif isinstance(reference, (list, tuple)):
        _sequence(reference, actual, ["exact"] * len(reference))
    elif reference != actual:
        _failure("exact value drift")


def _physics_reflectivity(reference: object, actual: object) -> None:
    import numpy as np

    reference_array, actual_array = _array_pair(
        reference,
        actual,
        "physics array dtype or shape drift",
    )
    if not np.issubdtype(reference_array.dtype, np.number) or not np.issubdtype(actual_array.dtype, np.number):
        _failure("physics arrays must be numeric")
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(actual_array)):
        _failure("physics arrays must be finite")
    tolerance = np.where(reference_array >= 1e-12, 1e-10 + 5e-7 * reference_array, 1e-12)
    if np.any(np.abs(actual_array - reference_array) > tolerance):
        _failure("physics reflectivity tolerance exceeded")


def _mapping(reference: object, actual: object, fields: object) -> None:
    if not isinstance(reference, Mapping) or not isinstance(actual, Mapping):
        _failure("mapping policy requires mappings")
    if not isinstance(fields, dict) or set(reference) != set(actual) or set(reference) != set(fields):
        _failure("mapping field set drift")
    for key in reference:
        compare_value(reference[key], actual[key], fields[key])


def _sequence_value(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _failure("sequence policy requires sequences")
    return value


def _sequence(reference: object, actual: object, items: object) -> None:
    reference = _sequence_value(reference)
    actual = _sequence_value(actual)
    if not isinstance(items, list) or len(reference) != len(actual) or len(reference) != len(items):
        _failure("sequence length drift")
    for reference_item, actual_item, policy in zip(reference, actual, items, strict=True):
        compare_value(reference_item, actual_item, policy)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _failure(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _failure(f"{label} must be finite")
    return number


def _scalar_tolerance(reference: object, actual: object, policy: dict[str, object]) -> None:
    if set(policy) != {"kind", "absolute", "relative"}:
        _failure("invalid scalar tolerance policy")
    expected = _finite_number(reference, "reference scalar")
    observed = _finite_number(actual, "actual scalar")
    absolute = _finite_number(policy["absolute"], "absolute tolerance")
    relative = _finite_number(policy["relative"], "relative tolerance")
    if absolute < 0.0 or relative < 0.0:
        _failure("scalar tolerances must be nonnegative")
    if abs(observed - expected) > absolute + relative * abs(expected):
        _failure("scalar tolerance exceeded")


def _scalar_bounds(reference: object, actual: object, policy: dict[str, object]) -> None:
    if set(policy) != {"kind", "minimum", "maximum"}:
        _failure("invalid scalar bounds policy")
    _finite_number(reference, "reference scalar")
    observed = _finite_number(actual, "actual scalar")
    minimum = _finite_number(policy["minimum"], "minimum bound")
    maximum = _finite_number(policy["maximum"], "maximum bound")
    if minimum > maximum or observed < minimum or observed > maximum:
        _failure("scalar bounds violated")


def _structured(reference: object, actual: object, policy: dict[str, object]) -> None:
    kind = policy.get("kind")
    if kind == "mapping" and set(policy) == {"kind", "fields"}:
        _mapping(reference, actual, policy["fields"])
    elif kind == "sequence" and set(policy) == {"kind", "items"}:
        _sequence(reference, actual, policy["items"])
    elif kind == "scalar_tolerance":
        _scalar_tolerance(reference, actual, policy)
    elif kind == "scalar_bounds":
        _scalar_bounds(reference, actual, policy)
    else:
        _failure(f"unknown comparison policy: {kind}")


def compare_value(reference: object, actual: object, policy: object = "exact") -> None:
    if policy == "exact":
        _exact(reference, actual)
    elif policy == "array_equal":
        _array_equal(reference, actual)
    elif policy == "physics_reflectivity":
        _physics_reflectivity(reference, actual)
    elif isinstance(policy, dict):
        _structured(reference, actual, policy)
    else:
        _failure(f"unknown comparison policy: {policy}")
