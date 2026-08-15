"""Shared strict-JSON primitives for the project codec."""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np


class ProjectSchemaError(ValueError):
    """Persisted project data violates the R22-compatible schema."""


class ProjectVersionError(ProjectSchemaError):
    """A project declares a schema version this runtime cannot read."""


OPTIONAL_FIELDS = frozenset(
    {
        "active_dataset_id",
        "beam_width_mm",
        "best_index",
        "best_objective",
        "bulk_density_g_cm3",
        "candidate_id",
        "checkpoint",
        "expanded_stack",
        "formula",
        "fit_group_id",
        "import_batch_id",
        "instrument_id",
        "intensity_sigma",
        "last_valid_result",
        "measurement_preset",
        "mcmc",
        "objective",
        "ranking_objective",
        "reason",
        "resolution",
        "resolution_kind",
        "sample_length_mm",
        "s_hat",
        "sharing_key",
        "sld_bands",
        "sld_override_a2",
        "structure",
        "structure_evidence",
        "tau_s_decades",
        "top_roughness_a",
        "uncertainty",
        "warning",
    }
)
NULLABLE_ARRAY_FIELDS = frozenset(
    {
        "acceptance_fraction",
        "correlation_matrix",
        "depth_a",
        "effective_sample_size",
        "imaginary",
        "log_probability",
        "log_residuals_decades",
        "model_normalized",
        "objectives",
        "parameter_sigma",
        "qz_a_inv",
        "real",
        "region_labels",
        "region_weights",
        "roughness_a",
        "samples_physical",
        "sld_a2",
        "sld_depth_a",
        "sld_profile_a2",
        "split_rhat",
        "thickness_a",
        "unit_vector",
        "values",
        "weighted_residuals",
    }
)


def _mapping(
    value: object,
    required: set[str],
    label: str,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectSchemaError(f"{label} must be a JSON object")
    allowed = required | (optional or set())
    missing = required - set(value)
    extra = set(value) - allowed
    if missing or extra:
        raise ProjectSchemaError(f"invalid {label} field set; missing={sorted(missing)!r}, extra={sorted(extra)!r}")
    return value


def _sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProjectSchemaError(f"{label} must be a JSON array")
    return value


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectSchemaError(f"duplicate project field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ProjectSchemaError(f"nonstandard JSON numeric constant: {value}")


def _allows_null(path: tuple[str | int, ...]) -> bool:
    field = path[-1]
    if isinstance(field, str) and field in OPTIONAL_FIELDS:
        return True
    return any(isinstance(part, str) and part in NULLABLE_ARRAY_FIELDS for part in path)


def _linked_path(path: tuple[str | int, ...]) -> object | None:
    linked: object | None = None
    for part in path:
        linked = (linked, part)
    return linked


def _materialize_path(linked_path: object | None) -> tuple[str | int, ...]:
    parts: list[str | int] = []
    cursor = linked_path
    while cursor is not None:
        cursor, part = cursor
        parts.append(part)
    return tuple(reversed(parts))


def _null_validation_children(
    value: object,
    linked_path: object | None,
) -> tuple[tuple[object, object], ...]:
    if isinstance(value, dict):
        return tuple((item, (linked_path, key)) for key, item in reversed(tuple(value.items())))
    if isinstance(value, list):
        return tuple((value[index], (linked_path, index)) for index in range(len(value) - 1, -1, -1))
    return ()


def _reject_disallowed_null(linked_path: object | None) -> None:
    path = _materialize_path(linked_path)
    if path and _allows_null(path):
        return
    location = ".".join(str(part) for part in path) or "<root>"
    raise ProjectSchemaError(f"required project value is null: {location}")


def _validate_nulls(value: object, path: tuple[str | int, ...] = ()) -> None:
    # Persisted constraint expressions are attacker-controlled recursive data.
    # Keep this schema-wide preflight iterative so their dedicated decoder can
    # enforce its explicit depth limit instead of Python's call stack doing so.
    pending: list[tuple[object, object | None]] = [(value, _linked_path(path))]
    while pending:
        current, linked_path = pending.pop()
        if current is None:
            _reject_disallowed_null(linked_path)
            continue
        pending.extend(_null_validation_children(current, linked_path))


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _real_array_to_list(value: np.ndarray) -> list[Any]:
    values = np.asarray(value)
    encoded = values.astype(object)
    encoded[~np.isfinite(values)] = None
    return encoded.tolist()


def _real_array_from_list(value: object, dtype: type = float) -> np.ndarray:
    return np.asarray(_sequence(value, "numeric array"), dtype=dtype)


def _complex_to_dict(value: complex | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"real": value.real, "imag": value.imag}


def _complex_from_dict(value: object, label: str) -> complex | None:
    if value is None:
        return None
    payload = _mapping(value, {"real", "imag"}, label)
    return complex(payload["real"], payload["imag"])


def _complex_array_to_list(value: np.ndarray) -> list[dict[str, float | None]]:
    result: list[dict[str, float | None]] = []
    for item in np.asarray(value, dtype=np.complex128).tolist():
        real = float(item.real) if np.isfinite(item.real) else None
        imag = float(item.imag) if np.isfinite(item.imag) else None
        result.append({"real": real, "imag": imag})
    return result


def _complex_array_from_list(value: object) -> np.ndarray:
    result: list[complex] = []
    for item in _sequence(value, "complex array"):
        payload = _mapping(item, {"real", "imag"}, "complex array item")
        real = np.nan if payload["real"] is None else payload["real"]
        imag = np.nan if payload["imag"] is None else payload["imag"]
        result.append(complex(real, imag))
    return np.asarray(result, dtype=np.complex128)
