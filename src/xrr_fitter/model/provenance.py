"""Canonical ownership seals for immutable fitting and analysis evidence.

Provenance is a pure model concern. It serializes complete immutable value
graphs into canonical JSON and hashes that representation without evaluating
physics, running optimization, or reading external state. Producers attach the
resulting SHA-256 to a published value; consumers recompute it against their
exact ``FitEvaluationContext`` before accepting the graph.

NumPy arrays contribute dtype, shape, and contiguous bytes. Dataclasses retain
their declared field names, mappings use sorted string keys, and nonfinite
floating-point values receive explicit tokens because canonical JSON forbids
NaN and infinity. ``PreparedData.source_path`` is excluded deliberately: source
bytes and every numerical input remain bound, while relocating an otherwise
identical project does not invalidate deterministic evidence.

Search seals cover the full candidate graph and stage history, including
archived invalid candidates. Bootstrap seals cover the complete winner snapshot
and all samples, intervals, and failure evidence; candidate identity is checked
separately by the analysis boundary.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite

import numpy as np

from xrr_fitter.model.fitting import FitEvaluationContext, FitSearchResult

POST_FREEZE_OMITTED_DEFAULTS: dict[tuple[str, str], object] = {
    ("ParameterDefinition", "constrained"): False,
}


def _identity_array(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": sha256(array.tobytes(order="C")).hexdigest(),
    }


def _dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _identity_dataclass(value: object) -> dict[str, object]:
    type_name = type(value).__name__
    return {
        item.name: _identity_value(getattr(value, item.name))
        for item in fields(value)
        if (
            type_name,
            item.name,
        )
        not in POST_FREEZE_OMITTED_DEFAULTS
        or getattr(value, item.name) != POST_FREEZE_OMITTED_DEFAULTS[(type_name, item.name)]
    }


def _identity_float(value: float) -> object:
    if isfinite(value):
        return value
    if np.isnan(value):
        return {"float": "nan"}
    label = "positive-infinity" if value > 0.0 else "negative-infinity"
    return {"float": label}


def _identity_atomic(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _identity_array(value)
    if isinstance(value, np.generic):
        return _identity_value(value.item())
    if isinstance(value, Enum):
        return _identity_value(value.value)
    if isinstance(value, complex):
        return {
            "real": _identity_value(value.real),
            "imag": _identity_value(value.imag),
        }
    raise TypeError(f"unsupported identity atom: {type(value).__name__}")


def _identity_collection(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _identity_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    return [_identity_value(item) for item in value]


def _identity_value(value: object) -> object:
    if _dataclass_instance(value):
        return _identity_dataclass(value)
    if isinstance(value, (np.ndarray, np.generic, Enum, complex)):
        return _identity_atomic(value)
    if isinstance(value, float):
        return _identity_float(value)
    if isinstance(value, (tuple, list, dict)):
        return _identity_collection(value)
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def _identity_sha256(value: object) -> str:
    encoded = json.dumps(
        _identity_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _context_identity(problem: FitEvaluationContext) -> dict[str, object]:
    data = {item.name: getattr(problem.data, item.name) for item in fields(problem.data) if item.name != "source_path"}
    identity = {
        "data": data,
        "structure": problem.structure,
        "instrument": problem.instrument,
        "config": problem.config,
        "parameter_definitions": problem.parameter_definitions,
        "variables": problem.variables,
        "region_labels": problem.region_labels,
        "weights": problem.weights,
        "scale_prior_center": problem.scale_prior_center,
        "scale_prior_tau_decades": problem.scale_prior_tau_decades,
        "scale_prior_reason": problem.scale_prior_reason,
        "warnings": problem.warnings,
    }
    if problem.constraint_rules:
        identity["constraint_rules"] = problem.constraint_rules
    return identity


def _dataclass_payload(value: object, excluded: frozenset[str]) -> dict[str, object]:
    return {item.name: getattr(value, item.name) for item in fields(value) if item.name not in excluded}


def _provenance_sha256(
    problem: FitEvaluationContext,
    payload: dict[str, object],
) -> str:
    return _identity_sha256({"context": _context_identity(problem), **payload})


def fit_search_provenance_sha256(
    problem: FitEvaluationContext,
    result: FitSearchResult,
    *,
    joint_layout_fingerprint: str | None = None,
) -> str:
    """Bind a complete fitting result graph to one evaluation context."""
    payload = _dataclass_payload(result, frozenset({"provenance_sha256"}))
    result_identity: dict[str, object] = {"result": payload}
    if joint_layout_fingerprint is not None:
        result_identity["joint_layout_fingerprint"] = joint_layout_fingerprint
    return _provenance_sha256(problem, result_identity)


def bootstrap_provenance_sha256(
    problem: FitEvaluationContext,
    candidate: object,
    bootstrap: object,
) -> str:
    """Bind complete bootstrap evidence to one context and winner snapshot."""
    excluded = frozenset({"candidate_id", "provenance_sha256"})
    payload = _dataclass_payload(bootstrap, excluded)
    return _provenance_sha256(
        problem,
        {"candidate": candidate, "bootstrap": payload},
    )
