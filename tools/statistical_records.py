"""Strict JSON records for the existing per-case statistical evidence."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, fields

from tests.support.synthetic_recovery_metrics import _metric_truth_is_eligible
from tests.support.synthetic_recovery_model import SyntheticCase
from tests.support.synthetic_recovery_runs import _CaseOutcome

COUNTERS = ("closed_included", "open_interval_covered", "open_interval_total")
FLAGS = ("downgraded", "production_acf_downgraded", "raw_acf_flag")


def require_schema(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} schema differs")
    return value


def nonnegative_real(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} is outside the finite range") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _metric_pair(value: object) -> tuple[str, tuple[float, ...]]:
    if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str):
        raise ValueError("statistical metric pair schema differs")
    family, errors = value
    return family, _metric_values(errors)


def _metric_values(errors: object) -> tuple[float, ...]:
    if not isinstance(errors, list) or not errors:
        raise ValueError("statistical metric errors must be nonempty")
    return tuple(nonnegative_real(error, "metric error") for error in errors)


def _metric_errors(value: object) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not isinstance(value, list):
        raise ValueError("statistical metric list schema differs")
    result = tuple(_metric_pair(pair) for pair in value)
    names = tuple(family for family, _errors in result)
    if names != tuple(sorted(set(names))):
        raise ValueError("statistical metric families must be unique and ordered")
    return result


def _scalar_fields(value: dict) -> None:
    if any(type(value[name]) is not int or value[name] < 0 for name in COUNTERS):
        raise ValueError("statistical counters must be nonnegative integers")
    if any(type(value[name]) is not bool for name in FLAGS):
        raise ValueError("statistical flags must be booleans")


def _category_fields(case: SyntheticCase, outcome: _CaseOutcome) -> None:
    allowed_flags = {
        "ambiguous": {"downgraded"},
        "model_error": {"production_acf_downgraded", "raw_acf_flag"},
    }.get(case.category, set())
    if any(getattr(outcome, name) for name in set(FLAGS) - allowed_flags):
        raise ValueError("statistical flags differ from case category")
    nonmetric = case.category in {"ambiguous", "model_error"} or case.expectation == "mixed_mono_mismatch"
    if nonmetric and (outcome.errors_by_family or any(getattr(outcome, name) for name in COUNTERS)):
        raise ValueError("statistical metrics differ from case category")


def _metric_counts(case: SyntheticCase, outcome: _CaseOutcome) -> None:
    eligible = Counter(metric.family for metric in case.metrics if _metric_truth_is_eligible(metric))
    observed = {family: len(errors) for family, errors in outcome.errors_by_family}
    if any(count > eligible[family] for family, count in observed.items()):
        raise ValueError("statistical metric counts exceed the canonical case")
    if sum(observed.values()) != outcome.closed_included:
        raise ValueError("statistical closed counter differs from metric samples")
    if outcome.open_interval_covered != outcome.open_interval_total:
        raise ValueError("statistical open intervals lack the required per-case evidence")
    if outcome.closed_included + outcome.open_interval_total > sum(eligible.values()):
        raise ValueError("statistical interval counters exceed the canonical case")


def decode_record(case: SyntheticCase, record: object) -> tuple[_CaseOutcome, float]:
    value = require_schema(record, {"outcome", "elapsed_seconds"}, "statistical record")
    outcome_value = require_schema(value["outcome"], {field.name for field in fields(_CaseOutcome)}, "outcome")
    if outcome_value["case_id"] != case.case_id:
        raise ValueError("statistical outcome case identity differs")
    _scalar_fields(outcome_value)
    outcome = _CaseOutcome(**{**outcome_value, "errors_by_family": _metric_errors(outcome_value["errors_by_family"])})
    _category_fields(case, outcome)
    _metric_counts(case, outcome)
    return outcome, nonnegative_real(value["elapsed_seconds"], "elapsed seconds")


def encode_record(outcome: _CaseOutcome, elapsed_seconds: float) -> dict:
    return {"outcome": asdict(outcome), "elapsed_seconds": nonnegative_real(elapsed_seconds, "elapsed seconds")}
