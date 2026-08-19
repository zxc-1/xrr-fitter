from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tests.support.synthetic_recovery_instrument_cases import (
    _ambiguous_cases,
    _footprint_cases,
    _instrument_effect_cases,
    _mixed_kalpha_cases,
)
from tests.support.synthetic_recovery_layer_cases import (
    _double_layer_cases,
    _oxide_cap_cases,
    _periodic_cases,
    _single_layer_cases,
)
from tests.support.synthetic_recovery_model import SyntheticCase
from tests.support.synthetic_recovery_model_error_cases import _model_error_cases
from tests.support.synthetic_recovery_runs import (
    _parallel_case_outcomes,
    _run_slow_ambiguous_corpus,
    _run_slow_model_error_corpus,
    _run_slow_statistical_recovery_corpus,
)


@dataclass(frozen=True, slots=True)
class CorpusReport:
    schema: str
    status: str
    case_count: int
    fit_count: int
    category_counts: tuple[tuple[str, int], ...]
    failed_case_ids: tuple[str, ...]


def _recovery_cases() -> tuple[SyntheticCase, ...]:
    return (
        *_single_layer_cases(),
        *_double_layer_cases(),
        *_periodic_cases(),
        *_oxide_cap_cases(),
        *_instrument_effect_cases(),
        *_footprint_cases(),
        *_mixed_kalpha_cases(),
    )


def build_corpus() -> tuple[SyntheticCase, ...]:
    return (*_recovery_cases(), *_ambiguous_cases(), *_model_error_cases())


def _validated_cases(cases: tuple[SyntheticCase, ...]) -> tuple[SyntheticCase, ...]:
    values = tuple(cases)
    case_ids = tuple(case.case_id for case in values)
    if len(values) != 220 or len(set(case_ids)) != len(case_ids):
        raise ValueError("synthetic corpus must contain 220 unique cases")
    return values


def _partition_cases(
    cases: tuple[SyntheticCase, ...],
) -> tuple[tuple[SyntheticCase, ...], tuple[SyntheticCase, ...], tuple[SyntheticCase, ...]]:
    ambiguous = tuple(case for case in cases if case.category == "ambiguous")
    model_error = tuple(case for case in cases if case.category == "model_error")
    excluded = {"ambiguous", "model_error"}
    recovery = tuple(case for case in cases if case.category not in excluded)
    return recovery, ambiguous, model_error


def run_corpus(cases: tuple[SyntheticCase, ...]) -> CorpusReport:
    values = _validated_cases(cases)
    recovery, ambiguous, model_error = _partition_cases(values)
    outcomes = _parallel_case_outcomes(values)
    recovery_outcomes = tuple(
        outcome
        for case, outcome in zip(values, outcomes, strict=True)
        if case.category not in {"ambiguous", "model_error"}
    )
    ambiguous_outcomes = tuple(
        outcome for case, outcome in zip(values, outcomes, strict=True) if case.category == "ambiguous"
    )
    model_error_outcomes = tuple(
        outcome for case, outcome in zip(values, outcomes, strict=True) if case.category == "model_error"
    )
    _run_slow_statistical_recovery_corpus(recovery, recovery_outcomes)
    _run_slow_ambiguous_corpus(ambiguous, ambiguous_outcomes)
    _run_slow_model_error_corpus(model_error, model_error_outcomes)
    counts = Counter(case.category for case in values)
    return CorpusReport(
        "xrr-r23-synthetic-recovery-v1",
        "PASS",
        len(values),
        len(values),
        tuple(sorted(counts.items())),
        (),
    )
