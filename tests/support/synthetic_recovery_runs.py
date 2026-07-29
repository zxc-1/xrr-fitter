from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import ceil
import multiprocessing
import os
import warnings

import numpy as np

from tests.support.synthetic_recovery_metrics import (
    _MetricAccumulator,
    _best_candidate,
    _metric_error,
    _values_by_name,
)
from tests.support.synthetic_recovery_model import SyntheticCase, _option_dict
from tests.support.synthetic_recovery_runtime import _fit_case
from xrr_fitter.analysis.diagnostics import residual_autocorrelation_flag
from xrr_fitter.model.analysis import ConfidenceClass
from xrr_fitter.model.data import PreparedData


@dataclass(frozen=True, slots=True)
class _CaseOutcome:
    case_id: str
    errors_by_family: tuple[tuple[str, tuple[float, ...]], ...] = ()
    closed_included: int = 0
    open_interval_covered: int = 0
    open_interval_total: int = 0
    downgraded: bool = False
    production_acf_downgraded: bool = False
    raw_acf_flag: bool = False


WORKER_CASES: dict[str, SyntheticCase] | None = None


def _assert_fit_output_contract(case: SyntheticCase, result, data: PreparedData) -> None:
    best = _best_candidate(result)
    evidence = {
        "case_id": case.case_id,
        "confidence": result.confidence.value,
        "warnings": result.warnings,
        "objective": best.objective,
    }
    model = np.asarray(best.model_normalized, dtype=float)
    assert np.all(np.isfinite(model[data.fit_mask])), evidence
    assert not np.any(np.isinf(model[~data.fit_mask])), evidence
    assert np.all(np.isfinite(best.log_residuals_decades[data.fit_mask])), evidence
    assert np.all(np.isnan(best.log_residuals_decades[~data.fit_mask])), evidence


def _residual_acf_flag(result, data: PreparedData) -> bool:
    best = _best_candidate(result)
    residual = np.asarray(best.log_residuals_decades[data.fit_mask], dtype=float)
    residual = residual[np.isfinite(residual)]
    return bool(residual.size >= 4 and residual_autocorrelation_flag(residual))


def _major_thickness_or_period_bias(case: SyntheticCase, result, limit: float = 0.05) -> bool:
    values = _values_by_name(result)
    for metric in case.metrics:
        if metric.family != "thickness_period":
            continue
        error = _metric_error(metric, values)
        if error > limit:
            return True
    return False


def _assert_footprint_case(case: SyntheticCase, result) -> None:
    values = _values_by_name(result)
    truth = _option_dict(case)["footprint_spill_angle_deg"]
    if case.expectation == "footprint_locked":
        if "instrument.footprint_spill_angle_deg" in values:
            assert abs(values["instrument.footprint_spill_angle_deg"] - truth) <= 1e-12, {
                "case_id": case.case_id,
                "truth": truth,
                "best": values.get("instrument.footprint_spill_angle_deg"),
            }
    elif case.expectation == "footprint_released":
        assert abs(values["instrument.footprint_spill_angle_deg"] - truth) <= 0.05, {
            "case_id": case.case_id,
            "truth": truth,
            "best": values["instrument.footprint_spill_angle_deg"],
        }


def _assert_mixed_mono_exposes_mismatch(case: SyntheticCase, result, data: PreparedData) -> None:
    biased = _major_thickness_or_period_bias(case, result, limit=0.05)
    acf_flag = _residual_acf_flag(result, data)
    uncertainty_flag = bool(result.uncertainty and result.uncertainty.systematic_residual)
    evidence = {
        "case_id": case.case_id,
        "confidence": result.confidence.value,
        "biased": biased,
        "acf_flag": acf_flag,
        "systematic_residual": uncertainty_flag,
        "warnings": result.warnings,
    }
    assert result.confidence is not ConfidenceClass.TRUSTED or not biased, evidence
    assert result.confidence is not ConfidenceClass.TRUSTED or acf_flag or uncertainty_flag, evidence


def _initialize_worker_cases() -> None:
    from tests.support.synthetic_recovery import build_corpus

    global WORKER_CASES
    cases = build_corpus()
    cases_by_id = {case.case_id: case for case in cases}
    if len(cases_by_id) != len(cases):
        raise ValueError("synthetic worker corpus contains duplicate case IDs")
    WORKER_CASES = cases_by_id


def _metric_outcome(case: SyntheticCase, result, data: PreparedData) -> _CaseOutcome:
    accumulator = _MetricAccumulator()
    accumulator.add_case(case, result, data)
    errors = tuple(
        (family, tuple(values))
        for family, values in sorted(accumulator.errors_by_family.items())
    )
    return _CaseOutcome(
        case_id=case.case_id,
        errors_by_family=errors,
        closed_included=accumulator.closed_included,
        open_interval_covered=accumulator.open_interval_covered,
        open_interval_total=accumulator.open_interval_total,
    )


def _recovery_outcome(case: SyntheticCase, result, data: PreparedData) -> _CaseOutcome:
    if case.expectation == "mixed_mono_mismatch":
        _assert_mixed_mono_exposes_mismatch(case, result, data)
        return _CaseOutcome(case.case_id)
    assert result.confidence is not ConfidenceClass.UNTRUSTED, {
        "case_id": case.case_id,
        "confidence": result.confidence.value,
        "warnings": result.warnings,
    }
    outcome = _metric_outcome(case, result, data)
    if case.expectation.startswith("footprint_"):
        _assert_footprint_case(case, result)
    return outcome


def _ambiguous_outcome(case: SyntheticCase, result) -> _CaseOutcome:
    assert result.confidence is not ConfidenceClass.TRUSTED, {
        "case_id": case.case_id,
        "confidence": result.confidence.value,
        "warnings": result.warnings,
    }
    downgraded = result.confidence in {
        ConfidenceClass.MULTIPLE,
        ConfidenceClass.UNTRUSTED,
    }
    return _CaseOutcome(case.case_id, downgraded=downgraded)


def _model_error_outcome(case: SyntheticCase, result, data: PreparedData) -> _CaseOutcome:
    biased = _major_thickness_or_period_bias(case, result, limit=0.05)
    acf_flag = _residual_acf_flag(result, data)
    production_flag = bool(
        result.uncertainty
        and result.uncertainty.residual_autocorrelation
    )
    if production_flag:
        assert result.confidence is not ConfidenceClass.TRUSTED, {
            "case_id": case.case_id,
            "confidence": result.confidence.value,
            "production_systematic_residual": production_flag,
        }
    assert not (result.confidence is ConfidenceClass.TRUSTED and biased), {
        "case_id": case.case_id,
        "model_error_class": case.model_error_class,
        "confidence": result.confidence.value,
        "biased": biased,
        "warnings": result.warnings,
    }
    return _CaseOutcome(
        case.case_id,
        production_acf_downgraded=production_flag,
        raw_acf_flag=acf_flag,
    )


def _fit_worker_case(case_id: str) -> _CaseOutcome:
    if WORKER_CASES is None:
        raise RuntimeError("synthetic worker cases are not initialized")
    case = WORKER_CASES[case_id]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, data = _fit_case(case)
    _assert_fit_output_contract(case, result, data)
    if case.category == "ambiguous":
        return _ambiguous_outcome(case, result)
    if case.category == "model_error":
        return _model_error_outcome(case, result, data)
    return _recovery_outcome(case, result, data)


def _parallel_case_outcomes(cases: tuple[SyntheticCase, ...]) -> tuple[_CaseOutcome, ...]:
    workers = min(5, os.cpu_count() or 1)
    context = multiprocessing.get_context("spawn")
    case_ids = (case.case_id for case in cases)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker_cases,
    ) as executor:
        return tuple(executor.map(_fit_worker_case, case_ids, chunksize=1))


def _assert_ordered_outcome(case: SyntheticCase, outcome: _CaseOutcome) -> None:
    assert outcome.case_id == case.case_id, {
        "expected_case_id": case.case_id,
        "actual_case_id": outcome.case_id,
    }


def _merge_metric_outcome(accumulator: _MetricAccumulator, outcome: _CaseOutcome) -> None:
    for family, values in outcome.errors_by_family:
        accumulator.errors_by_family[family].extend(values)
    accumulator.closed_included += outcome.closed_included
    accumulator.open_interval_covered += outcome.open_interval_covered
    accumulator.open_interval_total += outcome.open_interval_total


def _run_slow_statistical_recovery_corpus(cases: tuple[SyntheticCase, ...]) -> None:
    accumulator = _MetricAccumulator()
    recovery_count = 0
    mixed_mono_count = 0
    outcomes = _parallel_case_outcomes(cases)
    for case, outcome in zip(cases, outcomes, strict=True):
        _assert_ordered_outcome(case, outcome)
        if case.expectation == "mixed_mono_mismatch":
            mixed_mono_count += 1
            continue
        recovery_count += 1
        _merge_metric_outcome(accumulator, outcome)
    assert recovery_count == 160
    assert mixed_mono_count == 20
    accumulator.assert_thresholds()


def _run_slow_ambiguous_corpus(cases: tuple[SyntheticCase, ...]) -> None:
    downgraded = 0
    outcomes = _parallel_case_outcomes(cases)
    for case, outcome in zip(cases, outcomes, strict=True):
        _assert_ordered_outcome(case, outcome)
        downgraded += int(outcome.downgraded)
    assert downgraded >= ceil(0.90 * len(cases)), {
        "downgraded": downgraded,
        "total": len(cases),
    }


def _run_slow_model_error_corpus(cases: tuple[SyntheticCase, ...]) -> None:
    production_acf_downgraded = 0
    raw_acf_flags = 0
    outcomes = _parallel_case_outcomes(cases)
    for case, outcome in zip(cases, outcomes, strict=True):
        _assert_ordered_outcome(case, outcome)
        raw_acf_flags += int(outcome.raw_acf_flag)
        production_acf_downgraded += int(outcome.production_acf_downgraded)
    assert production_acf_downgraded >= ceil(0.70 * len(cases)), {
        "production_acf_downgraded": production_acf_downgraded,
        "raw_acf_flags": raw_acf_flags,
        "total": len(cases),
    }
