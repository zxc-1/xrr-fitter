"""Serialize existing failed-fit evidence without evaluating or fitting a model."""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite, isnan

import numpy as np

from tests.support.synthetic_recovery_model import SyntheticCase
from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.fitting import FitCandidate, candidate_selection_objective


def _nonfinite(value: float) -> str:
    if isnan(value):
        return "nan"
    return "+inf" if value > 0 else "-inf"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not isfinite(value):
        return _nonfinite(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _candidate(candidate: FitCandidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "seed_index": candidate.seed_index,
        "valid": candidate.valid,
        "objective": candidate.objective,
        "ranking_objective": candidate.ranking_objective,
        "selection_objective": candidate_selection_objective(candidate),
        "unit_vector": candidate.unit_vector,
        "parameters": [asdict(value) for value in candidate.parameters],
        "stop_reason": candidate.stop_reason,
        "nfev": candidate.nfev,
        "diagnostics": [asdict(value) for value in candidate.diagnostics],
        "search_evidence": [asdict(value) for value in candidate.search_evidence],
    }


def _bootstrap(evidence) -> dict | None:
    if evidence is None:
        return None
    return {
        "parameter_names": evidence.parameter_names,
        "attempted_count": evidence.attempted_count,
        "successful_samples": evidence.successful_samples,
        "failure_rate": evidence.failure_rate,
        "failure_reasons": evidence.failure_reasons,
        "method": evidence.method,
        "unavailable_reason": evidence.unavailable_reason,
        "diagnostic_unavailable_reason": evidence.diagnostic_unavailable_reason,
        "candidate_id": evidence.candidate_id,
        "provenance_sha256": evidence.provenance_sha256,
    }


def _uncertainty(report) -> dict | None:
    if report is None:
        return None
    return {
        "candidate_id": report.candidate_id,
        "bootstrap_failure_rate": report.bootstrap_failure_rate,
        "bootstrap_performed": report.bootstrap_performed,
        "bootstrap": _bootstrap(report.bootstrap_evidence),
        "boundary_hits": report.boundary_hits,
        "prior_conflicts": report.prior_conflicts,
        "systematic_residual": report.systematic_residual,
        "residual_autocorrelation": report.residual_autocorrelation,
        "diagnostics": [asdict(value) for value in report.diagnostics],
    }


def failure_diagnostics(case: SyntheticCase, result: FitResult) -> dict:
    """Keep classification inputs while avoiding large modeled-curve arrays."""
    return _json_value(
        {
            "case": {name: getattr(case, name) for name in ("case_id", "category", "seed", "fit_seed", "expectation")},
            "confidence": result.confidence.value,
            "classification_evidence": result.classification_evidence,
            "warnings": result.warnings,
            "best_index": result.best_index,
            "child_seeds": result.child_seeds,
            "skipped_stages": result.skipped_stages,
            "parameter_definitions": [asdict(value) for value in result.parameter_definitions],
            "candidates": [_candidate(value) for value in result.candidates],
            "stage_summaries": [asdict(value) for value in result.stage_summaries],
            "uncertainty": _uncertainty(result.uncertainty),
        }
    )
