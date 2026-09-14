"""Poisson draws and same-estimator refits, composed through runtime callbacks.

Only a requested calibration performs work. Raw advisory flags are retained
separately from the calibrated family decision. Fixed-size, ordered batches
bound live synthetic contexts and keep failures visible: no replacement draws,
optional stopping on a p-value, or saved execution callbacks are involved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256

import numpy as np

from xrr_fitter.analysis.bootstrap_generation import Recompile, bootstrap_source, draw_replicates, poll_cancelled
from xrr_fitter.analysis.bootstrap_samples import TaskRunner, run_tasks
from xrr_fitter.analysis.diagnostics import build_residual_evidence
from xrr_fitter.analysis.residual_statistics import residual_statistics, symmetric_max
from xrr_fitter.model.bootstrap import BootstrapResult, bootstrap_calibration_reason
from xrr_fitter.model.diagnostic_calibration import (
    RESIDUAL_ADVISORY_CODES,
    DiagnosticCalibration,
    DiagnosticRefit,
    DiagnosticStatistic,
)
from xrr_fitter.model.diagnostic_work import DiagnosticRefitWork, combine_refit_work
from xrr_fitter.model.evaluation import ModelEvaluation
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.inference import ResidualEvidence
from xrr_fitter.model.joint_bootstrap_provenance import seal_joint_bootstrap, validate_joint_bootstrap_content
from xrr_fitter.model.provenance import diagnostic_calibration_sha256

DIAGNOSTIC_SEED_DOMAIN = 0x504F495344494147
BATCH_SIZE = 16
MATCH_LIMITS = (1e-4, 1e-6, 1e-6)
ADVISORY_KIND = {
    "suspected_unmodeled_footprint": "footprint",
    "suspected_diffuse_background": "background",
    "surface_thin_layer_residual": "surface",
}
DiagnosticRefitter = Callable[[tuple[FitEvaluationContext, ...]], DiagnosticRefit]


def diagnostic_seed(config: object) -> int:
    sequence = np.random.SeedSequence([config.master_seed, DIAGNOSTIC_SEED_DOMAIN])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _bootstrap_diagnostic_reason(residuals):
    for evidence in residuals:
        if not evidence.executed:
            return evidence.unavailable_reason
        if evidence.systematic or evidence.autocorrelation or evidence.diagnostics:
            return "residual_diagnostics_failed"
    return None


def qualify_poisson_bootstrap(
    bootstrap: BootstrapResult | None,
    problems: tuple[FitEvaluationContext, ...],
    residuals: tuple[ResidualEvidence, ...],
) -> BootstrapResult | None:
    """Keep samples and quantiles, but do not label failed-model support as 95%."""
    if bootstrap is not None and bootstrap.joint_owner_sha256 is not None:
        validate_joint_bootstrap_content(bootstrap)
    if bootstrap is None or not any(problem.config.noise_model == "poisson" for problem in problems):
        return bootstrap
    reason = _bootstrap_diagnostic_reason(residuals)
    if reason == bootstrap.diagnostic_unavailable_reason:
        return bootstrap
    sampling_reason = bootstrap_calibration_reason(bootstrap.successful_samples, bootstrap.failure_rate)
    updated = replace(bootstrap, diagnostic_unavailable_reason=reason, unavailable_reason=sampling_reason or reason)
    if bootstrap.joint_owner_sha256 is not None:
        return seal_joint_bootstrap(updated, bootstrap.candidate_id, bootstrap.joint_owner_sha256)
    return updated


def validate_residual_owner(evidence: ResidualEvidence, owner: str, dataset_id: str | None) -> None:
    """Validate saved numerical ownership without executing physics or refits."""
    if not isinstance(evidence, ResidualEvidence):
        raise TypeError("residual_evidence must be ResidualEvidence")
    if evidence.dataset_id != dataset_id or evidence.owner_sha256 != owner:
        raise ValueError("residual evidence owner does not match the numerical winner")
    calibration = evidence.calibration
    if calibration is not None:
        if calibration.owner_sha256 != owner:
            raise ValueError("diagnostic calibration owner does not match residual evidence")
        if calibration.provenance_sha256 != diagnostic_calibration_sha256(calibration):
            raise ValueError("diagnostic calibration provenance does not match its evidence")


def _full_residuals(problem: FitEvaluationContext, evaluation: ModelEvaluation) -> np.ndarray:
    result = np.full(problem.data.fit_mask.shape, np.nan)
    if evaluation.valid:
        result[problem.data.fit_mask] = evaluation.fit_residuals
    return result


def _raw_evidence(problems, dataset_ids, evaluations, owner):
    return tuple(
        replace(
            build_residual_evidence(
                problem, _full_residuals(problem, evaluation), evaluation.diagnostics, dataset_id=dataset_id
            ),
            owner_sha256=owner,
        )
        for problem, dataset_id, evaluation in zip(problems, dataset_ids, evaluations, strict=True)
    )


def _all_statistics(problems, dataset_ids, evaluations) -> tuple[DiagnosticStatistic, ...]:
    return tuple(
        item
        for problem, dataset_id, evaluation in zip(problems, dataset_ids, evaluations, strict=True)
        for item in residual_statistics(problem, _full_residuals(problem, evaluation), dataset_id)
    )


def _physical(diagnostics):
    return tuple(item for item in diagnostics if item.code not in RESIDUAL_ADVISORY_CODES)


def _effective_member(raw: ResidualEvidence, calibration: DiagnosticCalibration) -> ResidualEvidence:
    if calibration.status == "unavailable":
        return replace(
            raw,
            executed=False,
            systematic=None,
            autocorrelation=None,
            diagnostics=_physical(raw.diagnostics),
            unavailable_reason=calibration.unavailable_reason,
            calibration=calibration,
        )
    rejected = {
        item.kind
        for item in calibration.statistics
        if item.dataset_id == raw.dataset_id and item.adjusted_p_value <= calibration.alpha
    }
    active = tuple(item for item in raw.advisories if ADVISORY_KIND[item.code] in rejected)
    return replace(
        raw,
        executed=True,
        systematic=bool(rejected),
        autocorrelation="acf" in rejected,
        diagnostics=(*_physical(raw.diagnostics), *active),
        unavailable_reason=None,
        calibration=calibration,
    )


def _attach(problems, raw, calibration):
    sealed = replace(calibration, provenance_sha256=diagnostic_calibration_sha256(calibration))
    return tuple(
        _effective_member(member, sealed) if problem.config.noise_model == "poisson" else member
        for problem, member in zip(problems, raw, strict=True)
    )


def _input_reason(problems, raw, evaluations, refit, recompile) -> str | None:
    if any(problem.config.noise_model != "poisson" for problem in problems):
        return "mixed_noise_diagnostic_calibration_unsupported"
    counts = tuple(problem.config.budget.diagnostic_samples for problem in problems)
    if len(set(counts)) != 1:
        return "joint_diagnostic_budget_mismatch"
    if counts[0] < 99:
        return "diagnostic_budget_insufficient"
    if refit is None:
        return "diagnostic_refitter_unavailable"
    if recompile is None:
        return "diagnostic_compiler_unavailable"
    if any(item.raw_systematic is None for item in raw):
        return "insufficient_finite_residuals"
    return _source_reason(problems, evaluations)


def _source_reason(problems, evaluations) -> str | None:
    for problem, evaluation in zip(problems, evaluations, strict=True):
        complete = problem.objective_point_count == np.count_nonzero(problem.data.fit_mask)
        if not complete or np.any(problem.sampling_multipliers != 1):
            return "diagnostic_requires_full_observation_grid"
        mean = evaluation.model_normalized[problem.data.fit_mask]
        if not evaluation.valid or np.any(~np.isfinite(mean)) or np.any(mean < 0.0):
            return "invalid_poisson_diagnostic_source"
    return None


def _call_refit(problems, refit, cancelled) -> DiagnosticRefit:
    poll_cancelled(cancelled)
    # The numerical boundary owns expected failures and their spent work.
    # An escaped exception has unknown accounting and must not become zero work.
    result = refit(problems)
    poll_cancelled(cancelled)
    if not isinstance(result, DiagnosticRefit):
        raise TypeError("diagnostic refitter must return DiagnosticRefit")
    if result.failure_reason is None and len(result.evaluations) != len(problems):
        raise ValueError("diagnostic refitter returned an incomplete member axis")
    return result


def _mean_distance(problems, original, fitted) -> float:
    distance = 0.0
    for problem, first, second in zip(problems, original, fitted, strict=True):
        mask, normalization = problem.data.fit_mask, problem.data.normalization
        left = first.model_normalized[mask] * normalization
        right = second.model_normalized[mask] * normalization
        if np.any(left < 0.0) or np.any(right < 0.0) or np.any((left == 0.0) != (right == 0.0)):
            return float("inf")
        positive = (left > 0.0) & (right > 0.0)
        difference = left[positive] - right[positive]
        # The symmetric Poisson KL sum has no factor of one half.
        distance += float(np.sum(difference * (np.log(left[positive]) - np.log(right[positive]))))
    return distance


def refit_discrepancy(problems, unit, evaluations, observed: DiagnosticRefit) -> tuple[float, float, float] | None:
    """Return the three finite distances, or no tuple for unrepresentable mismatch."""
    if np.asarray(unit).shape != observed.unit_vector.shape:
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            unit_distance = float(np.max(np.abs(unit - observed.unit_vector), initial=0.0))
            first = sum(
                problem.objective_point_count * value.objective
                for problem, value in zip(problems, evaluations, strict=True)
            )
            second = sum(
                problem.objective_point_count * value.objective
                for problem, value in zip(problems, observed.evaluations, strict=True)
            )
            distances = (
                unit_distance,
                abs(first - second),
                _mean_distance(problems, evaluations, observed.evaluations),
            )
    except FloatingPointError:
        return None
    return distances if all(np.isfinite(value) for value in distances) else None


def _observed_calibration(problems, dataset_ids, unit, evaluations, refit, cancelled, calibration):
    observed = _call_refit(problems, refit, cancelled)
    calibration = replace(calibration, refit_nfev=observed.nfev, observed_work=observed.work)
    if observed.failure_reason is not None:
        return replace(
            calibration,
            unavailable_reason=observed.failure_reason,
            failure_reasons=((-1, observed.failure_reason),),
        )
    discrepancy = refit_discrepancy(problems, unit, evaluations, observed)
    if discrepancy is None or any(value > limit for value, limit in zip(discrepancy, MATCH_LIMITS, strict=True)):
        return replace(
            calibration,
            unavailable_reason="diagnostic_refit_mismatch",
            refit_discrepancy=discrepancy,
            failure_reasons=((-1, "diagnostic_refit_mismatch"),),
        )
    try:
        statistics = _all_statistics(problems, dataset_ids, observed.evaluations)
    except FloatingPointError as error:
        reason = f"diagnostic_statistics_unavailable:{error}"
        return replace(
            calibration,
            unavailable_reason=reason,
            failure_reasons=((-1, reason),),
            refit_discrepancy=discrepancy,
        )
    return replace(calibration, statistics=statistics, refit_discrepancy=discrepancy)


@dataclass(frozen=True, slots=True)
class _Replicate:
    values: tuple[float, ...] = ()
    nfev: int = 0
    failure_reason: str | None = None
    work: DiagnosticRefitWork = DiagnosticRefitWork()


def _replicate(contexts, dataset_ids, refit, cancelled, columns) -> _Replicate:
    if isinstance(contexts, str):
        return _Replicate(failure_reason=contexts)
    fitted = _call_refit(contexts, refit, cancelled)
    if fitted.failure_reason is not None:
        return _Replicate(nfev=fitted.nfev, failure_reason=fitted.failure_reason, work=fitted.work)
    try:
        statistics = _all_statistics(contexts, dataset_ids, fitted.evaluations)
    except FloatingPointError as error:
        return _Replicate(
            nfev=fitted.nfev, failure_reason=f"diagnostic_statistics_unavailable:{error}", work=fitted.work
        )
    if tuple((item.dataset_id, item.kind) for item in statistics) != columns:
        return _Replicate(nfev=fitted.nfev, failure_reason="diagnostic_family_changed", work=fitted.work)
    return _Replicate(tuple(item.observed for item in statistics), fitted.nfev, work=fitted.work)


def _collect_batch(calibration, batch):
    offset = calibration.attempted_count
    failures = tuple(
        (offset + index, result.failure_reason)
        for index, result in enumerate(batch)
        if result.failure_reason is not None
    )
    return replace(
        calibration,
        attempted_count=offset + len(batch),
        successful_count=calibration.successful_count + len(batch) - len(failures),
        failure_reasons=calibration.failure_reasons + failures,
        refit_nfev=calibration.refit_nfev + sum(result.nfev for result in batch),
        null_work=combine_refit_work(calibration.null_work, *(result.work for result in batch)),
    )


def _complete(calibration, rows):
    matrix = np.asarray(rows, dtype="<f8")
    result = symmetric_max(matrix)
    statistics = tuple(
        replace(item, center=float(center), scale=float(scale), adjusted_p_value=p_value)
        for item, center, scale, p_value in zip(
            calibration.statistics,
            result.centers,
            result.scales,
            result.adjusted_p_values,
            strict=True,
        )
    )
    null_bytes = str(matrix[1:].shape).encode("ascii") + matrix[1:].tobytes(order="C")
    return replace(
        calibration,
        status="available",
        unavailable_reason=None,
        statistics=statistics,
        tail_count=result.tail_count,
        tie_count=result.tie_count,
        observed_score=result.observed_score,
        null_statistics_sha256=sha256(null_bytes).hexdigest(),
    )


def _simulate(problems, dataset_ids, evaluations, refit, recompile, cancelled, task_runner, progress, calibration):
    sources = tuple(
        bootstrap_source(problem, value.model_normalized, value.fit_residuals)
        for problem, value in zip(problems, evaluations, strict=True)
    )
    rng = np.random.default_rng(calibration.child_seed)
    columns = tuple((item.dataset_id, item.kind) for item in calibration.statistics)
    rows = [tuple(item.observed for item in calibration.statistics)]
    while calibration.attempted_count < calibration.sample_count:
        count = min(BATCH_SIZE, calibration.sample_count - calibration.attempted_count)
        draws = draw_replicates(sources, count, rng, recompile, cancelled)
        tasks = tuple(partial(_replicate, contexts, dataset_ids, refit, cancelled, columns) for contexts in draws)
        batch = run_tasks(tasks, task_runner)
        calibration = _collect_batch(calibration, batch)
        if progress is not None:
            progress(calibration.attempted_count, calibration.sample_count)
        poll_cancelled(cancelled)
        if calibration.failure_reasons:
            return replace(calibration, unavailable_reason="diagnostic_null_refit_failed")
        rows.extend(result.values for result in batch)
    try:
        return _complete(calibration, rows)
    except FloatingPointError as error:
        return replace(calibration, unavailable_reason=f"diagnostic_statistics_unavailable:{error}")


def _validate_member_axes(problems, dataset_ids, evaluations):
    aligned = bool(problems) and len(problems) == len(dataset_ids) == len(evaluations)
    if not aligned or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("diagnostic member axes must be nonempty, aligned, and unique")


def _calibration_requested(problems, raw):
    return any(
        problem.config.noise_model == "poisson" and evidence.raw_systematic
        for problem, evidence in zip(problems, raw, strict=True)
    )


def calibrate_residuals(
    problems: tuple[FitEvaluationContext, ...],
    dataset_ids: tuple[str | None, ...],
    unit_vector: np.ndarray,
    evaluations: tuple[ModelEvaluation, ...],
    *,
    owner_sha256: str,
    refit: DiagnosticRefitter | None = None,
    recompile: Recompile | None = None,
    cancelled: Callable[[], bool] | None = None,
    task_runner: TaskRunner | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[ResidualEvidence, ...]:
    """Calibrate all applicable member detectors with one ordered null family."""
    poll_cancelled(cancelled)
    _validate_member_axes(problems, dataset_ids, evaluations)
    raw = _raw_evidence(problems, dataset_ids, evaluations, owner_sha256)
    if not _calibration_requested(problems, raw):
        return raw
    config = problems[0].config
    calibration = DiagnosticCalibration(
        "unavailable",
        config.budget.diagnostic_samples,
        diagnostic_seed(config),
        owner_sha256,
        unavailable_reason="diagnostic_calibration_pending",
    )
    reason = _input_reason(problems, raw, evaluations, refit, recompile)
    if reason is not None:
        return _attach(problems, raw, replace(calibration, unavailable_reason=reason))
    if progress is not None:
        progress(0, calibration.sample_count)
    calibration = _observed_calibration(problems, dataset_ids, unit_vector, evaluations, refit, cancelled, calibration)
    if not calibration.failure_reasons:
        calibration = _simulate(
            problems,
            dataset_ids,
            evaluations,
            refit,
            recompile,
            cancelled,
            task_runner,
            progress,
            calibration,
        )
    poll_cancelled(cancelled)
    return _attach(problems, raw, calibration)
