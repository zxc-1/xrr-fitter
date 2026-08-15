"""Compose deterministic uncertainty evidence for one fitted candidate graph.

The request value is a process-safe handoff, not an execution container. Its
constructor validates immutable ownership seals and never evaluates physics.
This matters because the same constructor runs in the parent process and again
after worker unpickling.

Analysis consumes the complete retained candidate graph while selecting only
the final Stage-E scope. Bootstrap evidence is bound to the persisted winner;
profiles, correlations, residual diagnostics, and confidence classification all
use that same identity. Joint candidates retain their global ranking objective
when uncertainty progress and stage history are published.

The module performs no fitting continuation. It may identify uncertainty and
diagnostic evidence, while fit-owned code independently validates and executes
any later profile-basin recovery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from xrr_fitter.analysis.bootstrap import TaskRunner, bootstrap_problem_local
from xrr_fitter.analysis.classification import classify_result_with_evidence
from xrr_fitter.analysis.derivatives import (
    correlation_from_covariance,
    objective_information,
    physical_parameter_jacobian,
    strong_parameter_correlations,
)
from xrr_fitter.analysis.diagnostics import (
    diagnose_residual_patterns,
    ordered_fit_residuals,
    residual_autocorrelation_flag,
)
from xrr_fitter.analysis.mcmc import prior_conflicts, with_parameter_priors
from xrr_fitter.analysis.profiles import (
    _evidence_focused_layout,
    build_problem_profiles,
    select_profile_names,
)
from xrr_fitter.model.analysis import BootstrapResult, FitResult, UncertaintyReport
from xrr_fitter.model.fitting import (
    FitEvaluationContext,
    FitProgress,
    FitSearchResult,
    FitStageSummary,
    candidate_selection_objective,
)
from xrr_fitter.model.parameters import ParameterPrior
from xrr_fitter.model.provenance import (
    bootstrap_provenance_sha256,
    fit_search_provenance_sha256,
)

UNCERTAINTY_SEED_DOMAIN = 0x554E434552544149
WARNING_DIAGNOSTICS = {
    "gauss_hermite_unconverged",
    "ideal_reflectivity_above_one",
}


def _analysis_dataset_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("dataset_id must be a nonempty string or None")
    return value


def _analysis_profile_names(values: object) -> tuple[str, ...] | None:
    if values is None:
        return None
    names = tuple(values)
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError("profile_names must contain unique nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError("profile_names must contain unique nonempty strings")
    return names


def _validate_analysis_members(
    problem: object,
    search_result: object,
    bootstrap: object,
) -> None:
    if not isinstance(problem, FitEvaluationContext):
        raise TypeError("problem must be a FitEvaluationContext")
    if not isinstance(search_result, FitSearchResult):
        raise TypeError("search_result must be a FitSearchResult")
    if bootstrap is not None and not isinstance(bootstrap, BootstrapResult):
        raise TypeError("bootstrap must be a BootstrapResult or None")


def _analysis_parameter_priors(values: object) -> tuple[ParameterPrior, ...]:
    priors = tuple(values)
    if any(not isinstance(value, ParameterPrior) for value in priors):
        raise TypeError("parameter_priors must contain ParameterPrior values")
    return priors


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Pickle-safe data handoff for a service-owned analysis worker.

    ``parameter_priors`` is an analysis sidecar. Ownership is always validated
    against the unmodified fit problem and search result before the sidecar is
    used only for the final prior-conflict annotation.
    """

    dataset_id: str | None
    problem: FitEvaluationContext
    search_result: FitSearchResult
    profile_names: tuple[str, ...] | None = None
    bootstrap: BootstrapResult | None = None
    bootstrap_enabled: bool = True
    parameter_priors: tuple[ParameterPrior, ...] = ()

    def __post_init__(self) -> None:
        dataset_id = _analysis_dataset_id(self.dataset_id)
        names = _analysis_profile_names(self.profile_names)
        _validate_analysis_members(self.problem, self.search_result, self.bootstrap)
        _validate_analysis_ownership(self.problem, self.search_result, self.bootstrap)
        if not isinstance(self.bootstrap_enabled, bool):
            raise TypeError("bootstrap_enabled must be bool")
        priors = _analysis_parameter_priors(self.parameter_priors)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "profile_names", names)
        object.__setattr__(self, "parameter_priors", priors)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), (
            self.dataset_id,
            self.problem,
            self.search_result,
            self.profile_names,
            self.bootstrap,
            self.bootstrap_enabled,
            self.parameter_priors,
        )


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("cancelled")


def _select_candidate(problem: object, candidates: tuple[object, ...]) -> object:
    dimension = len(problem.variables)
    usable = tuple(
        candidate
        for candidate in candidates
        if bool(candidate.valid)
        and candidate.stop_reason != "early_eliminated"
        and isfinite(candidate.objective)
        and isfinite(candidate_selection_objective(candidate))
        and np.asarray(candidate.unit_vector).shape == (dimension,)
    )
    if not usable:
        raise ValueError("uncertainty requires at least one valid candidate")
    return min(usable, key=candidate_selection_objective)


def _candidate_id(candidates: tuple[object, ...], best: object) -> str | None:
    identity_required = any(hasattr(candidate, "candidate_id") for candidate in candidates)
    candidate_id = getattr(best, "candidate_id", None)
    if identity_required and candidate_id is None:
        raise AttributeError("uncertainty report data is missing candidate_id")
    return candidate_id


def _correlation_evidence(
    problem: object,
    unit_vector: np.ndarray,
    names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[str, str, float], ...], np.ndarray]:
    unit_covariance = np.linalg.pinv(
        objective_information(problem, unit_vector),
        rcond=1e-12,
    )
    physical_jacobian = physical_parameter_jacobian(problem, unit_vector)
    physical_covariance = physical_jacobian @ unit_covariance @ physical_jacobian.T
    correlation = correlation_from_covariance(physical_covariance)
    sigma = np.sqrt(np.clip(np.diag(physical_covariance), 0.0, np.inf))
    fraction = problem.config.confidence.boundary_fraction
    boundary_hits = tuple(
        name for name, value in zip(names, unit_vector, strict=True) if value <= fraction or value >= 1.0 - fraction
    )
    strong = strong_parameter_correlations(
        names,
        correlation,
        threshold=problem.config.confidence.strong_correlation,
    )
    return correlation, boundary_hits, strong, sigma


def _profiles(
    problem: object,
    unit_vector: np.ndarray,
    profile_names: tuple[str, ...],
    cancelled: Callable[[], bool] | None,
    progress: Callable[[int, int, str], None] | None = None,
    task_runner: TaskRunner | None = None,
) -> tuple[object, ...]:
    names = tuple(variable.name for variable in problem.variables)
    requested = tuple(name for name in profile_names if name in names or _validate_derived_profile(problem, name))
    if not requested:
        return ()
    profiles = build_problem_profiles(
        problem,
        unit_vector,
        requested,
        cancelled=cancelled,
        task_runner=task_runner,
    )
    for index, name in enumerate(requested, start=1):
        if progress is not None:
            progress(index, len(requested), name)
    return profiles


def _validate_derived_profile(problem: object, name: str) -> bool:
    from xrr_fitter.analysis.binary_profiles import binary_derived_profiles

    derived = {profile.name for profile in binary_derived_profiles(problem)}
    if name not in derived:
        raise ValueError(f"unknown profile parameter: {name}")
    return True


def _residual_evidence(
    problem: object,
    candidate: object,
) -> tuple[bool, tuple[object, ...], bool]:
    derived = diagnose_residual_patterns(problem, candidate)
    diagnostics = {
        (diagnostic.code, diagnostic.point_indices): diagnostic for diagnostic in (*candidate.diagnostics, *derived)
    }
    residuals = ordered_fit_residuals(problem, candidate)
    autocorrelation = bool(residuals.size >= 4 and residual_autocorrelation_flag(residuals))
    return bool(derived) or autocorrelation, tuple(diagnostics.values()), autocorrelation


def build_uncertainty_report(
    problem: FitEvaluationContext,
    candidates: tuple[object, ...],
    *,
    profile_names: tuple[str, ...] = (),
    bootstrap: BootstrapResult | None = None,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    task_runner: TaskRunner | None = None,
    parameter_priors: tuple[ParameterPrior, ...] = (),
) -> UncertaintyReport:
    """Build covariance, profile, bootstrap, and residual evidence."""
    parameter_priors = _analysis_parameter_priors(parameter_priors)
    _check_cancelled(cancelled)
    values = tuple(candidates)
    best = _select_candidate(problem, values)
    unit = np.asarray(best.unit_vector, dtype=float)
    names = tuple(variable.name for variable in problem.variables)
    correlation, boundary_hits, strong_correlations, sigma = _correlation_evidence(
        problem,
        unit,
        names,
    )
    profiles = _profiles(
        problem,
        unit,
        profile_names,
        cancelled,
        progress,
        task_runner,
    )
    _check_cancelled(cancelled)
    systematic, diagnostics, autocorrelation = _residual_evidence(problem, best)
    intervals = () if bootstrap is None else bootstrap.intervals
    failure_rate = 0.0 if bootstrap is None else bootstrap.failure_rate
    return UncertaintyReport(
        correlation_names=names,
        correlation_matrix=correlation,
        parameter_sigma=sigma,
        profiles=profiles,
        bootstrap_intervals=intervals,
        bootstrap_failure_rate=failure_rate,
        boundary_hits=boundary_hits,
        strong_correlations=strong_correlations,
        systematic_residual=systematic,
        diagnostics=diagnostics,
        residual_autocorrelation=autocorrelation,
        candidate_id=_candidate_id(values, best),
        bootstrap_performed=bootstrap is not None,
        prior_conflicts=prior_conflicts(
            with_parameter_priors(problem, parameter_priors),
            unit,
        ),
    )


def uncertainty_seed(config: object) -> int:
    """Derive the deterministic child stream reserved for uncertainty work."""
    return int(
        np.random.SeedSequence([config.master_seed, UNCERTAINTY_SEED_DOMAIN]).generate_state(1, dtype=np.uint64)[0]
    )


def _stage_e_candidates(search_result: FitSearchResult) -> tuple[object, ...]:
    summary = next(
        (value for value in reversed(search_result.stage_summaries) if value.stage in {"E", "stage-e"}),
        None,
    )
    if summary is None:
        raise ValueError("uncertainty requires completed Stage-E candidates")
    by_id = {candidate.candidate_id: candidate for candidate in search_result.candidates}
    if any(candidate_id not in by_id for candidate_id in summary.candidate_ids):
        raise ValueError("uncertainty Stage-E candidate lineage is incomplete")
    candidates = tuple(by_id[candidate_id] for candidate_id in summary.candidate_ids)
    if not candidates:
        raise ValueError("uncertainty requires completed Stage-E candidates")
    return candidates


def _validate_bootstrap_ownership(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
    bootstrap: BootstrapResult,
) -> None:
    expected_names = tuple(variable.name for variable in problem.variables)
    if bootstrap.parameter_names != expected_names:
        raise ValueError("bootstrap parameter names do not match analysis context")
    winner = search_result.best_candidate
    if winner is None:
        raise ValueError("bootstrap requires a valid search_result winner")
    if bootstrap.candidate_id != winner.candidate_id:
        raise ValueError("bootstrap candidate does not match search_result winner")
    expected_provenance = bootstrap_provenance_sha256(problem, winner, bootstrap)
    if bootstrap.provenance_sha256 != expected_provenance:
        raise ValueError("bootstrap provenance does not match analysis context")


def _validate_analysis_ownership(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
    bootstrap: BootstrapResult | None,
) -> None:
    if search_result.parameter_definitions != problem.parameter_definitions:
        raise ValueError("search_result parameter definitions do not match context")
    expected_provenance = fit_search_provenance_sha256(problem, search_result)
    if search_result.provenance_sha256 != expected_provenance:
        raise ValueError("search_result provenance does not match analysis context")
    if bootstrap is not None:
        _validate_bootstrap_ownership(problem, search_result, bootstrap)


def _selected_profile_names(
    problem: object,
    candidates: tuple[object, ...],
    requested: tuple[str, ...] | None,
    bootstrap: BootstrapResult,
    warnings: tuple[str, ...],
    cancelled: Callable[[], bool] | None,
) -> tuple[str, ...]:
    if requested is not None:
        return tuple(requested)
    preliminary = None
    if _evidence_focused_layout(problem):
        preliminary = build_uncertainty_report(
            problem,
            candidates,
            profile_names=(),
            bootstrap=bootstrap,
            cancelled=cancelled,
        )
    return select_profile_names(
        problem,
        preliminary,
        degeneracy_warnings=warnings,
    )


def _diagnostic_warning(problem: object, diagnostic: object) -> str | None:
    if diagnostic.code not in WARNING_DIAGNOSTICS:
        return None
    indices = tuple(int(index) for index in diagnostic.point_indices)
    valid = tuple(index for index in indices if 0 <= index < np.asarray(problem.data.qz_a_inv).size)
    if valid:
        qz = np.asarray(problem.data.qz_a_inv, dtype=float)[list(valid)]
        extent = f"[{float(np.min(qz)):.12g},{float(np.max(qz)):.12g}]"
    else:
        extent = "[]"
    index_text = ",".join(str(index) for index in indices)
    return f"{diagnostic.code}: {diagnostic.message}; full_data_indices=[{index_text}]; qz_a_inv_range={extent}"


def _enrich_search_result(
    problem: object,
    search_result: FitSearchResult,
    report: UncertaintyReport,
) -> FitSearchResult:
    candidate_id = report.candidate_id
    if candidate_id is None:
        raise ValueError("uncertainty report is missing candidate identity")
    winner = next(
        (candidate for candidate in search_result.candidates if candidate.candidate_id == candidate_id),
        None,
    )
    if winner is None:
        raise ValueError("uncertainty report references an unknown candidate")
    diagnostics = {
        (diagnostic.code, diagnostic.point_indices): diagnostic
        for diagnostic in (*winner.diagnostics, *report.diagnostics)
    }
    replacement = replace(winner, diagnostics=tuple(diagnostics.values()))
    candidates = tuple(replacement if candidate is winner else candidate for candidate in search_result.candidates)
    diagnostic_warnings = tuple(
        warning
        for diagnostic in report.diagnostics
        if (warning := _diagnostic_warning(problem, diagnostic)) is not None
    )
    warnings = tuple(dict.fromkeys((*search_result.warnings, *diagnostic_warnings)))
    return FitSearchResult(
        parameter_definitions=search_result.parameter_definitions,
        candidates=candidates,
        best_index=search_result.best_index,
        warnings=warnings,
        child_seeds=search_result.child_seeds,
        stage_summaries=search_result.stage_summaries,
        region_labels=search_result.region_labels,
        region_weights=search_result.region_weights,
    )


def _append_uncertainty_summary(
    search_result: FitSearchResult,
    candidates: tuple[object, ...],
) -> FitSearchResult:
    best = search_result.best_candidate
    if best is None:
        raise ValueError("uncertainty requires a valid Stage-E winner")
    summaries = tuple(value for value in search_result.stage_summaries if value.stage != "uncertainty") + (
        FitStageSummary(
            "uncertainty",
            tuple(candidate.candidate_id for candidate in candidates),
            candidate_selection_objective(best),
            0,
            ("completed",),
        ),
    )
    return FitSearchResult(
        parameter_definitions=search_result.parameter_definitions,
        candidates=search_result.candidates,
        best_index=search_result.best_index,
        warnings=search_result.warnings,
        child_seeds=search_result.child_seeds,
        stage_summaries=summaries,
        region_labels=search_result.region_labels,
        region_weights=search_result.region_weights,
    )


def analyze_search_result(
    problem: FitEvaluationContext,
    search_result: FitSearchResult,
    *,
    profile_names: tuple[str, ...] | None = None,
    bootstrap: BootstrapResult | None = None,
    bootstrap_enabled: bool = True,
    cancelled: Callable[[], bool] | None = None,
    dataset_id: str | None = None,
    progress: Callable[[FitProgress], None] | None = None,
    task_runner: TaskRunner | None = None,
    parameter_priors: tuple[ParameterPrior, ...] = (),
) -> FitResult:
    """Finalize a fitting-only search with deterministic uncertainty evidence."""
    _validate_analysis_members(problem, search_result, bootstrap)
    _validate_analysis_ownership(problem, search_result, bootstrap)
    parameter_priors = _analysis_parameter_priors(parameter_priors)
    candidates = _stage_e_candidates(search_result)
    best = search_result.best_candidate
    if best is None:
        raise ValueError("uncertainty requires a valid Stage-E winner")
    best_objective = candidate_selection_objective(best)

    def publish(stage: str, completed: int, total: int, message: str) -> None:
        if progress is not None:
            progress(
                FitProgress(
                    dataset_id,
                    stage,
                    completed,
                    total,
                    best_objective,
                    message,
                )
            )

    if bootstrap is None and bootstrap_enabled:
        bootstrap_total = problem.config.budget.bootstrap_samples
        publish("bootstrap", 0, bootstrap_total, f"bootstrap 0/{bootstrap_total}")

        def bootstrap_progress(completed: int, total: int) -> None:
            publish(
                "bootstrap",
                completed,
                total,
                f"bootstrap {completed}/{total}",
            )

        bootstrap = bootstrap_problem_local(
            problem,
            best,
            sample_count=bootstrap_total,
            child_seed=uncertainty_seed(problem.config),
            cancelled=cancelled,
            progress=bootstrap_progress,
            task_runner=task_runner,
        )
        _validate_bootstrap_ownership(problem, search_result, bootstrap)
    selected_profiles = _selected_profile_names(
        problem,
        candidates,
        profile_names,
        bootstrap,
        search_result.warnings,
        cancelled,
    )
    if selected_profiles:
        publish("profile", 0, len(selected_profiles), f"profile 0/{len(selected_profiles)}")

    def profile_progress(completed: int, total: int, name: str) -> None:
        publish("profile", completed, total, f"profile {completed}/{total}: {name}")

    report = build_uncertainty_report(
        problem,
        candidates,
        profile_names=selected_profiles,
        bootstrap=bootstrap,
        cancelled=cancelled,
        progress=profile_progress,
        task_runner=task_runner,
        parameter_priors=parameter_priors,
    )
    publish("finalizing", 0, 1, "finalizing")
    enriched = _enrich_search_result(problem, search_result, report)
    candidates = _stage_e_candidates(enriched)
    _check_cancelled(cancelled)
    confidence, evidence = classify_result_with_evidence(
        problem,
        candidates,
        report,
    )
    best = enriched.best_candidate
    if best is None:
        raise ValueError("uncertainty requires a valid Stage-E winner")
    completed = _append_uncertainty_summary(enriched, candidates)
    publish("finalizing", 1, 1, "completed")
    return FitResult.from_search(
        completed,
        confidence=confidence,
        uncertainty=report,
        classification_evidence=evidence,
    )


def run_analysis(
    request: AnalysisRequest,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[FitProgress], None] | None = None,
    task_runner: TaskRunner | None = None,
) -> FitResult:
    """Execute one validated worker request without storing runtime callbacks."""
    if not isinstance(request, AnalysisRequest):
        raise TypeError("request must be an AnalysisRequest")
    return analyze_search_result(
        request.problem,
        request.search_result,
        profile_names=request.profile_names,
        bootstrap=request.bootstrap,
        bootstrap_enabled=request.bootstrap_enabled,
        cancelled=cancelled,
        dataset_id=request.dataset_id,
        progress=progress,
        task_runner=task_runner,
        parameter_priors=request.parameter_priors,
    )
