"""Bounded automatic dataset search and publication."""

from __future__ import annotations

from collections.abc import Callable

from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.fitting import FitCheckpoint, FitSearchResult
from xrr_fitter.services.parallel import OrderedTaskRunner


from .common import (
    AutomaticPreparedResult,
    CancellationProbe,
    PreparedDatasetFit,
    ProgressCallback,
)


def _no_winner_result(search: FitSearchResult) -> FitResult:
    return FitResult.from_search(
        search,
        confidence=ConfidenceClass.UNTRUSTED,
        uncertainty=None,
    )


def _automatic_failure_result(
    prepared: PreparedDatasetFit,
    search: FitSearchResult,
) -> AutomaticPreparedResult | None:
    winner = search.best_candidate
    if winner is None:
        return AutomaticPreparedResult(
            prepared,
            _no_winner_result(search),
            False,
            "no valid candidate",
        )
    if not winner.valid:
        return AutomaticPreparedResult(
            prepared,
            _no_winner_result(search),
            False,
            "no valid candidate",
        )
    return None


def _automatic_reason(decision: object) -> str | None:
    if decision.passed:
        return None
    return "; ".join(decision.reasons) or "automatic quality review required"


def _completed_automatic_result(
    prepared: PreparedDatasetFit,
    fit_result: FitResult,
    decision: object,
) -> AutomaticPreparedResult:
    return AutomaticPreparedResult(
        prepared,
        fit_result,
        decision.passed,
        _automatic_reason(decision),
    )


def _automatic_fast_analysis(
    prepared: PreparedDatasetFit,
    search: FitSearchResult,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    task_runner: Callable,
    analysis_request: Callable,
    run_analysis: Callable,
) -> FitResult:
    return run_analysis(
        analysis_request(
            prepared.dataset_id,
            prepared.problem,
            search,
            profile_names=(),
            bootstrap_enabled=False,
        ),
        cancelled=cancelled,
        progress=progress,
        task_runner=task_runner,
    )

def _automatic_profile_recovery(
    prepared: PreparedDatasetFit,
    search: FitSearchResult,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    checkpoint: Callable[[FitCheckpoint], None] | None,
    task_runner: Callable,
    recover_profile_basin: Callable,
    continue_profile_basin: Callable,
) -> FitSearchResult:
    candidate = search.best_candidate
    if candidate is None:
        return search
    decision = recover_profile_basin(
        prepared.problem,
        candidate,
        cancelled=cancelled,
    )
    if decision is None:
        return search
    return continue_profile_basin(
        prepared.problem,
        search,
        decision.unit_vector,
        parameter_name=decision.parameter_name,
        cancelled=cancelled,
        checkpoint=checkpoint,
        task_runner=task_runner,
    )


def fit_automatic_prepared_dataset(
    prepared: PreparedDatasetFit,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
    local_workers: int | None = None,
    fit_search_request: Callable,
    run_fit_search: Callable,
    analysis_request: Callable,
    run_analysis: Callable,
    assess_automatic_quality: Callable,
    automatic_profile_recovery: Callable,
    automatic_absorption_search: Callable,
    task_runner_factory: Callable = OrderedTaskRunner,
) -> AutomaticPreparedResult:
    """Run the bounded automatic search, quality gates, and final report."""
    workers = prepared.problem.config.local_workers if local_workers is None else local_workers
    if local_workers is not None and local_workers > prepared.problem.config.local_workers:
        raise ValueError("local_workers must fit within the configured worker budget")
    with task_runner_factory(workers) as runner:
        search = run_fit_search(
            fit_search_request(
                prepared.dataset_id,
                prepared.problem,
                prepared.updated_dataset.checkpoint,
            ),
            cancelled=cancelled,
            progress=progress,
            checkpoint=checkpoint,
            task_runner=runner.run,
        )
        failure = _automatic_failure_result(prepared, search)
        if failure is not None:
            return failure
        fast_result = _automatic_fast_analysis(
            prepared,
            search,
            progress=progress,
            cancelled=cancelled,
            task_runner=runner.run,
            analysis_request=analysis_request,
            run_analysis=run_analysis,
        )
        decision = assess_automatic_quality(prepared.problem, fast_result)
        if decision.search_upgrade:
            search = automatic_profile_recovery(
                prepared,
                search,
                progress=progress,
                cancelled=cancelled,
                checkpoint=checkpoint,
                task_runner=runner.run,
            )
            fast_result = _automatic_fast_analysis(
                prepared,
                search,
                progress=progress,
                cancelled=cancelled,
                task_runner=runner.run,
                analysis_request=analysis_request,
                run_analysis=run_analysis,
            )
            decision = assess_automatic_quality(prepared.problem, fast_result)
        if decision.absorption_names:
            updated_prepared, updated = automatic_absorption_search(
                prepared,
                search,
                decision.absorption_names,
                cancelled=cancelled,
            )
            if updated is not search:
                prepared = updated_prepared
                search = updated
                fast_result = _automatic_fast_analysis(
                    prepared,
                    search,
                    progress=progress,
                    cancelled=cancelled,
                    task_runner=runner.run,
                    analysis_request=analysis_request,
                    run_analysis=run_analysis,
                )
                decision = assess_automatic_quality(prepared.problem, fast_result)
        final_result = run_analysis(
            analysis_request(
                prepared.dataset_id,
                prepared.problem,
                search,
                profile_names=decision.profile_names,
                bootstrap_enabled=False,
            ),
            cancelled=cancelled,
            progress=progress,
            task_runner=runner.run,
        )
        final_decision = assess_automatic_quality(prepared.problem, final_result)
        return _completed_automatic_result(prepared, final_result, final_decision)
