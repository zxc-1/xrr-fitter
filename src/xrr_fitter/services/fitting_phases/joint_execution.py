"""Automatic joint refinement and isolated recovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.fitting import (
    FitCheckpoint,
    FitProgress,
    candidate_selection_objective,
)
from xrr_fitter.model.parameters import SharingRule

from .common import (
    AutomaticPreparedResult,
    CancellationProbe,
    PreparedDatasetFit,
    ProgressCallback,
)
from .joint_analysis import _joint_checkpoints
from .joint_selection import (
    _accepted_material_values,
    _automatic_isolation_reasons,
    _automatic_joint_result,
    _best_candidates_by_dataset,
    _insufficient_joint_results,
    _joint_result_conflicts,
    _locked_material_prepared,
    _material_only_rules,
    _run_automatic_joint_refinement,
    _unlocked_joint_prepared,
    _validated_automatic_joint_inputs,
)
from .sharing import automatic_sharing_rules


def _qualified_joint_refinement(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    fit_group_id: str,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    checkpoint: Callable[[tuple[object, ...]], None] | None,
    compile_fit_problem: Callable,
    compile_joint_problem: Callable,
    consensus_joint_vector: Callable,
    joint_fit_request: Callable,
    run_joint_fit: Callable,
    analysis_request: Callable,
    run_analysis: Callable,
    assess_automatic_quality: Callable,
    analyze_joint_searches: Callable,
) -> tuple[
    tuple[PreparedDatasetFit, ...],
    tuple[FitResult, ...],
    tuple[object, ...],
    tuple[SharingRule, ...],
]:
    initial_rules = automatic_sharing_rules(
        prepared,
        fit_group_id,
        share_roughness=True,
    )
    joint_prepared = _unlocked_joint_prepared(
        prepared,
        prefits,
        initial_rules,
        compile_fit_problem=compile_fit_problem,
    )
    prefit_results = tuple(prefit.fit_result for prefit in prefits)
    _problem, joint_results, local_results, decisions = _run_automatic_joint_refinement(
        joint_prepared,
        initial_rules,
        _best_candidates_by_dataset(joint_prepared, prefit_results),
        progress=progress,
        cancelled=cancelled,
        checkpoint=checkpoint,
        compile_joint_problem=compile_joint_problem,
        consensus_joint_vector=consensus_joint_vector,
        joint_fit_request=joint_fit_request,
        run_joint_fit=run_joint_fit,
        analysis_request=analysis_request,
        run_analysis=run_analysis,
        assess_automatic_quality=assess_automatic_quality,
        analyze_joint_searches=analyze_joint_searches,
    )
    if len(joint_results) != len(joint_prepared):
        raise ValueError("automatic joint result batch size mismatch")
    material_rules = _material_only_rules(initial_rules)
    if _joint_result_conflicts(
        joint_prepared,
        prefits,
        joint_results,
        local_results,
    ):
        _problem, joint_results, _local_results, decisions = _run_automatic_joint_refinement(
            joint_prepared,
            material_rules,
            _best_candidates_by_dataset(joint_prepared, joint_results),
            progress=progress,
            cancelled=cancelled,
            checkpoint=checkpoint,
            compile_joint_problem=compile_joint_problem,
            consensus_joint_vector=consensus_joint_vector,
            joint_fit_request=joint_fit_request,
            run_joint_fit=run_joint_fit,
            analysis_request=analysis_request,
            run_analysis=run_analysis,
            assess_automatic_quality=assess_automatic_quality,
            analyze_joint_searches=analyze_joint_searches,
        )
    return joint_prepared, joint_results, decisions, material_rules


def _qualified_checkpoint_callback(
    checkpoint: Callable[[tuple[object, ...]], None] | None,
    qualified_indices: tuple[int, ...],
    total: int,
) -> Callable[[tuple[FitCheckpoint, ...]], None] | None:
    if checkpoint is None:
        return None

    def publish(values: tuple[FitCheckpoint, ...]) -> None:
        checkpoints = tuple(values)
        if len(checkpoints) != len(qualified_indices):
            raise ValueError("automatic joint checkpoint batch size mismatch")
        expanded: list[object | None] = [None] * total
        for index, value in zip(qualified_indices, checkpoints, strict=True):
            expanded[index] = value
        checkpoint(tuple(expanded))

    return publish


def _isolated_retry_result(
    isolated: PreparedDatasetFit,
    prefit: AutomaticPreparedResult,
    isolation_reason: str,
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    fit_automatic_prepared_dataset: Callable,
    cancellation_exceptions: tuple[type[BaseException], ...],
) -> AutomaticPreparedResult:
    try:
        result = fit_automatic_prepared_dataset(
            isolated,
            progress=progress,
            cancelled=cancelled,
            checkpoint=None,
        )
    except cancellation_exceptions:
        raise
    except Exception as error:
        reason = f"{isolation_reason}; isolated retry failed: {type(error).__name__}: {error}"
        return AutomaticPreparedResult(
            isolated,
            FitResult(
                parameter_definitions=isolated.problem.parameter_definitions,
                candidates=(),
                best_index=None,
                confidence=ConfidenceClass.UNTRUSTED,
                warnings=(reason,),
                child_seeds=(),
                stage_summaries=(),
                region_labels=isolated.problem.region_labels,
                region_weights=isolated.problem.weights,
                uncertainty=None,
            ),
            False,
            reason,
        )
    if result.passed:
        return result
    reasons = tuple(dict.fromkeys((isolation_reason, result.reason)))
    return replace(
        result,
        reason="; ".join(reason for reason in reasons if reason),
    )


def _retry_isolated_results(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    isolation_reasons: tuple[str | None, ...],
    accepted: dict[int, AutomaticPreparedResult],
    material_values: dict[tuple[str, str], float],
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    compile_fit_problem: Callable,
    fit_automatic_prepared_dataset: Callable,
    cancellation_exceptions: tuple[type[BaseException], ...],
) -> tuple[AutomaticPreparedResult, ...]:
    for index, isolation_reason in enumerate(isolation_reasons):
        if isolation_reason is None:
            continue
        isolated = _locked_material_prepared(
            prepared[index],
            material_values,
            isolation_reason,
            compile_fit_problem=compile_fit_problem,
        )
        accepted[index] = _isolated_retry_result(
            isolated,
            prefits[index],
            isolation_reason,
            progress=progress,
            cancelled=cancelled,
            fit_automatic_prepared_dataset=fit_automatic_prepared_dataset,
            cancellation_exceptions=cancellation_exceptions,
        )
    return tuple(accepted[index] for index in range(len(prepared)))


def fit_automatic_joint_group(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    fit_group_id: str,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None = None,
    compile_fit_problem: Callable,
    compile_joint_problem: Callable,
    consensus_joint_vector: Callable,
    joint_fit_request: Callable,
    run_joint_fit: Callable,
    analysis_request: Callable,
    run_analysis: Callable,
    assess_automatic_quality: Callable,
    analyze_joint_searches: Callable,
    fit_automatic_prepared_dataset: Callable,
    cancellation_exceptions: tuple[type[BaseException], ...],
) -> tuple[AutomaticPreparedResult, ...]:
    """Refine qualified prefits jointly and retry isolated points independently."""
    values, prefit_values = _validated_automatic_joint_inputs(
        prepared,
        prefits,
        fit_group_id,
    )
    isolation_reasons = _automatic_isolation_reasons(prefit_values)
    qualified_indices = tuple(index for index, reason in enumerate(isolation_reasons) if reason is None)
    if len(qualified_indices) < 2:
        return _insufficient_joint_results(
            values,
            prefit_values,
            isolation_reasons,
        )

    qualified = tuple(values[index] for index in qualified_indices)
    qualified_prefits = tuple(prefit_values[index] for index in qualified_indices)
    qualified_checkpoint = _qualified_checkpoint_callback(
        checkpoint,
        qualified_indices,
        len(values),
    )
    joint_prepared, joint_results, decisions, material_rules = _qualified_joint_refinement(
        qualified,
        qualified_prefits,
        fit_group_id,
        progress=progress,
        cancelled=cancelled,
        checkpoint=qualified_checkpoint,
        compile_fit_problem=compile_fit_problem,
        compile_joint_problem=compile_joint_problem,
        consensus_joint_vector=consensus_joint_vector,
        joint_fit_request=joint_fit_request,
        run_joint_fit=run_joint_fit,
        analysis_request=analysis_request,
        run_analysis=run_analysis,
        assess_automatic_quality=assess_automatic_quality,
        analyze_joint_searches=analyze_joint_searches,
    )
    accepted = {
        index: _automatic_joint_result(item, result, decision)
        for index, item, result, decision in zip(
            qualified_indices,
            joint_prepared,
            joint_results,
            decisions,
            strict=True,
        )
    }
    material_values = _accepted_material_values(
        joint_prepared,
        joint_results,
        material_rules,
    )
    return _retry_isolated_results(
        values,
        prefit_values,
        isolation_reasons,
        accepted,
        material_values,
        progress=progress,
        cancelled=cancelled,
        compile_fit_problem=compile_fit_problem,
        fit_automatic_prepared_dataset=fit_automatic_prepared_dataset,
        cancellation_exceptions=cancellation_exceptions,
    )


def fit_joint_datasets(
    prepared: tuple[PreparedDatasetFit, ...],
    sharing_rules: tuple,
    constraint_rules: tuple = (),
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationProbe | None = None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None = None,
    compile_joint_problem: Callable,
    joint_fit_request: Callable,
    run_joint_fit: Callable,
    analyze_joint_searches: Callable,
) -> tuple[FitResult, ...]:
    """Run and analyze one joint graph without independent fallback."""
    values = tuple(prepared)
    problem = compile_joint_problem(
        tuple(item.dataset_id for item in values),
        tuple(item.problem for item in values),
        tuple(sharing_rules),
        tuple(constraint_rules),
    )
    searches = run_joint_fit(
        joint_fit_request(problem, _joint_checkpoints(values)),
        cancelled=cancelled,
        progress=progress,
        checkpoint=checkpoint,
    )
    best = searches[0].best_candidate if searches else None
    objective = float("inf") if best is None else candidate_selection_objective(best)
    if progress is not None:
        progress(
            FitProgress(
                None,
                "finalizing",
                0,
                1,
                objective,
                "finalizing joint fit",
            )
        )
    results = analyze_joint_searches(
        problem,
        searches,
        tuple(item.updated_dataset.parameter_priors for item in values),
    )
    if progress is not None:
        progress(FitProgress(None, "finalizing", 1, 1, objective, "completed"))
    return results
