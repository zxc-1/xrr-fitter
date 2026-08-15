"""Automatic joint qualification and retry preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import isfinite
from statistics import median

from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.model.parameters import ParameterSetting, SharingRule

from .base import _scale_prior
from .common import (
    AutomaticPreparedResult,
    CancellationProbe,
    PreparedDatasetFit,
    ProgressCallback,
)
from .sharing import _automatic_material_occurrences


def _prefit_isolation_reason(
    prefit: AutomaticPreparedResult,
) -> str | None:
    best = prefit.fit_result.best_candidate
    if best is None:
        return "prefit has no valid candidate"
    if not best.valid or not isfinite(best.objective):
        return "prefit candidate is invalid"
    if best.diagnostics:
        return f"prefit physical diagnostic: {best.diagnostics[0].code}"
    if not prefit.passed:
        return f"prefit quality failed: {prefit.reason}"
    return None


def _mark_objective_outliers(
    prefits: tuple[AutomaticPreparedResult, ...],
    reasons: list[str | None],
) -> None:
    qualified = tuple(index for index, reason in enumerate(reasons) if reason is None)
    if len(qualified) < 3:
        return
    objectives = tuple(prefits[index].fit_result.best_candidate.objective for index in qualified)
    center = float(median(objectives))
    deviation = float(median(abs(value - center) for value in objectives))
    floor = prefits[qualified[0]].prepared.problem.config.confidence.equivalent_cost_floor
    limit = center + 3.0 * max(deviation, floor)
    for index, objective in zip(qualified, objectives, strict=True):
        if objective > limit:
            reasons[index] = f"prefit objective outlier: {objective:.17g} > {limit:.17g}"


def _automatic_isolation_reasons(
    prefits: tuple[AutomaticPreparedResult, ...],
) -> tuple[str | None, ...]:
    reasons = [_prefit_isolation_reason(prefit) for prefit in prefits]
    _mark_objective_outliers(prefits, reasons)
    return tuple(reasons)


def _automatic_role_prepared(
    prepared: PreparedDatasetFit,
    role: AutomaticRole,
    reason: str | None,
) -> PreparedDatasetFit:
    automation = replace(
        prepared.updated_dataset.automation,
        role=role,
        status=AutomaticStatus.REFINING,
        statistics_member=False,
        reason=reason,
    )
    return replace(
        prepared,
        updated_dataset=replace(
            prepared.updated_dataset,
            automation=automation,
        ),
    )


def _merged_settings(
    prepared: PreparedDatasetFit,
    replacements: dict[str, ParameterSetting],
) -> tuple[ParameterSetting, ...]:
    existing = {setting.name: setting for setting in prepared.updated_dataset.parameter_settings}
    existing.update(replacements)
    ordered_names = tuple(
        definition.name for definition in prepared.problem.parameter_definitions if definition.name in existing
    )
    return tuple(existing[name] for name in ordered_names)


def _recompiled_automatic_prepared(
    prepared: PreparedDatasetFit,
    settings: tuple[ParameterSetting, ...],
    *,
    compile_fit_problem: Callable,
) -> PreparedDatasetFit:
    problem = compile_fit_problem(
        prepared.problem.data,
        prepared.problem.structure,
        prepared.problem.instrument,
        prepared.problem.config,
        settings,
        prepared.problem.constraint_rules,
    )
    updated = replace(
        prepared.updated_dataset,
        parameter_settings=settings,
        scale_prior=_scale_prior(problem),
        last_valid_result=None,
        checkpoint=None,
    )
    return replace(prepared, updated_dataset=updated, problem=problem)


def _unlocked_joint_prepared(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    rules: tuple[SharingRule, ...],
    *,
    compile_fit_problem: Callable,
) -> tuple[PreparedDatasetFit, ...]:
    names_by_dataset: dict[str, set[str]] = {}
    for rule in rules:
        for member in rule.members:
            names_by_dataset.setdefault(member.dataset_id, set()).add(member.parameter_name)
    values = []
    for item, prefit in zip(prepared, prefits, strict=True):
        best = prefit.fit_result.best_candidate
        if best is None or not best.valid:
            raise ValueError(f"joint prefit candidate is invalid: {item.dataset_id}")
        physical = {parameter.name: parameter.value for parameter in best.parameters}
        definitions = {definition.name: definition for definition in item.problem.parameter_definitions}
        replacements = {
            name: ParameterSetting(
                name,
                physical[name],
                definitions[name].lower,
                definitions[name].upper,
                locked=False,
            )
            for name in names_by_dataset.get(item.dataset_id, ())
        }
        values.append(
            _recompiled_automatic_prepared(
                _automatic_role_prepared(item, AutomaticRole.JOINT, None),
                _merged_settings(item, replacements),
                compile_fit_problem=compile_fit_problem,
            )
        )
    return tuple(values)


def _run_automatic_joint_refinement(
    prepared: tuple[PreparedDatasetFit, ...],
    rules: tuple[SharingRule, ...],
    candidates_by_dataset: dict[str, object],
    *,
    progress: ProgressCallback | None,
    cancelled: CancellationProbe | None,
    checkpoint: Callable[[tuple[FitCheckpoint, ...]], None] | None,
    compile_joint_problem: Callable,
    consensus_joint_vector: Callable,
    joint_fit_request: Callable,
    run_joint_fit: Callable,
    analysis_request: Callable,
    run_analysis: Callable,
    assess_automatic_quality: Callable,
    analyze_joint_searches: Callable,
) -> tuple[
    object,
    tuple[FitResult, ...],
    tuple[FitResult, ...],
    tuple[object, ...],
]:
    problem = compile_joint_problem(
        tuple(item.dataset_id for item in prepared),
        tuple(item.problem for item in prepared),
        rules,
    )
    initial = consensus_joint_vector(problem, candidates_by_dataset)
    searches = run_joint_fit(
        joint_fit_request(problem, initial_unit_vector=initial),
        cancelled=cancelled,
        progress=progress,
        checkpoint=checkpoint,
    )
    local_results = tuple(
        run_analysis(
            analysis_request(
                item.dataset_id,
                item.problem,
                search,
                profile_names=(),
                bootstrap_enabled=False,
                parameter_priors=item.updated_dataset.parameter_priors,
            ),
            cancelled=cancelled,
            progress=progress,
        )
        for item, search in zip(prepared, searches, strict=True)
    )
    decisions = tuple(
        assess_automatic_quality(item.problem, result) for item, result in zip(prepared, local_results, strict=True)
    )
    return (
        problem,
        analyze_joint_searches(
            problem,
            searches,
            tuple(item.updated_dataset.parameter_priors for item in prepared),
        ),
        local_results,
        decisions,
    )


def _joint_result_conflicts(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    joint_results: tuple[FitResult, ...],
    local_results: tuple[FitResult, ...],
) -> bool:
    for item, prefit, result, local_result in zip(
        prepared,
        prefits,
        joint_results,
        local_results,
        strict=True,
    ):
        prefit_best = prefit.fit_result.best_candidate
        joint_best = result.best_candidate
        if prefit_best is None or joint_best is None or not joint_best.valid:
            return True
        thresholds = item.problem.config.confidence
        allowed = max(
            abs(prefit_best.objective) * thresholds.equivalent_cost_fraction,
            thresholds.equivalent_cost_floor,
        )
        report = local_result.uncertainty
        systematic = report is not None and report.systematic_residual
        if joint_best.objective > prefit_best.objective + allowed or systematic:
            return True
    return False


def _automatic_joint_result(
    prepared: PreparedDatasetFit,
    result: FitResult,
    decision: object,
) -> AutomaticPreparedResult:
    reason = None if decision.passed else "; ".join(decision.reasons)
    if not decision.passed and not reason:
        reason = "automatic quality review required"
    return AutomaticPreparedResult(prepared, result, decision.passed, reason)


def _material_name_by_path(
    prepared: PreparedDatasetFit,
) -> dict[str, str]:
    return {path: material.name for path, material in _automatic_material_occurrences(prepared)}


def _accepted_material_values(
    prepared: tuple[PreparedDatasetFit, ...],
    results: tuple[FitResult, ...],
    material_rules: tuple[SharingRule, ...],
) -> dict[tuple[str, str], float]:
    results_by_id = {item.dataset_id: result for item, result in zip(prepared, results, strict=True)}
    paths_by_id = {item.dataset_id: _material_name_by_path(item) for item in prepared}
    values = {}
    for rule in material_rules:
        member = rule.members[0]
        result = results_by_id[member.dataset_id]
        best = result.best_candidate
        if best is None or not best.valid:
            continue
        parameter = next(value for value in best.parameters if value.name == member.parameter_name)
        path, family = member.parameter_name.rsplit(".", 1)
        material_name = paths_by_id[member.dataset_id][path]
        values[(material_name, family)] = parameter.value
    return values


def _locked_material_prepared(
    prepared: PreparedDatasetFit,
    material_values: dict[tuple[str, str], float],
    reason: str,
    *,
    compile_fit_problem: Callable,
) -> PreparedDatasetFit:
    isolated = _automatic_role_prepared(
        prepared,
        AutomaticRole.ISOLATED_RETRY,
        reason,
    )
    definitions = {definition.name: definition for definition in isolated.problem.parameter_definitions}
    replacements = {}
    for path, material in _automatic_material_occurrences(isolated):
        for family in ("density_scale", "sld_real_a2", "sld_imag_a2"):
            key = (material.name, family)
            name = f"{path}.{family}"
            if key not in material_values or name not in definitions:
                continue
            definition = definitions[name]
            replacements[name] = ParameterSetting(
                name,
                material_values[key],
                definition.lower,
                definition.upper,
                locked=True,
            )
    return _recompiled_automatic_prepared(
        isolated,
        _merged_settings(isolated, replacements),
        compile_fit_problem=compile_fit_problem,
    )


def _insufficient_joint_results(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    isolation_reasons: tuple[str | None, ...],
) -> tuple[AutomaticPreparedResult, ...]:
    reason = "insufficient qualified points for joint refinement"
    values = []
    for item, prefit, isolation_reason in zip(
        prepared,
        prefits,
        isolation_reasons,
        strict=True,
    ):
        role = AutomaticRole.ISOLATED_RETRY if isolation_reason is not None else AutomaticRole.JOINT
        updated = _automatic_role_prepared(item, role, isolation_reason)
        values.append(AutomaticPreparedResult(updated, prefit.fit_result, False, reason))
    return tuple(values)


def _validated_automatic_joint_inputs(
    prepared: tuple[PreparedDatasetFit, ...],
    prefits: tuple[AutomaticPreparedResult, ...],
    fit_group_id: str,
) -> tuple[
    tuple[PreparedDatasetFit, ...],
    tuple[AutomaticPreparedResult, ...],
]:
    values = tuple(prepared)
    prefit_values = tuple(prefits)
    if not fit_group_id.strip():
        raise ValueError("fit_group_id must not be empty")
    if len(values) != len(prefit_values) or not values:
        raise ValueError("automatic joint inputs must be nonempty and aligned")
    if any(item.dataset_id != prefit.prepared.dataset_id for item, prefit in zip(values, prefit_values, strict=True)):
        raise ValueError("automatic joint dataset order mismatch")
    return values, prefit_values


def _material_only_rules(
    rules: tuple[SharingRule, ...],
) -> tuple[SharingRule, ...]:
    return tuple(
        rule for rule in rules if not any(member.parameter_name.endswith("roughness_a") for member in rule.members)
    )


def _best_candidates_by_dataset(
    prepared: tuple[PreparedDatasetFit, ...],
    results: tuple[FitResult, ...],
) -> dict[str, object]:
    return {item.dataset_id: result.best_candidate for item, result in zip(prepared, results, strict=True)}
