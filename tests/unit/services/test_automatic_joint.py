from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, prepared_data

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.joint_candidates import consensus_joint_vector
from xrr_fitter.fit.joint_pipeline import JointFitRequest
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
)
from xrr_fitter.model.fitting import FitConfig, FitSearchResult
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting
from xrr_fitter.services import fitting
from xrr_fitter.services.fitting_phases import joint_execution, sharing
from xrr_fitter.services.materials import automatic_structure

FIT_GROUP_ID = "fit-group-17"
LAYERS = ("Zr", "Zr", "CrSiC", "CrSiC")
DIRECT_INDICES = (2, 3)


def _prepared(
    dataset_id: str,
    dataset_index: int,
    *,
    released_imag: tuple[int, ...] = DIRECT_INDICES,
    free_imag: tuple[int, ...] = (),
) -> fitting.PreparedDatasetFit:
    structure, automatic_settings = automatic_structure(LAYERS, "sapphire")
    settings = (
        *automatic_settings,
        *tuple(
            ParameterSetting(
                f"component.{index}.sld_imag_a2",
                1e-6,
                0.0,
                20e-6,
                locked=index not in free_imag,
            )
            for index in released_imag
        ),
    )
    config = replace(
        FitConfig.fast(1701),
        local_workers=1,
        scale_prior_enabled=False,
    )
    problem = compile_fit_problem(
        prepared_data(),
        structure,
        InstrumentSpec(footprint_mode="none", instrument_id="automatic-lab"),
        config,
        settings,
    )
    automation = DatasetAutomation(
        import_batch_id="batch-1",
        fit_group_id=FIT_GROUP_ID,
        role=AutomaticRole.JOINT,
        status=AutomaticStatus.REFINING,
    )
    dataset = replace(
        dataset_project(dataset_id),
        structure=structure,
        instrument=problem.instrument,
        parameter_settings=settings,
        automation=automation,
    )
    return fitting.PreparedDatasetFit(
        dataset_id,
        dataset_index,
        dataset,
        problem,
    )


def _rule_with_suffix(rules, suffix: str):
    return next(rule for rule in rules if any(member.parameter_name.endswith(suffix) for member in rule.members))


def _fit_result(
    prepared: fitting.PreparedDatasetFit,
    objective: float,
    *,
    density_scale: float | None = None,
) -> FitResult:
    physical = {}
    if density_scale is not None:
        physical.update(
            {
                "component.0.density_scale": density_scale,
                "component.1.density_scale": density_scale,
            }
        )
    unit = encode_physical_vector(prepared.problem, physical)
    evaluation = evaluate_vector(prepared.problem, unit)
    assert evaluation.valid
    candidate = candidate_from_evaluation(
        prepared.problem,
        unit,
        evaluation,
        candidate_id=f"{prepared.dataset_id}-candidate",
        seed_index=0,
        stop_reason="converged",
        nfev=1,
    )
    candidate = replace(
        candidate,
        objective=objective,
        ranking_objective=objective,
    )
    search = FitSearchResult(
        prepared.problem.parameter_definitions,
        (candidate,),
        0,
        (),
        (),
        (),
        prepared.problem.region_labels,
        prepared.problem.weights,
    )
    return FitResult.from_search(
        search,
        confidence=ConfidenceClass.TRUSTED,
        uncertainty=None,
    )


def _prefit(
    prepared: fitting.PreparedDatasetFit,
    objective: float,
    *,
    passed: bool = True,
    reason: str | None = None,
) -> fitting.AutomaticPreparedResult:
    return fitting.AutomaticPreparedResult(
        prepared,
        _fit_result(prepared, objective),
        passed,
        reason,
    )


def _analysis_request(dataset_id, problem, search_result, **kwargs):
    return SimpleNamespace(
        dataset_id=dataset_id,
        problem=problem,
        search_result=search_result,
        **kwargs,
    )


def _passing_local_analysis(*_args, **_kwargs):
    return SimpleNamespace(uncertainty=None)


def _passing_quality(*_args):
    return SimpleNamespace(passed=True, reasons=())


def _unexpected_capability(*_args, **_kwargs):
    raise AssertionError("unexpected fitting capability call")


def _stub_joint_searches(request, **_kwargs):
    return tuple(object() for _item in request.problem.dataset_ids)


def _constant_joint_analysis(results):
    return lambda _problem, _searches, _priors: results


class _JointProjectionHarness:
    def __init__(self, searches, global_results, local_results):
        self.searches = searches
        self.global_results = global_results
        self.local_results = local_results
        self.requests = []
        self.analysis_requests = []

    def run_joint(self, request, **_kwargs):
        self.requests.append(request)
        return self.searches[len(self.requests) - 1]

    def run_analysis(self, request, **_kwargs):
        self.analysis_requests.append(request)
        run, index = divmod(len(self.analysis_requests) - 1, 2)
        return self.local_results[run][index]

    def analyze_joint(self, _problem, _searches, _priors):
        return self.global_results[len(self.requests) - 1]


def _raising_retry(error: Exception):
    def retry(*_args, **_kwargs):
        raise error

    return retry


def _fit_joint_group(
    prepared,
    prefits,
    *,
    run_joint_fit=_unexpected_capability,
    run_analysis=_passing_local_analysis,
    assess_automatic_quality=_passing_quality,
    analyze_joint_searches=_unexpected_capability,
    fit_automatic_prepared_dataset=_unexpected_capability,
    checkpoint=None,
):
    return joint_execution.fit_automatic_joint_group(
        prepared,
        prefits,
        FIT_GROUP_ID,
        checkpoint=checkpoint,
        compile_fit_problem=compile_fit_problem,
        compile_joint_problem=compile_joint_problem,
        consensus_joint_vector=consensus_joint_vector,
        joint_fit_request=JointFitRequest,
        run_joint_fit=run_joint_fit,
        analysis_request=_analysis_request,
        run_analysis=run_analysis,
        assess_automatic_quality=assess_automatic_quality,
        analyze_joint_searches=analyze_joint_searches,
        fit_automatic_prepared_dataset=fit_automatic_prepared_dataset,
        cancellation_exceptions=(SearchCancelled, InterruptedError),
    )


def _objective_outlier_group():
    prepared = tuple(
        _prepared(dataset_id, index, released_imag=()) for index, dataset_id in enumerate(("left", "middle", "outlier"))
    )
    prefits = tuple(_prefit(item, objective) for item, objective in zip(prepared, (1.0, 1.1, 100.0), strict=True))
    joint_results = (
        _fit_result(prepared[0], 0.8, density_scale=0.77),
        _fit_result(prepared[1], 0.9, density_scale=0.77),
    )
    return prepared, prefits, joint_results


def test_automatic_sharing_groups_material_occurrences_and_selected_roughness() -> None:
    prepared = (_prepared("left", 0), _prepared("right", 1))

    rules = sharing.automatic_sharing_rules(
        prepared,
        FIT_GROUP_ID,
        share_roughness=True,
    )

    density = next(rule for rule in rules if "density_scale:Zr" in rule.sharing_key)
    real_sld = next(rule for rule in rules if "sld_real_a2:CrSiC" in rule.sharing_key)
    first_roughness = next(rule for rule in rules if "roughness_a:component.0" in rule.sharing_key)
    assert (
        density.members,
        real_sld.members,
        first_roughness.members,
        all(FIT_GROUP_ID in rule.sharing_key for rule in rules),
        not any(
            member.parameter_name.endswith("thickness_a") or member.parameter_name.startswith("instrument.")
            for rule in rules
            for member in rule.members
        ),
    ) == (
        (
            ParameterReference("left", "component.0.density_scale"),
            ParameterReference("left", "component.1.density_scale"),
            ParameterReference("right", "component.0.density_scale"),
            ParameterReference("right", "component.1.density_scale"),
        ),
        (
            ParameterReference("left", "component.2.sld_real_a2"),
            ParameterReference("left", "component.3.sld_real_a2"),
            ParameterReference("right", "component.2.sld_real_a2"),
            ParameterReference("right", "component.3.sld_real_a2"),
        ),
        (
            ParameterReference("left", "component.0.roughness_a"),
            ParameterReference("right", "component.0.roughness_a"),
        ),
        True,
        True,
    )

    material_only = sharing.automatic_sharing_rules(
        prepared,
        FIT_GROUP_ID,
        share_roughness=False,
    )
    assert not any(member.parameter_name.endswith("roughness_a") for rule in material_only for member in rule.members)


def test_absorption_sharing_requires_release_evidence_for_every_occurrence() -> None:
    complete = (
        _prepared("left", 0, free_imag=DIRECT_INDICES),
        _prepared("right", 1, free_imag=DIRECT_INDICES),
    )
    incomplete = (
        _prepared("left", 0, free_imag=DIRECT_INDICES),
        _prepared("right", 1, released_imag=(2,), free_imag=(2,)),
    )

    complete_rules = sharing.automatic_sharing_rules(
        complete,
        FIT_GROUP_ID,
        share_roughness=False,
    )
    incomplete_rules = sharing.automatic_sharing_rules(
        incomplete,
        FIT_GROUP_ID,
        share_roughness=False,
    )

    imag_sld = _rule_with_suffix(complete_rules, "sld_imag_a2")
    assert imag_sld.members == (
        ParameterReference("left", "component.2.sld_imag_a2"),
        ParameterReference("left", "component.3.sld_imag_a2"),
        ParameterReference("right", "component.2.sld_imag_a2"),
        ParameterReference("right", "component.3.sld_imag_a2"),
    )
    joint = compile_joint_problem(
        tuple(item.dataset_id for item in complete),
        tuple(item.problem for item in complete),
        complete_rules,
    )
    assert joint.global_variables
    assert not any(
        member.parameter_name.endswith("sld_imag_a2") for rule in incomplete_rules for member in rule.members
    )


def test_absorption_sharing_unlocks_evidence_released_coordinates_for_joint_fit() -> None:
    prepared = (_prepared("left", 0), _prepared("right", 1))

    rules = sharing.automatic_sharing_rules(
        prepared,
        FIT_GROUP_ID,
        share_roughness=False,
    )

    imag_sld = _rule_with_suffix(rules, "sld_imag_a2")
    prefits = tuple(_prefit(item, 1.0) for item in prepared)
    unlocked = joint_execution._unlocked_joint_prepared(
        prepared,
        prefits,
        rules,
        compile_fit_problem=compile_fit_problem,
    )
    joint = compile_joint_problem(
        tuple(item.dataset_id for item in unlocked),
        tuple(item.problem for item in unlocked),
        rules,
    )

    assert imag_sld.members == (
        ParameterReference("left", "component.2.sld_imag_a2"),
        ParameterReference("left", "component.3.sld_imag_a2"),
        ParameterReference("right", "component.2.sld_imag_a2"),
        ParameterReference("right", "component.3.sld_imag_a2"),
    )
    assert joint.global_variables


def test_joint_conflict_releases_roughness_once_and_restarts_from_projection() -> None:
    prepared = (
        _prepared("left", 0, released_imag=()),
        _prepared("right", 1, released_imag=()),
    )
    prefits = (
        _prefit(prepared[0], 1.0),
        _prefit(prepared[1], 1.0),
    )
    first_joint = (
        _fit_result(prepared[0], 2.0, density_scale=0.72),
        _fit_result(prepared[1], 0.9, density_scale=0.72),
    )
    second_joint = (
        _fit_result(prepared[0], 0.8, density_scale=0.68),
        _fit_result(prepared[1], 0.8, density_scale=0.68),
    )
    requests = []

    def run_joint(request, **_kwargs):
        requests.append(request)
        return tuple(object() for _item in request.problem.dataset_ids)

    def analyze(_problem, _searches, _priors):
        return first_joint if len(requests) == 1 else second_joint

    results = _fit_joint_group(
        prepared,
        prefits,
        run_joint_fit=run_joint,
        analyze_joint_searches=analyze,
    )

    assert len(requests) == 2
    assert any(
        member.parameter_name.endswith("roughness_a")
        for rule in requests[0].problem.sharing_rules
        for member in rule.members
    )
    assert not any(
        member.parameter_name.endswith("roughness_a")
        for rule in requests[1].problem.sharing_rules
        for member in rule.members
    )
    projected = {
        dataset_id: result.best_candidate
        for dataset_id, result in zip(
            requests[1].problem.dataset_ids,
            first_joint,
            strict=True,
        )
    }
    expected = consensus_joint_vector(requests[1].problem, projected)
    np.testing.assert_allclose(requests[1].initial_unit_vector, expected)
    assert tuple(item.fit_result for item in results) == second_joint
    assert all(item.passed for item in results)


def test_joint_projection_uses_dataset_local_quality_for_release_and_status() -> None:
    prepared = (
        _prepared("left", 0, released_imag=()),
        _prepared("right", 1, released_imag=()),
    )
    prefits = (
        _prefit(prepared[0], 1.0),
        _prefit(prepared[1], 1.0),
    )
    local_results = (
        SimpleNamespace(
            uncertainty=SimpleNamespace(systematic_residual=False),
        ),
        SimpleNamespace(
            uncertainty=SimpleNamespace(systematic_residual=True),
        ),
    )
    harness = _JointProjectionHarness(
        ((object(), object()), (object(), object())),
        (
            (_fit_result(prepared[0], 0.8), _fit_result(prepared[1], 0.9)),
            (_fit_result(prepared[0], 0.7), _fit_result(prepared[1], 0.8)),
        ),
        (local_results, local_results),
    )

    decisions = iter(
        (
            SimpleNamespace(
                passed=True,
                reasons=(),
                systematic_residual=False,
            ),
            SimpleNamespace(
                passed=False,
                reasons=("systematic residual",),
                systematic_residual=True,
            ),
            SimpleNamespace(
                passed=True,
                reasons=(),
                systematic_residual=False,
            ),
            SimpleNamespace(
                passed=False,
                reasons=("systematic residual",),
                systematic_residual=True,
            ),
        )
    )
    results = _fit_joint_group(
        prepared,
        prefits,
        run_joint_fit=harness.run_joint,
        run_analysis=harness.run_analysis,
        assess_automatic_quality=lambda *_args: next(decisions),
        analyze_joint_searches=harness.analyze_joint,
    )

    assert len(harness.requests) == 2
    assert len(harness.analysis_requests) == 4
    assert all(request.profile_names == () for request in harness.analysis_requests)
    assert all(request.bootstrap_enabled is False for request in harness.analysis_requests)
    assert all(request.parameter_priors == () for request in harness.analysis_requests)
    assert tuple(request.dataset_id for request in harness.analysis_requests) == (
        "left",
        "right",
        "left",
        "right",
    )
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].reason == "systematic residual"


def test_objective_outlier_retries_with_joint_material_values_locked() -> None:
    prepared, prefits, joint_results = _objective_outlier_group()
    retried = []

    def retry(item, **_kwargs):
        retried.append(item)
        return fitting.AutomaticPreparedResult(
            item,
            _fit_result(item, 0.7),
            True,
            None,
        )

    results = _fit_joint_group(
        prepared,
        prefits,
        run_joint_fit=_stub_joint_searches,
        analyze_joint_searches=_constant_joint_analysis(joint_results),
        fit_automatic_prepared_dataset=retry,
    )

    assert tuple(item.dataset_id for item in retried) == ("outlier",)
    retry_settings = {setting.name: setting for setting in retried[0].updated_dataset.parameter_settings}
    assert tuple(
        (retry_settings[name].initial, retry_settings[name].locked)
        for name in (
            "component.0.density_scale",
            "component.1.density_scale",
        )
    ) == ((0.77, True), (0.77, True))
    automation = retried[0].updated_dataset.automation
    assert (
        automation.role,
        automation.status,
        automation.statistics_member,
        "objective outlier" in automation.reason,
        tuple(item.prepared.dataset_id for item in results),
        results[2].passed,
    ) == (
        AutomaticRole.ISOLATED_RETRY,
        AutomaticStatus.REFINING,
        False,
        True,
        ("left", "middle", "outlier"),
        True,
    )


def test_isolated_group_expands_qualified_joint_checkpoints_to_input_order() -> None:
    prepared, prefits, joint_results = _objective_outlier_group()
    published = []

    def run_joint(request, *, checkpoint, **_kwargs):
        checkpoint(("left-checkpoint", "middle-checkpoint"))
        return tuple(object() for _item in request.problem.dataset_ids)

    def retry(item, **_kwargs):
        return fitting.AutomaticPreparedResult(
            item,
            _fit_result(item, 0.7),
            True,
            None,
        )

    _fit_joint_group(
        prepared,
        prefits,
        run_joint_fit=run_joint,
        analyze_joint_searches=_constant_joint_analysis(joint_results),
        fit_automatic_prepared_dataset=retry,
        checkpoint=published.append,
    )

    assert published == [
        ("left-checkpoint", "middle-checkpoint", None),
    ]


def test_isolated_retry_review_combines_isolation_and_quality_reasons() -> None:
    prepared, prefits, joint_results = _objective_outlier_group()

    def retry(item, **_kwargs):
        return fitting.AutomaticPreparedResult(
            item,
            _fit_result(item, 0.7),
            False,
            "systematic residual",
        )

    results = _fit_joint_group(
        prepared,
        prefits,
        run_joint_fit=_stub_joint_searches,
        analyze_joint_searches=_constant_joint_analysis(joint_results),
        fit_automatic_prepared_dataset=retry,
    )

    assert results[2].passed is False
    assert "prefit objective outlier" in results[2].reason
    assert "systematic residual" in results[2].reason


def test_isolated_retry_exception_returns_an_unpublishable_failure() -> None:
    prepared, prefits, joint_results = _objective_outlier_group()

    results = _fit_joint_group(
        prepared,
        prefits,
        run_joint_fit=_stub_joint_searches,
        analyze_joint_searches=_constant_joint_analysis(joint_results),
        fit_automatic_prepared_dataset=_raising_retry(RuntimeError("isolated solver failed")),
    )

    assert results[2].fit_result.best_candidate is None
    assert results[2].passed is False
    assert "isolated solver failed" in results[2].reason


@pytest.mark.parametrize(
    "error",
    (
        InterruptedError("cancelled"),
        SearchCancelled("cancelled"),
        type("WrappedInterrupted", (InterruptedError,), {})("cancelled"),
        type("WrappedSearchCancelled", (SearchCancelled,), {})("cancelled"),
    ),
    ids=(
        "interrupted",
        "search-cancelled",
        "interrupted-subclass",
        "search-cancelled-subclass",
    ),
)
def test_isolated_retry_propagates_cancellation(
    error: Exception,
) -> None:
    prepared, prefits, joint_results = _objective_outlier_group()

    with pytest.raises(type(error), match="cancelled"):
        _fit_joint_group(
            prepared,
            prefits,
            run_joint_fit=_stub_joint_searches,
            analyze_joint_searches=_constant_joint_analysis(joint_results),
            fit_automatic_prepared_dataset=_raising_retry(error),
        )


def test_insufficient_qualified_points_keep_prefits_for_review_without_joint() -> None:
    prepared = (
        _prepared("qualified", 0, released_imag=()),
        _prepared("quality-failure", 1, released_imag=()),
    )
    prefits = (
        _prefit(prepared[0], 1.0),
        _prefit(
            prepared[1],
            1.1,
            passed=False,
            reason="systematic residual",
        ),
    )
    results = _fit_joint_group(
        prepared,
        prefits,
    )

    assert all(not item.passed for item in results)
    assert {item.reason for item in results} == {"insufficient qualified points for joint refinement"}
    assert results[1].prepared.updated_dataset.automation.role is (AutomaticRole.ISOLATED_RETRY)
    assert "systematic residual" in (results[1].prepared.updated_dataset.automation.reason)
