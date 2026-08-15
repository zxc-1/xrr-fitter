"""Joint fitting contracts for aligned search state and atomic replay.

The suite exercises real local projection for two datasets, shared-coordinate
ranking, atomic checkpoint batches, resume validation, pickle handoff, and batch
dispatch. Candidate-local objectives remain distinct from the persisted global
ranking objective. A real analysis pass verifies that uncertainty history keeps
that shared ranking instead of silently reverting to one dataset's objective.

The automatic path adds prefit projection, selective parameter sharing, and
roughness-release retries. Checkpoints represent one global attempt even though
the immutable project stores a projection per dataset. Tests keep this identity
visible across fresh runs, resume, cancellation, and result publication so work
cannot be duplicated or assigned to the wrong point.

All numerical fixtures are deliberately bounded and deterministic. Assertions
focus on lineage and ownership instead of optimizer wall time.
"""

from __future__ import annotations

import pickle
from dataclasses import fields, replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, prepared_data, project, simple_structure

from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import (
    FitConfig,
    SearchBudget,
    candidate_selection_objective,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting, SharingRule
from xrr_fitter.model.project import validate_project

SHARED_NAME = "component.0.density_scale"


def _problem(*, seed: int, size: int):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    base = compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none", instrument_id="shared-lab"),
        config,
    )
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name == SHARED_NAME else definition.initial,
            definition.upper if definition.name == SHARED_NAME else definition.initial,
            locked=definition.name != SHARED_NAME,
        )
        for definition in base.parameter_definitions
    )
    return compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        settings,
    )


def _sharing_rules() -> tuple[SharingRule, ...]:
    return (
        SharingRule(
            "film-thickness",
            (
                ParameterReference("left", SHARED_NAME),
                ParameterReference("right", SHARED_NAME),
            ),
        ),
    )


def _joint_problem():
    api = import_module("xrr_fitter.fit.joint_problem")
    return api.compile_joint_problem(
        ("left", "right"),
        (_problem(seed=863, size=40), _problem(seed=863, size=48)),
        _sharing_rules(),
    )


def _fully_locked_problem(*, seed: int, size: int):
    problem = _problem(seed=seed, size=size)
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial,
            definition.initial,
            locked=True,
        )
        for definition in problem.parameter_definitions
    )
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )


def _fully_locked_joint_problem():
    api = import_module("xrr_fitter.fit.joint_problem")
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _fully_locked_problem(seed=867, size=40),
            _fully_locked_problem(seed=867, size=48),
        ),
        (),
    )


def _staged_problem(*, seed: int, size: int):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    problem = compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none", instrument_id="shared-lab"),
        config,
    )
    free_names = {SHARED_NAME, "component.0.thickness_a"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name in free_names else definition.initial,
            definition.upper if definition.name in free_names else definition.initial,
            locked=definition.name not in free_names,
        )
        for definition in problem.parameter_definitions
    )
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
    )


def _staged_joint_problem():
    api = import_module("xrr_fitter.fit.joint_problem")
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _staged_problem(seed=869, size=40),
            _staged_problem(seed=869, size=48),
        ),
        _sharing_rules(),
    )


def _asymmetric_joint_problem():
    api = import_module("xrr_fitter.fit.joint_problem")
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _problem(seed=869, size=40),
            _staged_problem(seed=869, size=48),
        ),
        _sharing_rules(),
    )


def _assert_atomic_checkpoints(checkpoints) -> None:
    assert tuple(batch[0].stage for batch in checkpoints) == (
        "B",
        "C",
        "D",
        "E",
        "E",
        "E",
        "E",
    )
    assert tuple(len(batch[0].child_seeds) for batch in checkpoints) == (1, 1, 1, 2, 3, 4, 5)
    assert all(len(batch) == 2 for batch in checkpoints)
    for batch in checkpoints:
        assert len({value.stage for value in batch}) == 1
        assert len({value.joint_layout_fingerprint for value in batch}) == 1
        assert all(value.runtime_warnings == batch[0].runtime_warnings for value in batch)
        assert tuple(candidate.candidate_id for candidate in batch[0].candidates) == tuple(
            candidate.candidate_id for candidate in batch[1].candidates
        )


def _assert_aligned_results(results) -> None:
    left, right = results
    assert left.best_index == right.best_index
    assert left.warnings == right.warnings
    assert left.child_seeds == right.child_seeds
    assert left.stage_summaries == right.stage_summaries
    assert tuple(candidate.candidate_id for candidate in left.candidates) == tuple(
        candidate.candidate_id for candidate in right.candidates
    )
    for left_candidate, right_candidate in zip(left.candidates, right.candidates, strict=True):
        expected = np.mean((left_candidate.objective, right_candidate.objective))
        assert left_candidate.ranking_objective == pytest.approx(expected)
        assert right_candidate.ranking_objective == pytest.approx(expected)


def _assert_equivalent_results(fresh, resumed) -> None:
    for first, second in zip(fresh, resumed, strict=True):
        assert first.best_index == second.best_index
        assert first.child_seeds == second.child_seeds
        assert first.stage_summaries == second.stage_summaries
        assert tuple(candidate.candidate_id for candidate in first.candidates) == tuple(
            candidate.candidate_id for candidate in second.candidates
        )
        for left, right in zip(first.candidates, second.candidates, strict=True):
            assert left.objective == right.objective
            assert left.ranking_objective == right.ranking_objective
            np.testing.assert_array_equal(left.unit_vector, right.unit_vector)
            np.testing.assert_array_equal(left.model_normalized, right.model_normalized)


def test_joint_request_uses_prefit_consensus_instead_of_declared_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint_pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()
    consensus = np.full(len(joint.global_variables), 0.73)
    observed = []
    evaluate = joint_pipeline.evaluate_joint_vector

    def capture(problem, unit):
        observed.append(unit.copy())
        return evaluate(problem, unit)

    monkeypatch.setattr(joint_pipeline, "evaluate_joint_vector", capture)
    joint_pipeline.run_joint_fit(joint_pipeline.JointFitRequest(joint, initial_unit_vector=consensus))

    assert np.array_equal(observed[0], consensus)


def test_joint_request_schema_includes_optional_initial_vector() -> None:
    joint_pipeline = import_module("xrr_fitter.fit.joint_pipeline")

    assert [field.name for field in fields(joint_pipeline.JointFitRequest)] == [
        "problem",
        "resume_checkpoints",
        "initial_unit_vector",
    ]


def test_joint_request_copies_and_freezes_explicit_initial_vector() -> None:
    joint_pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()
    initial = np.full(len(joint.global_variables), 0.41)

    request = joint_pipeline.JointFitRequest(joint, initial_unit_vector=initial)
    initial[:] = 0.9

    assert np.all(request.initial_unit_vector == 0.41)
    assert not request.initial_unit_vector.flags.writeable


@pytest.mark.parametrize(
    "initial",
    (
        np.asarray(0.5),
        np.asarray([np.nan]),
        np.asarray([-0.1]),
        np.asarray([1.1]),
    ),
)
def test_joint_request_rejects_invalid_explicit_initial_vector(initial) -> None:
    joint_pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()
    if initial.ndim == 1 and initial.size == 1:
        initial = np.resize(initial, len(joint.global_variables))

    with pytest.raises(ValueError, match="initial|shape|finite|bounds"):
        joint_pipeline.JointFitRequest(joint, initial_unit_vector=initial)


def test_joint_request_rejects_explicit_initial_vector_when_resuming() -> None:
    joint_pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()

    with pytest.raises(ValueError, match="resume|initial"):
        joint_pipeline.JointFitRequest(
            joint,
            resume_checkpoints=(object(), object()),
            initial_unit_vector=np.full(len(joint.global_variables), 0.5),
        )


def _replace_first_checkpoint(checkpoints, **changes):
    return (replace(checkpoints[0], **changes), *checkpoints[1:])


def test_joint_pipeline_projects_aligned_results_and_atomic_checkpoints() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    checkpoints: list[tuple[object, ...]] = []
    progress: list[object] = []

    results = api.run_joint_fit(
        api.JointFitRequest(_joint_problem()),
        progress=progress.append,
        checkpoint=checkpoints.append,
    )

    assert len(results) == 2
    _assert_atomic_checkpoints(checkpoints)
    _assert_aligned_results(results)
    expected_ids = ("A-0", "B-0", "C-0", "D-0", "E-0", "E-1", "E-2", "E-3")
    expected_seeds = (
        11356364422455651221,
        2456937646021945087,
        13860652228729505114,
        6048956452989709052,
        7082095701162067864,
    )
    assert all(tuple(candidate.candidate_id for candidate in result.candidates) == expected_ids for result in results)
    assert all(result.child_seeds == expected_seeds for result in results)
    assert tuple(summary.candidate_ids for summary in results[0].stage_summaries) == (
        ("A-0",),
        ("B-0",),
        ("C-0",),
        ("D-0",),
        ("E-0", "E-1", "E-2", "E-3"),
    )
    assert tuple((value.stage, value.completed, value.total, value.message) for value in progress) == (
        ("A", 1, 1, "joint A"),
        ("B", 1, 1, "joint B"),
        ("C", 1, 1, "joint C"),
        ("D", 1, 1, "joint D"),
        ("E", 1, 4, "joint E"),
        ("E", 2, 4, "joint E"),
        ("E", 3, 4, "joint E"),
        ("E", 4, 4, "joint E"),
    )
    assert tuple(tuple(candidate.candidate_id for candidate in batch[0].candidates) for batch in checkpoints) == tuple(
        expected_ids[1:stop] for stop in (2, 3, 4, 5, 6, 7, 8)
    )
    assert tuple(tuple(summary.stage for summary in batch[0].stage_summaries) for batch in checkpoints) == (
        ("B",),
        ("B", "C"),
        ("B", "C", "D"),
        ("B", "C", "D", "E"),
        ("B", "C", "D", "E"),
        ("B", "C", "D", "E"),
        ("B", "C", "D", "E"),
    )
    assert tuple(batch[0].child_seeds for batch in checkpoints) == (
        expected_seeds[:1],
        expected_seeds[:1],
        expected_seeds[:1],
        expected_seeds[:2],
        expected_seeds[:3],
        expected_seeds[:4],
        expected_seeds[:5],
    )


def test_joint_analysis_publishes_coherent_shared_ranking_history() -> None:
    fit_api = import_module("xrr_fitter.fit.joint_pipeline")
    analysis_api = import_module("xrr_fitter.analysis.report")
    joint = _joint_problem()
    searches = fit_api.run_joint_fit(fit_api.JointFitRequest(joint))
    best_candidates = tuple(search.best_candidate for search in searches)
    assert all(best is not None for best in best_candidates)
    assert all(candidate_selection_objective(best) != best.objective for best in best_candidates)
    results = tuple(
        analysis_api.analyze_search_result(problem, search, profile_names=())
        for problem, search in zip(joint.problems, searches, strict=True)
    )

    summaries = tuple(result.stage_summaries[-1] for result in results)
    assert summaries[0] == summaries[1]
    assert all(
        summary.best_objective == candidate_selection_objective(best)
        for summary, best in zip(summaries, best_candidates, strict=True)
    )


def test_joint_analysis_results_are_publishable_as_one_project() -> None:
    fit_api = import_module("xrr_fitter.fit.joint_pipeline")
    analysis_api = import_module("xrr_fitter.analysis.report")
    joint = _joint_problem()
    searches = fit_api.run_joint_fit(fit_api.JointFitRequest(joint))
    results = tuple(
        analysis_api.analyze_search_result(problem, search, profile_names=())
        for problem, search in zip(joint.problems, searches, strict=True)
    )
    datasets = tuple(
        dataset_project(dataset_id, result=result)
        for dataset_id, result in zip(joint.dataset_ids, results, strict=True)
    )

    published = replace(project(*datasets), batch_mode="joint")
    validate_project(published)

    assert tuple(dataset.dataset_id for dataset in published.datasets) == joint.dataset_ids
    assert tuple(dataset.last_valid_result for dataset in published.datasets) == results


def test_joint_analysis_publishes_dataset_local_evidence_to_one_project() -> None:
    fit_api = import_module("xrr_fitter.fit.joint_pipeline")
    analysis_api = import_module("xrr_fitter.analysis.report")
    joint = _asymmetric_joint_problem()
    searches = fit_api.run_joint_fit(fit_api.JointFitRequest(joint))
    results = tuple(
        analysis_api.analyze_search_result(problem, search, profile_names=())
        for problem, search in zip(joint.problems, searches, strict=True)
    )
    assert tuple(len(problem.variables) for problem in joint.problems) == (1, 2)
    assert results[0].best_candidate.objective != results[1].best_candidate.objective
    assert results[0].best_candidate.ranking_objective == results[1].best_candidate.ranking_objective
    assert results[0].uncertainty is not results[1].uncertainty
    datasets = tuple(
        dataset_project(dataset_id, result=result)
        for dataset_id, result in zip(joint.dataset_ids, results, strict=True)
    )

    published = replace(project(*datasets), batch_mode="joint")
    validate_project(published)

    assert tuple(dataset.last_valid_result for dataset in published.datasets) == results


def test_joint_pipeline_preserves_fully_locked_lineage_through_stage_e() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")

    results = api.run_joint_fit(api.JointFitRequest(_fully_locked_joint_problem()))

    assert all(
        tuple(summary.stage for summary in result.stage_summaries) == ("A", "B", "C", "D", "E") for result in results
    )
    for result in results:
        final = tuple(candidate for candidate in result.candidates if candidate.candidate_id.startswith("E-"))
        assert tuple(candidate.candidate_id for candidate in final) == ("E-0", "E-1", "E-2", "E-3")
        assert all(candidate.stop_reason == "no_free_parameters" for candidate in final)
        assert all(candidate.unit_vector.size == 0 for candidate in final)
        assert result.best_candidate in final


def test_joint_pipeline_uses_one_global_layout_short_and_full_de_then_local_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    solvers = import_module("xrr_fitter.fit.joint_solvers")
    de_calls: list[tuple[int, int]] = []
    local_calls: list[int] = []

    def fake_de(objective, bounds, *, init, maxiter, **_kwargs):
        unit = np.array(init[0], dtype=float, copy=True)
        de_calls.append((len(bounds), maxiter))
        value = objective(unit)
        return SimpleNamespace(
            x=unit,
            population=np.array(init, dtype=float, copy=True),
            population_energies=np.full(len(init), value),
            message="captured DE",
            nfev=1,
        )

    def fake_local(residual, unit, *, jac, **_kwargs):
        start = np.array(unit, dtype=float, copy=True)
        local_calls.append(start.size)
        residual(start)
        jac(start)
        return SimpleNamespace(x=start, message="captured local", nfev=1)

    monkeypatch.setattr(solvers, "differential_evolution", fake_de)
    monkeypatch.setattr(solvers, "least_squares", fake_local)

    results = api.run_joint_fit(api.JointFitRequest(_staged_joint_problem()))

    assert de_calls == [(3, 0), (3, 0), (3, 0), (3, 0), (3, 0)]
    assert local_calls == [3, 3, 3, 3, 3, 3]
    assert tuple(summary.candidate_ids for summary in results[0].stage_summaries) == (
        ("A-0",),
        ("B-0",),
        ("C-0",),
        ("D-0",),
        ("E-0", "E-1", "E-2", "E-3"),
    )


def test_joint_request_handler_and_result_tuple_are_pickle_safe() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    request = api.JointFitRequest(_joint_problem())

    assert [field.name for field in fields(request)] == [
        "problem",
        "resume_checkpoints",
        "initial_unit_vector",
    ]
    restored_request = pickle.loads(pickle.dumps(request))
    assert restored_request.problem.dataset_ids == ("left", "right")
    assert pickle.loads(pickle.dumps(api.run_joint_fit)) is api.run_joint_fit

    results = api.run_joint_fit(request)
    restored_results = pickle.loads(pickle.dumps(results))
    assert len(restored_results) == 2
    for restored, original in zip(restored_results, results, strict=True):
        assert tuple(candidate.candidate_id for candidate in restored.candidates) == tuple(
            candidate.candidate_id for candidate in original.candidates
        )


def test_joint_resume_requires_one_checkpoint_for_every_dataset() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()

    with pytest.raises(ValueError, match="resume|checkpoint|dataset|all"):
        api.JointFitRequest(joint, resume_checkpoints=())


@pytest.mark.parametrize(
    ("checkpoint_index", "expected_suffix"),
    [
        pytest.param(0, ("C", "D", "E", "E", "E", "E"), id="from-b"),
        pytest.param(1, ("D", "E", "E", "E", "E"), id="from-c"),
        pytest.param(2, ("E", "E", "E", "E"), id="from-d"),
        pytest.param(3, ("E", "E", "E"), id="from-first-e-seed"),
        pytest.param(6, (), id="from-complete-e"),
    ],
)
def test_joint_resume_runs_only_the_suffix_and_matches_fresh_results(
    checkpoint_index: int,
    expected_suffix: tuple[str, ...],
) -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()
    fresh_checkpoints: list[tuple[object, ...]] = []
    fresh = api.run_joint_fit(
        api.JointFitRequest(joint),
        checkpoint=fresh_checkpoints.append,
    )
    resume_from = fresh_checkpoints[checkpoint_index]
    resumed_checkpoints: list[tuple[object, ...]] = []

    resumed = api.run_joint_fit(
        api.JointFitRequest(joint, resume_checkpoints=resume_from),
        checkpoint=resumed_checkpoints.append,
    )

    assert tuple(batch[0].stage for batch in resumed_checkpoints) == expected_suffix
    _assert_equivalent_results(fresh, resumed)


def test_joint_resume_rejects_cross_dataset_drift_before_callbacks() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()
    fresh_checkpoints: list[tuple[object, ...]] = []
    api.run_joint_fit(api.JointFitRequest(joint), checkpoint=fresh_checkpoints.append)
    stage_d = next(batch for batch in fresh_checkpoints if batch[0].stage == "D")
    first = stage_d[0]
    wrong_rank = replace(
        first.candidates[0],
        ranking_objective=(first.candidates[0].ranking_objective or first.candidates[0].objective) + 0.25,
    )
    changed_unit = np.array(first.candidates[0].unit_vector, copy=True)
    changed_unit[0] = 0.0 if changed_unit[0] > 0.5 else 1.0
    wrong_unit = replace(first.candidates[0], unit_vector=changed_unit)
    cases = (
        tuple(reversed(stage_d)),
        _replace_first_checkpoint(stage_d, joint_layout_fingerprint="0" * 64),
        _replace_first_checkpoint(stage_d, candidates=tuple(reversed(first.candidates))),
        _replace_first_checkpoint(
            stage_d,
            child_seeds=(first.child_seeds[0] + 1,),
        ),
        _replace_first_checkpoint(stage_d, stage="C"),
        _replace_first_checkpoint(
            stage_d,
            candidates=(wrong_rank, *first.candidates[1:]),
        ),
        _replace_first_checkpoint(
            stage_d,
            candidates=(wrong_unit, *first.candidates[1:]),
        ),
    )

    for resume_from in cases:
        progress: list[object] = []
        checkpoints: list[object] = []
        with pytest.raises(
            ValueError,
            match="joint|resume|checkpoint|dataset|candidate|ranking|seed|stage|unit|layout|q",
        ):
            api.run_joint_fit(
                api.JointFitRequest(joint, resume_checkpoints=resume_from),
                progress=progress.append,
                checkpoint=checkpoints.append,
            )
        assert progress == []
        assert checkpoints == []


def test_joint_checkpoint_callback_failure_propagates_exactly() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    sentinel = RuntimeError("atomic checkpoint failed")

    def fail(_checkpoints) -> None:
        raise sentinel

    with pytest.raises(RuntimeError, match="atomic checkpoint failed") as captured:
        api.run_joint_fit(api.JointFitRequest(_joint_problem()), checkpoint=fail)

    assert captured.value is sentinel


def test_independent_and_joint_modes_are_dispatched_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    pipeline = import_module("xrr_fitter.fit.pipeline")
    requests = (
        pipeline.FitSearchRequest("left", _problem(seed=881, size=40)),
        pipeline.FitSearchRequest("right", _problem(seed=881, size=48)),
    )
    calls: list[tuple[str, object]] = []
    independent_results = (object(), object())
    joint_results = (object(), object())

    def run_independent(request, **_kwargs):
        calls.append(("independent", request.dataset_id))
        return independent_results[len(calls) - 1]

    def compile_joint(dataset_ids, problems, rules, constraint_rules):
        calls.append(
            (
                "compile-joint",
                (dataset_ids, problems, rules, constraint_rules),
            )
        )
        return "joint-problem"

    def run_joint(request, **_kwargs):
        calls.append(("joint", request.problem))
        return joint_results

    monkeypatch.setattr(api, "run_fit_search", run_independent)
    monkeypatch.setattr(api, "compile_joint_problem", compile_joint)
    monkeypatch.setattr(api, "run_joint_fit", run_joint)

    independent = api.run_fit_batch(api.FitBatchRequest("independent", requests))
    joint = api.run_fit_batch(api.FitBatchRequest("joint", requests, sharing_rules=_sharing_rules()))

    assert independent == independent_results
    assert joint == joint_results
    assert calls[0:2] == [("independent", "left"), ("independent", "right")]
    assert calls[2][0] == "compile-joint"
    assert calls[3] == ("joint", "joint-problem")


def test_unknown_fit_mode_is_rejected_without_independent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    pipeline = import_module("xrr_fitter.fit.pipeline")
    requests = (pipeline.FitSearchRequest("left", _problem(seed=887, size=40)),)
    monkeypatch.setattr(
        api,
        "run_fit_search",
        lambda *_args, **_kwargs: pytest.fail("unknown mode fell back to independent"),
    )

    with pytest.raises(ValueError, match="mode|independent|joint"):
        api.run_fit_batch(api.FitBatchRequest("automatic", requests))
