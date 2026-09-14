"""Joint adaptive search must record the work that actually produced a candidate.

These tests use two real compiled reflectivity problems with 160 and 192 fitted
observations. Both therefore have a genuinely smaller first grid; a full-grid
optimizer cannot accidentally pass as a coarse search. Spies delegate to the
real solver and evaluator, retaining exact population costs and review calls.
The published candidates still contain the complete dataset axes.

Each optimizer allocation belongs to one global lineage, not to each projected
dataset. Stage E keeps its DE and local calls separately so neither budget nor
stop reason is lost. Fully locked problems record their single evaluation and
never invent a population search or a second local call.

Stagnation tests use a flat real population to exercise SciPy's actual stopping
boundary. A deterministic generation driver isolates the two reset conditions:
a new geometric representative and a material improvement within one geometry.
Its objective remains the real joint objective rather than a fabricated score.

Checkpoint tests retain only completed prefixes, compare the projected evidence,
and JSON-round-trip every resumable stage boundary. Resume must preserve all
completed work and execute only the remaining suffix, including partial E.
"""

from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.fit.adaptive_review import joint_grid_contexts
from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector
from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import initial_joint_vector
from xrr_fitter.fit.joint_solvers import solve_joint_global
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.io.codec_results import _checkpoint_from_dict, _checkpoint_to_dict
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting, SharingRule

SHARED_PARAMETER = "component.0.density_scale"


def _joint_member(size, config, locked):
    base = compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none", instrument_id="adaptive-joint"),
        config,
    )
    settings = []
    for definition in base.parameter_definitions:
        free = not locked and definition.name == SHARED_PARAMETER
        settings.append(
            ParameterSetting(
                definition.name,
                definition.initial,
                definition.lower if free else definition.initial,
                definition.upper if free else definition.initial,
                locked=not free,
            )
        )
    return compile_fit_problem(base.data, base.structure, base.instrument, base.config, tuple(settings))


def _joint_problem(*, locked=False):
    config = replace(
        FitConfig.fast(9601),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    members = tuple(_joint_member(size, config, locked) for size in (160, 192))
    sharing = (
        ()
        if locked
        else (
            SharingRule(
                "density",
                tuple(ParameterReference(dataset_id, SHARED_PARAMETER) for dataset_id in ("left", "right")),
            ),
        )
    )
    return compile_joint_problem(("left", "right"), members, sharing)


def _point_axes(problem):
    return tuple(int(np.count_nonzero(member.data.fit_mask)) for member in problem.problems)


def _calls_of_kind(calls, kind):
    return [call for call in calls if call["kind"] == kind]


def _observe_solvers(monkeypatch):
    pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    global_solver, local_solver = pipeline._solve_joint_global, pipeline._solve_joint
    calls = []

    def global_call(problem, start, population, *, seed, maxiter, cancelled):
        solved = global_solver(problem, start, population, seed=seed, maxiter=maxiter, cancelled=cancelled)
        calls.append(
            {
                "kind": "DE",
                "problem": problem,
                "start": start.copy(),
                "seed": seed,
                "limit": len(population) * (maxiter + 1),
                "solved": solved,
            }
        )
        return solved

    def local_call(problem, start, max_nfev, cancelled):
        solved = local_solver(problem, start, max_nfev, cancelled)
        calls.append(
            {
                "kind": "local",
                "problem": problem,
                "start": start.copy(),
                "limit": max_nfev,
                "solved": solved,
            }
        )
        return solved

    monkeypatch.setattr(pipeline, "_solve_joint_global", global_call)
    monkeypatch.setattr(pipeline, "_solve_joint", local_call)
    return calls


def _observe_reviews(monkeypatch):
    pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    evaluation_module = import_module("xrr_fitter.fit.joint_evaluation")
    original_review = pipeline.review_joint_population
    evaluations = []
    reviews = {}

    def observed_evaluation(context, unit, **kwargs):
        result = evaluate_joint_vector(context, unit, **kwargs)
        evaluations.append((_point_axes(context), result.objective))
        return result

    def observed_review(*args, **kwargs):
        before = len(evaluations)
        starts, evidence = original_review(*args, **kwargs)
        reviews[evidence.candidate_origin] = (starts, evidence, evaluations[before:])
        return starts, evidence

    monkeypatch.setattr(evaluation_module, "evaluate_joint_vector", observed_evaluation)
    monkeypatch.setattr(pipeline, "review_joint_population", observed_review)
    return reviews


def _assert_review_work(evidence, costs, solved):
    assert evidence.grid_points[0] == (128, 128)
    assert evidence.reviews[0].grid_evaluations == 0
    assert evidence.full_review_evaluations == sum(axes == (160, 192) for axes, _ in costs)
    assert evidence.grid_review_evaluations == sum(axes != (160, 192) for axes, _ in costs)
    assert evidence.optimizer_nfev == solved.nfev
    assert evidence.full_review_evaluations > 0


def _assert_review_costs(problem, coarse, solved, evidence):
    for identifier, coarse_cost, full_cost in zip(
        evidence.reviews[0].candidate_ids,
        evidence.reviews[0].coarse_objectives,
        evidence.reviews[0].full_objectives,
        strict=True,
    ):
        row = int(identifier.rsplit("-", 1)[1])
        unit = solved.population[row]
        assert coarse_cost == solved.population_energies[row]
        assert coarse_cost == evaluate_joint_vector(coarse, unit, fit_only=True).objective
        assert full_cost == evaluate_joint_vector(problem, unit, fit_only=True).objective


def _assert_reviewed_selection(candidate, starts, evidence, calls):
    if candidate.candidate_id == "B-0":
        np.testing.assert_array_equal(candidate.unit_vector, starts[0])
        assert candidate.ranking_objective == min(
            cost for review in evidence.reviews for cost in review.full_objectives
        )
    else:
        local_calls = _calls_of_kind(calls, "local")
        np.testing.assert_array_equal(local_calls[2]["start"], starts[0])


@pytest.mark.parametrize("candidate_id, global_index", [("B-0", 0), ("E-0", 1)])
def test_joint_de_uses_the_recorded_coarse_grid_before_full_review(monkeypatch, candidate_id, global_index):
    problem = _joint_problem()
    calls = _observe_solvers(monkeypatch)
    reviews = _observe_reviews(monkeypatch)
    results = run_joint_fit(JointFitRequest(problem))
    observed = _calls_of_kind(calls, "DE")[global_index]
    coarse = joint_grid_contexts(problem)[0]
    assert _point_axes(observed["problem"]) == _point_axes(coarse) == (128, 128)
    starts, evidence, costs = reviews[candidate_id]
    _assert_review_work(evidence, costs, observed["solved"])
    _assert_review_costs(problem, coarse, observed["solved"], evidence)
    candidate = next(value for value in results[0].candidates if value.candidate_id == candidate_id)
    assert candidate.search_evidence[0] == evidence
    _assert_reviewed_selection(candidate, starts, evidence, calls)
    assert candidate.model_normalized.size == 160


def _assert_allocation(allocation, call, origin, round_index):
    assert allocation.lineage_id == origin
    expected_id = origin + ("-local" if call["kind"] == "local" else "")
    assert allocation.candidate_id == expected_id
    assert allocation.round_index == round_index
    assert allocation.max_nfev == call["limit"]
    assert allocation.nfev == call["solved"].nfev


def _assert_optimizer_evidence(evidence, call, origin, seed, round_index):
    assert evidence.candidate_origin == origin
    assert evidence.seed == seed
    assert evidence.stop_reason == call["solved"].stop_reason
    assert len(evidence.budget_allocations) == 1
    _assert_allocation(evidence.budget_allocations[0], call, origin, round_index)


def _candidate_seed_index(candidate):
    return 1 + candidate.seed_index if candidate.candidate_id.startswith("E-") else 0


def _assert_candidate_calls(candidate, counterpart, calls, seeds):
    assert candidate.search_evidence == counterpart.search_evidence
    expected_count = 2 if candidate.candidate_id.startswith("E-") else 1
    assert len(candidate.search_evidence) == expected_count
    assert candidate.nfev == sum(value.optimizer_nfev for value in candidate.search_evidence)
    seed = seeds[_candidate_seed_index(candidate)]
    for round_index, evidence in enumerate(candidate.search_evidence):
        _assert_optimizer_evidence(evidence, calls[round_index], candidate.candidate_id, seed, round_index)
    return expected_count


def _assert_summary_work(result):
    for summary in result.stage_summaries[1:]:
        assert summary.total_nfev == sum(value.optimizer_nfev for value in summary.search_evidence)


def _assert_global_work_total(result, calls):
    assert sum(summary.total_nfev for summary in result.stage_summaries[1:]) == sum(
        call["solved"].nfev for call in calls
    )
    _assert_summary_work(result)


def test_joint_stages_record_each_live_optimizer_call_without_member_multiplication(monkeypatch):
    calls = _observe_solvers(monkeypatch)
    results = run_joint_fit(JointFitRequest(_joint_problem()))
    assert [call["kind"] for call in calls] == ["DE", "local", "local", *(["DE", "local"] * 4)]
    left, right = results
    assert left.stage_summaries == right.stage_summaries
    assert left.candidates[0].search_evidence == ()
    cursor = 0
    for candidate, counterpart in zip(left.candidates[1:], right.candidates[1:], strict=True):
        cursor += _assert_candidate_calls(candidate, counterpart, calls[cursor:], left.child_seeds)
    assert cursor == len(calls)
    _assert_global_work_total(left, calls)


def _assert_locked_evidence(candidate, seeds):
    assert len(candidate.search_evidence) == 1
    evidence = candidate.search_evidence[0]
    assert evidence.candidate_origin == candidate.candidate_id
    assert evidence.stop_reason == candidate.stop_reason == "no_free_parameters"
    assert evidence.seed == seeds[_candidate_seed_index(candidate)]
    assert evidence.reviews == ()
    assert evidence.optimizer_nfev == candidate.nfev == 1
    (allocation,) = evidence.budget_allocations
    assert allocation.lineage_id == candidate.candidate_id


def test_joint_locked_stages_record_the_one_evaluation_they_actually_perform(monkeypatch):
    problem = _joint_problem(locked=True)
    calls = _observe_solvers(monkeypatch)
    results = run_joint_fit(JointFitRequest(problem))
    assert [call["kind"] for call in calls] == ["local", "local"]
    for candidate in results[0].candidates[1:]:
        _assert_locked_evidence(candidate, results[0].child_seeds)
        expected_limit = 5 if candidate.candidate_id[0] in {"C", "D"} else 1
        assert candidate.search_evidence[0].optimizer_nfev_limit == expected_limit
    assert results[0].stage_summaries == results[1].stage_summaries
    _assert_summary_work(results[0])


@pytest.mark.parametrize("maxiter", [0, 1, 2, 7])
def test_joint_de_flat_real_population_obeys_three_generations_and_short_budget(maxiter):
    problem = _joint_problem()
    start = initial_joint_vector(problem)
    population = np.tile(start, (8, 1))
    solved = solve_joint_global(problem, start, population, seed=71, maxiter=maxiter, cancelled=None)
    assert solved.nfev == len(population) * (1 + min(maxiter, 3))
    if maxiter >= 3:
        assert solved.stop_reason == "three_generation_stagnation"
    else:
        assert solved.stop_reason == "Maximum number of iterations has been exceeded."


def _stagnation_reset_units(problem, reset):
    initial = np.asarray([0.5])
    changed = np.asarray([0.8 if reset == "new_geometry" else 0.52])
    initial_cost = evaluate_joint_vector(problem, initial, fit_only=True).objective
    changed_cost = evaluate_joint_vector(problem, changed, fit_only=True).objective
    if (changed_cost < initial_cost) == (reset == "new_geometry"):
        initial, changed = changed, initial
        initial_cost, changed_cost = changed_cost, initial_cost
    if reset == "material_gain":
        assert initial_cost - changed_cost >= 1e-4 * initial_cost
        assert np.linalg.norm(initial - changed) < 0.05
    else:
        assert changed_cost >= initial_cost
        assert np.linalg.norm(initial - changed) >= 0.05
    return initial, changed


@pytest.mark.parametrize("reset", ["new_geometry", "material_gain"])
def test_joint_de_real_objectives_reset_the_stagnation_window(monkeypatch, reset):
    problem = _joint_problem()
    initial, changed = _stagnation_reset_units(problem, reset)
    generations = []

    def scripted_de(objective, _bounds, *, init, maxiter, callback, **_kwargs):
        nfev = 0
        for row in init:
            objective(row)
            nfev += 1
        current = initial
        for generation in range(maxiter):
            current = changed if generation >= 2 else initial
            for _ in init:
                objective(current)
                nfev += 1
            generations.append(generation + 1)
            if callback(current, convergence=0.0):
                break
        return SimpleNamespace(
            x=current,
            population=np.tile(current, (len(init), 1)),
            population_energies=np.full(len(init), evaluate_joint_vector(problem, current, fit_only=True).objective),
            nfev=nfev,
            message="scripted generation budget",
        )

    module = import_module("xrr_fitter.fit.joint_solvers")
    monkeypatch.setattr(module, "differential_evolution", scripted_de)
    solved = solve_joint_global(problem, initial, np.tile(initial, (8, 1)), seed=73, maxiter=10, cancelled=None)
    assert generations == [1, 2, 3, 4, 5, 6]
    assert solved.stop_reason == "three_generation_stagnation"
    assert solved.nfev == 8 * 7


def _assert_aligned_batch(batch):
    assert batch[0].stage_summaries == batch[1].stage_summaries
    assert all(
        left.search_evidence == right.search_evidence
        for left, right in zip(batch[0].candidates, batch[1].candidates, strict=True)
    )


def _assert_first_e_prefix(checkpoint, locked):
    assert checkpoint.stage_summaries[-1].candidate_ids == ("E-0",)
    assert len(checkpoint.child_seeds) == 2
    assert len(checkpoint.candidates[-1].search_evidence) == (1 if locked else 2)


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("cancel_after", ["A", "first-E"])
def test_joint_cancellation_preserves_only_complete_atomic_evidence_prefixes(locked, cancel_after):
    problem = _joint_problem(locked=locked)
    cancelled = False
    batches = []

    def progress(value):
        nonlocal cancelled
        if cancel_after == "A" and value.stage == "A":
            cancelled = True

    def checkpoint(batch):
        nonlocal cancelled
        batches.append(batch)
        if cancel_after == "first-E" and batch[0].stage == "E":
            cancelled = True

    with pytest.raises(SearchCancelled):
        run_joint_fit(
            JointFitRequest(problem),
            cancelled=lambda: cancelled,
            progress=progress,
            checkpoint=checkpoint,
        )
    assert [batch[0].stage for batch in batches] == ([] if cancel_after == "A" else ["B", "C", "D", "E"])
    for batch in batches:
        _assert_aligned_batch(batch)
    if batches:
        _assert_first_e_prefix(batches[-1][0], locked)


@pytest.fixture(scope="module")
def completed_joint():
    problem = _joint_problem()
    batches = []
    result = run_joint_fit(JointFitRequest(problem), checkpoint=batches.append)
    return problem, result, batches


def _assert_same_candidate_state(original, continued):
    assert original.search_evidence == continued.search_evidence
    assert original.nfev == continued.nfev
    assert original.objective == continued.objective
    np.testing.assert_array_equal(original.unit_vector, continued.unit_vector)


def _assert_same_result_state(original, continued):
    assert original.best_index == continued.best_index
    assert original.child_seeds == continued.child_seeds
    assert original.stage_summaries == continued.stage_summaries
    for first, second in zip(original.candidates, continued.candidates, strict=True):
        _assert_same_candidate_state(first, second)


@pytest.mark.parametrize("checkpoint_index", [0, 1, 2, 3, 6])
def test_joint_resume_roundtrip_keeps_full_evidence_and_runs_only_the_suffix(completed_joint, checkpoint_index):
    problem, fresh, batches = completed_joint
    restored = tuple(
        _checkpoint_from_dict(json.loads(json.dumps(_checkpoint_to_dict(checkpoint))))
        for checkpoint in batches[checkpoint_index]
    )
    resumed_batches = []
    resumed = run_joint_fit(JointFitRequest(problem, restored), checkpoint=resumed_batches.append)
    assert [batch[0].stage for batch in resumed_batches] == [
        batch[0].stage for batch in batches[checkpoint_index + 1 :]
    ]
    for left, right in zip(fresh, resumed, strict=True):
        _assert_same_result_state(left, right)
