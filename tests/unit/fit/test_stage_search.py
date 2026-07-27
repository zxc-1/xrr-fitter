"""End-to-end contracts for deterministic A-through-E fit orchestration.

The tests keep stage behavior observable at the immutable fit boundary.
They verify named seed derivation independently from request ordering.
They verify fresh and resumed stage order through summaries and checkpoints.
They retain candidate lineage instead of checking private helper delegation.
They exercise coarse global search and full-resolution publication together.
They protect declared Stage-B baselines from a worse optimizer result.
They preserve archived evidence while routing only active cluster parents.
They verify Stage E population projection, local refinement, and restarts.
They verify elite carry only after a material objective improvement.
They reject partial progress publication when a seed is cancelled.

Characterization coverage also crosses the shared evaluation boundary.
Dynamic roughness values must survive physical-to-unit round trips.
Implicit periodic top interfaces must follow the first shared layer.
Ideal-reflectivity diagnostics must retain complete source-row indices.

Numerical solver replacements in this module are deliberately narrow.
They return the same result shape as the real solver and leave evaluation real.
This exposes stage resolution, ordering, and continuation decisions without
turning the tests into assertions about module layout or delegation details.

Scenario builders return complete physical cases, not precomputed assertions.
Comparison helpers are reused across stage progress and checkpoint sequences.
The resulting failures still identify the exact domain transition that drifted.

Resume coverage deliberately reuses checkpoints emitted by a fresh search.
This keeps candidate order, seed consumption, and stage summaries coupled.
Stage-specific tests replace only the solver boundary needed by the scenario.
Real encoding and evaluation continue to validate published candidate state.
Cancellation cases assert the last fully completed unit of observable work.
Archive cases distinguish retained evidence from eligible continuation parents.
Together these checks characterize replay, not implementation call structure.
"""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import encode_physical_vector, values_by_name
from xrr_fitter.fit.candidates import CandidateStart, candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem, compile_stage_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock


def _pipeline_api():
    return import_module("xrr_fitter.fit.pipeline")


def _stages_api():
    return import_module("xrr_fitter.fit.stages")


def _problem(*, seed: int = 727, size: int = 40, structure=None):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    return compile_fit_problem(
        prepared_data(size=size),
        structure or simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )


def _implicit_periodic_structure():
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="a", thickness_a=20.0, roughness_a=2.0),
            replace(film, name="b", thickness_a=30.0, roughness_a=3.0),
        ),
        repeats=2,
        top_roughness_a=None,
    )
    return replace(base, components=(block,), backing_roughness_a=2.0)


def _candidate(problem, candidate_id: str, unit: np.ndarray):
    evaluation = evaluate_vector(problem, unit)
    assert evaluation.valid
    return candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        candidate_id,
        0,
        "converged",
        1,
    )


def _stage_prefixes(result) -> tuple[str, ...]:
    return tuple(summary.stage for summary in result.stage_summaries)


def _event_values(events, kind: str) -> list[object]:
    return [value for observed_kind, value in events if observed_kind == kind]


def _assert_coherent_checkpoints(checkpoints, result) -> None:
    result_ids = {candidate.candidate_id for candidate in result.candidates}
    for checkpoint in checkpoints:
        checkpoint_ids = {candidate.candidate_id for candidate in checkpoint.candidates}
        assert checkpoint.stage_summaries[-1].stage == checkpoint.stage
        assert set(checkpoint.stage_summaries[-1].candidate_ids) <= checkpoint_ids
        assert checkpoint.child_seeds == result.child_seeds[: len(checkpoint.child_seeds)]
        assert checkpoint_ids <= result_ids


def _assert_stage_d_lineage(by_stage, candidate_ids) -> None:
    for value in by_stage["D"]:
        parent = value.split("-")[1]
        assert any(item.startswith(f"C-{parent}-") for item in candidate_ids)


def _assert_candidate_lineage(by_stage, candidate_ids) -> None:
    assert candidate_ids[: len(by_stage["B"])] == by_stage["B"]
    assert all(value.startswith("B-") for value in by_stage["B"])
    assert all(value.startswith("C-") for value in by_stage["C"])
    assert all(value.startswith("D-") for value in by_stage["D"])
    assert by_stage["E"] == ("E-0", "E-1", "E-2", "E-3")
    assert candidate_ids[-4:] == by_stage["E"]
    for value in by_stage["C"]:
        assert f"B-{value.split('-')[1]}" in candidate_ids
    _assert_stage_d_lineage(by_stage, candidate_ids)


def _assert_stage_progress(progress, stage: str, expected_total: int) -> None:
    stage_events = tuple(value for value in progress if value.stage == stage)
    assert tuple(value.completed for value in stage_events) == tuple(
        range(1, expected_total + 1)
    )
    assert {value.total for value in stage_events} == {expected_total}


def _assert_progress_schedule(progress) -> None:
    assert tuple(dict.fromkeys(value.stage for value in progress)) == (
        "A",
        "B",
        "C",
        "D",
        "E",
    )
    _assert_stage_progress(progress, "B", 2)
    _assert_stage_progress(progress, "E", 4)
    stage_a = tuple(value for value in progress if value.stage == "A")
    assert tuple(value.completed for value in stage_a) == tuple(
        range(1, len(stage_a) + 1)
    )
    assert {value.total for value in stage_a} == {len(stage_a)}


def _best_stage_b_values(stage_problem, trial_units: np.ndarray) -> dict[str, float]:
    trial_evaluations = tuple(evaluate_vector(stage_problem, unit) for unit in trial_units)
    best_index = min(
        (index for index, value in enumerate(trial_evaluations) if value.valid),
        key=lambda index: trial_evaluations[index].objective,
    )
    return {
        value.name: value.value
        for value in trial_evaluations[best_index].parameters
    }


def _stage_b_alternatives(problem, stage_problem, trial_units: np.ndarray):
    alternatives: list[tuple[float, np.ndarray, object]] = []
    for stage_unit in trial_units:
        stage_evaluation = evaluate_vector(stage_problem, stage_unit)
        if not stage_evaluation.valid:
            continue
        projected = encode_physical_vector(
            problem,
            {value.name: value.value for value in stage_evaluation.parameters},
        )
        full_evaluation = evaluate_vector(problem, projected)
        if full_evaluation.valid:
            alternatives.append((full_evaluation.objective, stage_unit, stage_evaluation))
    return alternatives


def _stage_b_baseline_case(problem):
    initial_values = {
        definition.name: definition.initial
        for definition in problem.parameter_definitions
    }
    initial_stage = compile_stage_problem(problem, "B", initial_values)
    trial_units = np.random.default_rng(751).random((64, len(initial_stage.variables)))
    values = _best_stage_b_values(initial_stage, trial_units)
    stage_problem = compile_stage_problem(problem, "B", values)
    baseline_unit = encode_physical_vector(problem, values)
    baseline = evaluate_vector(problem, baseline_unit)
    drift_objective, drift_unit, drift_evaluation = max(
        _stage_b_alternatives(problem, stage_problem, trial_units),
        key=lambda item: item[0],
    )
    assert drift_objective > baseline.objective
    return SimpleNamespace(
        start=CandidateStart(tuple(values.items()), "declared-baseline"),
        stage_problem=stage_problem,
        stage_start=encode_physical_vector(stage_problem, values),
        baseline_unit=baseline_unit,
        baseline=baseline,
        drift_unit=drift_unit,
        drift_evaluation=drift_evaluation,
    )


def test_child_seed_lineage_is_deterministic_and_order_independent() -> None:
    api = _stages_api()
    streams = ("B-0", "B-1", "E-0", "E-1", "E-2", "E-3")

    forward = api.reserve_child_seeds(20260723, streams)
    reverse = api.reserve_child_seeds(20260723, tuple(reversed(streams)))

    assert tuple(item.stream_id for item in forward) == streams
    assert tuple(item.stream_id for item in reverse) == tuple(reversed(streams))
    assert {item.stream_id: item.seed for item in forward} == {
        item.stream_id: item.seed for item in reverse
    }
    assert len({item.seed for item in forward}) == len(streams)
    assert all(0 <= item.seed < 2**64 for item in forward)
    assert tuple(item.seed for item in forward) == (
        16164323491089515154,
        9436610754940370787,
        14495158119691411689,
        11623762797650596694,
        18359781962598382080,
        9014141665841017941,
    )


def test_stage_graph_has_exact_a_through_e_order_and_resume_suffixes() -> None:
    api = _stages_api()

    assert api.STAGE_ORDER == ("A", "B", "C", "D", "E")
    assert api.remaining_stages(None) == api.STAGE_ORDER
    assert api.remaining_stages("B") == ("C", "D", "E")
    assert api.remaining_stages("D") == ("E",)
    assert api.remaining_stages("E") == ()
    with pytest.raises(ValueError, match="stage"):
        api.remaining_stages("uncertainty")


def test_fit_search_reports_ordered_history_progress_and_coherent_checkpoints() -> None:
    api = _pipeline_api()
    events: list[tuple[str, object]] = []

    result = api.run_fit_search(
        api.FitSearchRequest("curve", _problem()),
        progress=lambda value: events.append(("progress", value)),
        checkpoint=lambda value: events.append(("checkpoint", value)),
    )

    assert _stage_prefixes(result) == ("A", "B", "C", "D", "E")
    progress = _event_values(events, "progress")
    checkpoints = _event_values(events, "checkpoint")
    _assert_progress_schedule(progress)
    assert tuple(value.stage for value in checkpoints) == ("B", "C", "D", "E")
    _assert_coherent_checkpoints(checkpoints, result)


def test_stage_history_preserves_candidate_lineage_and_final_seed_order() -> None:
    api = _pipeline_api()
    result = api.run_fit_search(api.FitSearchRequest("curve", _problem(seed=733)))
    by_stage = {summary.stage: summary.candidate_ids for summary in result.stage_summaries}
    candidate_ids = tuple(candidate.candidate_id for candidate in result.candidates)

    _assert_candidate_lineage(by_stage, candidate_ids)
    final = tuple(candidate for candidate in result.candidates if candidate.candidate_id.startswith("E-"))
    assert tuple(candidate.seed_index for candidate in final) == (0, 1, 2, 3)


def test_fit_search_request_handler_and_result_are_pickle_safe() -> None:
    api = _pipeline_api()
    request = api.FitSearchRequest("curve", _problem(seed=739))

    assert [field.name for field in fields(request)] == [
        "dataset_id",
        "problem",
        "resume_checkpoint",
    ]
    restored_request = pickle.loads(pickle.dumps(request))
    assert restored_request.dataset_id == "curve"
    np.testing.assert_array_equal(
        restored_request.problem.data.qz_a_inv,
        request.problem.data.qz_a_inv,
    )
    assert pickle.loads(pickle.dumps(api.run_fit_search)) is api.run_fit_search

    result = api.run_fit_search(request)
    restored_result = pickle.loads(pickle.dumps(result))
    assert tuple(candidate.candidate_id for candidate in restored_result.candidates) == tuple(
        candidate.candidate_id for candidate in result.candidates
    )
    for left, right in zip(restored_result.candidates, result.candidates, strict=True):
        np.testing.assert_array_equal(left.unit_vector, right.unit_vector)
        np.testing.assert_array_equal(left.model_normalized, right.model_normalized)


def test_checkpoint_callback_failure_is_not_converted_to_search_success() -> None:
    api = _pipeline_api()
    sentinel = RuntimeError("checkpoint publication failed")

    def fail(_checkpoint) -> None:
        raise sentinel

    with pytest.raises(RuntimeError, match="checkpoint publication failed") as captured:
        api.run_fit_search(
            api.FitSearchRequest("curve", _problem(seed=743)),
            checkpoint=fail,
        )

    assert captured.value is sentinel


def test_dynamic_roughness_encoding_round_trips_initial_structure() -> None:
    problem = _problem(seed=747, structure=_implicit_periodic_structure())
    initial = {
        definition.name: definition.initial
        for definition in problem.parameter_definitions
    }

    unit = encode_physical_vector(problem, initial)
    decoded = values_by_name(problem, unit)

    for name, expected in initial.items():
        assert decoded[name] == pytest.approx(expected)


def test_implicit_periodic_top_roughness_tracks_first_layer() -> None:
    structure = _implicit_periodic_structure()
    problem = _problem(seed=749, structure=structure)
    first_roughness = "component.0.layer.0.roughness_a"
    unit = encode_physical_vector(problem, {first_roughness: 4.0})

    evaluation = evaluate_vector(problem, unit)

    assert evaluation.valid
    assert isinstance(structure.components[0], PeriodicBlock)
    assert structure.components[0].top_roughness_a is None
    assert next(
        value.value for value in evaluation.parameters if value.name == first_roughness
    ) == pytest.approx(4.0)
    assert evaluation.expanded_stack is not None
    np.testing.assert_allclose(
        evaluation.expanded_stack.roughness_a[[0, 2]],
        (4.0, 4.0),
        rtol=0.0,
        atol=1e-14,
    )


def test_ideal_reflectivity_above_one_records_full_dataset_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_api = import_module("xrr_fitter.evaluation")
    angles = np.concatenate((np.asarray([-0.4, 0.0]), np.linspace(0.1, 3.8, 38)))
    data = prepared_data(size=40, two_theta_deg=angles)
    problem = compile_fit_problem(
        data,
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(750), scale_prior_enabled=False),
    )
    unit = encode_physical_vector(problem, {})

    def reflectivity_with_two_violations(qz, _stack):
        result = np.ones_like(qz)
        result[[1, -2]] = 1.0 + 2e-6
        return result

    monkeypatch.setattr(evaluation_api, "parratt_reflectivity", reflectivity_with_two_violations)

    evaluation = evaluate_vector(problem, unit)

    diagnostic = next(
        value
        for value in evaluation.diagnostics
        if value.code == "ideal_reflectivity_above_one"
    )
    assert diagnostic.point_indices == (3, 38)


def test_stage_b_keeps_a_better_declared_baseline_than_the_de_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _stages_api()
    problem = _problem(seed=751)
    case = _stage_b_baseline_case(problem)

    def worse_global(problem_value, unit, **_kwargs):
        assert problem_value.parameter_definitions == case.stage_problem.parameter_definitions
        np.testing.assert_array_equal(unit, case.stage_start)
        return SimpleNamespace(
            unit_vector=case.drift_unit,
            evaluation=case.drift_evaluation,
            population=np.vstack((case.drift_unit,) * 5),
            trace=(),
            stop_reason="worse DE",
            nfev=5,
        )

    monkeypatch.setattr(api, "solve_global", worse_global)

    outcome = api.run_stage_b(
        problem,
        "curve",
        (case.start,),
        (123,),
        progress=None,
        cancelled=None,
    )

    assert len(outcome.candidates) == 1
    np.testing.assert_array_equal(outcome.candidates[0].unit_vector, case.baseline_unit)
    assert outcome.candidates[0].objective == case.baseline.objective


def test_stage_b_publishes_candidates_in_coarse_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _stages_api()
    problem = _problem(seed=753)
    starts = (
        CandidateStart((), "first"),
        CandidateStart((), "second"),
    )
    initial = encode_physical_vector(problem, {})
    by_index = (
        replace(_candidate(problem, "B-0", initial), objective=2.0),
        replace(_candidate(problem, "B-1", initial), objective=1.0),
    )

    monkeypatch.setattr(
        api,
        "_stage_b_candidate",
        lambda _problem, _start, index, _seed, _cancelled: by_index[index],
    )

    outcome = api.run_stage_b(
        problem,
        "curve",
        starts,
        (101, 202),
        progress=None,
        cancelled=None,
    )

    assert tuple(candidate.candidate_id for candidate in outcome.candidates) == (
        "B-0",
        "B-1",
    )
    assert outcome.summary.candidate_ids == ("B-0", "B-1")


def test_stage_a_propagates_unexpected_encoding_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _stages_api()
    sentinel = ValueError("unexpected implementation failure")

    def fail(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(api, "encode_physical_vector", fail)

    with pytest.raises(ValueError, match="unexpected implementation failure") as captured:
        api._stage_a_candidate(_problem(seed=755), CandidateStart((), "declared"), 0)

    assert captured.value is sentinel


def test_stage_a_rejects_an_out_of_bounds_candidate_as_invalid() -> None:
    api = _stages_api()
    start = CandidateStart((("component.0.roughness_a", 51.0),), "out-of-bounds")

    assert api._stage_a_candidate(_problem(seed=757), start, 0) is None


def test_pipeline_uses_coarse_global_and_full_resolution_local_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _stages_api()
    pipeline = _pipeline_api()
    problem = _problem(seed=757, size=160)
    full_size = problem.data.qz_a_inv.size
    global_sizes: list[int] = []
    local_sizes: list[int] = []

    def no_op_global(problem_value, start, *, population, **_kwargs):
        global_sizes.append(problem_value.data.qz_a_inv.size)
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            population=np.array(population, copy=True),
            trace=(),
            stop_reason="captured global",
            nfev=1,
        )

    def no_op_local(problem_value, start, **_kwargs):
        local_sizes.append(problem_value.data.qz_a_inv.size)
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            stop_reason="captured local",
            nfev=1,
        )

    monkeypatch.setattr(stages, "solve_global", no_op_global)
    monkeypatch.setattr(stages, "solve_local", no_op_local)

    result = pipeline.run_fit_search(pipeline.FitSearchRequest("curve", problem))

    assert global_sizes and all(size < full_size for size in global_sizes)
    assert local_sizes and all(size == full_size for size in local_sizes)
    assert all(candidate.qz_a_inv.size == full_size for candidate in result.candidates)


def test_pipeline_archives_stage_b_evidence_and_routes_only_active_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = import_module("xrr_fitter.fit.candidates")
    stages = _stages_api()
    pipeline = _pipeline_api()
    archive_calls: list[tuple[str, ...]] = []

    def no_op_global(problem_value, start, *, population, **_kwargs):
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            population=np.array(population, copy=True),
            population_energies=np.asarray(
                [evaluate_vector(problem_value, row).objective for row in population]
            ),
            trace=(),
            stop_reason="captured global",
            nfev=1,
        )

    def no_op_local(problem_value, start, **_kwargs):
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            stop_reason="captured local",
            nfev=1,
        )

    def forced_archive(values, **_kwargs):
        archive_calls.append(tuple(value.candidate_id for value in values))
        archived = replace(values[1], seed_index=-1, stop_reason="early_eliminated")
        return candidates.StageBArchive((values[0],), (archived,), (5,))

    monkeypatch.setattr(stages, "solve_global", no_op_global)
    monkeypatch.setattr(stages, "solve_local", no_op_local)
    monkeypatch.setattr(stages, "archive_stage_b_candidates", forced_archive)

    result = pipeline.run_fit_search(
        pipeline.FitSearchRequest("curve", _problem(seed=759))
    )
    candidate_ids = tuple(value.candidate_id for value in result.candidates)

    assert archive_calls == [("B-0", "B-1")]
    assert any(
        value.candidate_id == "B-1" and value.stop_reason == "early_eliminated"
        for value in result.candidates
    )
    assert tuple(value for value in candidate_ids if value.startswith("C-0-")) == tuple(
        f"C-0-{index}" for index in range(6)
    )
    assert not any(value.startswith(("C-1-", "D-1-")) for value in candidate_ids)


def test_stage_e_runs_de_four_ranked_locals_and_two_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _stages_api()
    problem = _problem(seed=761)
    declared_values = {
        definition.name: definition.initial
        for definition in problem.parameter_definitions
    }
    declared_unit = encode_physical_vector(problem, declared_values)
    shifted_unit = np.clip(declared_unit + 0.03, 0.0, 1.0)
    parents = (
        _candidate(problem, "D-0-0", declared_unit),
        _candidate(problem, "D-1-0", shifted_unit),
    )
    global_calls: list[int] = []
    local_starts: list[np.ndarray] = []

    def no_op_global(problem_value, start, *, population, **_kwargs):
        global_calls.append(problem_value.data.qz_a_inv.size)
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            population=np.array(population, copy=True),
            trace=(),
            stop_reason="captured global",
            nfev=1,
        )

    def no_op_local(problem_value, start, **_kwargs):
        local_starts.append(np.array(start, copy=True))
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            stop_reason="captured local",
            nfev=1,
        )

    monkeypatch.setattr(api, "solve_global", no_op_global)
    monkeypatch.setattr(api, "solve_local", no_op_local)

    outcome = api.run_stage_e(
        problem,
        "curve",
        parents,
        (991,),
        progress=None,
        cancelled=None,
    )

    assert len(global_calls) == 1
    assert len(local_starts) == 8
    assert outcome.summary.candidate_ids == ("E-0",)


def test_stage_e_carries_only_a_materially_improved_elite_to_later_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _stages_api()
    problem = _problem(seed=769)
    rng = np.random.default_rng(769)
    evaluated = tuple(
        (unit, evaluate_vector(problem, unit))
        for unit in rng.random((96, len(problem.variables)))
    )
    valid = sorted(
        (item for item in evaluated if item[1].valid),
        key=lambda item: item[1].objective,
    )
    assert len(valid) >= 8
    elite_unit, elite_evaluation = valid[0]
    baseline_unit, baseline_evaluation = valid[-1]
    required = max(
        problem.config.confidence.equivalent_cost_fraction
        * abs(baseline_evaluation.objective),
        problem.config.confidence.equivalent_cost_floor,
    )
    assert elite_evaluation.objective + required < baseline_evaluation.objective
    declared_unit = encode_physical_vector(
        problem,
        {definition.name: definition.initial for definition in problem.parameter_definitions},
    )
    parents = (
        _candidate(problem, "D-0-0", baseline_unit),
        _candidate(problem, "D-1-0", declared_unit),
    )
    active_seed = 0
    starts_by_seed: dict[int, list[np.ndarray]] = {index: [] for index in range(4)}

    def no_op_global(problem_value, start, *, population, **_kwargs):
        nonlocal active_seed
        current = active_seed
        active_seed += 1
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            population=np.array(population, copy=True),
            population_energies=np.asarray(
                [evaluate_vector(problem_value, row).objective for row in population]
            ),
            trace=(),
            stop_reason=f"captured global {current}",
            nfev=1,
        )

    def controlled_local(problem_value, start, **_kwargs):
        seed_index = max(0, active_seed - 1)
        starts_by_seed[seed_index].append(np.array(start, copy=True))
        unit = elite_unit if seed_index == 2 else baseline_unit
        return SimpleNamespace(
            unit_vector=np.array(unit, copy=True),
            evaluation=evaluate_vector(problem_value, unit),
            stop_reason="captured local",
            nfev=1,
        )

    monkeypatch.setattr(api, "solve_global", no_op_global)
    monkeypatch.setattr(api, "solve_local", controlled_local)

    api.run_stage_e(
        problem,
        "curve",
        parents,
        (101, 202, 303, 404),
        progress=None,
        cancelled=None,
    )

    assert tuple(len(starts_by_seed[index]) for index in range(4)) == (8, 8, 8, 9)
    seed_three_incumbents = starts_by_seed[3][:3]
    assert any(
        not np.array_equal(start, elite_unit)
        and np.max(np.abs(start - elite_unit)) <= 0.01
        for start in seed_three_incumbents
    )


def test_stage_e_does_not_publish_progress_for_a_cancelled_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _stages_api()
    problem = _problem(seed=773)
    unit = encode_physical_vector(
        problem,
        {definition.name: definition.initial for definition in problem.parameter_definitions},
    )
    parents = (_candidate(problem, "D-0-0", unit),)
    global_calls = 0
    progress = []

    def cancel_second_global(problem_value, start, *, population, **_kwargs):
        nonlocal global_calls
        global_calls += 1
        if global_calls == 2:
            raise api.SearchCancelled("cancelled second Stage-E seed")
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            population=np.array(population, copy=True),
            population_energies=np.asarray(
                [evaluate_vector(problem_value, row).objective for row in population]
            ),
            trace=(),
            stop_reason="captured global",
            nfev=1,
        )

    def no_op_local(problem_value, start, **_kwargs):
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            stop_reason="captured local",
            nfev=1,
        )

    monkeypatch.setattr(api, "solve_global", cancel_second_global)
    monkeypatch.setattr(api, "solve_local", no_op_local)

    with pytest.raises(api.SearchCancelled, match="cancelled second"):
        api.run_stage_e(
            problem,
            "curve",
            parents,
            (505, 606),
            progress=progress.append,
            cancelled=None,
        )

    assert global_calls == 2
    assert tuple(value.completed for value in progress) == (1,)
