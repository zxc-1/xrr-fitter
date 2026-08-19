"""Ordered task batches at fit orchestration boundaries.

These tests isolate parallel dispatch from the larger A-through-E replay suite.
The runner executes synchronously so assertions cover batch ownership and order
without making wall-clock scheduling part of the behavioral contract.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.unit.fit.test_stage_search import (
    _candidate,
    _pipeline_api,
    _problem,
    _stages_api,
    _unchanged_local_solution,
)

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector


class _BatchRunner:
    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, tasks):
        batch = tuple(tasks)
        self.sizes.append(len(batch))
        return tuple(task() for task in batch)


def test_pipeline_routes_stage_work_through_injected_task_runner() -> None:
    pipeline = _pipeline_api()
    runner = _BatchRunner()

    pipeline.run_fit_search(
        pipeline.FitSearchRequest("curve", _problem(seed=758)),
        task_runner=runner,
    )

    assert runner.sizes


def test_profile_continuation_forwards_task_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline_api()
    problem = _problem(seed=741)
    search = pipeline.run_fit_search(pipeline.FitSearchRequest("curve", problem))
    best = search.best_candidate
    assert best is not None
    required = max(
        problem.config.confidence.equivalent_cost_fraction * abs(best.objective),
        problem.config.confidence.equivalent_cost_floor,
    )
    runner = _BatchRunner()
    observed = []

    monkeypatch.setattr(
        pipeline,
        "evaluate_model",
        lambda *_args: SimpleNamespace(
            valid=True,
            objective=best.objective - 2.0 * required,
        ),
    )

    def reconverge(_problem, candidates, *_args, **kwargs):
        observed.append(kwargs["task_runner"])
        return candidates

    monkeypatch.setattr(pipeline, "reconverge_profile_basin", reconverge)

    pipeline.continue_profile_basin(
        problem,
        search,
        best.unit_vector,
        parameter_name="component.0.thickness_a",
        task_runner=runner,
    )

    assert observed == [runner]


def test_local_stage_batches_each_parents_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _stages_api()
    problem = _problem(seed=762)
    parent = _candidate(problem, "B-0", np.full(len(problem.variables), 0.5))
    runner = _BatchRunner()
    monkeypatch.setattr(stages, "solve_local", _unchanged_local_solution)

    outcome = stages.run_local_stage(
        problem,
        "curve",
        "C",
        (parent,),
        perturbation_counts=(2,),
        progress=None,
        cancelled=None,
        task_runner=runner,
    )

    assert runner.sizes == [3]
    assert tuple(candidate.candidate_id for candidate in outcome.candidates) == (
        "C-0-0",
        "C-0-1",
        "C-0-2",
    )


def test_stage_e_batches_ranked_locals_before_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _stages_api()
    problem = _problem(seed=761)
    declared = {definition.name: definition.initial for definition in problem.parameter_definitions}
    declared_unit = encode_physical_vector(problem, declared)
    parents = (
        _candidate(problem, "D-0-0", declared_unit),
        _candidate(problem, "D-1-0", np.clip(declared_unit + 0.03, 0.0, 1.0)),
    )
    runner = _BatchRunner()

    def no_op_global(problem_value, start, *, population, **_kwargs):
        return SimpleNamespace(
            unit_vector=np.array(start, copy=True),
            evaluation=evaluate_vector(problem_value, start),
            population=np.array(population, copy=True),
            trace=(),
            stop_reason="captured global",
            nfev=1,
        )

    monkeypatch.setattr(stages, "solve_global", no_op_global)
    monkeypatch.setattr(stages, "solve_local", _unchanged_local_solution)

    stages.run_stage_e(
        problem,
        "curve",
        parents,
        (991,),
        progress=None,
        cancelled=None,
        task_runner=runner,
    )

    assert runner.sizes == [6, 2]


def test_profile_rescue_batches_all_four_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _stages_api()
    problem = _problem(seed=775)
    center = encode_physical_vector(
        problem,
        {definition.name: definition.initial for definition in problem.parameter_definitions},
    )
    candidates = tuple(replace(_candidate(problem, f"E-{index}", center), seed_index=index) for index in range(4))
    runner = _BatchRunner()
    monkeypatch.setattr(stages, "solve_local", _unchanged_local_solution)
    monkeypatch.setattr(
        stages,
        "_publish_profile_rescue",
        lambda _problem, originals, *_args: originals,
    )

    rebuilt = stages.reconverge_profile_basin(
        problem,
        candidates,
        center,
        (101, 202, 303, 404),
        parameter_name="component.0.thickness_a",
        task_runner=runner,
    )

    assert rebuilt == candidates
    assert runner.sizes == [4]
