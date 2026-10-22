"""Stage-E elite admission uses the best selectable full-data parent."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.unit.fit.test_stage_search import _candidate, _problem, _unchanged_local_solution

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit import stages
from xrr_fitter.fit.candidates import materially_improves
from xrr_fitter.fit.objective import evaluate_vector


@pytest.fixture
def elite_case():
    problem = _problem(seed=769)
    declared = encode_physical_vector(problem, {})
    good = np.ones(len(problem.variables))
    poor_parent = _candidate(problem, "D-0-0", declared)
    good_parent = _candidate(problem, "D-1-0", good)
    return problem, declared, good, poor_parent, good_parent


def _capture_local_starts(monkeypatch, problem, parents, results):
    starts: list[list[np.ndarray]] = [[] for _ in results]
    active_seed = -1

    def global_solution(context, start, *, population, **_kwargs):
        nonlocal active_seed
        active_seed += 1
        return SimpleNamespace(
            **vars(_unchanged_local_solution(context, start)),
            population=np.array(population, copy=True),
            population_energies=np.asarray([evaluate_vector(context, row).objective for row in population]),
            trace=(),
        )

    def local_solution(context, start, **_kwargs):
        starts[active_seed].append(np.array(start, copy=True))
        return _unchanged_local_solution(context, results[active_seed])

    monkeypatch.setattr(stages, "solve_global", global_solution)
    monkeypatch.setattr(stages, "solve_local", local_solution)
    outcome = stages.run_stage_e(problem, "curve", parents, (101, 202, 303), progress=None, cancelled=None)
    assert outcome.summary.candidate_ids == ("E-0", "E-1", "E-2")
    return starts


def test_first_seed_material_gain_ignores_ineligible_parents(elite_case, monkeypatch) -> None:
    problem, _declared, good, poor_parent, good_parent = elite_case
    excluded = (
        replace(good_parent, candidate_id="D-invalid", valid=False),
        replace(good_parent, candidate_id="D-archived", stop_reason="early_eliminated"),
    )
    assert materially_improves(problem, poor_parent, good_parent)

    starts = _capture_local_starts(monkeypatch, problem, (*excluded, poor_parent), (good, good, good))

    assert [len(values) for values in starts] == [7, 8, 8]
    assert np.max(np.abs(starts[1][0] - good)) > 0.01
    assert np.max(np.abs(starts[1][1] - good)) <= 0.01
    assert not np.array_equal(starts[1][1], good)


def test_first_seed_nonmaterial_gain_uses_best_parent_not_publication_order(elite_case, monkeypatch) -> None:
    problem, declared, good, poor_parent, good_parent = elite_case
    first = good + 1e-7 * (declared - good)
    evaluation = evaluate_vector(problem, first)
    assert evaluation.objective < good_parent.objective
    assert not materially_improves(problem, good_parent, evaluation)
    assert materially_improves(problem, poor_parent, evaluation)

    starts = _capture_local_starts(monkeypatch, problem, (poor_parent, good_parent), (first, good, good))

    assert [len(values) for values in starts] == [8, 8, 8]


def test_unselectable_first_seed_preserves_parent_reference_for_next_result(elite_case, monkeypatch) -> None:
    problem, _declared, good, poor_parent, good_parent = elite_case
    elites = []

    def seed_result(_problem, _setup, index, _seed, elite, *_args, **_kwargs):
        elites.append(elite)
        return replace(
            good_parent,
            candidate_id=f"E-{index}",
            seed_index=index,
            valid=index > 0,
            objective=good_parent.objective if index > 0 else float("inf"),
        )

    monkeypatch.setattr(stages, "_stage_e_seed", seed_result)
    progress = []
    outcome = stages.run_stage_e(
        problem, "curve", (poor_parent,), (101, 202, 303), progress=progress.append, cancelled=None
    )

    assert elites[:2] == [None, None]
    np.testing.assert_array_equal(elites[2], good)
    assert progress[0].best_objective == float("inf")
    assert not outcome.candidates[0].valid
