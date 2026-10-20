"""Exact Stage-E local reuse preserves full-data evidence and lineage."""

import numpy as np
import pytest
from tests.unit.fit.test_stage_search import _candidate, _problem, _unchanged_local_solution

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit import stages
from xrr_fitter.fit.local_search import SearchCancelled


def _case():
    problem = _problem(size=1200)
    unit = encode_physical_vector(problem, {})
    parent = _candidate(problem, "D-0-0", unit)
    return problem, stages._stage_e_setup(problem, (parent,)), unit


def _run(problem, setup, starts, seed=0, cancelled=None, runner=None):
    return stages._run_stage_e_locals(problem, setup, starts, seed, "local", cancelled, runner, source_seed=100 + seed)


def _assert_reused_evidence(candidate, original):
    np.testing.assert_array_equal(candidate.model_normalized, original.model_normalized)
    assert candidate.objective == original.objective
    assert candidate.qz_a_inv.size == 1200
    assert "reused_exact_start:E-0-local-0" in candidate.search_evidence[0].stop_reason


def test_exact_starts_solve_once_across_batches_with_honest_work_counts(monkeypatch):
    problem, setup, unit = _case()
    calls = []

    def solve(context, start, **kwargs):
        calls.append((context, start.copy()))
        return _unchanged_local_solution(context, start)

    monkeypatch.setattr(stages, "solve_local", solve)
    first = _run(problem, setup, (unit, unit.copy()))
    later = _run(problem, setup, (unit.copy(),), seed=1)

    assert len(calls) == 1
    assert calls[0][0].data.qz_a_inv.size == 1200
    assert [c.candidate_id for c in first + later] == ["E-0-local-0", "E-0-local-1", "E-1-local-0"]
    assert [c.seed_index for c in first + later] == [0, 0, 1]
    assert first[0].nfev > 0
    assert first[1].nfev == later[0].nfev == 0
    for candidate in first[1:] + later:
        _assert_reused_evidence(candidate, first[0])
    assert later[0].search_evidence[0].seed == 101


def test_nearby_starts_are_not_merged_and_cache_does_not_cross_setups(monkeypatch):
    problem, setup, unit = _case()
    calls = []

    def solve(context, start, **kwargs):
        calls.append(start.copy())
        return _unchanged_local_solution(context, start)

    monkeypatch.setattr(stages, "solve_local", solve)
    near = unit.copy()
    near[0] = np.nextafter(near[0], 1.0)
    _run(problem, setup, (unit, near))
    independent = stages._stage_e_setup(problem, (_candidate(problem, "D-0-0", unit),))
    _run(problem, independent, (unit,))
    assert len(calls) == 3


def test_reuse_polls_cancellation_before_returning_cached_result(monkeypatch):
    problem, setup, unit = _case()
    monkeypatch.setattr(stages, "solve_local", _unchanged_local_solution)
    _run(problem, setup, (unit,))
    with pytest.raises(SearchCancelled):
        _run(problem, setup, (unit,), seed=1, cancelled=lambda: True)


def test_cancelled_batch_does_not_commit_partial_cache(monkeypatch):
    problem, setup, unit = _case()
    calls = []

    def solve(context, start, **kwargs):
        calls.append(start.copy())
        return _unchanged_local_solution(context, start)

    def interrupted(tasks):
        tasks[0]()
        raise SearchCancelled("cancelled batch")

    monkeypatch.setattr(stages, "solve_local", solve)
    with pytest.raises(SearchCancelled):
        _run(problem, setup, (unit,), runner=interrupted)
    _run(problem, setup, (unit,))
    assert len(calls) == 2
