"""Cancellation before publication must not become a completed joint prefix.

The numerical solvers and full reflectivity evaluations here are real. Narrow
wrappers request cancellation only after the selected call has computed its
result, exposing the window between numerical completion and stage publication.
C/D exercise both free and locked layouts. E exercises the last local return and
the final full-grid curve evaluation, where no later seed can notice cancellation.

Progress callbacks are intentionally different: their event announces an already
completed seed. Cancellation requested by that event must retain its atomic
checkpoint, including when the event belongs to the final seed of the run.
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest
from tests.unit.fit.test_adaptive_joint_search import _joint_problem

from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.local_search import SearchCancelled


def _cancel_after_local_return(monkeypatch, target_call):
    pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    original = pipeline._solve_joint
    state = SimpleNamespace(cancelled=False, calls=0)

    def solve(problem, start, maximum, cancelled):
        result = original(problem, start, maximum, cancelled)
        state.calls += 1
        if state.calls == target_call:
            state.cancelled = True
        return result

    monkeypatch.setattr(pipeline, "_solve_joint", solve)
    return state


def _candidate_evidence(checkpoint):
    return tuple(candidate.search_evidence for candidate in checkpoint.candidates)


def _assert_atomic_batches(batches):
    for left, right in batches:
        assert left.stage_summaries == right.stage_summaries
        assert left.child_seeds == right.child_seeds
        assert _candidate_evidence(left) == _candidate_evidence(right)


def _assert_e_prefix(batches, completed):
    final = batches[-1][0]
    expected_ids = tuple(f"E-{index}" for index in range(completed))
    assert final.stage == "E"
    assert final.stage_summaries[-1].candidate_ids == expected_ids
    assert len(final.child_seeds) == completed + 1
    _assert_atomic_batches(batches)


@pytest.mark.parametrize("locked", [False, True])
def test_joint_b_full_publication_cancellation_does_not_publish_a_checkpoint(monkeypatch, locked):
    problem = _joint_problem(locked=locked)
    pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    original = pipeline.evaluate_joint_vector
    state = SimpleNamespace(cancelled=False, calls=0)
    batches, progress = [], []

    def evaluate(context, unit):
        result = original(context, unit)
        state.calls += 1
        if state.calls == 2:
            state.cancelled = True
        return result

    monkeypatch.setattr(pipeline, "evaluate_joint_vector", evaluate)
    with pytest.raises(SearchCancelled):
        run_joint_fit(
            JointFitRequest(problem),
            cancelled=lambda: state.cancelled,
            checkpoint=batches.append,
            progress=progress.append,
        )

    assert state.calls == 2
    assert batches == []
    assert [event.stage for event in progress] == ["A"]


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("stage, target_call, expected_stages", [("C", 1, ["B"]), ("D", 2, ["B", "C"])])
def test_joint_c_d_cancelled_on_local_return_do_not_publish_the_stage(
    monkeypatch,
    locked,
    stage,
    target_call,
    expected_stages,
):
    problem = _joint_problem(locked=locked)
    state = _cancel_after_local_return(monkeypatch, target_call)
    batches, progress = [], []

    with pytest.raises(SearchCancelled):
        run_joint_fit(
            JointFitRequest(problem),
            cancelled=lambda: state.cancelled,
            checkpoint=batches.append,
            progress=progress.append,
        )

    assert state.cancelled
    assert [batch[0].stage for batch in batches] == expected_stages
    assert stage not in [event.stage for event in progress]
    _assert_atomic_batches(batches)


def test_joint_last_e_local_return_cancellation_does_not_publish_a_complete_result(monkeypatch):
    problem = _joint_problem()
    final_count = problem.problems[0].config.final_seed_count
    state = _cancel_after_local_return(monkeypatch, 2 + final_count)
    batches, progress = [], []

    with pytest.raises(SearchCancelled):
        run_joint_fit(
            JointFitRequest(problem),
            cancelled=lambda: state.cancelled,
            checkpoint=batches.append,
            progress=progress.append,
        )

    assert state.cancelled
    assert progress[-1].stage == "E"
    assert progress[-1].completed == final_count - 1
    _assert_e_prefix(batches, final_count - 1)


def _cancel_during_last_e_publication(monkeypatch, problem, locked):
    pipeline = import_module("xrr_fitter.fit.joint_pipeline")
    solvers = import_module("xrr_fitter.fit.joint_solvers")
    original = solvers.evaluate_joint_vector
    state = SimpleNamespace(cancelled=False, armed=False, full_evaluations=0)
    target_count = 1 if locked else 2

    def evaluate(context, unit, *, fit_only=False):
        result = original(context, unit, fit_only=fit_only)
        if state.armed and context is problem and not fit_only:
            state.full_evaluations += 1
            if state.full_evaluations == target_count:
                state.cancelled = True
        return result

    monkeypatch.setattr(pipeline, "evaluate_joint_vector", evaluate)
    monkeypatch.setattr(solvers, "evaluate_joint_vector", evaluate)
    return state


@pytest.mark.parametrize("locked", [False, True])
def test_joint_last_e_full_publication_cancellation_keeps_the_previous_prefix(monkeypatch, locked):
    problem = _joint_problem(locked=locked)
    final_count = problem.problems[0].config.final_seed_count
    state = _cancel_during_last_e_publication(monkeypatch, problem, locked)
    batches, progress = [], []

    def completed(event):
        progress.append(event)
        if event.stage == "E" and event.completed == final_count - 1:
            state.armed = True

    with pytest.raises(SearchCancelled):
        run_joint_fit(
            JointFitRequest(problem),
            cancelled=lambda: state.cancelled,
            checkpoint=batches.append,
            progress=completed,
        )

    assert state.cancelled
    assert state.full_evaluations == (1 if locked else 2)
    assert progress[-1].completed == final_count - 1
    _assert_e_prefix(batches, final_count - 1)


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("completed_seeds", [1, 4])
def test_joint_completed_seed_progress_cancellation_retains_that_atomic_checkpoint(locked, completed_seeds):
    problem = _joint_problem(locked=locked)
    cancelled = False
    batches = []

    def progress(event):
        nonlocal cancelled
        if event.stage == "E" and event.completed == completed_seeds:
            cancelled = True

    def run():
        return run_joint_fit(
            JointFitRequest(problem),
            cancelled=lambda: cancelled,
            checkpoint=batches.append,
            progress=progress,
        )

    if completed_seeds == problem.problems[0].config.final_seed_count:
        results = run()
        assert len(results) == 2
    else:
        with pytest.raises(SearchCancelled):
            run()
    assert cancelled
    _assert_e_prefix(batches, completed_seeds)
