"""Joint path histograms retain exits and permit independent budget verification."""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, replace
from importlib import import_module, util

import pytest


def _model():
    name = "xrr_fitter.model.diagnostic_work"
    assert util.find_spec(name) is not None, "immutable two-stage work evidence is required"
    return import_module(name)


def _exit(phase="localize", **changes):
    values = dict(
        phase=phase, status=0 if phase == "localize" else 3, success=True, message="native stop", nfev=2, kind="native"
    )
    if phase == "handoff":
        values.update(status=None, nfev=1, kind="validation", message="diagnostic_handoff_validated")
    return _model().DiagnosticSolverExit(**(values | changes))


def _path(*, count=1, budget=9, stages=None):
    stages = tuple(_exit(phase) for phase in ("localize", "handoff", "refine")) if stages is None else stages
    return _model().DiagnosticPathSummary(budget, stages, count)


def _work(*, count=4):
    return _model().DiagnosticRefitWork(4, (_path(count=count),))


def test_joint_histogram_derives_exact_work_and_max_without_repeated_trajectories():
    work = _work()
    assert work.path_count == 4 and work.nfev == 20
    assert work.path_budget == 9 and work.max_path_nfev == 5
    assert work.completed_paths == 4
    assert len(work.paths) == 1
    assert pickle.loads(pickle.dumps(work)) == work
    with pytest.raises(FrozenInstanceError):
        work.declared_paths = 3


def test_merge_counts_every_path_and_preserves_distinct_native_exit_reasons():
    model = _model()
    abnormal = _exit(status=2, success=False, message="ABNORMAL")
    second = model.DiagnosticRefitWork(4, (_path(stages=(abnormal, _exit("handoff"), _exit("refine"))),))
    merged = model.combine_refit_work(_work(count=2), second, _work(count=3))
    assert merged.path_count == 6 and merged.nfev == 30 and merged.completed_paths == 6
    assert len(merged.paths) == 2
    assert sorted(row.count for row in merged.paths) == [1, 5]
    assert {row.stages[0].message for row in merged.paths} == {"native stop", "ABNORMAL"}
    assert model.combine_refit_work(model.DiagnosticRefitWork(), merged) == merged


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "unknown"},
        {"status": True},
        {"nfev": True},
        {"nfev": -1},
        {"nfev": 0},
        {"success": 1},
        {"message": " "},
        {"kind": "unknown"},
        {"kind": "validation"},
        {"status": None},
        {"kind": "not_entered", "status": None, "success": False},
    ],
)
def test_exit_rejects_ambiguous_or_inconsistent_native_evidence(changes):
    with pytest.raises((TypeError, ValueError)):
        _exit(**changes)


@pytest.mark.parametrize(
    ("phase", "status", "success"),
    [
        ("localize", 0, False),
        ("localize", 1, True),
        ("localize", 2, True),
        ("localize", -1, False),
        ("localize", 3, True),
        ("refine", -2, True),
        ("refine", -1, True),
        ("refine", 0, True),
        ("refine", 1, False),
        ("refine", 2, False),
        ("refine", 3, False),
        ("refine", 4, False),
        ("refine", 999, True),
    ],
)
def test_native_status_must_have_its_actual_solver_success_semantics(phase, status, success):
    with pytest.raises(ValueError, match="status"):
        _exit(phase, status=status, success=success)


@pytest.mark.parametrize(
    ("phase", "status", "success"),
    [
        ("localize", 0, True),
        ("localize", 1, False),
        ("localize", 2, False),
        ("refine", -2, False),
        ("refine", -1, False),
        ("refine", 0, False),
        ("refine", 1, True),
        ("refine", 2, True),
        ("refine", 3, True),
        ("refine", 4, True),
    ],
)
def test_native_status_preserves_every_known_exit(phase, status, success):
    stage = _exit(phase, status=status, success=success)
    assert (stage.status, stage.success) == (status, success)


def test_native_refinement_budget_exit_cannot_underreport_its_charged_work():
    stages = (_exit(), _exit("handoff"), _exit("refine", status=0, success=False))
    with pytest.raises(ValueError, match="budget"):
        _path(stages=stages)
    exhausted = _path(stages=(*stages[:2], replace(stages[2], nfev=6)))
    assert exhausted.nfev == exhausted.budget == 9
    interrupted = _path(stages=(*stages[:2], replace(stages[2], status=-2)))
    assert interrupted.nfev == 5 and not interrupted.completed


def test_path_requires_ordered_three_phase_axis_and_enforces_each_actual_cap():
    model = _model()
    stages = tuple(_exit(phase) for phase in ("localize", "handoff", "refine"))
    for malformed in (stages[:2], stages[::-1], (stages[0], stages[0], stages[2])):
        with pytest.raises(ValueError):
            _path(stages=malformed)
    with pytest.raises(ValueError, match="localization"):
        _path(budget=4)
    with pytest.raises(ValueError, match="budget"):
        _path(stages=(*stages[:2], replace(stages[2], nfev=8)))
    with pytest.raises(ValueError):
        model.DiagnosticPathSummary(True, stages)
    with pytest.raises(ValueError):
        _path(count=True)


def test_zero_dimensions_and_failed_prefixes_have_explicit_unentered_stages():
    model = _model()
    zero = tuple(
        model.DiagnosticSolverExit(phase, None, False, "locked", 0, "zero_dimensional")
        for phase in ("localize", "handoff", "refine")
    )
    work = model.DiagnosticRefitWork(1, (_path(stages=zero, count=100),))
    assert work.nfev == work.max_path_nfev == 0
    assert work.path_count == work.completed_paths == 100
    failure = model.DiagnosticSolverExit("localize", None, False, "FloatingPointError:bad", 1, "numerical_failure")
    skipped = tuple(
        model.DiagnosticSolverExit(phase, None, False, "not_entered", 0, "not_entered")
        for phase in ("handoff", "refine")
    )
    failed = model.DiagnosticRefitWork(4, (_path(stages=(failure, *skipped)),))
    assert failed.path_count == failed.nfev == 1 and failed.completed_paths == 0
    with pytest.raises(ValueError):
        _path(stages=(failure, _exit("handoff"), _exit("refine")))
    with pytest.raises(ValueError):
        _path(stages=(_exit(), *skipped))


def test_small_budget_is_recorded_without_any_phase_evaluation():
    model = _model()
    localize = model.DiagnosticSolverExit("localize", None, False, "insufficient budget", 0, "budget")
    skipped = tuple(
        model.DiagnosticSolverExit(phase, None, False, "not_entered", 0, "not_entered")
        for phase in ("handoff", "refine")
    )
    work = model.DiagnosticRefitWork(4, (_path(budget=1, stages=(localize, *skipped)),))
    assert work.path_count == 1 and work.nfev == 0 and work.completed_paths == 0


def test_histogram_cannot_duplicate_bins_change_declared_paths_or_mix_path_budgets():
    model = _model()
    with pytest.raises(ValueError, match="unique"):
        model.DiagnosticRefitWork(4, (_path(), _path()))
    with pytest.raises(ValueError):
        model.DiagnosticRefitWork(0, (_path(),))
    with pytest.raises(ValueError):
        model.DiagnosticRefitWork(True, (_path(),))
    with pytest.raises(ValueError):
        model.combine_refit_work(_work(), model.DiagnosticRefitWork(3, (_path(),)))
    with pytest.raises(ValueError):
        model.combine_refit_work(_work(), model.DiagnosticRefitWork(4, (_path(budget=11),)))


@pytest.mark.parametrize("declared_paths", [1, 2])
def test_nonzero_dimension_requires_three_or_four_deduplicated_starts(declared_paths):
    with pytest.raises(ValueError, match="declared|dimensional"):
        _model().DiagnosticRefitWork(declared_paths, (_path(),))


@pytest.mark.parametrize("declared_paths", [3, 4])
def test_zero_dimension_cannot_claim_a_multistart_axis(declared_paths):
    model = _model()
    stages = tuple(
        model.DiagnosticSolverExit(phase, None, False, "locked", 0, "zero_dimensional") for phase in model.PHASES
    )
    with pytest.raises(ValueError, match="declared|dimensional"):
        model.DiagnosticRefitWork(declared_paths, (_path(stages=stages),))


def test_one_work_histogram_cannot_mix_locked_and_nonzero_dimensional_paths():
    model = _model()
    stages = tuple(
        model.DiagnosticSolverExit(phase, None, False, "locked", 0, "zero_dimensional") for phase in model.PHASES
    )
    paths = tuple(sorted((_path(), _path(stages=stages)), key=model.path_key))
    with pytest.raises(ValueError, match="dimensional"):
        model.DiagnosticRefitWork(4, paths)


def test_nested_frozen_records_own_and_revalidate_their_values():
    model = _model()
    stage = _exit()
    stages = [stage, _exit("handoff"), _exit("refine")]
    path = _path(stages=stages)
    stages.clear()
    work = model.DiagnosticRefitWork(4, (path,))
    object.__setattr__(stage, "nfev", 900)
    object.__setattr__(path, "count", True)
    assert work.nfev == 5 and work.path_count == 1
    corrupted = work.paths[0].stages[0]
    object.__setattr__(corrupted, "nfev", True)
    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(work))
