from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting, SharingRule


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


def _assert_atomic_checkpoints(checkpoints) -> None:
    assert tuple(batch[0].stage for batch in checkpoints) == ("B", "C", "D", "E")
    assert all(len(batch) == 2 for batch in checkpoints)
    for batch in checkpoints:
        assert len({value.stage for value in batch}) == 1
        assert len({value.joint_layout_fingerprint for value in batch}) == 1
        assert tuple(candidate.candidate_id for candidate in batch[0].candidates) == tuple(
            candidate.candidate_id for candidate in batch[1].candidates
        )


def _assert_aligned_results(results) -> None:
    left, right = results
    assert left.best_index == right.best_index
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


def _replace_first_checkpoint(checkpoints, **changes):
    return (replace(checkpoints[0], **changes), *checkpoints[1:])


def test_joint_pipeline_projects_aligned_results_and_atomic_checkpoints() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    checkpoints: list[tuple[object, ...]] = []

    results = api.run_joint_fit(
        api.JointFitRequest(_joint_problem()),
        checkpoint=checkpoints.append,
    )

    assert len(results) == 2
    _assert_atomic_checkpoints(checkpoints)
    _assert_aligned_results(results)


def test_joint_pipeline_preserves_fully_locked_lineage_through_stage_e() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")

    results = api.run_joint_fit(api.JointFitRequest(_fully_locked_joint_problem()))

    assert all(
        tuple(summary.stage for summary in result.stage_summaries) == ("A", "B", "C", "D", "E")
        for result in results
    )
    for result in results:
        final = tuple(
            candidate
            for candidate in result.candidates
            if candidate.candidate_id.startswith("E-")
        )
        assert tuple(candidate.candidate_id for candidate in final) == ("E-0", "E-1", "E-2", "E-3")
        assert all(candidate.stop_reason == "no_free_parameters" for candidate in final)
        assert all(candidate.unit_vector.size == 0 for candidate in final)
        assert result.best_candidate in final


def test_joint_pipeline_uses_stage_layouts_short_and_full_de_then_local_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    de_calls: list[tuple[int, int]] = []
    local_calls: list[int] = []

    def fake_de(objective, bounds, *, x0, init, maxiter, **_kwargs):
        unit = np.array(x0, dtype=float, copy=True)
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

    monkeypatch.setattr(api, "differential_evolution", fake_de, raising=False)
    monkeypatch.setattr(api, "least_squares", fake_local)

    results = api.run_joint_fit(api.JointFitRequest(_staged_joint_problem()))

    assert de_calls == [(2, 0), (2, 0), (3, 0), (3, 0), (3, 0), (3, 0)]
    assert local_calls == [1, 1, 3, 3, 3, 3]
    assert tuple(summary.candidate_ids for summary in results[0].stage_summaries[1:]) == (
        ("B-0", "B-1"),
        ("C-0-0", "C-1-0"),
        ("D-0-0", "D-1-0"),
        ("E-0", "E-1", "E-2", "E-3"),
    )


def test_joint_request_handler_and_result_tuple_are_pickle_safe() -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    request = api.JointFitRequest(_joint_problem())

    assert [field.name for field in fields(request)] == ["problem", "resume_checkpoints"]
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
    ("completed_stage", "expected_suffix"),
    [
        pytest.param("B", ("C", "D", "E"), id="from-b"),
        pytest.param("C", ("D", "E"), id="from-c"),
        pytest.param("D", ("E",), id="from-d"),
        pytest.param("E", (), id="from-e"),
    ],
)
def test_joint_resume_runs_only_the_suffix_and_matches_fresh_results(
    completed_stage: str,
    expected_suffix: tuple[str, ...],
) -> None:
    api = import_module("xrr_fitter.fit.joint_pipeline")
    joint = _joint_problem()
    fresh_checkpoints: list[tuple[object, ...]] = []
    fresh = api.run_joint_fit(
        api.JointFitRequest(joint),
        checkpoint=fresh_checkpoints.append,
    )
    resume_from = next(
        batch for batch in fresh_checkpoints if batch[0].stage == completed_stage
    )
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
        ranking_objective=(first.candidates[0].ranking_objective or first.candidates[0].objective)
        + 0.25,
    )
    wrong_unit = replace(
        first.candidates[0],
        unit_vector=np.clip(first.candidates[0].unit_vector + 0.01, 0.0, 1.0),
    )
    cases = (
        tuple(reversed(stage_d)),
        _replace_first_checkpoint(stage_d, joint_layout_fingerprint="0" * 64),
        _replace_first_checkpoint(stage_d, candidates=tuple(reversed(first.candidates))),
        _replace_first_checkpoint(stage_d, child_seeds=tuple(reversed(first.child_seeds))),
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

    def compile_joint(dataset_ids, problems, rules):
        calls.append(("compile-joint", (dataset_ids, problems, rules)))
        return "joint-problem"

    def run_joint(request, **_kwargs):
        calls.append(("joint", request.problem))
        return joint_results

    monkeypatch.setattr(api, "run_fit_search", run_independent)
    monkeypatch.setattr(api, "compile_joint_problem", compile_joint)
    monkeypatch.setattr(api, "run_joint_fit", run_joint)

    independent = api.run_fit_batch(api.FitBatchRequest("independent", requests))
    joint = api.run_fit_batch(
        api.FitBatchRequest("joint", requests, sharing_rules=_sharing_rules())
    )

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
