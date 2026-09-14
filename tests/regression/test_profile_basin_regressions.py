from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure
from tests.support.synthetic_recovery_layer_cases import _oxide_cap_cases
from tests.support.synthetic_recovery_runtime import _fit_config, _generate_case_intensity, _prepared_case_data

from xrr_fitter.analysis.profiles import recover_profile_basin
from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    encode_physical_vector,
    evaluate_model,
)
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import ProfileBasinDecision
from xrr_fitter.model.fitting import (
    FitCheckpoint,
    FitConfig,
    FitSearchResult,
    FitStageSummary,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.provenance import fit_search_provenance_sha256


def _problem():
    initial = compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(947), scale_prior_enabled=False),
    )
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name != "component.0.thickness_a",
        )
        for definition in initial.parameter_definitions
    )
    return compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        settings,
    )


def _state(problem):
    unit = encode_physical_vector(problem, {})
    evaluation = evaluate_model(problem, unit)
    candidates = tuple(
        replace(
            candidate_from_evaluation(
                problem,
                unit,
                evaluation,
                candidate_id=f"E-{index}",
                seed_index=index,
                stop_reason="original stage E",
                nfev=1,
            ),
            objective=evaluation.objective + max(0.10, abs(evaluation.objective) * (0.10 + 0.01 * index)),
        )
        for index in range(4)
    )
    summary = FitStageSummary(
        "E",
        tuple(candidate.candidate_id for candidate in candidates),
        candidates[0].objective,
        4,
        tuple(candidate.stop_reason for candidate in candidates),
    )
    unsealed = FitSearchResult(
        problem.parameter_definitions,
        candidates,
        0,
        (),
        (101, 102, 103, 104),
        (summary,),
        problem.region_labels,
        problem.weights,
    )
    search = replace(
        unsealed,
        provenance_sha256=fit_search_provenance_sha256(problem, unsealed),
    )
    decision = ProfileBasinDecision(
        "component.0.thickness_a",
        unit,
        evaluation.objective,
        ("materially_better_profile_basin",),
    )
    return search, decision, evaluation


def _successful_local(problem, start, *, max_nfev, cancelled=None):
    del max_nfev
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")
    unit = np.asarray(start, dtype=float)
    return SimpleNamespace(
        unit_vector=unit,
        evaluation=evaluate_model(problem, unit),
        stop_reason="profile reconverged",
        nfev=2,
    )


def _continue_profile_basin(pipeline, problem, search, decision, **kwargs):
    return pipeline.continue_profile_basin(
        problem,
        search,
        decision.unit_vector,
        parameter_name=decision.parameter_name,
        **kwargs,
    )


def _stage_e_candidates(result: FitSearchResult):
    return tuple(candidate for candidate in result.candidates if candidate.candidate_id.startswith("E-"))


def _assert_reconverged_identity(stage_e) -> None:
    assert tuple(candidate.candidate_id for candidate in stage_e) == (
        "E-0",
        "E-1",
        "E-2",
        "E-3",
    )
    assert tuple(candidate.seed_index for candidate in stage_e) == (0, 1, 2, 3)
    assert all(candidate.stop_reason.startswith("profile_basin_rescue:") for candidate in stage_e)


def _assert_reconverged_stage_e(
    result: FitSearchResult,
    search: FitSearchResult,
    starts: list[np.ndarray],
    expected_objective: float,
) -> None:
    _assert_reconverged_identity(_stage_e_candidates(result))
    assert len(starts) == 4
    assert len({start.tobytes() for start in starts}) == 4
    assert result.best_candidate.objective == pytest.approx(expected_objective, rel=1e-3)
    assert result.child_seeds == search.child_seeds
    assert result.stage_summaries[-1].candidate_ids == ("E-0", "E-1", "E-2", "E-3")


def test_fit_dataset_attempts_profile_basin_rescue_before_bootstrap(monkeypatch) -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    profile_rescue = import_module("xrr_fitter.fit.profile_rescue")
    problem = _problem()
    search, decision, _evaluation = _state(problem)
    calls: list[str] = []

    def observed(*args, **kwargs):
        calls.append("reconverge")
        return profile_rescue.reconverge_profile_basin(*args, **kwargs)

    monkeypatch.setattr(profile_rescue, "solve_local", _successful_local)
    monkeypatch.setattr(pipeline, "reconverge_profile_basin", observed)

    continued = _continue_profile_basin(pipeline, problem, search, decision)
    calls.append("bootstrap")

    assert continued is not search
    assert calls == ["reconverge", "bootstrap"]


def test_fit_dataset_reconverges_profile_basin_across_four_stage_e_seeds(
    monkeypatch,
) -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    profile_rescue = import_module("xrr_fitter.fit.profile_rescue")
    problem = _problem()
    search, decision, evaluation = _state(problem)
    starts: list[np.ndarray] = []

    def observed(problem_value, start, **kwargs):
        starts.append(np.asarray(start, dtype=float).copy())
        return _successful_local(problem_value, start, **kwargs)

    monkeypatch.setattr(profile_rescue, "solve_local", observed)
    result = _continue_profile_basin(pipeline, problem, search, decision)

    _assert_reconverged_stage_e(result, search, starts, evaluation.objective)


@pytest.mark.parametrize(
    "revalidation",
    (
        pytest.param("equivalent", id="equivalent"),
        pytest.param("invalid", id="invalid"),
        pytest.param("nonfinite", id="nonfinite"),
    ),
)
def test_fit_dataset_rejects_unverified_profile_basin_rescue(
    revalidation: str,
    monkeypatch,
) -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    problem = _problem()
    search, decision, evaluation = _state(problem)

    if revalidation == "invalid":
        observed = replace(evaluation, valid=False, objective=float("inf"))
    elif revalidation == "nonfinite":
        observed = SimpleNamespace(valid=True, objective=float("nan"))
    else:
        observed = replace(evaluation, objective=search.best_candidate.objective)
    monkeypatch.setattr(pipeline, "evaluate_model", lambda *_args: observed)

    result = _continue_profile_basin(pipeline, problem, search, decision)

    assert result is search


def test_fit_dataset_rejects_constraint_failed_profile_basin_decision(
    monkeypatch,
) -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    problem = _problem()
    search, decision, _evaluation = _state(problem)

    def constrained(*_args):
        raise EvaluationConstraintError("constraint_violation:ValueError")

    monkeypatch.setattr(pipeline, "evaluate_model", constrained)

    result = _continue_profile_basin(pipeline, problem, search, decision)

    assert result is search


def test_profile_continuation_rejects_tampered_historical_candidate(
    monkeypatch,
) -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    problem = _problem()
    search, decision, _evaluation = _state(problem)
    historical = replace(search.candidates[0], candidate_id="B-0")
    unsealed = replace(
        search,
        candidates=(historical, *search.candidates),
        best_index=1,
        provenance_sha256=None,
    )
    owned = replace(
        unsealed,
        provenance_sha256=fit_search_provenance_sha256(problem, unsealed),
    )
    tampered = replace(
        historical,
        sld_profile_a2=np.zeros_like(historical.sld_profile_a2),
    )
    changed = replace(owned, candidates=(tampered, *owned.candidates[1:]))
    monkeypatch.setattr(
        pipeline,
        "reconverge_profile_basin",
        lambda _problem, candidates, *_args, **_kwargs: candidates,
    )

    with pytest.raises(ValueError, match="search_result|provenance|context"):
        _continue_profile_basin(pipeline, problem, changed, decision)


def test_fit_dataset_cancels_profile_reconvergence_before_state_commit(
    monkeypatch,
) -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    profile_rescue = import_module("xrr_fitter.fit.profile_rescue")
    problem = _problem()
    search, decision, _evaluation = _state(problem)
    checkpoints: list[FitCheckpoint] = []
    monkeypatch.setattr(profile_rescue, "solve_local", _successful_local)

    with pytest.raises(SearchCancelled, match="cancelled"):
        _continue_profile_basin(
            pipeline,
            problem,
            search,
            decision,
            cancelled=lambda: True,
            checkpoint=checkpoints.append,
        )

    assert checkpoints == []
    assert search.stage_summaries[-1].stop_reasons == ("original stage E",) * 4


def test_fit_dataset_maps_profile_cancellation_to_cancelled_result(monkeypatch) -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")
    profile_rescue = import_module("xrr_fitter.fit.profile_rescue")
    problem = _problem()
    search, decision, _evaluation = _state(problem)

    def interrupted(*_args, **_kwargs):
        raise SearchCancelled("search cancelled")

    monkeypatch.setattr(profile_rescue, "solve_local", interrupted)

    with pytest.raises(SearchCancelled, match="cancelled"):
        _continue_profile_basin(pipeline, problem, search, decision)


@pytest.mark.parametrize(
    "outcome",
    (
        pytest.param("incomplete", id="incomplete"),
        pytest.param("invalid", id="invalid"),
        pytest.param("nonfinite", id="nonfinite"),
    ),
)
def test_profile_reconvergence_rejects_unpublishable_local_paths(
    outcome: str,
    monkeypatch,
) -> None:
    profile_rescue = import_module("xrr_fitter.fit.profile_rescue")
    problem = _problem()
    search, decision, evaluation = _state(problem)
    calls = 0

    def local(problem_value, start, **_kwargs):
        nonlocal calls
        calls += 1
        if outcome == "incomplete" and calls == 4:
            return None
        if outcome == "invalid":
            observed = replace(evaluation, valid=False, objective=float("inf"))
        elif outcome == "nonfinite":
            observed = SimpleNamespace(valid=True, objective=float("nan"))
        else:
            observed = evaluate_model(problem_value, start)
        return SimpleNamespace(
            unit_vector=np.asarray(start, dtype=float),
            evaluation=observed,
            stop_reason="local",
            nfev=1,
        )

    monkeypatch.setattr(profile_rescue, "solve_local", local)

    result = profile_rescue.reconverge_profile_basin(
        problem,
        search.candidates,
        decision.unit_vector,
        search.child_seeds,
        parameter_name=decision.parameter_name,
    )

    assert result is None


def test_profile_reconvergence_rejects_four_valid_paths_without_material_gain(
    monkeypatch,
) -> None:
    profile_rescue = import_module("xrr_fitter.fit.profile_rescue")
    problem = _problem()
    search, decision, _evaluation = _state(problem)

    def stable_no_gain(problem_value, start, **_kwargs):
        del problem_value, _kwargs
        return SimpleNamespace(
            unit_vector=np.asarray(start, dtype=float),
            evaluation=SimpleNamespace(
                valid=True,
                objective=search.best_candidate.objective,
            ),
            stop_reason="no gain",
            nfev=1,
        )

    monkeypatch.setattr(profile_rescue, "solve_local", stable_no_gain)

    rebuilt = profile_rescue.reconverge_profile_basin(
        problem,
        search.candidates,
        decision.unit_vector,
        search.child_seeds,
        parameter_name=decision.parameter_name,
    )

    assert rebuilt is None


def test_oxide_cap_recovery_finds_the_basin_seen_by_reported_profiles() -> None:
    case = next(case for case in _oxide_cap_cases() if case.case_id == "oxide-cap-14017")
    data = _prepared_case_data(case, _generate_case_intensity(case))
    problem = compile_fit_problem(data, case.fit_structure, case.fit_instrument, _fit_config(case))
    # Frozen failed Stage-E center; exercise the real inference handoff without
    # rerunning the global search or changing the corpus input, seed, or budget.
    unit = np.asarray(
        [
            0.11687643352138376,
            0.9573901326699007,
            0.00082410047806986,
            0.5450205227856092,
            0.521080557016504,
            0.8414340132570994,
            0.019400872852358714,
            0.32138728325358856,
            0.49470604876855967,
            3.4755484498953376e-05,
            0.055184774877327786,
        ]
    )
    observed = evaluate_model(problem, unit)
    candidate = SimpleNamespace(valid=observed.valid, objective=observed.objective, unit_vector=unit)

    decision = recover_profile_basin(problem, candidate)

    assert decision is not None
    assert decision.parameter_name == "component.0.thickness_a"
    assert decision.objective < 0.5 * candidate.objective
    assert evaluate_model(problem, decision.unit_vector).objective == pytest.approx(decision.objective)
