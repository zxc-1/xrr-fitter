"""Static type contracts for the stage orchestration boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_type_hints

from xrr_fitter.fit import stages
from xrr_fitter.fit.global_search import GlobalSearchResult
from xrr_fitter.fit.local_search import LocalSearchResult
from xrr_fitter.fit.tasking import TaskRunner
from xrr_fitter.model.fitting import FitCandidate, FitEvaluationContext, FitProgress, ModelEvaluation


def test_stage_helpers_use_the_compiled_fit_context() -> None:
    function_names = (
        "_parameter_settings",
        "compile_coarse_problem",
        "_stage_problems",
        "_complete_values",
        "_published_candidate",
        "_coarse_log_curve",
        "_stage_a_candidate",
        "_evaluate_stage_a_pool",
        "run_stage_a",
        "_stage_b_candidate",
        "_stage_b_launch_evidence",
        "_stage_b_geometry_indices",
        "_stage_b_geometry_group",
        "_stage_b_representatives",
        "run_stage_b",
        "_local_stage_candidate",
        "_local_stage_starts",
        "run_local_stage",
        "_stage_e_setup",
        "_population_energies",
        "_stage_e_local_candidate",
        "_run_stage_e_locals",
        "_stage_e_seed",
        "_materially_improves",
        "_profile_rescue_start",
        "_profile_rescue_paths",
        "_solve_profile_rescue_path",
        "_valid_profile_rescue_result",
        "_publish_profile_rescue",
        "reconverge_profile_basin",
        "run_stage_e",
    )

    for name in function_names:
        assert get_type_hints(getattr(stages, name))["problem"] is FitEvaluationContext


def test_stage_setup_uses_concrete_fit_contexts() -> None:
    setup_hints = get_type_hints(stages._StageESetup)
    assert setup_hints["coarse_problem"] is FitEvaluationContext
    assert setup_hints["full_problem"] is FitEvaluationContext


def test_stage_population_uses_global_solver_results() -> None:
    population_hints = get_type_hints(stages._population_energies)
    population_starts_hints = get_type_hints(stages._population_starts)
    assert population_hints["solved"] is GlobalSearchResult
    assert population_starts_hints["solved"] is GlobalSearchResult


def test_stage_rescue_uses_local_solver_results() -> None:
    rescue_hints = get_type_hints(stages._solve_profile_rescue_path)
    valid_hints = get_type_hints(stages._valid_profile_rescue_result)
    paths_hints = get_type_hints(stages._profile_rescue_paths)
    publish_hints = get_type_hints(stages._publish_profile_rescue)
    assert rescue_hints["return"] is LocalSearchResult
    assert valid_hints["result"] is LocalSearchResult
    assert paths_hints["return"] == tuple[LocalSearchResult, ...] | None
    assert publish_hints["refined"] == tuple[LocalSearchResult, ...]


def test_stage_evaluation_annotations_preserve_domain_boundaries() -> None:
    published_hints = get_type_hints(stages._published_candidate)
    assert published_hints["return"] is FitCandidate
    assert get_type_hints(stages._materially_improves)["candidate"] == FitCandidate | ModelEvaluation


def test_stage_callbacks_preserve_domain_boundaries() -> None:
    stage_hints = get_type_hints(stages.run_stage_e)
    assert stage_hints["progress"] == Callable[[FitProgress], None] | None
    assert stage_hints["cancelled"] == Callable[[], bool] | None
    assert stage_hints["task_runner"] == TaskRunner | None
