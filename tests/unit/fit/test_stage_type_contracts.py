"""Seed scheduling ownership and typed stage orchestration contracts."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import get_type_hints

import numpy as np
import pytest

from xrr_fitter.fit import adaptive_review, candidates, local_budget, profile_rescue, stage_schedule, stages
from xrr_fitter.fit.global_search import GlobalSearchResult
from xrr_fitter.fit.local_search import LocalSearchResult
from xrr_fitter.fit.tasking import TaskRunner
from xrr_fitter.model.evaluation import ModelEvaluation
from xrr_fitter.model.fitting import FitCandidate, FitEvaluationContext, FitProgress
from xrr_fitter.model.search import SearchEvidence


def test_child_seed_lineage_is_deterministic_and_order_independent() -> None:
    api = stage_schedule
    streams = ("B-0", "B-1", "E-0", "E-1", "E-2", "E-3")

    forward = api.reserve_child_seeds(20260723, streams)
    reverse = api.reserve_child_seeds(20260723, tuple(reversed(streams)))

    assert tuple(item.stream_id for item in forward) == streams
    assert tuple(item.stream_id for item in reverse) == tuple(reversed(streams))
    assert {item.stream_id: item.seed for item in forward} == {item.stream_id: item.seed for item in reverse}
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


def test_stage_schedule_owns_seed_and_resume_contracts() -> None:
    import inspect

    schedule = import_module("xrr_fitter.fit.stage_schedule")

    assert inspect.getsourcefile(schedule.reserve_child_seeds) == inspect.getsourcefile(schedule)
    assert inspect.getsourcefile(schedule.remaining_stages) == inspect.getsourcefile(schedule)


def test_stage_graph_has_exact_a_through_e_order_and_resume_suffixes() -> None:
    api = stage_schedule

    assert api.STAGE_ORDER == ("A", "B", "C", "D", "E")
    assert api.remaining_stages(None) == api.STAGE_ORDER
    assert api.remaining_stages("B") == ("C", "D", "E")
    assert api.remaining_stages("D") == ("E",)
    assert api.remaining_stages("E") == ()
    with pytest.raises(ValueError, match="stage"):
        api.remaining_stages("uncertainty")


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
        "_screen_stage_a",
        "run_stage_a",
        "_stage_b_candidate",
        "_stage_b_launch_evidence",
        "_stage_b_geometry_indices",
        "_stage_b_geometry_group",
        "_stage_b_representatives",
        "run_stage_b",
        "_local_stage_candidate",
        "run_local_stage",
        "_stage_e_setup",
        "_stage_e_local_candidate",
        "_run_stage_e_locals",
        "_stage_e_seed",
        "run_stage_e",
    )
    boundaries = (
        (stages, function_names),
        (adaptive_review, ("compile_grid_problem", "single_grid_contexts", "review_single_population")),
        (local_budget, ("local_stage_seed", "local_stage_setups")),
        (candidates, ("materially_improves",)),
        (
            profile_rescue,
            (
                "_profile_rescue_start",
                "_profile_rescue_paths",
                "_solve_profile_rescue_path",
                "_valid_profile_rescue_result",
                "_publish_profile_rescue",
                "reconverge_profile_basin",
            ),
        ),
    )
    for module, names in boundaries:
        for name in names:
            assert get_type_hints(getattr(module, name))["problem"] is FitEvaluationContext


def test_stage_setup_uses_concrete_fit_contexts() -> None:
    setup_hints = get_type_hints(stages._StageESetup)
    assert setup_hints["coarse_problem"] is FitEvaluationContext
    assert setup_hints["full_problem"] is FitEvaluationContext
    assert get_type_hints(local_budget.LocalStageSetup)["problem"] is FitEvaluationContext
    continuation_hints = get_type_hints(stages._ordered_stage_b_continuation)
    assert continuation_hints["archive"] is candidates.StageBArchive
    assert continuation_hints["return"] == tuple[tuple[FitCandidate, ...], tuple[int, ...]]


def test_stage_population_uses_global_solver_results() -> None:
    population_hints = get_type_hints(adaptive_review.review_single_population)
    assert population_hints["solved"] is GlobalSearchResult
    assert population_hints["return"] == tuple[tuple[np.ndarray, ...], SearchEvidence]
    assert get_type_hints(adaptive_review.single_grid_contexts)["return"] == tuple[FitEvaluationContext, ...]


def test_stage_rescue_uses_local_solver_results() -> None:
    rescue_hints = get_type_hints(profile_rescue._solve_profile_rescue_path)
    valid_hints = get_type_hints(profile_rescue._valid_profile_rescue_result)
    paths_hints = get_type_hints(profile_rescue._profile_rescue_paths)
    publish_hints = get_type_hints(profile_rescue._publish_profile_rescue)
    assert rescue_hints["return"] is LocalSearchResult
    assert valid_hints["result"] is LocalSearchResult
    assert paths_hints["return"] == tuple[LocalSearchResult, ...] | None
    assert publish_hints["refined"] == tuple[LocalSearchResult, ...]


def test_stage_evaluation_annotations_preserve_domain_boundaries() -> None:
    published_hints = get_type_hints(stages._published_candidate)
    assert published_hints["return"] is FitCandidate
    assert get_type_hints(candidates.materially_improves)["candidate"] == FitCandidate | ModelEvaluation


def test_stage_callbacks_preserve_domain_boundaries() -> None:
    stage_hints = get_type_hints(stages.run_stage_e)
    assert stage_hints["progress"] == Callable[[FitProgress], None] | None
    assert stage_hints["cancelled"] == Callable[[], bool] | None
    assert stage_hints["task_runner"] == TaskRunner | None
