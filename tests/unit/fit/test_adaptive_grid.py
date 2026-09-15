"""Progressive grids, full-cost review, and deterministic search decisions."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure
from tests.unit.fit.test_global_solver import _problem

from xrr_fitter.fit.feature_grid import feature_grid_indices
from xrr_fitter.fit.global_search import solve_global
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.fit.screening import fringe_extrema_qz
from xrr_fitter.fit.stages import compile_coarse_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec


def _adaptive():
    assert find_spec("xrr_fitter.fit.adaptive_grid") is not None, "progressive grid decisions must be implemented"
    return import_module("xrr_fitter.fit.adaptive_grid")


@pytest.mark.parametrize(
    ("count", "expected"),
    [(60, (60,)), (128, (128,)), (200, (128, 200)), (512, (128, 256, 512)), (1200, (128, 256, 512, 1200))],
)
def test_grid_levels_never_create_observations(count, expected) -> None:
    assert _adaptive().grid_levels(count) == expected


def _fringed_data():
    data = prepared_data(size=1200, two_theta_deg=np.linspace(0.5, 3.2, 1200))
    qz = data.qz_a_inv
    phase = (qz - qz[0]) / np.ptp(qz)
    curve = (1.0 + 0.6 * np.cos(2 * np.pi * 40 * phase)) / (qz / qz[0]) ** 4
    return replace(data, intensity_raw=curve, intensity_normalized=curve, normalization=1.0)


def test_initial_grid_resolves_detected_fringes_and_retains_real_rows() -> None:
    data = _fringed_data()
    extrema = fringe_extrema_qz(data.qz_a_inv, data.intensity_normalized, data.r_floor)
    fringes = extrema.size // 2
    assert fringes >= 30
    selected_count = _adaptive().initial_grid_points(data)
    assert selected_count in (128, 256, 512, 1200)
    assert selected_count / fringes >= 8 or selected_count == data.fit_mask.sum()
    indices = feature_grid_indices(data, selected_count)
    assert len(indices) == selected_count
    assert indices[0] == 0 and indices[-1] == 1199


def test_stage_coarse_compilation_uses_fringe_density_without_changing_frozen_evidence() -> None:
    full = compile_fit_problem(_fringed_data(), simple_structure(), InstrumentSpec(), FitConfig.fast(17))
    coarse = compile_coarse_problem(full)
    assert coarse.data.fit_mask.sum() >= 512
    assert coarse.objective_point_count == full.objective_point_count
    assert coarse.scale_prior_center == full.scale_prior_center
    assert coarse.scale_prior_tau_decades == full.scale_prior_tau_decades
    assert np.sum(coarse.sampling_multipliers) == pytest.approx(np.sum(full.sampling_multipliers))


@pytest.mark.parametrize(
    ("coarse", "full", "expected"),
    [
        ([1.05], [1.0], False),
        ([1.050001], [1.0], True),
        ([0.5e-12], [0.0], True),
        ([1.0], [np.inf], True),
        ([np.inf], [np.inf], False),
        ([1.0, 1.01], [1.03, 1.0], True),
        ([1.0, 1.01], [1.019, 1.0], False),
    ],
)
def test_promotion_uses_relative_error_and_material_full_cost_rank_reversal(coarse, full, expected) -> None:
    assert _adaptive().should_promote(np.asarray(coarse), np.asarray(full)) is expected


def test_review_selection_keeps_cost_leaders_and_structurally_distinct_candidates() -> None:
    units = np.r_[np.linspace(0.1, 0.101, 8), [0.3, 0.5, 0.7, 0.9]][:, None]
    costs = np.arange(len(units), dtype=float)
    identifiers = tuple(f"candidate-{index:02d}" for index in range(len(units)))
    selected = _adaptive().review_candidate_indices(units, costs, identifiers)
    assert len(selected) == 8
    assert set(range(4)) <= set(selected)
    assert 11 in selected
    assert selected == _adaptive().review_candidate_indices(units.copy(), costs.copy(), identifiers)


def test_review_selection_checks_all_candidates_when_fewer_than_eight() -> None:
    selected = _adaptive().review_candidate_indices(
        np.array([[0.2], [0.5], [0.9]]), np.array([1.0, 2.0, 3.0]), ("a", "b", "c")
    )
    assert set(selected) == {0, 1, 2}


def test_budget_round_prioritizes_full_cost_then_id_and_never_repeats_a_lineage() -> None:
    api = _adaptive()
    costs = {"c": 2.0, "b": 1.0, "a": 1.0}
    assert api.next_budget_lineage(costs, ()) == "a"
    assert api.next_budget_lineage(costs, ("a",)) == "b"
    assert api.next_budget_lineage(costs, ("a", "b")) == "c"
    assert api.next_budget_lineage(costs, ("a", "b", "c")) is None


def test_stagnation_requires_three_generations_without_material_gain_or_new_geometry() -> None:
    tracker = _adaptive().GenerationStagnation()
    tracker.observe(np.array([0.5]), 1.0)
    tracker.start_generations()
    for _ in range(2):
        tracker.observe(np.array([0.5]), 1.0)
        assert tracker.finish_generation() is False
    tracker.observe(np.array([0.55]), 1.0)
    assert tracker.finish_generation() is False
    for _ in range(2):
        tracker.observe(np.array([0.55]), 1.0)
        assert tracker.finish_generation() is False
    tracker.observe(np.array([0.55]), 1.0)
    assert tracker.finish_generation() is True


def test_material_gain_resets_stagnation_and_tracker_owns_candidate_arrays() -> None:
    tracker = _adaptive().GenerationStagnation()
    value = np.array([0.5])
    tracker.observe(value, 1.0)
    tracker.start_generations()
    value[:] = 0.9
    for _ in range(2):
        tracker.observe(np.array([0.5]), 1.0)
        assert tracker.finish_generation() is False
    tracker.observe(np.array([0.5]), 0.999)
    assert tracker.finish_generation() is False
    for _ in range(2):
        assert tracker.finish_generation() is False
    assert tracker.finish_generation() is True


def test_real_de_cannot_stop_after_one_flat_generation() -> None:
    problem = _problem()
    solved = solve_global(problem, np.array([0.5]), population=np.full((5, 1), 0.5), seed=17, maxiter=8)
    assert solved.nfev == 20
    assert solved.stop_reason == "three_generation_stagnation"


def test_real_de_short_budget_is_respected_even_before_three_generations() -> None:
    problem = _problem()
    solved = solve_global(problem, np.array([0.5]), population=np.full((5, 1), 0.5), seed=17, maxiter=1)
    assert solved.nfev == 10
    assert solved.stop_reason != "three_generation_stagnation"


def test_grid_review_promotes_before_selection_and_reuses_complete_objectives() -> None:
    module_name = "xrr_fitter.fit.adaptive_review"
    assert find_spec(module_name) is not None, "full-grid review must precede candidate elimination"
    module = import_module(module_name)
    units = np.linspace(0.0, 1.0, 12)[:, None]
    identifiers = tuple(f"p-{index:02d}" for index in range(12))
    full = np.r_[1.03, 1.0, np.arange(2.0, 12.0)]
    coarse = full.copy()
    coarse[:2] = (1.0, 1.01)
    calls = []

    def evaluate(level, unit):
        index = int(np.argmin(np.abs(units[:, 0] - unit[0])))
        calls.append((level, index))
        return (coarse if level < 2 else full)[index]

    reviewed = module.review_on_grids(units, coarse, identifiers, ((128,), (256,), (512,), (1200,)), evaluate)
    assert reviewed.indices[0] == 1
    assert tuple(item.grid_points for item in reviewed.reviews) == ((128,), (256,), (512,))
    assert tuple(item.promoted for item in reviewed.reviews) == (True, True, False)
    assert len(reviewed.reviews[0].candidate_ids) >= 8
    assert sum(item.full_evaluations for item in reviewed.reviews) == sum(level == 3 for level, _ in calls) == 8
    assert sum(item.grid_evaluations for item in reviewed.reviews) == sum(level != 3 for level, _ in calls) == 24
    np.testing.assert_array_equal(reviewed.objectives, full[list(reviewed.indices)])


def test_real_stage_a_records_complete_objective_review_before_choosing_launches() -> None:
    from tests.unit.fit.test_stage_search import _problem as stage_problem

    from xrr_fitter.fit.stages import run_stage_a

    problem = stage_problem(size=160)
    _starts, summary, _warnings = run_stage_a(problem, None, progress=None, cancelled=None)
    assert summary.search_evidence, "Stage A must retain its full-grid review decisions"
    evidence = summary.search_evidence[0]
    assert evidence.seed == problem.config.master_seed
    assert evidence.full_review_evaluations >= 8
    assert all(len(review.candidate_ids) >= 8 for review in evidence.reviews)


def test_real_single_stage_b_and_e_keep_full_review_and_optimizer_budget_evidence() -> None:
    from tests.unit.fit.test_stage_search import _problem as stage_problem

    from xrr_fitter.fit.pipeline import FitSearchRequest, run_fit_search

    problem = stage_problem(size=160)
    result = run_fit_search(FitSearchRequest(None, problem))
    for stage in ("B", "E"):
        summary = next(item for item in result.stage_summaries if item.stage == stage)
        evidence = [item for item in summary.search_evidence if item.reviews]
        assert evidence, f"stage {stage} must publish actual full-grid review"
        assert all(item.full_review_evaluations >= 8 for item in evidence)
        assert all(item.optimizer_nfev <= item.optimizer_nfev_limit for item in evidence)
        assert all(item.budget_allocations for item in evidence)


def test_real_joint_stage_e_counts_global_work_and_persists_grid_reviews() -> None:
    from tests.unit.fit.test_joint_pipeline import _staged_joint_problem

    from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit

    problem = _staged_joint_problem()
    results = run_joint_fit(JointFitRequest(problem))
    summary = next(item for item in results[0].stage_summaries if item.stage == "E")
    for candidate in results[0].candidates:
        if candidate.candidate_id.startswith("E-"):
            assert candidate.nfev >= 64 + 1, "joint Stage E cannot discard its DE nfev"
            assert candidate.search_evidence
    assert summary.search_evidence == results[1].stage_summaries[-1].search_evidence
    assert sum(item.full_review_evaluations for item in summary.search_evidence) >= 8 * 4
