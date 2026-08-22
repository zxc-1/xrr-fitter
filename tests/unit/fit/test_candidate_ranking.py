from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
from tests.support.model_cases import fit_candidate, prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec, PhysicsDiagnostic
from xrr_fitter.model.slab_stack import SlabStack


def _api():
    return import_module("xrr_fitter.fit.candidates")


def _candidate(
    candidate_id: str,
    objective: float,
    *,
    ranking_objective: float | None = None,
    unit: tuple[float, ...] = (0.5,),
):
    return replace(
        fit_candidate(candidate_id, objective),
        unit_vector=np.asarray(unit, dtype=float),
        ranking_objective=ranking_objective,
    )


def test_candidate_ranking_uses_effective_objective_and_preserves_exact_ties() -> None:
    api = _api()
    candidates = (
        _candidate("first-tie", 3.0, ranking_objective=1.0),
        _candidate("second-tie", 2.0, ranking_objective=1.0),
        _candidate("local-objective", 1.5),
    )

    assert api.rank_candidate_indices(candidates) == (0, 1, 2)
    assert api.best_candidate_index(candidates) == 0


def test_candidate_ranking_excludes_invalid_and_eliminated_evidence() -> None:
    api = _api()
    invalid = replace(_candidate("invalid", 0.1), valid=False)
    eliminated = replace(_candidate("eliminated", 0.2), stop_reason="early_eliminated")
    selectable = _candidate("selectable", 4.0)
    candidates = (invalid, eliminated, selectable)

    assert api.rank_candidate_indices(candidates) == (2,)
    assert api.best_candidate_index(candidates) == 2
    assert tuple(candidate.candidate_id for candidate in candidates) == (
        "invalid",
        "eliminated",
        "selectable",
    )


def test_final_stage_eligibility_cannot_resurrect_an_earlier_candidate() -> None:
    api = _api()
    candidates = (
        _candidate("B-0", 0.01),
        _candidate("E-0", 2.0),
        _candidate("E-1", 3.0),
    )
    eligible = ("E-0", "E-1")

    assert api.rank_candidate_indices(candidates, eligible_ids=eligible) == (1, 2)
    assert api.best_candidate_index(candidates, eligible_ids=eligible) == 1


def test_empty_or_entirely_unselectable_scope_has_no_winner() -> None:
    api = _api()
    invalid = replace(_candidate("invalid", 1.0), valid=False)

    assert api.rank_candidate_indices(()) == ()
    assert api.best_candidate_index(()) is None
    assert api.rank_candidate_indices((invalid,)) == ()
    assert api.best_candidate_index((invalid,)) is None


def test_candidate_clustering_is_stable_and_retains_member_lineage() -> None:
    api = _api()
    candidates = (
        _candidate("B-0", 2.0, unit=(0.10, 0.20)),
        _candidate("B-1", 1.0, unit=(0.11, 0.19)),
        _candidate("B-2", 3.0, unit=(0.80, 0.90)),
        _candidate("B-3", 4.0, unit=(0.82, 0.88)),
    )

    clusters = api.cluster_candidate_indices(candidates, distance=0.05)

    assert clusters == ((0, 1), (2, 3))
    assert tuple(min(cluster, key=lambda index: (candidates[index].objective, index)) for cluster in clusters) == (1, 2)


def test_candidate_clustering_and_ranking_do_not_mutate_published_arrays() -> None:
    api = _api()
    candidates = (
        _candidate("first", 2.0, unit=(0.2, 0.3)),
        _candidate("second", 1.0, unit=(0.7, 0.8)),
    )
    before = tuple(candidate.unit_vector.copy() for candidate in candidates)

    assert api.cluster_candidate_indices(candidates, distance=0.1) == ((0,), (1,))
    assert api.rank_candidate_indices(candidates) == (1, 0)

    for candidate, expected in zip(candidates, before, strict=True):
        np.testing.assert_array_equal(candidate.unit_vector, expected)
        assert not candidate.unit_vector.flags.writeable


def test_stage_b_archive_uses_full_objective_and_reclaims_local_budget() -> None:
    api = _api()
    candidates = tuple(
        _candidate(f"B-{index}", objective) for index, objective in enumerate((1.0, 5.0, 10.0, 10.000001, 20.0))
    )

    archive = api.archive_stage_b_candidates(
        candidates,
        threshold_ratio=10.0,
        base_perturbations=2,
    )

    assert tuple(value.objective for value in archive.active) == (1.0, 5.0, 10.0)
    assert archive.perturbation_counts == (4, 4, 4)
    assert tuple(value.objective for value in archive.archived) == (10.000001, 20.0)
    assert all(value.stop_reason == "early_eliminated" for value in archive.archived)
    assert all(value.seed_index == -1 for value in archive.archived)
    assert sum(count + 1 for count in archive.perturbation_counts) == len(candidates) * 3


def test_physics_diagnostics_survive_candidate_conversion() -> None:
    api = _api()
    problem = compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(857), scale_prior_enabled=False),
    )
    unit = encode_physical_vector(problem, {})
    diagnostic = PhysicsDiagnostic(
        "ideal_reflectivity_above_one",
        "ideal reflectivity exceeded its physical ceiling",
        (2, 7),
    )
    evaluation = replace(evaluate_vector(problem, unit), diagnostics=(diagnostic,))

    candidate = api.candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        "B-0",
        0,
        "converged",
        3,
    )

    assert candidate.diagnostics == (diagnostic,)
    assert candidate.diagnostics[0] is diagnostic


def test_candidate_conversion_bounds_long_stack_reporting_profile() -> None:
    api = _api()
    problem = compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(859), scale_prior_enabled=False),
    )
    unit = encode_physical_vector(problem, {})
    total_depth_a = 600_000.0
    evaluation = replace(
        evaluate_vector(problem, unit),
        expanded_stack=SlabStack(
            [0.0, total_depth_a, 0.0],
            [0.0j, 2.0e-5 + 1.0e-7j, 4.0e-6 + 0.0j],
            [2.0, 3.0],
        ),
    )

    candidate = api.candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        "E-0",
        0,
        "converged",
        3,
    )

    assert candidate.valid
    assert candidate.sld_depth_a.size <= api.MAX_CANDIDATE_SLD_PROFILE_POINTS
    assert candidate.sld_depth_a[0] <= -10.0
    assert candidate.sld_depth_a[-1] >= total_depth_a + 10.0
    assert np.all(np.isfinite(candidate.sld_profile_a2))
