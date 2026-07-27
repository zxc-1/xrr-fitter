from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting


def _api():
    return import_module("xrr_fitter.fit.global_search")


def _problem(*, seed: int = 709):
    config = replace(
        FitConfig.fast(seed),
        budget=SearchBudget(1, 1, 8, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    base = compile_fit_problem(
        prepared_data(size=44),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    target = "component.0.thickness_a"
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name == target else definition.initial,
            definition.upper if definition.name == target else definition.initial,
            locked=definition.name != target,
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


def _population() -> np.ndarray:
    return np.asarray([[0.05], [0.25], [0.45], [0.65], [0.85]], dtype=float)


def test_global_solver_replays_seed_population_and_trace_deterministically() -> None:
    api = _api()
    problem = _problem()
    population = _population()

    first = api.solve_global(
        problem,
        np.asarray([0.5]),
        population=population,
        seed=911,
        maxiter=1,
    )
    second = api.solve_global(
        problem,
        np.asarray([0.5]),
        population=population,
        seed=911,
        maxiter=1,
    )

    np.testing.assert_array_equal(first.unit_vector, second.unit_vector)
    np.testing.assert_array_equal(first.population, second.population)
    np.testing.assert_array_equal(first.population_energies, second.population_energies)
    assert first.trace == second.trace
    assert first.evaluation.objective == second.evaluation.objective
    assert not first.unit_vector.flags.writeable
    assert not first.population.flags.writeable
    assert not first.population_energies.flags.writeable


def test_global_solver_passes_explicit_population_and_fixed_scipy_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    population = _population()
    observed: dict[str, object] = {}

    def fake_differential_evolution(function, bounds, **kwargs):
        observed.update(kwargs)
        assert tuple(bounds) == ((0.0, 1.0),)
        objectives = np.asarray([function(row) for row in population])
        best = int(np.argmin(objectives))
        return SimpleNamespace(
            x=population[best],
            fun=float(objectives[best]),
            message="captured differential evolution",
            nfev=len(population),
            success=True,
            population=population,
            population_energies=objectives,
        )

    monkeypatch.setattr(api, "differential_evolution", fake_differential_evolution)

    result = api.solve_global(
        _problem(),
        np.asarray([0.5]),
        population=population,
        seed=919,
        maxiter=3,
    )

    np.testing.assert_array_equal(observed["init"], population)
    assert observed["updating"] == "deferred"
    assert observed["polish"] is False
    assert observed["workers"] == 1
    assert observed["seed"] == 919
    assert result.nfev == len(population)
    np.testing.assert_array_equal(
        result.population_energies,
        np.asarray(result.trace[: len(population)]),
    )


def test_global_solver_polls_cancellation_inside_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 4

    def exercising_solver(function, bounds, **kwargs):
        del bounds, kwargs
        for row in _population():
            function(row)
        pytest.fail("global solver returned after cancellation")

    monkeypatch.setattr(api, "differential_evolution", exercising_solver)

    with pytest.raises(api.SearchCancelled, match="cancel"):
        api.solve_global(
            _problem(),
            np.asarray([0.5]),
            population=_population(),
            seed=929,
            maxiter=10,
            cancelled=cancelled,
        )

    assert polls == 4


@pytest.mark.parametrize(
    ("start", "population"),
    [
        pytest.param(np.asarray(0.5), _population(), id="scalar-start"),
        pytest.param(np.asarray([[0.5]]), _population(), id="matrix-start"),
        pytest.param(np.asarray([np.nan]), _population(), id="nonfinite-start"),
        pytest.param(np.asarray([-0.1]), _population(), id="start-below-unit-bounds"),
        pytest.param(np.asarray([0.5]), np.zeros(5), id="population-not-matrix"),
        pytest.param(np.asarray([0.5]), np.zeros((5, 2)), id="population-wrong-width"),
        pytest.param(np.asarray([0.5]), np.zeros((4, 1)), id="population-too-small"),
        pytest.param(
            np.asarray([0.5]),
            np.asarray([[0.1], [0.3], [np.nan], [0.7], [0.9]]),
            id="population-nonfinite",
        ),
        pytest.param(
            np.asarray([0.5]),
            np.asarray([[0.1], [0.3], [0.5], [0.7], [1.1]]),
            id="population-outside-unit-bounds",
        ),
    ],
)
def test_global_solver_rejects_invalid_unit_layouts(
    start: np.ndarray,
    population: np.ndarray,
) -> None:
    api = _api()

    with pytest.raises(ValueError, match="start|population|unit|shape|finite|bounds"):
        api.solve_global(
            _problem(),
            start,
            population=population,
            seed=937,
            maxiter=0,
        )


def test_global_solver_propagates_unexpected_objective_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    sentinel = RuntimeError("unexpected objective failure")

    def fail(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(api, "evaluate_vector", fail)

    with pytest.raises(RuntimeError, match="unexpected objective failure") as captured:
        api.solve_global(
            _problem(),
            np.asarray([0.5]),
            population=_population(),
            seed=941,
            maxiter=0,
        )

    assert captured.value is sentinel


def test_de_population_preserves_the_start_and_local_search_bias() -> None:
    api = _api()
    start = np.asarray([0.2, 0.5, 0.8])

    first = api.build_de_population(start, seed=1234, population_size=17)
    second = api.build_de_population(start, seed=1234, population_size=17)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], start)
    assert first.shape == (17, 3)
    assert not first.flags.writeable
    assert np.all((first >= 0.0) & (first <= 1.0))
    gaussian_count = round(0.75 * (first.shape[0] - 1))
    gaussian = first[1 : 1 + gaussian_count]
    latin_hypercube = first[1 + gaussian_count :]
    assert gaussian.size and latin_hypercube.size
    assert np.mean(np.abs(gaussian - start)) < np.mean(
        np.abs(latin_hypercube - start)
    )


def test_stage_e_population_contains_centers_perturbations_and_lhs() -> None:
    api = _api()
    centers = (
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.7, 0.8, 0.9]),
    )

    first = api.build_stage_e_population(
        centers,
        seed=12345,
        population_size=10,
        perturbations_per_center=2,
    )
    second = api.build_stage_e_population(
        centers,
        seed=12345,
        population_size=10,
        perturbations_per_center=2,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (10, 3)
    assert not first.flags.writeable
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert all(any(np.array_equal(row, center) for row in first) for center in centers)
    assert first.shape[0] - (1 + 2) * len(centers) == 4


def test_feature_grid_preserves_a_narrow_reflectivity_peak() -> None:
    api = _api()
    data = prepared_data(size=512)
    intensity = np.geomspace(1.0, 1e-8, data.qz_a_inv.size)
    peak_index = 137
    intensity[peak_index] = 100.0 * max(
        intensity[peak_index - 1],
        intensity[peak_index + 1],
    )
    peaked = replace(
        data,
        intensity_raw=intensity,
        intensity_normalized=intensity,
    )

    first = api.feature_grid_indices(peaked, max_points=32)
    second = api.feature_grid_indices(peaked, max_points=32)

    np.testing.assert_array_equal(first, second)
    assert first.size <= 32
    assert first[0] == 0
    assert first[-1] == data.qz_a_inv.size - 1
    assert peak_index in first


def test_downsampled_search_data_keeps_every_row_field_aligned() -> None:
    api = _api()
    data = prepared_data(size=160)
    row_values = np.arange(data.qz_a_inv.size, dtype=float)
    enriched = replace(
        data,
        intensity_sigma_raw=row_values + 1.0,
        resolution_raw=row_values + 2.0,
        intensity_sigma_normalized=row_values + 3.0,
        sigma_q_a_inv=row_values + 4.0,
    )
    selected = api.feature_grid_indices(enriched, max_points=32)

    coarse = api.downsample_prepared_data(enriched, selected)

    assert coarse.source_row_groups == tuple(
        enriched.source_row_groups[index] for index in selected
    )
    for field in (
        "two_theta_deg",
        "intensity_raw",
        "intensity_sigma_raw",
        "resolution_raw",
        "qz_a_inv",
        "intensity_normalized",
        "intensity_sigma_normalized",
        "sigma_q_a_inv",
        "validation_mask",
        "fit_mask",
    ):
        np.testing.assert_array_equal(getattr(coarse, field), getattr(enriched, field)[selected])
    assert coarse.fit_ready
