from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import encode_physical_vector, evaluate_model
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting


def _api():
    return import_module("xrr_fitter.analysis.bootstrap")


def bootstrap_local(*args, **kwargs):
    return _api().bootstrap_local(*args, **kwargs)


def bootstrap_problem_local(*args, **kwargs):
    return _api().bootstrap_problem_local(*args, **kwargs)


def residual_block_length(*args, **kwargs):
    return _api().residual_block_length(*args, **kwargs)


def _problem(*, explicit_errors: bool = False):
    data = prepared_data(size=40)
    if explicit_errors:
        sigma = np.full(data.qz_a_inv.size, 1e-3)
        data = replace(
            data,
            intensity_sigma_raw=sigma,
            intensity_sigma_normalized=sigma,
        )
    initial = compile_fit_problem(
        data,
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(919), scale_prior_enabled=False),
    )
    targets = {"component.0.thickness_a", "component.0.density_scale"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in targets,
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


def _candidate(problem):
    unit = encode_physical_vector(problem, {})
    return candidate_from_evaluation(
        problem,
        unit,
        evaluate_model(problem, unit),
        candidate_id="bootstrap-center",
        seed_index=0,
        stop_reason="test center",
        nfev=1,
    )


def test_bootstrap_failure_rate_over_twenty_percent_suppresses_intervals() -> None:
    def fit_sample(rng: np.random.Generator, sample_index: int):
        if sample_index in {1, 4, 8}:
            return None
        return np.asarray([0.4, 0.6]) + rng.normal(0.0, 0.01, size=2)

    result = bootstrap_local(
        fit_sample,
        ("x", "y"),
        sample_count=10,
        child_seed=123,
    )

    assert result.failure_rate == 0.3
    assert result.intervals == ()
    assert result.samples.shape == (7, 2)


def test_bootstrap_failure_rate_gate_is_strictly_greater_than_twenty_percent() -> None:
    def fit_sample(rng: np.random.Generator, sample_index: int):
        if sample_index in {1, 4}:
            return None
        return np.asarray([0.4]) + rng.normal(0.0, 0.01, size=1)

    result = bootstrap_local(
        fit_sample,
        ("x",),
        sample_count=10,
        child_seed=124,
    )

    assert result.failure_rate == 0.2
    assert len(result.intervals) == 1


def test_bootstrap_reports_every_completed_sample_including_failures() -> None:
    events: list[tuple[int, int]] = []

    result = bootstrap_local(
        lambda _rng, sample_index: None if sample_index == 1 else np.asarray([0.5]),
        ("x",),
        sample_count=3,
        child_seed=125,
        progress=lambda completed, total: events.append((completed, total)),
    )

    assert events == [(1, 3), (2, 3), (3, 3)]
    assert result.failure_rate == pytest.approx(1.0 / 3.0)


def test_residual_bootstrap_block_length_uses_first_zero_crossing_with_clamps() -> None:
    alternating = np.tile(np.asarray([1.0, -1.0]), 100)
    persistent = np.linspace(-1.0, 1.0, 200)

    assert residual_block_length(alternating) == 3
    assert residual_block_length(persistent) == 25


def test_problem_bootstrap_is_deterministic_and_reports_physical_parameters() -> None:
    problem = _problem()
    candidate = _candidate(problem)

    first = bootstrap_problem_local(
        problem,
        candidate,
        sample_count=4,
        child_seed=9981,
    )
    second = bootstrap_problem_local(
        problem,
        candidate,
        sample_count=4,
        child_seed=9981,
    )

    assert first.parameter_names == tuple(variable.name for variable in problem.variables)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.failure_rate == 0.0
    assert len(first.intervals) == len(problem.variables)
    assert np.max(first.samples[:, 0]) > 1.0
    assert first.candidate_id == candidate.candidate_id
    assert len(first.provenance_sha256) == 64
    assert first.provenance_sha256 == second.provenance_sha256


def test_problem_bootstrap_with_explicit_errors_uses_parametric_draws(
    monkeypatch,
) -> None:
    module = _api()
    problem = _problem(explicit_errors=True)
    candidate = _candidate(problem)

    def unexpected_block_draw(*_args, **_kwargs):
        raise AssertionError("moving-block draw used despite explicit errors")

    monkeypatch.setattr(module, "_moving_block_draw", unexpected_block_draw)
    result = module.bootstrap_problem_local(
        problem,
        candidate,
        sample_count=3,
        child_seed=309,
    )

    assert result.failure_rate == 0.0
    assert result.samples.shape == (3, len(problem.variables))
