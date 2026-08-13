from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector, evaluate_model, values_by_name
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import EnsembleSamples, McmcConfig
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterPrior, ParameterSetting, PriorSpec


def _api():
    return import_module("xrr_fitter.analysis.mcmc")


def run_affine_invariant(*args, **kwargs):
    return _api().run_affine_invariant(*args, **kwargs)


def run_problem_mcmc(*args, **kwargs):
    return _api().run_problem_mcmc(*args, **kwargs)


def _problem(*targets: str, scale_prior: bool = False):
    initial = compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(
            FitConfig.fast(929),
            scale_prior_enabled=scale_prior,
            scale_prior_tau_decades=0.2,
        ),
    )
    selected = set(targets or ("component.0.thickness_a", "component.0.density_scale"))
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in selected,
        )
        for definition in initial.parameter_definitions
    )
    problem = compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        settings,
    )
    if scale_prior:
        problem = replace(problem, scale_prior_center=1.1, scale_prior_reason=None)
    return problem


def _candidate(problem, candidate_id: str = "mcmc-center"):
    unit = encode_physical_vector(problem, {})
    return candidate_from_evaluation(
        problem,
        unit,
        evaluate_model(problem, unit),
        candidate_id=candidate_id,
        seed_index=0,
        stop_reason="test center",
        nfev=1,
    )


def test_affine_sampler_recovers_bounded_gaussian_and_is_deterministic() -> None:
    mean = np.asarray([0.45, 0.60])
    covariance = np.asarray([[0.010, 0.004], [0.004, 0.020]])
    precision = np.linalg.inv(covariance)

    def log_probability(unit: np.ndarray) -> float:
        if np.any((unit < 0.0) | (unit > 1.0)):
            return -np.inf
        delta = unit - mean
        return float(-0.5 * delta @ precision @ delta)

    initial = np.random.default_rng(991).multivariate_normal(mean, covariance * 0.1, size=16)
    config = McmcConfig(walkers=16, burn_in=200, production_steps=500, thin=2)
    first = run_affine_invariant(log_probability, initial, config, child_seed=8128)
    second = run_affine_invariant(log_probability, initial, config, child_seed=8128)

    np.testing.assert_array_equal(first.samples_unit, second.samples_unit)
    np.testing.assert_array_equal(first.log_probability, second.log_probability)
    np.testing.assert_allclose(first.samples_unit.mean(axis=(0, 1)), mean, atol=0.05)
    assert np.all((first.samples_unit >= 0.0) & (first.samples_unit <= 1.0))


def test_mcmc_config_rejects_invalid_ensemble_shape() -> None:
    with pytest.raises(ValueError, match=r"walkers must be even and at least 2\*nfree\+2"):
        run_affine_invariant(
            lambda value: -float(value @ value),
            np.full((4, 2), 0.5),
            McmcConfig(walkers=4, burn_in=2, production_steps=4),
            child_seed=1,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        pytest.param(
            {"burn_in": -1},
            "invalid MCMC step configuration",
            id="negative-burn-in",
        ),
        pytest.param(
            {"production_steps": 1},
            "invalid MCMC step configuration",
            id="too-few-production-steps",
        ),
        pytest.param(
            {"thin": 0},
            "invalid MCMC step configuration",
            id="zero-thin",
        ),
        pytest.param(
            {"thin": 4},
            "invalid MCMC step configuration",
            id="thin-exceeds-production",
        ),
        pytest.param(
            {"stretch_scale": 1.0},
            r"stretch_scale must be in \(1,10\]",
            id="stretch-scale-lower-bound",
        ),
        pytest.param(
            {"stretch_scale": np.inf},
            r"stretch_scale must be in \(1,10\]",
            id="nonfinite-stretch-scale",
        ),
    ),
)
def test_affine_sampler_rejects_invalid_step_and_stretch_configuration(
    overrides: dict[str, float | int], message: str
) -> None:
    values: dict[str, float | int] = {
        "walkers": 6,
        "burn_in": 2,
        "production_steps": 4,
        "thin": 1,
        "stretch_scale": 2.0,
    }
    values.update(overrides)
    config = SimpleNamespace(**values)

    with pytest.raises(ValueError, match=message):
        run_affine_invariant(
            lambda value: -float(value @ value),
            np.full((6, 2), 0.5),
            config,
            child_seed=1,
        )


@pytest.mark.parametrize(
    "invalid_log_probability",
    (
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(np.nan, id="nan"),
    ),
)
def test_affine_sampler_rejects_nonfinite_initial_log_probability(
    invalid_log_probability: float,
) -> None:
    with pytest.raises(ValueError, match="initial walkers must have finite log probability"):
        run_affine_invariant(
            lambda _value: invalid_log_probability,
            np.full((6, 2), 0.5),
            McmcConfig(walkers=6, burn_in=2, production_steps=4),
            child_seed=1,
        )


def test_mcmc_split_rhat_and_ess_match_independent_numpy_calculations() -> None:
    module = _api()
    rng = np.random.default_rng(314)
    draws, walkers, dimension = 64, 6, 2
    innovations = rng.normal(size=(draws, walkers, dimension))
    samples = np.empty_like(innovations)
    samples[0] = innovations[0]
    for draw in range(1, draws):
        samples[draw] = 0.65 * samples[draw - 1] + innovations[draw]

    half = draws // 2
    chains = np.concatenate(
        (np.transpose(samples[:half], (1, 0, 2)), np.transpose(samples[-half:], (1, 0, 2))),
        axis=0,
    )
    within = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(chains, axis=1), axis=0, ddof=1)
    expected_rhat = np.sqrt((((half - 1.0) / half) * within + between / half) / within)
    expected_ess = np.empty(dimension)
    by_walker = np.transpose(samples, (1, 0, 2))
    for parameter_index in range(dimension):
        correlations = []
        for lag in range(draws):
            values = []
            for walker in range(walkers):
                series = by_walker[walker, :, parameter_index]
                centered = series - np.mean(series)
                variance = np.dot(centered, centered) / draws
                covariance = np.dot(centered[: draws - lag], centered[lag:]) / (draws - lag)
                values.append(covariance / variance)
            correlations.append(float(np.mean(values)))
        positive_sum = 0.0
        for lag in range(1, draws, 2):
            pair = correlations[lag] + (correlations[lag + 1] if lag + 1 < draws else 0.0)
            if pair <= 0.0:
                break
            positive_sum += pair
        expected_ess[parameter_index] = np.clip(
            draws * walkers / max(1.0, 1.0 + 2.0 * positive_sum),
            1.0,
            draws * walkers,
        )

    np.testing.assert_allclose(module.split_rhat(samples), expected_rhat, atol=1e-13)
    np.testing.assert_allclose(module.effective_sample_size(samples), expected_ess, atol=1e-11)


def test_affine_sampler_reports_progress_and_honors_cancellation() -> None:
    initial = np.linspace(0.45, 0.55, 12).reshape(6, 2)
    config = McmcConfig(walkers=6, burn_in=2, production_steps=4)
    events: list[tuple[int, int]] = []
    run_affine_invariant(
        lambda value: -float(np.sum((value - 0.5) ** 2)),
        initial,
        config,
        child_seed=315,
        progress=lambda completed, total: events.append((completed, total)),
    )
    assert events == [(index, 6) for index in range(1, 7)]

    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(InterruptedError, match="cancelled"):
        run_affine_invariant(
            lambda value: -float(np.sum((value - 0.5) ** 2)),
            initial,
            config,
            child_seed=315,
            cancelled=cancelled,
        )


def test_mcmc_scale_prior_is_a_separate_standard_gaussian_term() -> None:
    module = _api()
    problem = _problem("component.0.thickness_a", "instrument.scale", scale_prior=True)
    scale = 0.75 * problem.scale_prior_center
    unit = encode_physical_vector(problem, {"instrument.scale": scale})
    evaluation = evaluate_model(problem, unit)
    residual = evaluation.fit_log_residuals_decades
    weights = problem.weights[problem.data.fit_mask]
    c = problem.config.c_decades
    data_sum = np.sum(weights**2 * 2.0 * c**2 * (np.sqrt(1.0 + (residual / c) ** 2) - 1.0))
    prior_z = (np.log10(scale) - np.log10(problem.scale_prior_center)) / problem.scale_prior_tau_decades

    actual = module.problem_log_probability(problem, unit)

    np.testing.assert_allclose(actual, -data_sum / (2.0 * c**2) - 0.5 * prior_z**2)


def test_problem_mcmc_applies_mapping_and_diagnostic_warning_thresholds() -> None:
    module = _api()
    problem = _problem("component.0.thickness_a", "component.0.roughness_a")
    names = tuple(variable.name for variable in problem.variables)
    dimension = len(names)
    retained = np.linspace(0.2, 0.8, dimension)
    retained[0] = 0.0
    retained[-1] = 1.0
    samples_unit = np.broadcast_to(retained, (2, 6, dimension)).copy()
    acceptance = np.full(6, 0.5)
    acceptance[:2] = (0.10, 0.80)
    rhat = np.full(dimension, 1.09)
    rhat[0] = 1.10
    ess = np.full(dimension, 100.0)
    ess[-1] = np.nextafter(100.0, 0.0)
    ensemble = EnsembleSamples(
        samples_unit,
        np.zeros((2, 6)),
        acceptance,
        rhat,
        ess,
    )

    flat_unit, physical = module.map_problem_samples(problem, ensemble)
    expected = values_by_name(problem, retained)

    np.testing.assert_allclose(physical[0], [expected[name] for name in names])
    assert module.problem_mcmc_warnings(ensemble, names) == (
        "mcmc_acceptance_outside_0.10_0.80:walkers=0,1",
        f"mcmc_split_rhat_at_least_1.10:parameters={names[0]}",
        f"mcmc_effective_sample_size_below_100:parameters={names[-1]}",
    )
    assert module.mcmc_boundary_hits(problem, flat_unit, physical) == names


def test_problem_mcmc_honors_cancellation_during_initialization_and_mapping() -> None:
    module = _api()
    problem = _problem()
    candidate = _candidate(problem)
    config = McmcConfig(walkers=6, burn_in=2, production_steps=4, thin=2)

    with pytest.raises(InterruptedError, match="cancelled"):
        run_problem_mcmc(
            problem,
            candidate,
            config,
            child_seed=443,
            cancelled=lambda: True,
        )

    dimension = len(problem.variables)
    center = candidate.unit_vector
    ensemble = EnsembleSamples(
        np.broadcast_to(center, (2, 6, dimension)).copy(),
        np.zeros((2, 6)),
        np.full(6, 0.5),
        np.ones(dimension),
        np.full(dimension, 200.0),
    )
    with pytest.raises(InterruptedError, match="cancelled"):
        module.map_problem_samples(problem, ensemble, cancelled=lambda: True)


def test_problem_mcmc_maps_retained_samples_to_physical_space() -> None:
    problem = _problem()
    candidate = _candidate(problem)
    config = McmcConfig(walkers=6, burn_in=2, production_steps=4, thin=2)
    progress: list[tuple[int, int]] = []

    report = run_problem_mcmc(
        problem,
        candidate,
        config,
        child_seed=441,
        progress=lambda completed, total: progress.append((completed, total)),
    )
    repeated = run_problem_mcmc(problem, candidate, config, child_seed=441)

    assert report.parameter_names == tuple(variable.name for variable in problem.variables)
    assert report.candidate_id == candidate.candidate_id
    assert report.samples_physical.shape == (12, len(problem.variables))
    np.testing.assert_array_equal(report.samples_physical, repeated.samples_physical)
    assert progress[-1] == (6, 6)
    assert np.max(report.samples_physical) > 1.0


def test_physical_roughness_thickness_correlation_is_not_unconditionally_hidden() -> None:
    derivatives = import_module("xrr_fitter.analysis.derivatives")
    names = ("component.0.thickness_a", "component.0.roughness_a")
    correlation = np.asarray([[1.0, 0.99], [0.99, 1.0]])

    assert derivatives.strong_parameter_correlations(names, correlation) == ((names[0], names[1], 0.99),)


def test_fit_dataset_never_runs_expert_mcmc() -> None:
    pipeline = import_module("xrr_fitter.fit.pipeline")

    assert "run_problem_mcmc" not in pipeline.__dict__
    assert not any(name.startswith("xrr_fitter.analysis") for name in pipeline.__dict__)


def test_parameter_prior_overlay_preserves_empty_context_and_original_definitions() -> None:
    module = _api()
    problem = _problem()
    prior = ParameterPrior(
        "component.0.density_scale",
        PriorSpec("normal", (0.6, 0.05)),
    )

    overlaid = module.with_parameter_priors(problem, (prior,))

    assert module.with_parameter_priors(problem, ()) is problem
    assert all(definition.prior is None for definition in problem.parameter_definitions)
    assert overlaid is not problem
    assert (
        next(definition for definition in overlaid.parameter_definitions if definition.name == prior.name).prior
        == prior.prior
    )


def _inject_priors(problem, priors):
    definitions = tuple(
        replace(definition, prior=priors[definition.name]) if definition.name in priors else definition
        for definition in problem.parameter_definitions
    )
    return replace(problem, parameter_definitions=definitions)


def _center_ensemble(problem, candidate, config):
    dimension = len(problem.variables)
    ensemble = EnsembleSamples(
        np.broadcast_to(candidate.unit_vector, (2, config.walkers, dimension)).copy(),
        np.zeros((2, config.walkers)),
        np.full(config.walkers, 0.5),
        np.ones(dimension),
        np.full(dimension, 200.0),
    )
    return lambda *args, **kwargs: ensemble


def test_mcmc_report_flags_a_parameter_pulled_from_its_prior(monkeypatch) -> None:
    module = _api()
    problem = _inject_priors(
        _problem(),
        {
            "component.0.density_scale": PriorSpec("normal", (0.6, 0.05)),
            "component.0.thickness_a": PriorSpec("normal", (20.0, 5.0)),
        },
    )
    candidate = _candidate(problem)
    config = McmcConfig(walkers=6, burn_in=2, production_steps=4, thin=2)
    monkeypatch.setattr(module, "run_affine_invariant", _center_ensemble(problem, candidate, config))

    report = run_problem_mcmc(problem, candidate, config, child_seed=441)

    assert "component.0.density_scale" in report.prior_conflicts
    assert "component.0.thickness_a" not in report.prior_conflicts


def test_mcmc_report_no_conflict_when_posterior_agrees_with_prior(monkeypatch) -> None:
    module = _api()
    problem = _inject_priors(
        _problem(),
        {
            "component.0.density_scale": PriorSpec("normal", (1.0, 0.05)),
            "component.0.thickness_a": PriorSpec("normal", (20.0, 5.0)),
        },
    )
    candidate = _candidate(problem)
    config = McmcConfig(walkers=6, burn_in=2, production_steps=4, thin=2)
    monkeypatch.setattr(module, "run_affine_invariant", _center_ensemble(problem, candidate, config))

    report = run_problem_mcmc(problem, candidate, config, child_seed=441)

    assert report.prior_conflicts == ()


def test_mcmc_prior_conflicts_use_the_physical_sample_median_for_log_parameters(monkeypatch) -> None:
    module = _api()
    problem = _problem("component.0.thickness_a")
    definition = next(
        definition for definition in problem.parameter_definitions if definition.name == "component.0.thickness_a"
    )
    physical_center = (definition.lower + definition.upper) / 2.0
    problem = _inject_priors(
        problem,
        {definition.name: PriorSpec("normal", (physical_center, 10.0))},
    )
    candidate = _candidate(problem)
    config = McmcConfig(walkers=4, burn_in=2, production_steps=2, thin=1)
    samples_unit = np.asarray(
        (
            ((0.0,), (0.0,), (1.0,), (1.0,)),
            ((0.0,), (0.0,), (1.0,), (1.0,)),
        )
    )
    ensemble = EnsembleSamples(
        samples_unit,
        np.zeros((2, 4)),
        np.full(4, 0.5),
        np.ones(1),
        np.full(1, 200.0),
    )
    monkeypatch.setattr(module, "run_affine_invariant", lambda *args, **kwargs: ensemble)

    report = run_problem_mcmc(problem, candidate, config, child_seed=441)

    assert np.median(report.samples_physical[:, 0]) == physical_center
    assert report.prior_conflicts == ()
