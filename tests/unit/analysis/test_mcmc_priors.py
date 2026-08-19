from __future__ import annotations

from tests.unit.analysis.mcmc_cases import *


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


def test_mcmc_prior_conflict_median_avoids_finite_extreme_sum_overflow() -> None:
    maximum = np.finfo(float).max
    problem = SimpleNamespace(
        variables=(SimpleNamespace(name="extreme", parameter_index=0),),
        parameter_definitions=(
            SimpleNamespace(
                transform="linear",
                prior=PriorSpec("normal", (maximum, 1.0)),
            ),
        ),
        config=SimpleNamespace(
            confidence=SimpleNamespace(prior_conflict_sigmas=3.0),
        ),
    )
    physical = np.full((2, 1), maximum)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conflicts = _api().mcmc_prior_conflicts(
            problem,
            np.ones_like(physical),
            physical,
        )

    assert conflicts == ()
    assert not any(item.category is RuntimeWarning for item in caught)
