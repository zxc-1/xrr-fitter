from __future__ import annotations

from tests.unit.analysis.mcmc_cases import *


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


def test_mcmc_log_boundary_hits_use_unit_distance_not_linear_physical_distance() -> None:
    module = _api()
    problem = SimpleNamespace(
        variables=(SimpleNamespace(name="thickness", parameter_index=0),),
        parameter_definitions=(
            SimpleNamespace(
                transform="log",
                lower=1.0,
                upper=1e6,
            ),
        ),
        config=SimpleNamespace(
            confidence=SimpleNamespace(boundary_fraction=0.005),
        ),
    )
    flat_unit = np.array([[0.5]])
    physical = np.array([[1e3]])

    assert module.mcmc_boundary_hits(problem, flat_unit, physical) == ()

    near_lower = np.array([[0.001]])
    near_lower_physical = np.array([[1.0 * (1e6 / 1.0) ** 0.001]])
    assert module.mcmc_boundary_hits(problem, near_lower, near_lower_physical) == ("thickness",)


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


def test_problem_mcmc_records_the_compiled_gradient_topology() -> None:
    base = simple_structure()
    structure = replace(
        base,
        components=(
            GradientLayerSpec(
                "gradient",
                upper_sld_a2=10e-6 + 0.5e-6j,
                lower_sld_a2=50e-6 + 2.0e-6j,
                thickness_a=20.0,
                roughness_a=0.0,
                microslab_max_a=10.0,
            ),
        ),
        backing_roughness_a=0.0,
    )
    initial = compile_fit_problem(
        prepared_data(size=40),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(930), scale_prior_enabled=False),
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
    problem = compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        settings,
    )
    definition = next(item for item in problem.parameter_definitions if item.name == "component.0.thickness_a")

    report = run_problem_mcmc(
        problem,
        _candidate(problem),
        McmcConfig(walkers=4, burn_in=0, production_steps=5),
        child_seed=442,
    )

    assert report.gradient_slab_counts == (("component.0", int(np.ceil(definition.upper / 10.0))),)
