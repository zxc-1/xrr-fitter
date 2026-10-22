"""Elite restarts stay local in residual space despite wide parameter bounds."""

from dataclasses import replace

import numpy as np
import pytest
from tests.unit.fit.test_stage_search import _candidate, _problem

from xrr_fitter.evaluation import encode_physical_vector, least_squares_system
from xrr_fitter.fit import stages
from xrr_fitter.fit.local_search import solve_local
from xrr_fitter.fit.perturbations import bounded_perturbations
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.fit.stage_schedule import reserve_child_seeds
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
@pytest.mark.parametrize("with_prior", (False, True))
def test_elite_uses_the_solver_residual_metric_with_one_optional_prior(mode, with_prior) -> None:
    from tests.unit.fit.test_evaluation_workload import _masked_pair

    problem, _compact = _masked_pair(mode)
    problem = replace(problem, scale_prior_center=1.0 if with_prior else None)
    center = encode_physical_vector(problem, {})
    setup = stages._stage_e_setup(problem, (_candidate(problem, "D-0", center),))
    residual, jacobian = least_squares_system(setup.full_problem, center)
    assert residual.size == np.count_nonzero(problem.data.fit_mask) + int(with_prior)
    seed = int(
        np.random.SeedSequence([202, ord("I"), len(setup.full_incumbents)]).generate_state(1, dtype=np.uint64)[0]
    )
    raw = bounded_perturbations(center, 1, seed=seed, sigma=0.002)[0]
    rms = np.linalg.norm(jacobian, axis=0) / np.sqrt(residual.size)
    sensitive = rms > 1.0
    assert sensitive.any()

    start = stages._incumbent_starts(setup, 1, 202, center)[-1]

    np.testing.assert_array_equal(start[~sensitive], raw[~sensitive])
    np.testing.assert_allclose(
        start[sensitive], center[sensitive] + (raw[sensitive] - center[sensitive]) / rms[sensitive], rtol=1e-14, atol=0
    )
    assert not np.array_equal(start, center)


def test_elite_derivative_failure_keeps_zero_data_columns_and_exactly_one_prior(monkeypatch) -> None:
    import xrr_fitter.evaluation as evaluation

    problem = replace(_problem(seed=769), scale_prior_center=1.0, scale_prior_tau_decades=0.01)
    center = encode_physical_vector(problem, {})
    setup = stages._stage_e_setup(problem, (_candidate(problem, "D-0", center),))

    def derivative_failure(*_args):
        raise FloatingPointError("derivative unavailable")

    monkeypatch.setattr(evaluation, "_model_residual_jacobian", derivative_failure)
    residual, jacobian = least_squares_system(setup.full_problem, center)
    rows = int(np.count_nonzero(problem.data.fit_mask))
    assert jacobian.shape == (rows + 1, center.size)
    np.testing.assert_array_equal(jacobian[:rows], np.zeros((rows, center.size)))
    np.testing.assert_array_equal(residual, evaluation.least_squares_residual(setup.full_problem, center))
    scale_index = next(
        i for i, variable in enumerate(setup.full_problem.variables) if variable.name == "instrument.scale"
    )
    rms = abs(jacobian[-1, scale_index]) / np.sqrt(rows + 1)
    assert rms > 1.0
    seed = int(
        np.random.SeedSequence([202, ord("I"), len(setup.full_incumbents)]).generate_state(1, dtype=np.uint64)[0]
    )
    raw = bounded_perturbations(center, 1, seed=seed, sigma=0.002)[0]
    expected = raw.copy()
    expected[scale_index] = center[scale_index] + (raw[scale_index] - center[scale_index]) / rms

    start = stages._incumbent_starts(setup, 1, 202, center)[-1]

    np.testing.assert_allclose(start, expected, rtol=1e-14, atol=0)
    unchanged = np.arange(center.size) != scale_index
    np.testing.assert_array_equal(start[unchanged], raw[unchanged])


def test_elite_displacement_contracts_only_sensitive_coordinates_and_preserves_streams(monkeypatch) -> None:
    problem = _problem(seed=769)
    center = np.full(len(problem.variables), 0.5)
    setup = stages._stage_e_setup(problem, (_candidate(problem, "D-0", center),))
    jacobian = np.tile(np.array([10.0, 2.0, 1.0, 0.0, 0.5, 0.0, 1.0, 1.0]), (4, 1))
    monkeypatch.setattr(stages, "least_squares_system", lambda *_args: (np.zeros(4), jacobian), raising=False)
    original = stages._incumbent_starts(setup, 1, 202, None)

    actual = stages._incumbent_starts(setup, 1, 202, center)

    for before, after in zip(original, actual[:-1], strict=True):
        np.testing.assert_array_equal(after, before)
    seed = int(
        np.random.SeedSequence([202, ord("I"), len(setup.full_incumbents)]).generate_state(1, dtype=np.uint64)[0]
    )
    raw = bounded_perturbations(center, 1, seed=seed, sigma=0.002)[0]
    expected = raw.copy()
    expected[:2] = center[:2] + (raw[:2] - center[:2]) / [10.0, 2.0]
    np.testing.assert_array_equal(actual[-1], expected)
    assert not np.array_equal(actual[-1], center)
    assert not actual[-1].flags.writeable


def _periodic_truth_problem():
    from tests.support.synthetic_recovery import build_corpus
    from tests.support.synthetic_recovery_runtime import _generate_case_intensity, _prepared_case_data

    case = next(case for case in build_corpus() if case.case_id == "periodic-13018-n95")
    with pytest.warns(UserWarning, match="Gauss-Hermite"):
        data = _prepared_case_data(case, _generate_case_intensity(case))
    options = dict(case.generation_options)
    instrument = InstrumentSpec(footprint_mode="none")
    problem = compile_fit_problem(
        data, case.generating_structure, instrument, replace(FitConfig.fast(case.fit_seed), scale_prior_enabled=False)
    )
    center = encode_physical_vector(problem, {f"instrument.{name}": value for name, value in options.items()})
    return problem, center, case.fit_seed


def test_periodic_elite_restart_reconverges_within_the_original_local_budget() -> None:
    problem, center, master_seed = _periodic_truth_problem()
    parent = _candidate(problem, "D-0", center)
    setup = stages._stage_e_setup(problem, (parent,))
    seed = reserve_child_seeds(master_seed, ("E-0", "E-1", "E-2", "E-3"))[1].seed
    start = stages._incumbent_starts(setup, 1, seed, center)[-1]
    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * (len(problem.variables) + 1),
    )
    assert not np.array_equal(start, center)

    result = solve_local(setup.full_problem, start, max_nfev=maximum)

    assert result.evaluation.valid
    assert result.nfev <= maximum
    assert result.evaluation.objective < 1e-12


@pytest.mark.parametrize("jacobian_kind", ("zero", "repeated", "large", "empty"))
def test_sensitivity_scaling_handles_degenerate_and_extreme_columns(jacobian_kind) -> None:
    from xrr_fitter.fit import perturbations

    center = np.array([0.0, 0.5, 1.0])
    proposal = np.array([0.01, 0.49, 0.99])
    jacobian = np.tile([10.0, 0.0, 2.0], (4, 1))
    if jacobian_kind == "zero":
        jacobian[:] = 0.0
    elif jacobian_kind == "large":
        jacobian[:] = np.finfo(float).max
    elif jacobian_kind == "empty":
        center, proposal, jacobian = center[:0], proposal[:0], jacobian[:, :0]

    actual = perturbations.sensitivity_scaled_start(center, proposal, jacobian)

    assert np.all(np.isfinite(actual))
    assert np.all((actual >= np.minimum(center, proposal)) & (actual <= np.maximum(center, proposal)))
    assert not actual.flags.writeable
    if jacobian_kind == "zero":
        np.testing.assert_array_equal(actual, proposal)
    elif jacobian_kind == "repeated":
        repeat = perturbations.sensitivity_scaled_start(center, proposal, np.tile(jacobian, (3, 1)))
        np.testing.assert_array_equal(actual, repeat)
        np.testing.assert_array_equal(actual, [0.001, 0.49, 0.995])


@pytest.mark.parametrize(
    "jacobian", (np.zeros((0, 2)), np.zeros((3, 1)), np.zeros(2), [[np.nan, 0.0]], [[np.inf, 0.0]])
)
def test_sensitivity_scaling_rejects_invalid_derivative_axes_and_nonfinite_values(jacobian) -> None:
    from xrr_fitter.fit import perturbations

    with pytest.raises(ValueError, match="Jacobian"):
        perturbations.sensitivity_scaled_start(np.array([0.5, 0.5]), np.array([0.51, 0.49]), jacobian)


def test_elite_rounding_to_exact_center_does_not_add_a_false_independent_path(monkeypatch) -> None:
    problem = _problem(seed=769)
    center = np.full(len(problem.variables), 0.5)
    setup = stages._stage_e_setup(problem, (_candidate(problem, "D-0", center),))
    jacobian = np.full((4, center.size), np.finfo(float).max)
    monkeypatch.setattr(stages, "least_squares_system", lambda *_args: (np.zeros(4), jacobian), raising=False)

    without_elite = stages._incumbent_starts(setup, 1, 202, None)
    with_elite = stages._incumbent_starts(setup, 1, 202, center)

    assert len(with_elite) == len(without_elite)
    for before, after in zip(without_elite, with_elite, strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("elite_state", ("collapsed", "cached"))
def test_stage_e_charges_the_elite_system_evaluation_without_enlarging_optimizer_budgets(
    monkeypatch, elite_state
) -> None:
    from types import SimpleNamespace

    from tests.unit.fit.test_stage_search import _unchanged_local_solution

    problem = _problem(seed=769)
    center = np.full(len(problem.variables), 0.5)
    setup = stages._stage_e_setup(problem, (_candidate(problem, "D-0", center),))
    calls = []
    optimized = []
    jacobian = np.full((4, center.size), np.finfo(float).max if elite_state == "collapsed" else 10.0)

    def system(context, unit):
        calls.append(unit.copy())
        return np.zeros(4), jacobian

    def global_solution(context, start, *, population, **_kwargs):
        return SimpleNamespace(**vars(_unchanged_local_solution(context, start)), population=population, trace=())

    def local_solution(context, start, *, max_nfev, **_kwargs):
        optimized.append((start.tobytes(), max_nfev))
        return _unchanged_local_solution(context, start)

    monkeypatch.setattr(stages, "least_squares_system", system, raising=False)
    monkeypatch.setattr(stages, "solve_global", global_solution)
    monkeypatch.setattr(stages, "solve_local", local_solution)
    cached_keys = set()
    if elite_state == "cached":
        start = stages._incumbent_starts(setup, 1, 202, center)[-1]
        key = start.tobytes()
        setup.local_candidates[key] = _candidate(setup.full_problem, "E-0-local-0", start)
        cached_keys.add(key)
        calls.clear()

    candidate = stages._stage_e_seed(problem, setup, 1, 202, center, None, None)

    assert len(calls) == 1
    assert optimized and not cached_keys.intersection(key for key, _budget in optimized)
    maximum = max(
        problem.config.budget.local_min_nfev, problem.config.budget.local_nfev_per_parameter * (center.size + 1)
    )
    assert all(budget == maximum for _key, budget in optimized)
    if elite_state == "cached":
        assert any(record.stop_reason == "reused_exact_start:E-0-local-0" for record in candidate.search_evidence)
    # Actual optimizer counts and each local's publication evaluation precede
    # the system evaluation; neither an omitted nor cached elite adds a solve.
    expected = 1 + sum(value.nfev for key, value in setup.local_candidates.items() if key not in cached_keys) + 1
    assert candidate.nfev == expected
