"""Real inner physics uses fitted rows; published candidates keep the source axis."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.fit.global_search import downsample_prepared_data
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.data import BeamSpec, DataColumnMapping
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterFreedom, ParameterSetting


def _masked_pair(mode="robust_log", layout="plain"):
    mask = np.arange(1200) % 10 == 0
    beam = BeamSpec("monochromatic" if layout == "plain" else "mixed_kalpha")
    data = prepared_data(size=1200, intensity_raw=np.rint(np.linspace(1000, 10, 1200)), fit_mask=mask, beam=beam)
    data = replace(data, intensity_sigma_normalized=np.full(1200, 0.02))
    if layout == "point_resolution":
        data = replace(
            data,
            resolution_raw=np.linspace(0.001, 0.003, 1200),
            column_mapping=DataColumnMapping(resolution=2, resolution_kind="sigma_two_theta_deg"),
        )
    instrument = InstrumentSpec(footprint_mode="fit", resolution_domain="theta" if layout == "theta" else "q")
    config = replace(FitConfig.fast(123), noise_model=mode, scale_prior_enabled=False)
    masked = compile_fit_problem(data, simple_structure(), instrument, config)
    compact = compile_fit_problem(
        downsample_prepared_data(data, np.flatnonzero(mask)), masked.structure, instrument, config
    )
    return masked, compact


def _instrument_counter(monkeypatch):
    module = import_module("xrr_fitter.evaluation_model")
    original = module.instrument_reflectivity
    points = []

    def counted(theta, *args, **kwargs):
        points.append(theta.size)
        return original(theta, *args, **kwargs)

    monkeypatch.setattr(module, "instrument_reflectivity", counted)
    return points


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
@pytest.mark.parametrize("layout", ["plain", "point_resolution", "theta"])
def test_inner_residual_and_jacobian_only_evaluate_selected_physics(monkeypatch, mode, layout) -> None:
    masked, compact = _masked_pair(mode, layout)
    unit = evaluation.encode_physical_vector(masked, {})
    expected_residual, expected_jacobian = evaluation.least_squares_system(compact, unit)
    original = evaluation._instrument_model_jacobian.__wrapped__
    points = []

    def counted(problem, theta, *args, **kwargs):
        points.append(theta.size)
        return original(problem, theta, *args, **kwargs)

    monkeypatch.setattr(evaluation, "_instrument_model_jacobian", counted)
    residual, jacobian = evaluation.least_squares_system(masked, unit)
    np.testing.assert_allclose(residual, expected_residual, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-12, atol=1e-12)
    assert points == [120]


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
@pytest.mark.parametrize("operation", ["residual", "objective", "log_probability"])
def test_scalar_and_residual_inner_paths_skip_excluded_points(monkeypatch, mode, operation) -> None:
    masked, compact = _masked_pair(mode)
    unit = evaluation.encode_physical_vector(masked, {})
    operation = {
        "residual": evaluation.least_squares_residual,
        "objective": evaluation.problem_objective_total,
        "log_probability": evaluation.problem_log_probability,
    }[operation]
    expected = operation(compact, unit)
    points = _instrument_counter(monkeypatch)
    np.testing.assert_allclose(operation(masked, unit), expected, rtol=1e-13, atol=1e-13)
    assert points == [120]


def test_publication_still_evaluates_the_full_model_axis(monkeypatch) -> None:
    masked, compact = _masked_pair()
    unit = evaluation.encode_physical_vector(masked, {})
    expected = evaluation.evaluate_model(compact, unit)
    points = _instrument_counter(monkeypatch)
    published = evaluation.evaluate_model(masked, unit)
    assert points == [1200]
    assert published.model_normalized.shape == (1200,)
    assert np.all(np.isfinite(published.model_normalized))
    np.testing.assert_array_equal(published.model_normalized[masked.data.fit_mask], expected.model_normalized)
    assert published.objective == expected.objective


def test_fit_only_still_rejects_an_invalid_fitted_angle_but_ignores_excluded_angle() -> None:
    mask = np.ones(48, dtype=bool)
    mask[0] = False
    data = prepared_data(size=48, fit_mask=mask)
    problem = compile_fit_problem(data, simple_structure(), InstrumentSpec(footprint_mode="none"), FitConfig.fast(9))
    unit = evaluation.encode_physical_vector(problem, {"instrument.angle_offset_deg": -0.08})
    assert evaluation.evaluate_model(problem, unit, fit_only=True).valid
    invalid = evaluation.encode_physical_vector(problem, {"instrument.angle_offset_deg": -0.09})
    with pytest.raises(evaluation.EvaluationConstraintError, match="nonpositive_fitted_incident_angle"):
        evaluation.evaluate_model(problem, invalid, fit_only=True)


def test_global_search_uses_fit_rows_until_final_publication(monkeypatch) -> None:
    module = import_module("xrr_fitter.fit.global_search")
    masked, _compact = _masked_pair()
    unit = evaluation.encode_physical_vector(masked, {})
    points = _instrument_counter(monkeypatch)
    result = module.solve_global(masked, unit, population=np.tile(unit, (5, 1)), seed=17, maxiter=0)
    assert points == [120] * result.nfev + [1200]
    assert result.evaluation.model_normalized.size == 1200


def test_inner_derivative_failure_reuses_only_selected_primal_rows(monkeypatch) -> None:
    masked, _compact = _masked_pair()
    unit = evaluation.encode_physical_vector(masked, {})
    expected = evaluation.least_squares_residual(masked, unit)
    points = _instrument_counter(monkeypatch)

    def unavailable(*_args):
        raise FloatingPointError("derivative unavailable")

    monkeypatch.setattr(evaluation, "_model_residual_jacobian", unavailable)
    residual, jacobian = evaluation.least_squares_system(masked, unit)
    np.testing.assert_array_equal(residual, expected)
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))
    assert points == [120]


@pytest.mark.parametrize("operation", [evaluation.least_squares_residual, evaluation.least_squares_system])
def test_inner_angular_point_resolution_uses_the_same_selected_rows(monkeypatch, operation) -> None:
    module = import_module("xrr_fitter.evaluation_model")
    masked, _compact = _masked_pair(layout="point_resolution")
    unit = evaluation.encode_physical_vector(masked, {})
    original = module.resolution_to_sigma_q
    angles = []

    def counted(two_theta, *args):
        angles.append(two_theta.copy())
        return original(two_theta, *args)

    monkeypatch.setattr(module, "resolution_to_sigma_q", counted)
    operation(masked, unit)
    assert len(angles) == 2
    for observed in angles:
        np.testing.assert_array_equal(observed, masked.data.two_theta_deg[masked.data.fit_mask])


def test_profile_scan_only_evaluates_fitted_physics_rows(monkeypatch) -> None:
    from xrr_fitter.analysis.profiles import build_problem_profile

    masked, _compact = _masked_pair()
    settings = tuple(
        ParameterSetting(
            item.name,
            item.initial,
            item.lower,
            item.upper,
            freedom=ParameterFreedom.from_locked(item.name != "instrument.scale"),
        )
        for item in masked.parameter_definitions
    )
    masked = compile_fit_problem(masked.data, masked.structure, masked.instrument, masked.config, settings)
    unit = evaluation.encode_physical_vector(masked, {})
    points = _instrument_counter(monkeypatch)
    profile = build_problem_profile(masked, unit, "instrument.scale")
    assert profile.successful_samples > 0
    assert points and set(points) == {120}


def test_binary_profile_scan_only_evaluates_fitted_physics_rows(monkeypatch) -> None:
    from tests.unit.analysis.test_binary_profiles import _periodic_problem, build_binary_profile

    problem = _periodic_problem()
    mask = np.arange(problem.data.fit_mask.size) % 2 == 0
    problem = replace(
        problem,
        data=replace(problem.data, fit_mask=mask),
        objective_point_count=int(mask.sum()),
        weights=np.where(mask, problem.weights, 0.0),
        region_labels=np.where(mask, problem.region_labels, -1),
    )
    unit = evaluation.encode_physical_vector(problem, {})
    points = _instrument_counter(monkeypatch)
    profile = build_binary_profile(problem, unit, "component.0.period_a")
    assert profile.successful_samples > 0
    assert points and set(points) == {32}


def test_objective_derivative_inputs_only_evaluate_fitted_primal_rows(monkeypatch) -> None:
    from xrr_fitter.analysis.derivatives import objective_gradient

    masked, compact = _masked_pair()
    unit = evaluation.encode_physical_vector(masked, {})
    expected = objective_gradient(compact, unit)
    points = _instrument_counter(monkeypatch)
    np.testing.assert_allclose(objective_gradient(masked, unit), expected, rtol=1e-12, atol=1e-12)
    assert points == [120]


def test_profile_path_cost_only_evaluates_fitted_rows(monkeypatch) -> None:
    from xrr_fitter.analysis.profile_paths import _path_objective

    masked, compact = _masked_pair()
    unit = evaluation.encode_physical_vector(masked, {})
    expected = evaluation.evaluate_model(compact, unit).objective
    points = _instrument_counter(monkeypatch)
    assert _path_objective(masked)(unit) == expected
    assert points == [120]
