from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import encode_physical_vector, evaluate_model
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting


def _api():
    return import_module("xrr_fitter.analysis.derivatives")


def correlation_from_covariance(*args, **kwargs):
    return _api().correlation_from_covariance(*args, **kwargs)


def objective_gradient(*args, **kwargs):
    return _api().objective_gradient(*args, **kwargs)


def objective_information(*args, **kwargs):
    return _api().objective_information(*args, **kwargs)


def physical_parameter_jacobian(*args, **kwargs):
    return _api().physical_parameter_jacobian(*args, **kwargs)


def strong_parameter_correlations(*args, **kwargs):
    return _api().strong_parameter_correlations(*args, **kwargs)


def thickness_density_pairs(*args, **kwargs):
    return _api().thickness_density_pairs(*args, **kwargs)


def _problem(*targets: str, scale_prior: bool = False):
    first = compile_fit_problem(
        prepared_data(size=56),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(
            FitConfig.fast(911),
            scale_prior_enabled=scale_prior,
            scale_prior_tau_decades=0.2,
        ),
    )
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in targets,
        )
        for definition in first.parameter_definitions
    )
    result = compile_fit_problem(
        first.data, first.structure, first.instrument, first.config, settings
    )
    if scale_prior:
        result = replace(result, scale_prior_center=1.15, scale_prior_reason=None)
    return result


def test_problem_objective_gradient_matches_central_differences() -> None:
    problem = _problem("component.0.thickness_a", "component.0.density_scale")
    unit = encode_physical_vector(problem, {})
    observed = objective_gradient(problem, unit)
    expected = np.empty(unit.size)
    step = 1e-5
    for index in range(unit.size):
        plus, minus = unit.copy(), unit.copy()
        plus[index] += step
        minus[index] -= step
        expected[index] = (
            evaluate_model(problem, plus).objective
            - evaluate_model(problem, minus).objective
        ) / (2.0 * step)

    np.testing.assert_allclose(observed, expected, rtol=2e-4, atol=2e-7)


def test_problem_objective_information_uses_robust_weights_and_scale_prior() -> None:
    without = _problem("component.0.thickness_a", "instrument.scale")
    with_prior = replace(without, scale_prior_center=1.2, scale_prior_reason=None)
    unit = encode_physical_vector(without, {})

    plain = objective_information(without, unit)
    regularized = objective_information(with_prior, unit)

    assert plain.shape == regularized.shape == (2, 2)
    assert regularized[1, 1] > plain[1, 1]
    assert np.allclose(regularized, regularized.T)


def test_physical_parameter_jacobian_maps_unit_coordinates_in_declared_order() -> None:
    problem = _problem("component.0.thickness_a", "component.0.density_scale")
    unit = encode_physical_vector(problem, {})
    jacobian = physical_parameter_jacobian(problem, unit)

    assert jacobian.shape == (2, 2)
    assert np.all(np.diag(jacobian) > 0.0)
    assert np.count_nonzero(jacobian - np.diag(np.diag(jacobian))) == 0


def test_local_parameter_correlation_detects_near_collinear_thickness_density() -> None:
    covariance = np.asarray([[4.0, 5.9], [5.9, 9.0]])
    correlation = correlation_from_covariance(covariance)

    assert correlation[0, 1] > 0.98
    assert strong_parameter_correlations(("thickness", "density"), correlation) == (
        ("thickness", "density", float(correlation[0, 1])),
    )


def test_thickness_density_pairs_share_the_same_layer_prefix() -> None:
    names = (
        "component.0.thickness_a",
        "component.0.density_scale",
        "component.1.thickness_a",
        "instrument.scale",
    )

    assert thickness_density_pairs(names) == (
        ("component.0.thickness_a", "component.0.density_scale"),
    )
