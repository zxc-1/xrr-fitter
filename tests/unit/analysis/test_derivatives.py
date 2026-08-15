from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
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


def covariance_from_correlation(*args, **kwargs):
    return _api().covariance_from_correlation(*args, **kwargs)


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
    result = compile_fit_problem(first.data, first.structure, first.instrument, first.config, settings)
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
        expected[index] = (evaluate_model(problem, plus).objective - evaluate_model(problem, minus).objective) / (
            2.0 * step
        )

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

    assert thickness_density_pairs(names) == (("component.0.thickness_a", "component.0.density_scale"),)


def test_covariance_from_correlation_round_trips_on_positive_definite_blocks() -> None:
    # 往返不能用 ==:correlation_from_covariance 是有损的(plan 修正 2)——
    #   (1) where=denominator>0 把非正对角对应的整行整列零化;
    #   (2) np.clip(correlation, -1, 1) 截断超界元素;
    #   (3) 对角被覆写成精确 1.0/0.0,原对角信息只留在 sigma 里;
    #   (4) 一次浮点除法 + 后续一次浮点乘法,往返本身带舍入。
    # 严格对角占优的对称阵是正定的,前三条不触发,只剩第 4 条的浮点往返,
    # 故用显式容差 atol=1e-12 而非逐位相等。
    covariance = np.asarray([[4.0, 1.0, 0.5], [1.0, 9.0, 1.5], [0.5, 1.5, 16.0]])
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    correlation = correlation_from_covariance(covariance)

    reconstructed = covariance_from_correlation(sigma, correlation)

    assert np.allclose(reconstructed, covariance, rtol=0.0, atol=1e-12)


def test_covariance_from_correlation_zeroes_non_positive_variance_rows() -> None:
    # 对角 <= 0 的坐标(锁定参数 / pinv 零模):sigma 为 0,correlation 的该
    # 行列也已被 correlation_from_covariance 零化,故重建的协方差在该行列
    # 整体为 0——与 .ort 里"协方差元素写 0"的既有语义一致。
    covariance = np.asarray([[4.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 9.0]])
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    correlation = correlation_from_covariance(covariance)

    reconstructed = covariance_from_correlation(sigma, correlation)

    assert np.all(reconstructed[1, :] == 0.0)
    assert np.all(reconstructed[:, 1] == 0.0)
    # 正定子块仍逐坐标往返一致。
    assert np.isclose(reconstructed[0, 0], 4.0, atol=1e-12)
    assert np.isclose(reconstructed[2, 2], 9.0, atol=1e-12)
    assert np.isclose(reconstructed[0, 2], 1.0, atol=1e-12)


def test_covariance_from_correlation_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        covariance_from_correlation(np.ones(2), np.eye(3))
