from __future__ import annotations

import warnings
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

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


def _wide_log_scale_problem(*, scale_prior: bool):
    problem = _problem("instrument.scale", scale_prior=scale_prior)
    definitions = tuple(
        replace(
            definition,
            initial=1.0,
            lower=1e-308,
            upper=1e308,
        )
        if definition.name == "instrument.scale"
        else definition
        for definition in problem.parameter_definitions
    )
    return replace(problem, parameter_definitions=definitions)


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


def test_problem_objective_gradient_handles_log_bounds_whose_ratio_overflows() -> None:
    problem = _wide_log_scale_problem(scale_prior=True)
    unit = encode_physical_vector(problem, {"instrument.scale": 1.0})

    observed = objective_gradient(problem, unit)
    step = 1e-6
    plus, minus = unit.copy(), unit.copy()
    plus[0] += step
    minus[0] -= step
    expected = (evaluate_model(problem, plus).objective - evaluate_model(problem, minus).objective) / (2.0 * step)

    assert np.all(np.isfinite(observed))
    assert observed[0] == pytest.approx(expected, rel=2e-4)


def test_problem_objective_information_handles_log_bounds_whose_ratio_overflows() -> None:
    with_prior = _wide_log_scale_problem(scale_prior=True)
    without_prior = replace(with_prior, scale_prior_center=None)
    unit = encode_physical_vector(with_prior, {"instrument.scale": 1.0})

    plain = objective_information(without_prior, unit)
    regularized = objective_information(with_prior, unit)
    definition = next(item for item in with_prior.parameter_definitions if item.name == "instrument.scale")
    decades_per_unit = np.log10(definition.upper) - np.log10(definition.lower)
    expected_increment = (
        2.0 * (decades_per_unit / with_prior.scale_prior_tau_decades) ** 2 / np.count_nonzero(with_prior.data.fit_mask)
    )

    assert np.all(np.isfinite(regularized))
    assert regularized[0, 0] - plain[0, 0] == pytest.approx(expected_increment)


def test_problem_objective_gradient_at_prior_center_handles_extreme_tau_without_overflow() -> None:
    with_prior = replace(
        _problem("instrument.scale", scale_prior=True),
        scale_prior_center=1.0,
        scale_prior_tau_decades=np.nextafter(0.0, 1.0),
    )
    without_prior = replace(with_prior, scale_prior_center=None)
    unit = encode_physical_vector(with_prior, {"instrument.scale": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = objective_gradient(with_prior, unit)

    np.testing.assert_array_equal(observed, objective_gradient(without_prior, unit))
    assert np.all(np.isfinite(observed))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_problem_objective_information_rejects_extreme_prior_tau_without_warning() -> None:
    problem = replace(
        _problem("instrument.scale", scale_prior=True),
        scale_prior_center=1.0,
        scale_prior_tau_decades=np.nextafter(0.0, 1.0),
    )
    unit = encode_physical_vector(problem, {"instrument.scale": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="scale prior derivative"):
            objective_information(problem, unit)

    assert not any(item.category is RuntimeWarning for item in caught)


def test_problem_objective_information_rejects_prior_curvature_overflow_without_warning() -> None:
    problem = replace(
        _problem("instrument.scale", scale_prior=True),
        scale_prior_center=1.0,
        scale_prior_tau_decades=1e-200,
    )
    unit = encode_physical_vector(problem, {"instrument.scale": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="scale prior information"):
            objective_information(problem, unit)

    assert not any(item.category is RuntimeWarning for item in caught)


def test_problem_objective_gradient_handles_extreme_positive_robust_scale() -> None:
    baseline = _problem("instrument.scale")
    problem = replace(baseline, config=replace(baseline.config, c_decades=1e-200))
    unit = encode_physical_vector(problem, {"instrument.scale": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = objective_gradient(problem, unit)
    step = 1e-6
    plus, minus = unit.copy(), unit.copy()
    plus[0] += step
    minus[0] -= step
    expected = (evaluate_model(problem, plus).objective - evaluate_model(problem, minus).objective) / (2.0 * step)

    assert observed[0] == pytest.approx(expected, rel=2e-4, abs=0.0)
    assert np.all(np.isfinite(observed))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_problem_objective_information_handles_extreme_positive_robust_scale() -> None:
    baseline = _problem("instrument.scale")
    problem = replace(baseline, config=replace(baseline.config, c_decades=1e-200))
    unit = encode_physical_vector(problem, {"instrument.scale": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = objective_information(problem, unit)

    assert np.all(np.isfinite(observed))
    assert not any(item.category is RuntimeWarning for item in caught)


@pytest.mark.parametrize(
    ("function_name", "message"),
    (
        ("objective_gradient", "objective gradient"),
        ("objective_information", "objective information"),
    ),
)
def test_objective_derivatives_reject_unrepresentable_matrix_products(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    message: str,
) -> None:
    module = _api()
    problem = SimpleNamespace(
        variables=(SimpleNamespace(name="x", parameter_index=0),),
        weights=np.ones(1),
        data=SimpleNamespace(fit_mask=np.array([True])),
        config=SimpleNamespace(c_decades=1.0),
        scale_prior_center=None,
        parameter_definitions=(SimpleNamespace(),),
    )
    observed = SimpleNamespace(
        valid=True,
        objective=1.0,
        fit_log_residuals_decades=np.ones(1),
        parameters=(),
    )
    monkeypatch.setattr(module, "evaluate_model", lambda *_args: observed)
    monkeypatch.setattr(
        module,
        "evaluate_model_jacobian",
        lambda *_args: np.array([[np.finfo(float).max]]),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match=message):
            getattr(module, function_name)(problem, np.array([0.5]))

    assert not any(item.category is RuntimeWarning for item in caught)


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


@pytest.mark.parametrize(
    "covariance",
    (
        np.array([[1.0, np.inf], [np.inf, 1.0]]),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
    ),
)
def test_correlation_from_covariance_rejects_nonfinite_entries_without_warning(
    covariance: np.ndarray,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="finite"):
            correlation_from_covariance(covariance)

    assert not any(item.category is RuntimeWarning for item in caught)


@pytest.mark.parametrize(
    ("sigma", "correlation"),
    (
        (np.array([-1.0, 1.0]), np.eye(2)),
        (np.array([1.0, np.inf]), np.eye(2)),
        (np.ones(2), np.array([[1.0, np.nan], [np.nan, 1.0]])),
    ),
)
def test_covariance_from_correlation_rejects_invalid_numeric_entries_without_warning(
    sigma: np.ndarray,
    correlation: np.ndarray,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="finite|nonnegative"):
            covariance_from_correlation(sigma, correlation)

    assert not any(item.category is RuntimeWarning for item in caught)


def test_covariance_from_correlation_rejects_unrepresentable_result_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="finite"):
            covariance_from_correlation(np.array([1e308, 1.0]), np.eye(2))

    assert not any(item.category is RuntimeWarning for item in caught)
