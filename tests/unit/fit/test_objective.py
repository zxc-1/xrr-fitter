from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import xrr_fitter.evaluation as evaluation_module
from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import (
    assign_fit_regions,
    encode_physical_vector,
    log_residuals,
    region_weights,
    robust_log_cost,
    scale_prior_penalty,
)
from xrr_fitter.fit.objective import evaluate_jacobian, evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec


def _problem(*, size: int = 64):
    config = replace(FitConfig.fast(master_seed=7), scale_prior_enabled=False)
    return compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )


def _reliable_plateau_data():
    theta_deg = np.linspace(0.01, 2.0, 600)
    intensity = 0.82 / (1.0 + np.exp((theta_deg - 0.35) / 0.006)) + 1e-8
    return prepared_data(
        size=theta_deg.size,
        two_theta_deg=2.0 * theta_deg,
        intensity_raw=intensity,
    )


def _richardson(problem, unit: np.ndarray) -> np.ndarray:
    def residual(value: np.ndarray) -> np.ndarray:
        result = evaluate_vector(problem, value)
        assert result.valid
        return result.fit_log_residuals_decades

    output = np.empty((np.count_nonzero(problem.data.fit_mask), unit.size))
    step = 5e-5
    for index in range(unit.size):
        coarse_plus = unit.copy()
        coarse_minus = unit.copy()
        fine_plus = unit.copy()
        fine_minus = unit.copy()
        coarse_plus[index] += step
        coarse_minus[index] -= step
        fine_plus[index] += step / 2.0
        fine_minus[index] -= step / 2.0
        coarse = (residual(coarse_plus) - residual(coarse_minus)) / (2.0 * step)
        fine = (residual(fine_plus) - residual(fine_minus)) / step
        output[:, index] = (4.0 * fine - coarse) / 3.0
    return output


def test_region_weights_give_each_present_region_equal_quadratic_mass() -> None:
    labels = np.array([0, 0, 1, 2, 2, 2])

    weights = region_weights(labels)

    masses = [np.sum(weights[labels == label] ** 2) for label in np.unique(labels)]
    np.testing.assert_allclose(masses, np.full(3, labels.size / 3.0))


def test_log_residual_uses_background_floor_and_is_unweighted() -> None:
    model = np.array([1.0, 1e-9, 0.0])
    observed = np.array([0.5, 2e-9, 1e-10])
    floor = 1e-8

    actual = log_residuals(model, observed, floor)

    expected = np.log10(model + floor) - np.log10(observed + floor)
    np.testing.assert_allclose(actual, expected)


def test_robust_cost_places_weights_outside_the_loss() -> None:
    delta = np.array([0.01, 0.2])
    weights = np.array([3.0, 0.5])
    c = 0.05

    actual = robust_log_cost(delta, weights, c)

    pointwise = 2.0 * c**2 * (np.sqrt(1.0 + (delta / c) ** 2) - 1.0)
    assert actual == pytest.approx(np.mean(weights**2 * pointwise))


def test_robust_cost_equals_pointwise_threshold_scaling() -> None:
    delta = np.array([-0.1, 0.0, 0.1])
    weights = np.ones(3)
    c = 0.05

    actual = robust_log_cost(delta, weights, c)

    expected = np.mean(2.0 * c**2 * (np.sqrt(1.0 + (delta / c) ** 2) - 1.0))
    assert actual == pytest.approx(expected)


def test_scale_prior_penalty_matches_versioned_form() -> None:
    scale = 1.2
    estimate = 1.05
    tau = 0.1
    count = 200

    actual = scale_prior_penalty(scale, estimate, tau, count)

    expected = ((np.log10(scale) - np.log10(estimate)) / tau) ** 2 / count
    assert actual == pytest.approx(expected)


def test_scale_prior_disabled_returns_zero() -> None:
    assert scale_prior_penalty(1.0, None, 0.1, 10) == 0.0


def test_reliable_plateau_adds_the_versioned_scale_prior() -> None:
    problem = compile_fit_problem(
        _reliable_plateau_data(),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        FitConfig.fast(master_seed=14),
    )
    assert problem.scale_prior_center is not None
    assert problem.scale_prior_reason is None
    assert problem.warnings == ()
    physical = {
        definition.name: definition.initial
        for definition in problem.parameter_definitions
    }
    physical["instrument.scale"] = 0.75 * problem.scale_prior_center

    result = evaluate_vector(problem, encode_physical_vector(problem, physical))
    data_cost = robust_log_cost(
        result.fit_log_residuals_decades,
        problem.weights[problem.data.fit_mask],
        problem.config.c_decades,
    )
    prior = scale_prior_penalty(
        physical["instrument.scale"],
        problem.scale_prior_center,
        problem.scale_prior_tau_decades,
        int(np.count_nonzero(problem.data.fit_mask)),
    )

    assert result.valid
    assert result.objective == pytest.approx(data_cost + prior)


def test_expert_scale_prior_off_switch_preserves_data_and_reason() -> None:
    config = replace(FitConfig.fast(master_seed=16), scale_prior_enabled=False)
    problem = compile_fit_problem(
        _reliable_plateau_data(),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    expected_reason = "\u4e13\u5bb6\u914d\u7f6e\u5df2\u5173\u95ed\u5c3a\u5ea6\u5f31\u5148\u9a8c"
    assert problem.scale_prior_center is None
    assert problem.scale_prior_tau_decades == config.scale_prior_tau_decades
    assert problem.scale_prior_reason == expected_reason
    assert problem.warnings == (expected_reason,)
    physical = {
        definition.name: definition.initial
        for definition in problem.parameter_definitions
    }

    result = evaluate_vector(problem, encode_physical_vector(problem, physical))
    data_cost = robust_log_cost(
        result.fit_log_residuals_decades,
        problem.weights[problem.data.fit_mask],
        problem.config.c_decades,
    )

    assert result.valid
    assert result.objective == pytest.approx(data_cost)


def test_nonpositive_angle_outside_fit_mask_does_not_reject_candidate() -> None:
    data = prepared_data(size=48)
    fit_mask = data.fit_mask.copy()
    fit_mask[0] = False
    masked = replace(data, fit_mask=fit_mask)
    config = replace(FitConfig.fast(master_seed=31), scale_prior_enabled=False)
    problem = compile_fit_problem(
        masked,
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    physical = {
        definition.name: definition.initial
        for definition in problem.parameter_definitions
    }
    physical["instrument.angle_offset_deg"] = -0.08

    result = evaluate_vector(problem, encode_physical_vector(problem, physical))

    assert masked.two_theta_deg[0] / 2.0 - 0.08 <= 0.0
    assert np.all(masked.two_theta_deg[fit_mask] / 2.0 - 0.08 > 0.0)
    assert result.valid
    assert np.isnan(result.qz_a_inv[0])
    assert np.isnan(result.model_normalized[0])
    assert np.all(np.isfinite(result.model_normalized[fit_mask]))


def test_region_fallback_uses_deterministic_equal_width_q_quartiles() -> None:
    qz = np.array([0.0, 0.1, 0.24, 0.25, 0.49, 0.5, 0.74, 0.75, 1.0])

    labels = assign_fit_regions(qz)

    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 2, 2, 3, 3])


def test_region_fallback_keeps_four_quartile_bins_for_short_vectors() -> None:
    np.testing.assert_array_equal(assign_fit_regions(np.array([0.0, 1.0])), [0, 3])


@pytest.mark.parametrize(
    "labels",
    [[0.0, 1.5], [0.0, np.nan], [0.0, np.inf]],
    ids=("fractional", "nan", "infinity"),
)
def test_region_weights_reject_fractional_and_nonfinite_labels(labels) -> None:
    with pytest.raises(ValueError, match="finite integer"):
        region_weights(np.asarray(labels))


@pytest.mark.parametrize(
    ("model", "observed", "floor"),
    [
        ([1.0, np.nan], [1.0, 1.0], 1e-8),
        ([1.0, 1.0], [1.0, np.inf], 1e-8),
        ([1.0], [1.0], np.nan),
    ],
    ids=("model-nan", "observed-infinity", "floor-nan"),
)
def test_log_residuals_reject_nonfinite_inputs(model, observed, floor) -> None:
    with pytest.raises(ValueError, match="finite"):
        log_residuals(np.asarray(model), np.asarray(observed), floor)


@pytest.mark.parametrize(
    ("delta", "weights", "c"),
    [
        ([np.nan], [1.0], 0.05),
        ([0.0], [np.inf], 0.05),
        ([0.0], [1.0], np.nan),
        ([], [], 0.05),
    ],
    ids=("delta-nan", "weights-infinity", "threshold-nan", "empty"),
)
def test_robust_log_cost_returns_infinity_for_nonfinite_numeric_inputs(
    delta, weights, c
) -> None:
    assert np.isinf(robust_log_cost(np.asarray(delta), np.asarray(weights), c))


@pytest.mark.parametrize(
    ("scale", "estimate", "tau", "count"),
    [
        (np.nan, 1.0, 0.1, 10),
        (1.0, np.inf, 0.1, 10),
        (1.0, 1.0, np.nan, 10),
        (1.0, 1.0, 0.1, 0),
    ],
    ids=("scale-nan", "estimate-infinity", "tau-nan", "count-zero"),
)
def test_scale_prior_rejects_nonfinite_numeric_inputs(
    scale, estimate, tau, count
) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        scale_prior_penalty(scale, estimate, tau, count)


def test_evaluate_jacobian_matches_richardson_central_differences() -> None:
    problem = _problem()
    unit = np.full(len(problem.variables), 0.4)

    analytic = evaluate_jacobian(problem, unit)
    reference = _richardson(problem, unit)

    np.testing.assert_allclose(analytic, reference, rtol=1e-6, atol=5e-11)


def test_evaluate_jacobian_is_accurate_near_the_critical_edge() -> None:
    problem = _problem(size=160)
    unit = np.full(len(problem.variables), 0.5)
    fitted = problem.data.qz_a_inv <= 0.08
    critical = compile_fit_problem(
        replace(problem.data, fit_mask=fitted, fit_ready=True),
        problem.structure,
        problem.instrument,
        problem.config,
    )

    analytic = evaluate_jacobian(critical, unit)
    reference = _richardson(critical, unit)

    np.testing.assert_allclose(analytic, reference, rtol=1e-6, atol=5e-11)


def test_evaluate_jacobian_output_is_read_only() -> None:
    problem = _problem()
    jacobian = evaluate_jacobian(problem, np.full(len(problem.variables), 0.5))

    assert not jacobian.flags.writeable


def test_evaluate_jacobian_does_not_recompute_the_primal_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    unit = np.full(len(problem.variables), 0.5)
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_model",
        lambda *_args, **_kwargs: pytest.fail("primal model was recomputed"),
    )

    jacobian = evaluate_jacobian(problem, unit)

    assert jacobian.shape == (np.count_nonzero(problem.data.fit_mask), len(unit))


def test_periodic_jacobian_expansion_reuses_shared_layer_sld_tangent() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="a", thickness_a=20.0),
            replace(film, name="b", thickness_a=30.0),
        ),
        repeats=5,
        top_roughness_a=1.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=64),
        StructureSpec(base.fronting, (block,), base.backing),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=9), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.5)

    differentiable = evaluation_module.expanded_structure_jacobian(problem, unit)

    for layer_offset in (0, 1):
        rows = differentiable.sld_jacobian[1 + layer_offset : 11 : 2]
        expected = np.repeat(rows[:1], rows.shape[0], axis=0)
        np.testing.assert_array_equal(rows, expected)
