from __future__ import annotations

from tests.unit.fit.objective_cases import *


def test_scale_prior_penalty_preserves_adjacent_extreme_log_delta() -> None:
    center = 1e308
    scale = np.nextafter(center, np.inf)

    actual = scale_prior_penalty(scale, center, 1.0, 1)

    expected_delta = np.log1p((scale - center) / center) / np.log(10.0)
    assert actual == pytest.approx(expected_delta**2, rel=1e-15, abs=0.0)
    assert actual > 0.0


def test_scale_prior_penalty_divides_before_an_overflowing_square() -> None:
    standardized = 1e200
    tau = np.log10(2.0) / standardized
    count = 10**100

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = scale_prior_penalty(2.0, 1.0, tau, count)

    expected = standardized * (standardized / count)
    assert actual == pytest.approx(expected, rel=1e-15)
    assert np.isfinite(actual)
    assert not any(item.category is RuntimeWarning for item in caught)


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
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
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
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}

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
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
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
