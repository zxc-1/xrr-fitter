"""Bootstrap generation follows the declared observation distribution."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import scale_candidate, scale_problem

from xrr_fitter.fit.problem import recompile_resampled_problem


def _capture(monkeypatch, problem, *, seed=31):
    module = import_module("xrr_fitter.analysis.bootstrap")
    contexts = []

    def refit(context, start):
        contexts.append(context)
        return start.copy()

    monkeypatch.setattr(module, "_local_bootstrap_fit", refit)
    candidate = scale_candidate(problem)
    result = module.bootstrap_problem_local(
        problem, candidate, sample_count=3, child_seed=seed, recompile=recompile_resampled_problem
    )
    return candidate, contexts, result


def test_gaussian_draws_preserve_negative_observations_and_known_sigma(monkeypatch) -> None:
    problem = scale_problem("gaussian")
    candidate, contexts, result = _capture(monkeypatch, problem)
    expected = np.random.default_rng(31).normal(candidate.model_normalized, problem.data.intensity_sigma_normalized)
    assert np.any(expected < 0)
    np.testing.assert_array_equal(contexts[0].data.intensity_normalized, expected)
    assert result.method == "gaussian_parametric"


def test_poisson_draws_keep_zero_and_integer_counts_without_round_trip(monkeypatch) -> None:
    problem = scale_problem("poisson")
    problem = replace(
        problem,
        data=replace(problem.data, normalization=997.3, intensity_normalized=problem.data.intensity_raw / 997.3),
    )
    candidate, contexts, result = _capture(monkeypatch, problem)
    expected = np.random.default_rng(31).poisson(997.3 * candidate.model_normalized)
    assert np.any(expected == 0)
    np.testing.assert_array_equal(contexts[0].data.intensity_raw, expected)
    np.testing.assert_array_equal(contexts[0].data.intensity_normalized, expected / 997.3)
    assert result.method == "poisson_parametric"


def test_robust_resamples_ordered_log_blocks_even_when_sigma_is_present(monkeypatch) -> None:
    from xrr_fitter.analysis.residual_resampling import moving_block_draw, residual_block_length

    problem = scale_problem()
    candidate, contexts, result = _capture(monkeypatch, problem)
    residuals = candidate.residuals - np.mean(candidate.residuals)
    sampled = moving_block_draw(residuals, residual_block_length(residuals), np.random.default_rng(31))
    expected = (candidate.model_normalized + problem.data.r_floor) * 10 ** (-sampled) - problem.data.r_floor
    np.testing.assert_allclose(contexts[0].data.intensity_normalized, np.maximum(expected, 0), rtol=1e-13)
    assert result.method == "robust_log_moving_block"


def test_resampling_preserves_excluded_raw_points_and_identity(monkeypatch) -> None:
    problem = scale_problem("poisson")
    mask = problem.data.fit_mask.copy()
    mask[::5] = False
    problem = replace(
        problem,
        data=replace(problem.data, fit_mask=mask),
        weights=mask.astype(float),
        region_labels=np.where(mask, problem.region_labels, -1),
        objective_point_count=int(np.count_nonzero(mask)),
        sampling_multipliers=np.ones(mask.size),
    )
    _candidate, contexts, _result = _capture(monkeypatch, problem)
    for context in contexts:
        np.testing.assert_array_equal(context.data.fit_mask, mask)
        np.testing.assert_array_equal(context.data.intensity_raw[~mask], problem.data.intensity_raw[~mask])
        assert context.data.source_sha256 == problem.data.source_sha256
        assert context.data.normalization == problem.data.normalization


def test_each_synthetic_curve_reestimates_data_derived_scale_prior(monkeypatch) -> None:
    problem = scale_problem()
    problem = replace(
        problem,
        config=replace(problem.config, scale_prior_enabled=True),
        scale_prior_center=123.0,
        scale_prior_reason=None,
    )
    _candidate, contexts, _result = _capture(monkeypatch, problem)
    compiler = import_module("xrr_fitter.fit.problem")
    for context in contexts:
        center, reason = compiler._scale_prior_state(context.data, context.instrument, context.config)
        assert context.scale_prior_center == center
        assert context.scale_prior_reason == reason
        assert context.scale_prior_center != 123.0


def test_reestimated_prior_uses_a_real_eligible_platform(monkeypatch) -> None:
    from tests.unit.fit.test_objective_contract import _plateau_problem

    problem = _plateau_problem()
    assert problem.scale_prior_center is not None
    problem = replace(
        problem,
        config=replace(problem.config, noise_model="gaussian"),
        data=replace(problem.data, intensity_sigma_normalized=np.full(1200, 1e-4)),
    )
    _candidate, contexts, _result = _capture(monkeypatch, problem)
    centers = [context.scale_prior_center for context in contexts]
    assert all(center is not None and 0.49 < center < 0.51 for center in centers)
    assert len(set(centers)) == 3
    assert all(context.scale_prior_reason is None for context in contexts)


def test_optimizer_nonconvergence_is_not_a_successful_bootstrap_fit(monkeypatch) -> None:
    from types import SimpleNamespace

    module = import_module("xrr_fitter.analysis.bootstrap")
    problem = scale_problem()
    candidate = scale_candidate(problem)
    monkeypatch.setattr(
        module,
        "least_squares",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=candidate.unit_vector, success=False, message="maximum evaluations exceeded"
        ),
    )
    result = module.bootstrap_problem_local(
        problem, candidate, sample_count=3, child_seed=31, recompile=recompile_resampled_problem
    )
    assert result.successful_samples == 0
    assert result.failure_rate == 1.0
    assert all("maximum evaluations" in reason for _index, reason in result.failure_reasons)


def test_robust_log_inverse_does_not_clip_valid_subfloor_draws() -> None:
    from xrr_fitter.analysis.bootstrap_generation import bootstrap_source, synthetic_context
    from xrr_fitter.analysis.residual_resampling import moving_block_draw, residual_block_length

    problem = scale_problem()
    residuals = np.tile(np.array([-0.1, 0.1]), 40)
    source = bootstrap_source(problem, np.zeros(80), residuals)
    context = synthetic_context(source, np.random.default_rng(31), recompile_resampled_problem)
    sampled = moving_block_draw(residuals, residual_block_length(residuals), np.random.default_rng(31))
    expected = problem.data.r_floor * (10 ** (-sampled) - 1)
    assert np.any(expected < 0)
    np.testing.assert_allclose(context.data.intensity_normalized, expected, atol=0, rtol=1e-13)


def test_unrepresentable_poisson_draw_is_recorded_as_a_failed_replicate(monkeypatch) -> None:
    problem = scale_problem("poisson")
    problem = replace(
        problem, data=replace(problem.data, normalization=1e30, intensity_normalized=problem.data.intensity_raw / 1e30)
    )
    _candidate, contexts, result = _capture(monkeypatch, problem)
    assert contexts == []
    assert result.failure_rate == 1.0
    assert all("Poisson" in reason for _index, reason in result.failure_reasons)


def test_cancellation_during_last_refit_does_not_publish_a_partial_result(monkeypatch) -> None:
    problem = scale_problem()
    candidate = scale_candidate(problem)
    module = import_module("xrr_fitter.analysis.bootstrap")
    cancelled = False

    def refit(_problem, start):
        nonlocal cancelled
        cancelled = True
        return start

    monkeypatch.setattr(module, "_local_bootstrap_fit", refit)
    with pytest.raises(InterruptedError, match="cancelled"):
        module.bootstrap_problem_local(
            problem,
            candidate,
            sample_count=1,
            child_seed=31,
            recompile=recompile_resampled_problem,
            cancelled=lambda: cancelled,
        )
