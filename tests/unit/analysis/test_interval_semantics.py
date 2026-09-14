"""Formal intervals require explicit calibration, independent of work limits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import scale_candidate, scale_problem

from xrr_fitter.analysis.bootstrap import bootstrap_local
from xrr_fitter.analysis.profiles import build_problem_profile, profile_covers_value
from xrr_fitter.model.analysis import BootstrapResult
from xrr_fitter.model.fitting import FitConfig


@pytest.mark.parametrize(
    ("count", "kind", "confidence", "reason"),
    [
        (8, "exploratory_bootstrap", None, "insufficient_successful_samples"),
        (199, "exploratory_bootstrap", None, "insufficient_successful_samples"),
        (200, "percentile_bootstrap", 0.95, None),
    ],
)
def test_percentile_confidence_requires_two_hundred_successful_fits(count, kind, confidence, reason) -> None:
    result = bootstrap_local(lambda rng, _index: rng.normal(size=1), ("scale",), sample_count=count, child_seed=17)
    assert bool(result.intervals) is (count >= 200)
    assert result.confidence_level == confidence
    assert result.successful_samples == count
    assert result.attempted_count == count
    assert result.method == "custom_resampling"
    assert result.interval_kind == kind
    assert result.unavailable_reason == reason


def test_bootstrap_gate_counts_successes_and_preserves_failed_replicate_indices() -> None:
    result = bootstrap_local(
        lambda _rng, index: None if index == 73 else np.array([float(index)]),
        ("scale",),
        sample_count=200,
        child_seed=17,
    )
    assert result.intervals == ()
    assert result.successful_samples == 199
    assert result.attempted_count == 200
    assert result.failure_reasons == ((73, "fit_failed"),)
    assert result.failure_rate == 1 / 200


def test_bootstrap_evidence_rejects_an_empty_parameter_axis() -> None:
    with pytest.raises(ValueError, match="parameter names"):
        BootstrapResult((), np.empty((8, 0)), (), 0.0, 8, unavailable_reason="insufficient_successful_samples")


def test_profile_grid_uses_its_own_configuration() -> None:
    config = FitConfig.fast(17)
    assert hasattr(config, "profile_steps"), "profile resolution must not borrow bootstrap_samples"
    problem = scale_problem()
    profile_config = replace(problem.config, profile_steps=17)
    profiles = []
    for count in (8, 200):
        local = replace(
            problem, config=replace(profile_config, budget=replace(profile_config.budget, bootstrap_samples=count))
        )
        profiles.append(build_problem_profile(local, scale_candidate(local).unit_vector, "instrument.scale"))
    np.testing.assert_array_equal(profiles[0].values, profiles[1].values)
    np.testing.assert_array_equal(profiles[0].objectives, profiles[1].objectives)


@pytest.mark.parametrize("steps", [True, 4, 5.5])
def test_profile_steps_reject_invalid_resolution(steps) -> None:
    with pytest.raises(ValueError, match="profile_steps"):
        replace(FitConfig.fast(17), profile_steps=steps)


def test_robust_profile_is_support_without_a_confidence_level() -> None:
    problem = scale_problem()
    profile = build_problem_profile(problem, scale_candidate(problem).unit_vector, "instrument.scale")
    assert profile.interval_kind == "loss_support"
    assert profile.confidence_level is None
    assert profile.method == "objective_tolerance"
    assert profile.delta_total > 0


@pytest.mark.parametrize("mode", ["gaussian", "poisson"])
def test_regular_likelihood_profile_uses_total_chi_square_threshold(mode) -> None:
    problem = scale_problem(mode)
    candidate = scale_candidate(problem)
    if mode == "poisson":
        normalization = 1e7
        counts = np.random.default_rng(17).poisson(normalization * candidate.model_normalized)
        problem = replace(
            problem,
            data=replace(
                problem.data,
                normalization=normalization,
                intensity_raw=counts,
                intensity_normalized=counts / normalization,
            ),
        )
    profile = build_problem_profile(problem, candidate.unit_vector, "instrument.scale")
    assert hasattr(profile, "delta_total"), "profile must preserve its total-objective threshold"
    assert profile.delta_total == pytest.approx(3.841458820694124)
    assert profile.interval_kind == "likelihood_ratio"
    assert profile.confidence_level == 0.95
    assert profile.objective_point_count == problem.objective_point_count
    assert profile.objective_delta == pytest.approx(profile.delta_total / problem.objective_point_count)


def test_profile_support_uses_the_recorded_threshold_for_coverage() -> None:
    from xrr_fitter.model.analysis import ParameterProfile

    probe = ParameterProfile("probe", np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), True, True, delta_total=0.6)
    assert profile_covers_value(probe, 0.5)


@pytest.mark.parametrize("reason", ["prior", "boundary", "diagnostics"])
def test_nonregular_likelihood_profile_withholds_formal_confidence(reason) -> None:
    problem = scale_problem("gaussian")
    candidate = scale_candidate(problem)
    if reason == "prior":
        problem = replace(problem, scale_prior_center=0.5, scale_prior_reason=None)
    elif reason == "boundary":
        candidate = scale_candidate(
            problem, scale=problem.parameter_definitions[problem.variables[0].parameter_index].lower
        )
    else:
        observed = candidate.model_normalized + 0.02 * 4 * np.sin(np.linspace(0, 5 * np.pi, 80))
        problem = replace(problem, data=replace(problem.data, intensity_raw=observed, intensity_normalized=observed))
    profile = build_problem_profile(problem, candidate.unit_vector, "instrument.scale")
    assert profile.confidence_level is None
    assert profile.interval_kind == "loss_support"
    assert profile.unavailable_reason


def test_low_count_poisson_requires_calibration_before_claiming_likelihood_coverage() -> None:
    problem = scale_problem("poisson")
    profile = build_problem_profile(problem, scale_candidate(problem).unit_vector, "instrument.scale")
    assert profile.interval_kind == "loss_support"
    assert profile.confidence_level is None
    assert profile.unavailable_reason == "poisson_diagnostic_calibration_required"
