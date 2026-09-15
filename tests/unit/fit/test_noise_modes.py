from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from scipy.special import xlogy
from tests.unit.fit.noise_mode_cases import _config, _data, _problem, _unit

from xrr_fitter.analysis.derivatives import objective_gradient
from xrr_fitter.evaluation import (
    evaluate_model,
    least_squares_loss,
    least_squares_residual,
    least_squares_system,
    problem_log_probability,
)
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector, joint_least_squares_loss
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.io.project_codec import project_from_dict, project_to_dict
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.data import BeamSpec, DataColumnMapping, with_fit_mask
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.project import XrrProject


def test_noise_model_is_explicit_default_and_rejects_unknown_values() -> None:
    assert _config("robust_log").noise_model == "robust_log"
    for value in ("auto", "counts_rate", "background_subtracted", "unknown"):
        with pytest.raises(ValueError, match="noise_model"):
            _config(value)


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
def test_mode_is_preserved_by_the_current_project_codec(mode: str) -> None:
    project = replace(XrrProject.new((), master_seed=17), fit_config=_config(mode))
    assert project_from_dict(project_to_dict(project)).fit_config.noise_model == mode


def test_gaussian_cost_residual_and_known_sigma_scaling() -> None:
    problem = _problem("gaussian")
    unit = _unit(problem)
    result = evaluate_model(problem, unit)
    expected = (result.model_normalized - problem.data.intensity_normalized) / 0.02
    np.testing.assert_allclose(result.fit_residuals, expected)
    assert result.residual_name == "standardized_intensity"
    assert result.residual_unit == "1"
    assert result.objective * 80 == pytest.approx(np.sum(expected**2))
    np.testing.assert_array_equal(problem.weights, np.ones(80))
    doubled = _problem(
        "gaussian",
        replace(problem.data, intensity_sigma_raw=np.full(80, 0.04), intensity_sigma_normalized=np.full(80, 0.04)),
    )
    assert evaluate_model(doubled, unit).objective == pytest.approx(result.objective / 4)


def test_poisson_uses_raw_counts_and_normalization() -> None:
    problem = _problem("poisson")
    result = evaluate_model(problem, _unit(problem))
    counts = problem.data.intensity_raw
    mu = problem.data.normalization * result.model_normalized
    expected = 2 * (mu - counts + xlogy(counts, counts / mu))
    np.testing.assert_allclose(result.fit_residuals**2, expected, rtol=1e-12)
    assert result.fit_residuals[0] ** 2 == pytest.approx(2 * mu[0])
    assert result.objective * 80 == pytest.approx(np.sum(expected))
    assert result.residual_name == "signed_poisson_deviance"
    np.testing.assert_array_equal(problem.weights, np.ones(80))


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
def test_three_mode_derivatives_solver_cost_and_sampling_target_agree(mode: str) -> None:
    problem = _problem(mode)
    unit = _unit(problem)
    residual, jacobian = least_squares_system(problem, unit)
    np.testing.assert_allclose(residual, least_squares_residual(problem, unit), rtol=1e-11)
    gradient = objective_gradient(problem, unit)
    for name in ("instrument.scale", "component.0.thickness_a", "instrument.background"):
        column = next(i for i, value in enumerate(problem.variables) if value.name == name)
        if not 1e-6 < unit[column] < 1 - 1e-6:
            continue
        plus, minus = unit.copy(), unit.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        expected = (least_squares_residual(problem, plus) - least_squares_residual(problem, minus)) / 2e-6
        np.testing.assert_allclose(jacobian[:, column], expected, rtol=2e-5, atol=1e-7)
        scalar = (evaluate_model(problem, plus).objective - evaluate_model(problem, minus).objective) / 2e-6
        assert gradient[column] == pytest.approx(scalar, rel=2e-5, abs=1e-7)
    total = evaluate_model(problem, unit).objective * problem.objective_point_count
    assert np.sum(least_squares_loss(problem)(residual**2)[0]) / 2 == pytest.approx(total)
    assert problem_log_probability(problem, unit) == pytest.approx(-total / 2)


@pytest.mark.parametrize("sigma", [None, 0.0, -1.0, np.nan, np.inf])
def test_gaussian_compile_rejects_missing_or_invalid_selected_sigma(sigma: float | None) -> None:
    data = _data("gaussian")
    values = None if sigma is None else np.full(80, sigma)
    with pytest.raises(ValueError, match="sigma|uncertainty"):
        _problem("gaussian", replace(data, intensity_sigma_normalized=values))


@pytest.mark.parametrize("count", [-1.0, 0.25, np.nan, np.inf])
def test_poisson_compile_rejects_invalid_selected_raw_counts(count: float) -> None:
    data = _data("poisson")
    raw = data.intensity_raw.copy()
    raw[0] = count
    with pytest.raises(ValueError, match="count"):
        _problem("poisson", replace(data, intensity_raw=raw))


@pytest.mark.parametrize("mode", ["gaussian", "poisson"])
def test_compile_ignores_invalid_values_only_outside_the_selected_mask(mode: str) -> None:
    data = _data(mode)
    mask = data.fit_mask.copy()
    mask[0] = False
    values = data.intensity_raw.copy() if mode == "poisson" else data.intensity_sigma_normalized.copy()
    values[0] = np.nan
    field = "intensity_raw" if mode == "poisson" else "intensity_sigma_normalized"
    problem = _problem(mode, replace(with_fit_mask(data, mask), **{field: values}))
    assert problem.objective_point_count == 79


def test_poisson_rejects_averaged_duplicate_rows_even_if_the_mean_is_integer() -> None:
    data = _data("poisson")
    groups = ((0, 1), *data.source_row_groups[1:])
    with pytest.raises(ValueError, match="merged|duplicate|raw count"):
        _problem("poisson", replace(data, source_row_groups=groups))


@pytest.mark.parametrize("mode", ["gaussian", "poisson"])
def test_likelihood_joint_objective_adds_information_without_dataset_balancing(mode: str) -> None:
    members = (_problem(mode, size=80), _problem(mode, size=800))
    joint = compile_joint_problem(("small", "large"), members, ())
    result = evaluate_joint_vector(joint, np.full(len(joint.global_variables), 0.4))
    assert result.valid
    expected = sum(
        value.objective * member.objective_point_count
        for member, value in zip(members, result.local_evaluations, strict=True)
    )
    assert result.objective * 880 == pytest.approx(expected)
    assert np.sum(joint_least_squares_loss(joint)(result.residuals**2)[0]) / 2 == pytest.approx(expected)


def test_real_gaussian_import_keeps_zero_negative_and_user_excluded_points() -> None:
    _config("gaussian")
    content = "\n".join(
        f"{angle:.8f} {value:.8f} 0.1"
        for angle, value in zip(np.linspace(0.1, 4.0, 80), np.linspace(-10, 1, 80), strict=True)
    ).encode()
    data = read_xy_bytes(
        content,
        source_path="gaussian.xy",
        beam=BeamSpec("monochromatic"),
        column_mapping=DataColumnMapping(intensity_sigma=2),
        noise_model="gaussian",
    )
    assert np.all(data.fit_mask)
    mask = data.fit_mask.copy()
    mask[10] = False
    problem = _problem("gaussian", with_fit_mask(data, mask))
    assert not problem.data.fit_mask[10]
    assert np.count_nonzero(problem.data.fit_mask) == 79


def test_poisson_deviance_zero_and_near_equal_counts_are_stable() -> None:
    boundary = import_module("xrr_fitter.evaluation")
    assert hasattr(boundary, "poisson_deviance"), "shared evaluation must define Poisson deviance"
    np.testing.assert_array_equal(boundary.poisson_deviance(np.zeros(3), np.array([0.0, 1.0, 2.0])), [0.0, 2.0, 4.0])
    counts = np.array([1.0, 1e6, 1e12])
    mu = counts * (1 + 1e-8)
    expected = (mu - counts) ** 2 / counts * (1 - 2 * (mu - counts) / counts / 3)
    np.testing.assert_allclose(boundary.poisson_deviance(counts, mu), expected, rtol=1e-14)


def test_poisson_positive_counts_reject_nonpositive_prediction() -> None:
    boundary = import_module("xrr_fitter.evaluation")
    assert hasattr(boundary, "poisson_deviance"), "shared evaluation must define Poisson deviance"
    with pytest.raises(ValueError, match="Poisson|count|prediction"):
        boundary.poisson_deviance(np.array([1.0]), np.array([0.0]))


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
def test_published_mode_residuals_are_distinct_from_log_plot_residuals(mode: str) -> None:
    problem = _problem(mode)
    result = evaluate_model(problem, _unit(problem))
    candidate = candidate_from_evaluation(problem, _unit(problem), result, "noise-mode", 0, "test", 1)
    assert hasattr(candidate, "residuals"), "published candidates need honest mode residuals"
    np.testing.assert_array_equal(candidate.residuals[problem.data.fit_mask], result.fit_residuals)
    assert candidate.noise_model == mode
    assert (candidate.residual_name, candidate.residual_unit) == (result.residual_name, result.residual_unit)
    nonpositive = problem.data.intensity_normalized <= 0.0
    if mode != "robust_log":
        assert np.all(np.isnan(candidate.log_residuals_decades[nonpositive]))
    codec = import_module("xrr_fitter.io.codec_candidates")
    restored = codec._candidate_from_dict(codec._candidate_to_dict(candidate))
    np.testing.assert_array_equal(restored.residuals, candidate.residuals)
    assert not restored.residuals.flags.writeable
    assert restored.noise_model == mode


def test_public_gaussian_import_and_reload_preserve_selected_negative_points(tmp_path) -> None:
    import xrr_fitter.api as api
    from xrr_fitter.services.datasets import _prepared_current

    path = tmp_path / "sample.xy"
    path.write_text(
        "\n".join(
            f"{angle:.8f} {value:.8f} 0.1"
            for angle, value in zip(np.linspace(0.1, 4.0, 80), np.linspace(-10, 1, 80), strict=True)
        )
    )
    project = replace(api.new_project(), fit_config=_config("gaussian"))
    project = api.add_dataset(project, path, InstrumentSpec(), column_mapping=DataColumnMapping(intensity_sigma=2))
    assert all(project.datasets[0].fit_mask)
    mask = np.array(project.datasets[0].fit_mask)
    mask[10] = False
    project = api.set_fit_mask(project, project.datasets[0].dataset_id, mask)
    data = _prepared_current(project, project.datasets[0])
    np.testing.assert_array_equal(data.fit_mask, mask)


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
def test_joint_resume_validates_the_declared_total_target(mode: str) -> None:
    from xrr_fitter.fit.joint_candidates import validate_joint_candidate_alignment

    joint = compile_joint_problem(("small", "large"), (_problem(mode), _problem(mode, size=800)), ())
    result = evaluate_joint_vector(joint, np.full(len(joint.global_variables), 0.4))
    candidates = tuple(
        (
            replace(
                candidate_from_evaluation(member, unit, evaluation, "A-0", 0, "test", 1),
                ranking_objective=result.objective,
            ),
        )
        for member, unit, evaluation in zip(
            joint.problems, result.local_unit_vectors, result.local_evaluations, strict=True
        )
    )
    validate_joint_candidate_alignment(joint, candidates, ())


def test_gaussian_diagnostics_use_all_standardized_residuals_not_only_log_domain() -> None:
    from xrr_fitter.analysis.diagnostics import ordered_fit_residuals

    problem = _problem("gaussian")
    result = evaluate_model(problem, _unit(problem))
    candidate = candidate_from_evaluation(problem, _unit(problem), result, "gaussian", 0, "test", 1)
    np.testing.assert_array_equal(ordered_fit_residuals(problem, candidate), result.fit_residuals)


def test_poisson_import_does_not_reaverage_single_source_counts() -> None:
    values = np.arange(80, dtype=float)
    content = "\n".join(
        f"{angle:.8f} {value:.0f} 0.3" for angle, value in zip(np.linspace(0.1, 4, 80), values, strict=True)
    ).encode()
    data = read_xy_bytes(
        content,
        source_path="counts.xy",
        beam=BeamSpec("monochromatic"),
        column_mapping=DataColumnMapping(intensity_sigma=2),
        noise_model="poisson",
    )
    np.testing.assert_array_equal(data.intensity_raw, values)
    assert _problem("poisson", data).objective_point_count == 80


@pytest.mark.parametrize("mode", ["robust_log", "gaussian", "poisson"])
def test_mode_prior_retains_the_same_total_q_across_single_and_joint(mode: str) -> None:
    members = tuple(
        replace(_problem(mode, size=size), scale_prior_center=1.0, scale_prior_reason=None) for size in (80, 800)
    )
    joint = compile_joint_problem(("small", "large"), members, ())
    result = evaluate_joint_vector(joint, np.full(len(joint.global_variables), 0.4))
    solver = joint_least_squares_loss(joint)(result.residuals**2)
    assert np.sum(solver[0]) / 2 == pytest.approx(result.objective * 880)
    for member, unit, local in zip(joint.problems, result.local_unit_vectors, result.local_evaluations, strict=True):
        residual, jacobian = least_squares_system(member, unit)
        rho = least_squares_loss(member)(residual**2)
        np.testing.assert_allclose(
            objective_gradient(member, unit),
            jacobian.T @ (rho[1] * residual) / member.objective_point_count,
            rtol=1e-11,
        )
        assert problem_log_probability(member, unit) == pytest.approx(
            -0.5 * local.objective * member.objective_point_count
        )
