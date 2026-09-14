"""Full-score and active-prior arithmetic boundaries of the two-stage refitter.

These integer-plateau fixtures keep the physical scale prior genuinely active.
Representable narrow priors must reach their exact center; unrepresentable
squares or gradients must fail with actual charged work, not warning suppression.
The independent full-objective finite difference also covers global joint
weights and dimensions, rather than a fabricated least-squares normalization.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data
from tests.unit.fit.test_diagnostic_refit import (
    _api,
    _context,
    _declared,
    _mock_refiner,
    _objective,
    _problem,
    _run,
    _solution,
    _stationary_refiner,
)

from xrr_fitter.evaluation import encode_physical_vector, least_squares_system
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting, SharingRule


def _compiled_prior_context(kind, tau):
    """Compile an active prior from an integer-count plateau, without context surgery.

    The ordinary low-count fixture can disable its prior for a short or noisy
    plateau. These data deliberately satisfy the compiler's plateau checks so
    extreme positive tau is reachable through the real configuration boundary.
    """
    base = _problem()
    theta = np.linspace(0.01, 2.0, 600)
    counts = np.rint(400 / (1 + np.exp((theta - 0.35) / 0.006)))
    data = prepared_data(size=theta.size, two_theta_deg=2 * theta, intensity_raw=counts)
    config = replace(base.config, scale_prior_enabled=True, scale_prior_tau_decades=tau)
    settings = tuple(
        ParameterSetting(value.name, value.initial, value.lower, value.upper, locked=value.locked)
        for value in base.parameter_definitions
    )
    problem = compile_fit_problem(data, base.structure, base.instrument, config, settings)
    if kind == "single":
        return problem
    rule = SharingRule(
        "common-scale", tuple(ParameterReference(name, "instrument.scale") for name in ("left", "right"))
    )
    return compile_joint_problem(("left", "right"), (problem, problem), (rule,))


def _assert_finite_active_prior(problem):
    assert problem.config.scale_prior_enabled
    assert problem.scale_prior_center == 1.0
    assert problem.scale_prior_reason is None
    residual, jacobian = least_squares_system(problem, encode_physical_vector(problem, {}))
    assert np.all(np.isfinite(residual))
    assert np.all(np.isfinite(jacobian))


def _score_context(kind, active_prior):
    problem = _compiled_prior_context(kind, 0.25) if active_prior else _context(kind, multivariate=True)
    members = (problem,) if kind == "single" else problem.problems
    if active_prior:
        for member in members:
            _assert_finite_active_prior(member)
    return problem, sum(member.objective_point_count for member in members)


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("active_prior", [False, True])
def test_localization_score_and_gradient_match_the_complete_physical_objective(monkeypatch, kind, active_prior):
    api = _api()
    problem, count = _score_context(kind, active_prior)
    captured = []

    def localize(fun, initial, **_kwargs):
        score, gradient = fun(initial)
        captured.append((initial.copy(), score, gradient.copy()))
        return _solution(initial, status=0)

    monkeypatch.setattr(api.diagnostic_solver, "minimize", localize)
    monkeypatch.setattr(api.diagnostic_solver, "least_squares", _stationary_refiner)
    result = _run(api, kind, problem)
    unit, score, gradient = captured[0]
    np.testing.assert_array_equal(unit, _declared(kind, problem))
    assert score == pytest.approx(count * _objective(kind, problem, unit), rel=1e-14)
    h = 1e-6
    offsets = np.eye(unit.size) * h
    finite_difference = np.array(
        [
            count * (_objective(kind, problem, unit + offset) - _objective(kind, problem, unit - offset)) / (2 * h)
            for offset in offsets
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-6)
    assert len(captured) == result.attempted_paths == 4
    assert result.nfev == 12
    assert all(path.stages[0].nfev == 1 for path in result.work.paths)


def _record_optimizer_requests(monkeypatch, api):
    """Audit both real optimizers and the charged full handoff validation."""
    requests = []

    def record_solver(original):
        def optimizer(fun, start, **kwargs):
            def counted(unit):
                requests.append(unit.copy())
                return fun(unit)

            return original(counted, start, **kwargs)

        return optimizer

    for name in ("minimize", "least_squares"):
        monkeypatch.setattr(api.diagnostic_solver, name, record_solver(getattr(api.diagnostic_solver, name)))
    original = api._validate_handoff

    def validate(problem, unit):
        requests.append(unit.copy())
        return original(problem, unit)

    monkeypatch.setattr(api, "_validate_handoff", validate)
    return requests


def _assert_numeric_refit_failure(result, expected_nfev):
    """The first failed path must discard its point but retain its actual work.

    A later successful Sobol path cannot repair this incomplete estimator; the
    observed and null refits must obey the same all-paths-success requirement.
    """
    assert result.failure_reason is not None
    assert result.failure_reason.startswith("diagnostic_refit_numerical_failure:FloatingPointError:")
    assert result.unit_vector is None
    assert result.evaluations == ()
    assert result.attempted_paths == 1
    assert result.nfev == expected_nfev
    assert result.nfev > 0


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("tau", [1e-200, 1e-155], ids=["residual_square", "complete_gradient"])
def test_finite_prior_arithmetic_overflow_is_explicit_failure(monkeypatch, kind, tau):
    """Finite callback elements alone do not prove later arithmetic is representable.

    These widths overflow residual squares or the complete score gradient. None may escape as a generic ValueError or
    publish a warning-tainted successful fit. The ambient NumPy policy is local
    to the caller and must be restored after the bounded optimizer finishes.
    """
    api = _api()
    problem = _compiled_prior_context(kind, tau)
    member = problem if kind == "single" else problem.problems[0]
    _assert_finite_active_prior(member)
    requests = _record_optimizer_requests(monkeypatch, api)
    original_policy = np.geterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(api, kind, problem)
    _assert_numeric_refit_failure(result, len(requests))
    assert not any(item.category is RuntimeWarning for item in caught)
    assert np.geterr() == original_policy


def _assert_exact_prior_center(member, result):
    # Localization reaches the exact representable prior center before TRF.
    # The active narrow prior is neither clipped nor disabled.
    assert member.scale_prior_center == 1.0
    for evaluation in result.evaluations:
        scale = next(value.value for value in evaluation.parameters if value.name == "instrument.scale")
        assert scale == 1.0
        assert np.isfinite(evaluation.objective)


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("tau", [0.25, 1e-150], ids=["ordinary", "narrow_representable"])
def test_representable_compiled_scale_prior_keeps_the_declared_estimator(kind, tau):
    """The overflow boundary must not disable or change an ordinary active prior.

    Use the same plateau and declarations as the extreme cases so the only
    changed input is tau. The prior remains active and all four paths complete.
    """
    api = _api()
    problem = _compiled_prior_context(kind, tau)
    member = problem if kind == "single" else problem.problems[0]
    _assert_finite_active_prior(member)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(api, kind, problem)
    assert result.failure_reason is None
    assert result.attempted_paths == 4
    assert result.nfev > 0
    assert not any(item.category is RuntimeWarning for item in caught)
    if tau == 1e-150:
        _assert_exact_prior_center(member, result)


@pytest.mark.parametrize("kind", ["single", "joint"])
def test_cancellation_takes_precedence_over_optimizer_arithmetic_failure(monkeypatch, kind):
    """A concurrent cancellation must not be recast as a numeric unavailable result.

    A real residual call completes before cancellation and the overflowing
    square. This narrow optimizer replacement positions the otherwise racy
    request inside the numeric exception boundary rather than testing only
    cancellation that arrives before any numerical work starts.
    """
    api = _api()
    problem = _context(kind)
    state = SimpleNamespace(cancelled=False)
    original_policy = np.geterr()

    def optimizer(fun, start, **_kwargs):
        fun(start)
        state.cancelled = True
        np.square(np.float64(1e200))
        return _solution(start)

    _mock_refiner(monkeypatch, api, optimizer)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SearchCancelled):
            _run(api, kind, problem, cancelled=lambda: state.cancelled)
    assert not any(item.category is RuntimeWarning for item in caught)
    assert np.geterr() == original_policy
