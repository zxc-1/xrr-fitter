"""The diagnostic estimator is fixed, bounded, and distinct from search winners.

Real low-count problems exercise the residual/analytic-Jacobian chain. Narrow
optimizer fakes expose stop status, work accounting, and cancellation boundaries
without manufacturing a second objective or a separate joint scatter policy.

The real cases start from declared parameters rather than a previously fitted
winner. Poisson draws use a fixed normalization, include exact zero counts, and
retain missing source rows outside the fit mask. Multivariate joint cases share
scale while retaining separate density coordinates, so the path budget must use
the global dimension rather than either member's local dimension.

Failure probes distinguish three boundaries: a solver that returns success=False,
an expected numeric exception without an OptimizeResult, and an unexpected
programming exception that must escape. Partial solutions cannot be published in
either of the first two cases. Full reporting evaluations are checked separately
from optimizer callbacks because their work is deliberately excluded from nfev.

Cancellation is injected after computation as well as before it. A callback
cache hit, successful last optimizer return, or locked full-axis evaluation must
not turn a cancellation request into completed evidence. Joint recompilation is
spied only to assert the original declarations and changed member contexts; its
actual compiler and shared-parameter scatter remain in use.

The active-prior cases use an independent, sufficiently long integer plateau.
Changing tau on a noisy low-count fixture alone is insufficient: the compiler
can legitimately disable its prior before the numerical boundary is reached.
The narrow widths below expose different arithmetic stages, not new accepted
parameter limits. Finite residuals may overflow when squared; even representable
residual squares do not guarantee representable Jacobian scales or later steps.
An ordinary-width prior provides the corresponding unchanged-estimator control.

Only NumPy overflow/invalid operations inside the numerical optimizer are meant
to change failure classification. The tests retain separate programmer-error
and cancellation contracts and verify that the caller's NumPy policy survives
the numerical boundary. No prior clipping or solver-warning suppression is used.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.joint_constraint_cases import cross_constraint_joint
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import initial_joint_vector
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
    ParameterSetting,
    SharingRule,
)


def _api():
    name = "xrr_fitter.fit.diagnostic_refit"
    assert find_spec(name) is not None, "the fixed declared diagnostic refitter must be implemented"
    return import_module(name)


def _problem(*, size=64, seed=73, locked=False, missing=False, minimum=80, per_parameter=30, multivariate=False):
    angles = np.linspace(0.1, 2.4, size)
    if missing:
        angles[0] = np.nan
    data = prepared_data(size=size, two_theta_deg=angles, intensity_raw=np.full(size, 400.0))
    config = replace(
        FitConfig.fast(seed),
        noise_model="poisson",
        scale_prior_enabled=False,
        local_workers=1,
        budget=SearchBudget(0, 0, minimum, per_parameter, 1),
    )
    base = compile_fit_problem(
        data, simple_structure(), InstrumentSpec(footprint_mode="none", instrument_id="diagnostic-lab"), config
    )
    truth = evaluate_vector(base, encode_physical_vector(base, {}))
    counts = np.zeros(size)
    selected = data.fit_mask
    counts[selected] = np.random.default_rng(seed).poisson(truth.model_normalized[selected] * data.normalization)
    data = replace(data, intensity_raw=counts, intensity_normalized=counts / data.normalization)
    settings = []
    for definition in base.parameter_definitions:
        if definition.name == "instrument.scale" and not locked:
            settings.append(ParameterSetting(definition.name, 0.9, 0.3, 1.8))
        elif definition.name == "component.0.density_scale" and multivariate and not locked:
            settings.append(ParameterSetting(definition.name, 1.0, 0.8, 1.2))
        else:
            settings.append(
                ParameterSetting(
                    definition.name, definition.initial, definition.initial, definition.initial, locked=True
                )
            )
    return compile_fit_problem(data, base.structure, base.instrument, config, tuple(settings))


def _joint(*, locked=False, missing=False, minimum=80, per_parameter=30, multivariate=False):
    members = tuple(
        _problem(
            size=size,
            seed=seed,
            locked=locked,
            missing=missing,
            minimum=minimum,
            per_parameter=per_parameter,
            multivariate=multivariate,
        )
        for size, seed in ((64, 73), (80, 79))
    )
    sharing = (
        ()
        if locked
        else (
            SharingRule(
                "common-scale", tuple(ParameterReference(name, "instrument.scale") for name in ("left", "right"))
            ),
        )
    )
    return compile_joint_problem(("left", "right"), members, sharing)


def _context(kind, **kwargs):
    return _problem(**kwargs) if kind == "single" else _joint(**kwargs)


def _run(api, kind, problem, **kwargs):
    if kind == "single":
        return api.refit_diagnostic_single(problem, **kwargs)
    return api.refit_diagnostic_joint(problem, problem.problems, **kwargs)


def _solution(unit, *, nfev=1, success=True, status=None):
    return SimpleNamespace(
        x=np.array(unit, copy=True),
        success=success,
        status=(1 if success else -2) if status is None else status,
        nfev=nfev,
        message="synthetic optimizer stop",
    )


def _stationary_refiner(fun, unit, *, jac, **_kwargs):
    fun(unit)
    jac(unit)
    return _solution(unit)


def _mock_refiner(monkeypatch, api, optimizer):
    """Isolate refinement semantics after one real declared localization request."""

    def localize(fun, unit, **_kwargs):
        fun(unit)
        return _solution(unit, status=0)

    monkeypatch.setattr(api.diagnostic_solver, "minimize", localize)
    monkeypatch.setattr(api.diagnostic_solver, "least_squares", optimizer)


def _declared(kind, problem):
    return encode_physical_vector(problem, {}) if kind == "single" else initial_joint_vector(problem)


def _objective(kind, problem, unit):
    evaluation = evaluate_vector(problem, unit) if kind == "single" else evaluate_joint_vector(problem, unit)
    return evaluation.objective


def _assert_deterministic(first, second):
    assert first.failure_reason is None
    assert first.attempted_paths == 4
    assert first.nfev > 0
    assert first.nfev == second.nfev
    np.testing.assert_array_equal(first.unit_vector, second.unit_vector)


def _assert_full_axes(member, evaluation):
    assert evaluation.valid
    assert evaluation.noise_model == "poisson"
    assert evaluation.model_normalized.shape == member.data.qz_a_inv.shape
    assert np.isnan(evaluation.qz_a_inv[0]) and np.isnan(evaluation.model_normalized[0])
    assert evaluation.fit_residuals.size == np.count_nonzero(member.data.fit_mask)
    assert np.all(np.isfinite(evaluation.model_normalized[member.data.fit_mask]))


def _assert_locked_parameters(member, evaluation):
    observed = {value.name: value.value for value in evaluation.parameters}
    for definition in member.parameter_definitions:
        if definition.locked:
            assert observed[definition.name] == definition.initial


def _assert_shared_parameter(evaluations, name):
    values = [next(value.value for value in result.parameters if value.name == name) for result in evaluations]
    assert values[0] == values[1]


def _assert_same_evaluations(actual, expected):
    for result, local in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result.fit_residuals, local.fit_residuals)
        assert result.objective == local.objective


def test_declared_starts_are_fixed_sobol_points_and_own_their_arrays():
    api = _api()
    initial = np.array([0.4, 0.6])
    first = api.diagnostic_starts(initial)
    second = api.diagnostic_starts(initial.copy())
    expected = [[0.4, 0.6], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(first, second)
    initial[:] = 0.8
    np.testing.assert_array_equal(first, expected)


def test_declared_starts_stably_remove_exact_duplicates_and_handle_zero_dimensions():
    api = _api()
    np.testing.assert_array_equal(api.diagnostic_starts(np.array([0.5])), [[0.5], [0.75], [0.25]])
    empty = api.diagnostic_starts(np.array([]))
    assert len(empty) == 1
    assert empty[0].shape == (0,)


@pytest.mark.parametrize("initial", [[-0.1], [1.1], [np.nan], [np.inf], [[0.5]], 0.5])
def test_declared_starts_reject_invalid_units(initial):
    with pytest.raises(ValueError, match="unit"):
        _api().diagnostic_starts(np.asarray(initial))


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("multivariate", [False, True])
def test_real_refit_is_deterministic_and_preserves_full_axes_and_locks(kind, multivariate):
    api = _api()
    problem = _context(kind, missing=True, multivariate=multivariate)
    first = _run(api, kind, problem)
    second = _run(api, kind, problem)
    _assert_deterministic(first, second)
    assert _objective(kind, problem, first.unit_vector) < _objective(kind, problem, _declared(kind, problem))
    members = (problem,) if kind == "single" else problem.problems
    for member, evaluation, repeated in zip(members, first.evaluations, second.evaluations, strict=True):
        _assert_full_axes(member, evaluation)
        _assert_locked_parameters(member, evaluation)
        np.testing.assert_array_equal(evaluation.fit_residuals, repeated.fit_residuals)
    if kind == "joint":
        _assert_shared_parameter(first.evaluations, "instrument.scale")


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("minimum, per_parameter", [(7, 11), (13, 2)])
@pytest.mark.parametrize("multivariate", [False, True])
def test_each_path_uses_the_same_analytic_estimator_and_its_own_budget(
    monkeypatch, kind, minimum, per_parameter, multivariate
):
    """Check the actual callbacks, not a stand-in quadratic objective.

    Analytic columns must agree with finite differences, and SciPy's half-sum
    loss must equal the existing full objective after restoring its point mass.
    """
    api = _api()
    problem = _context(kind, minimum=minimum, per_parameter=per_parameter, multivariate=multivariate)
    starts, objectives = [], []

    def optimizer(fun, x0, *, jac, loss, **kwargs):
        starts.append(x0.copy())
        residual, analytic = fun(x0), jac(x0)
        directions = np.eye(x0.size) * 1e-6
        finite = np.column_stack([(fun(x0 + direction) - fun(x0 - direction)) / 2e-6 for direction in directions])
        np.testing.assert_allclose(analytic, finite, rtol=2e-5, atol=2e-7)
        assert kwargs == {
            "bounds": (0.0, 1.0),
            "method": "trf",
            "x_scale": "jac",
            "ftol": None,
            "xtol": 1e-10,
            "gtol": 1e-10,
            "max_nfev": max(minimum, per_parameter * x0.size) - 2,
            "callback": kwargs["callback"],
        }
        assert callable(kwargs["callback"])
        kwargs["callback"](x0)
        evaluated = evaluate_vector(problem, x0) if kind == "single" else evaluate_joint_vector(problem, x0)
        count = (
            problem.objective_point_count
            if kind == "single"
            else sum(member.objective_point_count for member in problem.problems)
        )
        assert np.sum(loss(residual**2)[0]) / 2 == pytest.approx(evaluated.objective * count)
        objectives.append(evaluated.objective)
        return _solution(x0, nfev=1 + 2 * x0.size)

    _mock_refiner(monkeypatch, api, optimizer)
    result = _run(api, kind, problem)
    declared = encode_physical_vector(problem, {}) if kind == "single" else initial_joint_vector(problem)
    np.testing.assert_array_equal(starts, api.diagnostic_starts(declared))
    assert result.failure_reason is None
    assert result.nfev == (3 + 2 * declared.size) * len(starts)
    assert result.attempted_paths == len(starts)
    np.testing.assert_array_equal(result.unit_vector, starts[int(np.argmin(objectives))])


@pytest.mark.parametrize("kind", ["single", "joint"])
def test_zero_dimensional_refit_evaluates_once_without_optimizer_work(monkeypatch, kind):
    """A locked model still has a full evaluation but zero optimizer nfev.

    Counting this publication traversal as a solver call would inflate the
    calibration cost and obscure the zero-dimensional estimator boundary.
    """
    api = _api()
    problem = _context(kind, locked=True)
    name = "evaluate_vector" if kind == "single" else "evaluate_joint_vector"
    original, calls = getattr(api, name), []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(api, name, evaluate)
    _mock_refiner(monkeypatch, api, lambda *_args, **_kwargs: pytest.fail("locked refit must not optimize"))
    result = _run(api, kind, problem)
    assert result.failure_reason is None
    assert result.unit_vector.size == 0
    assert result.attempted_paths == 1
    assert result.nfev == 0
    assert calls == [{}]


@pytest.mark.parametrize("kind", ["single", "joint"])
def test_one_nonconverged_path_invalidates_all_previous_solutions(monkeypatch, kind):
    api = _api()
    problem = _context(kind)
    starts = []

    def optimizer(fun, x0, **_kwargs):
        starts.append(x0.copy())
        for _index in range(4):
            fun(x0)
        return _solution(x0, nfev=4, success=len(starts) != 3)

    _mock_refiner(monkeypatch, api, optimizer)
    result = _run(api, kind, problem)
    assert result.failure_reason.startswith("diagnostic_refit_nonconverged")
    assert result.unit_vector is None
    assert result.evaluations == ()
    assert result.nfev == 18
    assert result.attempted_paths == 3
    assert len(starts) == 3


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize(
    "error", [FloatingPointError("numeric failure"), OverflowError("overflow"), np.linalg.LinAlgError("svd failed")]
)
def test_expected_optimizer_numeric_failure_keeps_spent_work(monkeypatch, kind, error):
    """Preserve all stages of prior paths and the interrupted path's work.

    The extra Jacobian callback must not increase nfev. No fourth replacement
    path may erase the failed second path or restore an available result.
    """
    api = _api()
    problem = _context(kind)
    calls = []

    def optimizer(fun, x0, *, jac, **_kwargs):
        calls.append(x0.copy())
        fun(x0)
        jac(x0)
        if len(calls) == 2:
            raise error
        return _solution(x0, nfev=1)

    _mock_refiner(monkeypatch, api, optimizer)
    result = _run(api, kind, problem)
    assert type(error).__name__ in result.failure_reason
    assert result.unit_vector is None and result.evaluations == ()
    assert result.nfev == 6
    assert result.attempted_paths == 2
    assert result.work.path_count == 2 and result.work.completed_paths == 1
    assert result.work.nfev == 6 and result.work.max_path_nfev == 3
    assert sorted(path.stages[-1].kind for path in result.work.paths) == ["native", "numerical_failure"]
    assert all(tuple(stage.nfev for stage in path.stages) == (1, 1, 1) for path in result.work.paths)


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("error", [ValueError("malformed system"), RuntimeError("bug"), TypeError("wrong callback")])
def test_programming_errors_are_not_calibration_failures(monkeypatch, error, kind):
    api = _api()

    def optimizer(*_args, **_kwargs):
        raise error

    _mock_refiner(monkeypatch, api, optimizer)
    with pytest.raises(type(error)) as caught:
        _run(api, kind, _context(kind))
    assert caught.value is error


@pytest.mark.parametrize("kind", ["single", "joint"])
def test_cancellation_before_work_uses_the_existing_search_boundary(monkeypatch, kind):
    api = _api()
    problem = _context(kind)
    _mock_refiner(monkeypatch, api, lambda *_args, **_kwargs: pytest.fail("cancelled refit must not optimize"))
    with pytest.raises(SearchCancelled, match="search cancelled"):
        _run(api, kind, problem, cancelled=lambda: True)


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("boundary", ["jacobian_cache_hit", "optimizer_return", "handoff", "publication"])
def test_cancellation_never_publishes_a_completed_or_partial_refit(monkeypatch, kind, boundary):
    """Cancellation wins even when numerical work just finished successfully.

    The cache-hit case specifically requires polling around callback access,
    not merely inside the physical evaluator that a cache hit would bypass.
    """
    api = _api()
    problem = _context(kind)
    state = SimpleNamespace(cancelled=False)

    def optimizer(fun, x0, *, jac, **_kwargs):
        fun(x0)
        if boundary == "jacobian_cache_hit":
            state.cancelled = True
            jac(x0.copy())
        if boundary == "optimizer_return":
            state.cancelled = True
        return _solution(x0)

    if boundary in {"handoff", "publication"}:
        name = "evaluate_vector" if kind == "single" else "evaluate_joint_vector"
        original, calls = getattr(api, name), []

        def evaluate(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(None)
            state.cancelled = len(calls) == (1 if boundary == "handoff" else 2)
            return result

        monkeypatch.setattr(api, name, evaluate)
    _mock_refiner(monkeypatch, api, optimizer)
    with pytest.raises(SearchCancelled):
        _run(api, kind, problem, cancelled=lambda: state.cancelled)


def test_joint_refit_recompiles_changed_members_with_original_local_and_joint_constraints(monkeypatch):
    api = _api()
    base = cross_constraint_joint()
    left, right = base.problems
    local_rule = ConstraintRule(ParameterReference("left", "instrument.scale"), ConstraintNode("const", value=1.0))
    left = replace(left, constraint_rules=(local_rule,))
    template = compile_joint_problem(base.dataset_ids, (left, right), base.sharing_rules, base.joint_constraint_rules)
    assert template.constraint_rules != template.joint_constraint_rules
    changed = tuple(replace(member, scale_prior_center=0.8, scale_prior_reason=None) for member in template.problems)
    captured = []
    original = api.compile_joint_problem

    def compile_joint(ids, members, sharing, constraints):
        assert ids == template.dataset_ids
        assert members is changed
        assert sharing == template.sharing_rules
        assert constraints == template.joint_constraint_rules
        generated = original(ids, members, sharing, constraints)
        assert generated is not template
        captured.append(generated)
        return generated

    monkeypatch.setattr(api, "compile_joint_problem", compile_joint)
    _mock_refiner(monkeypatch, api, _stationary_refiner)
    result = api.refit_diagnostic_joint(template, changed)
    assert result.failure_reason is None
    assert len(captured) == 1
    assert captured[0].constraint_rules == template.constraint_rules
    expected = evaluate_joint_vector(captured[0], result.unit_vector)
    _assert_same_evaluations(result.evaluations, expected.local_evaluations)
    _assert_shared_parameter(result.evaluations, "component.0.density_scale")


@pytest.mark.parametrize("kind", ["single", "joint"])
def test_equal_objectives_keep_the_first_declared_path(monkeypatch, kind):
    api = _api()
    problem = _context(kind)
    name = "evaluate_vector" if kind == "single" else "evaluate_joint_vector"
    original = getattr(api, name)
    monkeypatch.setattr(api, name, lambda *args, **kwargs: replace(original(*args, **kwargs), objective=1.0))
    _mock_refiner(monkeypatch, api, _stationary_refiner)
    result = _run(api, kind, problem)
    declared = encode_physical_vector(problem, {}) if kind == "single" else initial_joint_vector(problem)
    assert result.failure_reason is None
    assert result.attempted_paths == 4
    np.testing.assert_array_equal(result.unit_vector, declared)


def _damaged(evaluation, defect):
    active_nan = evaluation.qz_a_inv.copy()
    active_nan[1] = np.nan
    missing_mean = evaluation.model_normalized.copy()
    missing_mean[1] = np.nan
    updates = {
        "residual": {"fit_residuals": np.full(evaluation.fit_residuals.shape, np.nan)},
        "weighted": {"fit_weighted_residuals": np.full(evaluation.fit_residuals.shape, np.inf)},
        "mean": {"model_normalized": np.full(evaluation.model_normalized.shape, np.inf)},
        "fitted_nan": {"qz_a_inv": active_nan, "model_normalized": missing_mean},
        "full_axis": {"qz_a_inv": evaluation.qz_a_inv[1:], "model_normalized": evaluation.model_normalized[1:]},
        "fit_axis": {
            "fit_residuals": evaluation.fit_residuals[:-1],
            "fit_weighted_residuals": evaluation.fit_weighted_residuals[:-1],
        },
        "noise": {"noise_model": "gaussian"},
    }
    return replace(evaluation, **updates[defect])


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("defect", ["residual", "weighted", "mean", "fitted_nan", "full_axis", "fit_axis", "noise"])
@pytest.mark.parametrize("phase", ["handoff", "publication"])
def test_invalid_published_evaluation_becomes_explicit_failure_not_partial_success(monkeypatch, kind, defect, phase):
    """Numerically invalid returned evidence is not a valid fit or a programmer error.

    The context owns the required full and fitted axes. A generic immutable
    ModelEvaluation cannot independently validate those context-specific lengths
    or decide whether a paired missing row is actually excluded from fitting.
    """
    api = _api()
    problem = _context(kind, missing=True)
    name = "evaluate_vector" if kind == "single" else "evaluate_joint_vector"
    original, calls = getattr(api, name), []

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(None)
        if phase == "publication" and len(calls) == 1:
            return result
        if kind == "single":
            return _damaged(result, defect)
        locals_ = (_damaged(result.local_evaluations[0], defect),) + result.local_evaluations[1:]
        return replace(result, local_evaluations=locals_)

    monkeypatch.setattr(api, name, evaluate)
    _mock_refiner(monkeypatch, api, _stationary_refiner)
    result = _run(api, kind, problem)
    assert result.failure_reason is not None
    assert result.unit_vector is None and result.evaluations == ()
    assert result.attempted_paths == 1
    assert result.nfev == (2 if phase == "handoff" else 3)


def test_joint_refit_rejects_incomplete_member_results(monkeypatch):
    api = _api()
    problem = _joint()
    original = api.evaluate_joint_vector

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, local_evaluations=result.local_evaluations[:-1])

    monkeypatch.setattr(api, "evaluate_joint_vector", evaluate)
    _mock_refiner(monkeypatch, api, _stationary_refiner)
    result = _run(api, "joint", problem)
    assert result.failure_reason is not None
    assert result.unit_vector is None and result.evaluations == ()
    assert result.attempted_paths == 1
    assert result.nfev == 2


def test_physical_failure_reason_is_not_replaced_with_generic_success(monkeypatch):
    api = _api()
    problem = _problem()
    original = api.evaluate_vector

    def evaluate(*args, **kwargs):
        return replace(
            original(*args, **kwargs), valid=False, objective=float("inf"), reason="declared_physical_failure"
        )

    monkeypatch.setattr(api, "evaluate_vector", evaluate)
    _mock_refiner(monkeypatch, api, _stationary_refiner)
    result = api.refit_diagnostic_single(problem)
    assert "declared_physical_failure" in result.failure_reason
    assert result.attempted_paths == 1 and result.nfev == 2


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("index", [0, 1], ids=["residual", "jacobian"])
def test_nonfinite_optimizer_callbacks_are_numeric_failure_not_scipy_value_errors(monkeypatch, kind, index):
    """Reject known numeric failures before SciPy's generic ValueError boundary.

    Only the nonfinite numerical value is translated; unrelated ValueError,
    RuntimeError, and TypeError retain the separate propagation tests above.
    """
    api = _api()
    problem = _context(kind)
    name = "least_squares_system" if kind == "single" else "joint_least_squares_system"
    original = getattr(api, name)

    def system(*args):
        result = original(*args)
        result[index][:] = np.nan
        return result

    monkeypatch.setattr(api, name, system)
    result = _run(api, kind, problem)
    assert "nonfinite" in result.failure_reason
    assert result.unit_vector is None and result.evaluations == ()
    assert result.attempted_paths == 1 and result.nfev == 1


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("minimum", [1, 3])
def test_real_budget_exhaustion_cannot_publish_a_converged_refit(kind, minimum):
    api = _api()
    result = _run(api, kind, _context(kind, minimum=minimum, per_parameter=1))
    prefix = "diagnostic_refit_insufficient_two_stage_budget" if minimum == 1 else "diagnostic_refit_nonconverged"
    assert result.failure_reason.startswith(prefix)
    assert result.attempted_paths == 1 and result.nfev == (0 if minimum == 1 else 3)
    assert result.unit_vector is None and result.evaluations == ()


@pytest.mark.parametrize("kind", ["single", "joint"])
def test_declared_physical_encoding_failure_does_not_claim_an_optimization_path(monkeypatch, kind):
    api = _api()
    problem = _context(kind)

    def encode(*_args):
        raise EvaluationConstraintError("declared_constraint_failure")

    name = "encode_physical_vector" if kind == "single" else "initial_joint_vector"
    monkeypatch.setattr(api, name, encode)
    result = _run(api, kind, problem)
    assert "declared_constraint_failure" in result.failure_reason
    assert result.attempted_paths == 0 and result.nfev == 0
    assert result.unit_vector is None and result.evaluations == ()


def test_joint_compile_programming_failure_is_not_swallowed(monkeypatch):
    api = _api()
    problem = _joint()
    error = ValueError("malformed sharing declarations")

    def compile_joint(*_args):
        raise error

    monkeypatch.setattr(api, "compile_joint_problem", compile_joint)
    with pytest.raises(ValueError) as caught:
        _run(api, "joint", problem)
    assert caught.value is error


@pytest.mark.parametrize("kind", ["single", "joint"])
def test_cancelled_locked_full_evaluation_is_not_publishable(monkeypatch, kind):
    api = _api()
    problem = _context(kind, locked=True)
    state = SimpleNamespace(cancelled=False)
    name = "evaluate_vector" if kind == "single" else "evaluate_joint_vector"
    original = getattr(api, name)

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        state.cancelled = True
        return result

    monkeypatch.setattr(api, name, evaluate)
    with pytest.raises(SearchCancelled):
        _run(api, kind, problem, cancelled=lambda: state.cancelled)
