"""Joint sampling refits every member through one shared coordinate system."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from tests.unit.analysis.covariance_cases import joint_scale_searches

from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_solvers import solve_joint
from xrr_fitter.fit.problem import recompile_resampled_problem


def _scale(evaluation):
    return next(value.value for value in evaluation.parameters if value.name == "instrument.scale")


@pytest.mark.parametrize("mode", ["gaussian", "poisson", "robust_log"])
def test_joint_bootstrap_generates_all_members_then_refits_the_shared_problem(mode) -> None:
    module = import_module("xrr_fitter.analysis.joint_bootstrap")
    problem, searches = joint_scale_searches(mode)
    start = searches[0].best_candidate.unit_vector
    evaluation = evaluate_joint_vector(problem, start)
    assert evaluation.valid
    prepared = []
    fitted = []

    def recompile(local, data):
        prepared.append(data)
        return recompile_resampled_problem(local, data)

    def refit(members):
        assert len(prepared) == 8 * 2
        assert len(members) == 2
        joint = compile_joint_problem(problem.dataset_ids, members, problem.sharing_rules, problem.constraint_rules)
        fitted.append(joint)
        solved = solve_joint(joint, start, 200, None)
        assert solved.evaluation.valid
        scale = _scale(solved.evaluation.local_evaluations[0])
        other = _scale(solved.evaluation.local_evaluations[1])
        assert scale == other
        return np.array([scale])

    result = module.bootstrap_joint_local(
        problem,
        tuple(search.best_candidate for search in searches),
        start,
        sample_count=8,
        child_seed=41,
        recompile=recompile,
        refit=refit,
    )
    assert len(fitted) == 8
    assert result.samples.shape == (8, 1)
    assert np.std(result.samples[:, 0]) > 0
    assert result.confidence_level is None
    assert result.method.startswith("joint_")


def test_joint_service_records_requested_bootstrap_without_enabling_it_by_default() -> None:
    fitting = import_module("xrr_fitter.services.fitting")
    problem, searches = joint_scale_searches("gaussian")
    default = fitting._analyze_joint_searches(problem, searches, ((), ()))[0]
    assert default.uncertainty.bootstrap_performed is False
    import inspect

    assert "bootstrap_enabled" in inspect.signature(fitting._analyze_joint_searches).parameters
    results = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)
    report = results[0].uncertainty
    assert results[1].uncertainty is report
    assert report.bootstrap_performed is True
    assert report.bootstrap_evidence.successful_samples == 8
    assert report.bootstrap_evidence.method == "joint_gaussian_parametric"
    assert report.bootstrap_evidence.confidence_level is None


def test_joint_service_emits_bootstrap_attempt_progress() -> None:
    fitting = import_module("xrr_fitter.services.fitting")
    problem, searches = joint_scale_searches("gaussian")
    events = []
    fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True, progress=events.append)
    assert [(event.stage, event.completed, event.total) for event in events] == [
        ("bootstrap", count, 8) for count in range(9)
    ]


def test_two_hundred_real_joint_gaussian_refits_publish_calibrated_percentiles() -> None:
    fitting = import_module("xrr_fitter.services.fitting")
    problem, searches = joint_scale_searches("gaussian")
    members = tuple(
        replace(local, config=replace(local.config, budget=replace(local.config.budget, bootstrap_samples=200)))
        for local in problem.problems
    )
    problem = compile_joint_problem(problem.dataset_ids, members, problem.sharing_rules, problem.constraint_rules)
    report = fitting._analyze_joint_searches(problem, searches, ((), ()), bootstrap_enabled=True)[0].uncertainty
    bootstrap = report.bootstrap_evidence
    assert bootstrap.successful_samples == 200
    assert bootstrap.confidence_level == 0.95
    assert bootstrap.interval_kind == "percentile_bootstrap"
    assert np.std(bootstrap.samples[:, 0], ddof=1) == pytest.approx(report.parameter_sigma[0], rel=0.2)


def test_joint_bootstrap_rejects_residuals_from_a_different_noise_model() -> None:
    module = import_module("xrr_fitter.analysis.joint_bootstrap")
    problem, searches = joint_scale_searches("gaussian")
    evaluation = evaluate_joint_vector(problem, searches[0].best_candidate.unit_vector)
    assert evaluation.valid
    members = tuple(replace(search.best_candidate, noise_model="robust_log") for search in searches)
    with pytest.raises(ValueError, match="noise model"):
        module.bootstrap_joint_local(
            problem,
            members,
            searches[0].best_candidate.unit_vector,
            sample_count=1,
            child_seed=41,
            recompile=recompile_resampled_problem,
            refit=lambda _members: np.array([0.5]),
        )


def test_joint_bootstrap_cancellation_during_last_refit_cannot_publish() -> None:
    module = import_module("xrr_fitter.analysis.joint_bootstrap")
    problem, searches = joint_scale_searches("gaussian")
    evaluation = evaluate_joint_vector(problem, searches[0].best_candidate.unit_vector)
    assert evaluation.valid
    cancelled = False

    def refit(_members):
        nonlocal cancelled
        cancelled = True
        return np.array([0.5])

    with pytest.raises(InterruptedError, match="cancelled"):
        module.bootstrap_joint_local(
            problem,
            tuple(search.best_candidate for search in searches),
            searches[0].best_candidate.unit_vector,
            sample_count=1,
            child_seed=41,
            recompile=recompile_resampled_problem,
            refit=refit,
            cancelled=lambda: cancelled,
        )
