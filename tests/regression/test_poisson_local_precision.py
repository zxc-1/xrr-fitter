"""Large irreducible Poisson loss must not hide a resolvable parameter error.

These real seed37 curves are existing development fixtures, never new holdout
observations. All optimizations retain the original four declaration/Sobol
starts, analytical systems, physical constraints and hard local budget. A
shared-thickness duplicate tests the joint numerical boundary without making
any claim of independent statistical evidence.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from xrr_fitter.analysis.residual_calibration import refit_discrepancy
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit import diagnostic_refit, joint_solvers, local_search
from xrr_fitter.fit.diagnostic_refit import diagnostic_starts
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_sharing import initial_joint_vector
from xrr_fitter.fit.joint_solvers import solve_joint
from xrr_fitter.fit.local_search import solve_local
from xrr_fitter.model.parameters import ParameterReference, SharingRule
from xrr_fitter.services.datasets import service_seed_branches
from xrr_fitter.services.fitting import prepare_dataset_fit


def _case(load_tool_module, tmp_path, group, kind):
    tool = load_tool_module("check_poisson_diagnostics")
    fixture = tool.cases.build_observations(tmp_path, group, 37)
    project = fixture.project
    seeds, _, _ = service_seed_branches(project)
    dataset_id = project.datasets[0].dataset_id
    problem = prepare_dataset_fit(project, dataset_id, seeds[dataset_id]).problem
    if kind == "single":
        return problem, (problem,), encode_physical_vector(problem, {})
    sharing = SharingRule(
        "precision-thickness",
        tuple(ParameterReference(name, tool.cases.TARGET) for name in ("left", "right")),
    )
    joint = compile_joint_problem(("left", "right"), (problem, problem), (sharing,))
    return joint, joint.problems, initial_joint_vector(joint)


def _solve(problem, kind, start):
    if kind == "single":
        result = solve_local(problem, start, max_nfev=80)
        values = (result.evaluation,)
    else:
        result = solve_joint(problem, start, 80, None)
        assert result.converged
        values = result.evaluation.local_evaluations
    assert 0 < result.nfev <= 80
    assert all(value.valid for value in values)
    return SimpleNamespace(unit_vector=result.unit_vector, evaluations=values, nfev=result.nfev)


def _record_product_outcomes(monkeypatch, kind):
    solver_module = local_search if kind == "single" else joint_solvers
    original_solver = solver_module.least_squares
    outcomes = []

    def recorded_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        outcomes.append((bool(result.success), str(result.message)))
        return result

    monkeypatch.setattr(solver_module, "least_squares", recorded_solver)
    return outcomes


def _total_score(members, result):
    return sum(
        member.objective_point_count * value.objective
        for member, value in zip(members, result.evaluations, strict=True)
    )


@pytest.mark.parametrize("group", ["footprint", "surface"])
@pytest.mark.parametrize("kind", ["single", "joint"])
def test_poisson_local_starts_agree_despite_large_irreducible_cost(
    load_tool_module, tmp_path, monkeypatch, group, kind
):
    problem, members, initial = _case(load_tool_module, tmp_path, group, kind)
    outcomes = _record_product_outcomes(monkeypatch, kind)
    solutions = tuple(_solve(problem, kind, start) for start in diagnostic_starts(initial))
    assert len(outcomes) == 4
    assert all(success for success, _message in outcomes), outcomes
    best = min(solutions, key=lambda value: _total_score(members, value))
    assert _total_score(members, best) > 1e6
    for first, second in combinations(solutions, 2):
        discrepancy = refit_discrepancy(members, first.unit_vector, first.evaluations, second)
        assert discrepancy is not None
        assert np.all(np.asarray(discrepancy) <= (1e-4, 1e-6, 1e-6)), discrepancy


@pytest.mark.parametrize(
    "modes",
    [
        ("robust_log",),
        ("gaussian",),
        ("poisson",),
        ("robust_log", "gaussian"),
        ("gaussian", "poisson"),
        ("poisson", "poisson"),
    ],
)
def test_only_poisson_containing_local_objectives_disable_relative_cost_stop(
    load_tool_module, tmp_path, monkeypatch, modes
):
    kind = "single" if len(modes) == 1 else "joint"
    problem, members, initial = _case(load_tool_module, tmp_path, "footprint", kind)
    changed = tuple(
        replace(
            member,
            data=replace(member.data, intensity_sigma_normalized=np.full(member.data.fit_mask.size, 0.02)),
            config=replace(member.config, noise_model=mode),
        )
        for member, mode in zip(members, modes, strict=True)
    )
    problem = (
        changed[0]
        if kind == "single"
        else compile_joint_problem(problem.dataset_ids, changed, problem.sharing_rules, problem.joint_constraint_rules)
    )
    captured = {}

    def inspected_solver(_fun, start, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(x=start, success=True, message="inspected solver options", nfev=1)

    monkeypatch.setattr(local_search if kind == "single" else joint_solvers, "least_squares", inspected_solver)
    _solve(problem, kind, initial)
    expected = None if "poisson" in modes else 1e-10
    assert captured["ftol"] == expected
    assert captured["xtol"] == captured["gtol"] == 1e-10
    assert captured["max_nfev"] == 80


@pytest.mark.parametrize("group", ["footprint", "surface"])
@pytest.mark.parametrize("kind", ["single", "joint"])
def test_poisson_diagnostic_refit_matches_the_polished_product(load_tool_module, tmp_path, monkeypatch, group, kind):
    problem, members, initial = _case(load_tool_module, tmp_path, group, kind)
    outcomes = _record_product_outcomes(monkeypatch, kind)
    solutions = tuple(_solve(problem, kind, start) for start in diagnostic_starts(initial))
    assert len(outcomes) == 4
    assert all(success for success, _message in outcomes), outcomes
    best = min(solutions, key=lambda value: _total_score(members, value))
    observed = (
        diagnostic_refit.refit_diagnostic_single(problem)
        if kind == "single"
        else diagnostic_refit.refit_diagnostic_joint(problem, members)
    )
    assert observed.failure_reason is None
    assert observed.attempted_paths == len(solutions)
    discrepancy = refit_discrepancy(members, best.unit_vector, best.evaluations, observed)
    assert discrepancy is not None
    assert np.all(np.asarray(discrepancy) <= (1e-4, 1e-6, 1e-6)), discrepancy
