from __future__ import annotations

from importlib import import_module

import numpy as np
from tests.unit.analysis.test_mcmc import (
    _candidate,
    _problem,
    run_problem_mcmc,
)

from xrr_fitter.evaluation import values_by_name
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import McmcConfig
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
    ParameterSetting,
)


def _constrained_problem():
    problem = _problem("component.0.density_scale")
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            definition.locked,
        )
        for definition in problem.parameter_definitions
    )
    rule = ConstraintRule(
        ParameterReference("curve", "component.0.thickness_a"),
        ConstraintNode(
            "mul",
            operands=(
                ConstraintNode(
                    "ref",
                    reference=ParameterReference("curve", "component.0.density_scale"),
                ),
                ConstraintNode("const", value=40.0),
            ),
        ),
    )
    return compile_fit_problem(
        problem.data,
        problem.structure,
        problem.instrument,
        problem.config,
        settings,
        (rule,),
    )


def test_problem_mcmc_maps_local_constraint_targets_into_replay_samples(monkeypatch) -> None:
    module = import_module("xrr_fitter.analysis.mcmc")
    problem = _constrained_problem()
    candidate = _candidate(problem)
    config = McmcConfig(walkers=4, burn_in=0, production_steps=4, thin=2)
    samples_unit = np.asarray(
        (
            ((0.25,), (0.25,), (0.75,), (0.75,)),
            ((0.25,), (0.25,), (0.75,), (0.75,)),
        ),
        dtype=float,
    )
    monkeypatch.setattr(
        module,
        "run_affine_invariant",
        lambda *args, **kwargs: module.EnsembleSamples(
            samples_unit,
            np.zeros((2, 4)),
            np.full(4, 0.5),
            np.ones(1),
            np.full(1, 200.0),
        ),
    )

    report = run_problem_mcmc(problem, candidate, config, child_seed=441)

    assert report.parameter_names == ("component.0.density_scale",)
    assert report.derived_parameter_names == ("component.0.thickness_a",)
    expected = tuple(values_by_name(problem, unit)["component.0.thickness_a"] for unit in samples_unit.reshape(-1, 1))
    np.testing.assert_allclose(report.derived_samples_physical[:, 0], expected)
