from __future__ import annotations

from dataclasses import replace
from typing import get_type_hints

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec


def test_model_evaluation_recomputes_qz_and_shared_periodic_layers() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="first", thickness_a=24.0),
            replace(film, name="second", thickness_a=36.0),
        ),
        repeats=4,
        top_roughness_a=1.5,
    )
    problem = compile_fit_problem(
        prepared_data(size=80),
        StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=17), scale_prior_enabled=False),
    )
    physical = {
        definition.name: definition.initial for definition in problem.parameter_definitions
    }
    physical.update(
        {
            "component.0.layer.0.thickness_a": 27.0,
            "component.0.layer.1.thickness_a": 43.0,
            "instrument.angle_offset_deg": 0.025,
        }
    )
    unit = encode_physical_vector(problem, physical)

    evaluation = evaluate_vector(problem, unit)

    expected_qz = (
        4.0
        * np.pi
        * np.sin(
            np.deg2rad(problem.data.two_theta_deg / 2.0 + physical["instrument.angle_offset_deg"])
        )
        / problem.data.beam.effective_wavelength_a
    )
    assert evaluation.valid
    np.testing.assert_allclose(evaluation.qz_a_inv, expected_qz)
    assert evaluation.expanded_stack is not None
    np.testing.assert_allclose(
        evaluation.expanded_stack.thickness_a[1:-1],
        np.tile([27.0, 43.0], 4),
    )


def test_shared_solver_primitives_match_the_compiled_objective() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=18), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.55)
    residual = evaluation.least_squares_residual(problem, unit)
    jacobian = evaluation.least_squares_residual_jacobian(problem, unit)
    rho = evaluation.least_squares_loss(problem)(residual**2)

    assert jacobian.shape == (residual.size, unit.size)
    optimizer_objective = 0.5 * float(np.sum(rho[0])) / residual.size
    assert optimizer_objective == pytest.approx(
        evaluate_vector(problem, unit).objective,
        rel=1e-12,
        abs=1e-14,
    )


def test_shared_problem_log_probability_uses_the_soft_l1_data_likelihood() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=19), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.45)
    residual = evaluation.least_squares_residual(problem, unit)
    weights = problem.weights[problem.data.fit_mask]
    c = problem.config.c_decades
    expected = -float(
        np.sum(weights**2 * 2.0 * c**2 * (np.sqrt(1.0 + (residual / c) ** 2) - 1.0))
    ) / (2.0 * c**2)

    assert evaluation.problem_log_probability(problem, unit) == expected


@pytest.mark.parametrize(
    "name",
    (
        "values_and_jacobians",
        "values_by_name",
        "encode_physical_vector",
        "evaluate_model",
        "expanded_structure_jacobian",
        "evaluate_model_jacobian",
        "least_squares_residual",
        "least_squares_residual_jacobian",
        "least_squares_loss",
        "problem_log_probability",
    ),
)
def test_public_evaluation_entries_require_the_typed_context(name: str) -> None:
    hints = get_type_hints(getattr(evaluation, name))

    assert hints["problem"] is FitEvaluationContext


def test_analytic_stack_roughness_failure_is_a_candidate_constraint() -> None:
    structure = simple_structure()
    layer = replace(
        structure.components[0],
        thickness_a=2.0,
        roughness_a=1.0,
    )
    structure = replace(structure, components=(layer,))
    config = replace(FitConfig.fast(20), scale_prior_enabled=False)
    initial = compile_fit_problem(
        prepared_data(size=40),
        structure,
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    locked = {"component.0.thickness_a", "component.0.roughness_a"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial if definition.name in locked else definition.lower,
            definition.initial if definition.name in locked else definition.upper,
            locked=definition.name in locked or definition.locked,
        )
        for definition in initial.parameter_definitions
    )
    problem = compile_fit_problem(
        initial.data,
        structure,
        initial.instrument,
        config,
        settings,
    )
    unit = np.full(len(problem.variables), 0.5)

    with pytest.raises(EvaluationConstraintError, match="constraint_violation"):
        evaluation.evaluate_model_jacobian(problem, unit)

    jacobian = evaluation.least_squares_residual_jacobian(problem, unit)
    np.testing.assert_array_equal(
        jacobian,
        np.zeros((np.count_nonzero(problem.data.fit_mask), len(problem.variables))),
    )
