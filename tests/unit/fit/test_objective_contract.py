from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.analysis.derivatives import objective_gradient
from xrr_fitter.evaluation import (
    encode_physical_vector,
    evaluate_model,
    least_squares_residual,
    least_squares_system,
    problem_log_probability,
    robust_log_cost,
    robust_loss_rho,
)
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.fit.stages import compile_coarse_problem
from xrr_fitter.io.project_codec import ProjectSchemaError, ProjectVersionError, project_from_dict, project_to_dict
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ConstraintNode, ConstraintRule, ParameterReference
from xrr_fitter.model.project import XrrProject


def _plateau_problem(*, constrained: bool = False):
    data = prepared_data(size=1200, two_theta_deg=np.linspace(0.02, 4.0, 1200))
    structure = simple_structure()
    instrument = InstrumentSpec(footprint_mode="none")
    config = FitConfig.fast(17)
    initial = compile_fit_problem(data, structure, instrument, replace(config, scale_prior_enabled=False))
    signal = evaluate_model(initial, encode_physical_vector(initial, {})).model_normalized
    data = replace(data, intensity_raw=signal, intensity_normalized=signal, normalization=1.0)
    rules = ()
    if constrained:
        rules = (
            ConstraintRule(
                ParameterReference("curve", "instrument.scale"),
                ConstraintNode(
                    "div",
                    operands=(
                        ConstraintNode("ref", reference=ParameterReference("curve", "component.0.thickness_a")),
                        ConstraintNode("const", value=20.0),
                    ),
                ),
            ),
        )
    return compile_fit_problem(data, structure, instrument, config, constraint_rules=rules)


def test_robust_cost_is_dimensionless_per_point_total() -> None:
    residual = np.array([0.0, 0.04, -0.2])
    weights = np.array([1.0, 2.0, 0.5])
    expected = np.mean(2 * weights**2 * (np.sqrt(1 + (residual / 0.05) ** 2) - 1))
    assert robust_log_cost(residual, weights, 0.05) == pytest.approx(expected)


def test_large_robust_scale_preserves_representable_subnormal_cost() -> None:
    residual = np.array([1e154])
    expected = (residual[0] / 1e308) ** 2
    assert robust_log_cost(residual, np.ones(1), 1e308) == pytest.approx(expected, rel=1e-14, abs=0.0)
    rho = robust_loss_rho(residual**2, np.ones(1), 1e308)
    assert rho[0, 0] == pytest.approx(2 * expected, rel=1e-14, abs=0.0)


def test_robust_rho_derivatives_match_the_dimensionless_value() -> None:
    squared = np.array([0.01, 0.04, 0.09])
    weights = np.array([1.0, 1.5, 2.0])
    step = 1e-6
    rho = robust_loss_rho(squared, weights, 0.05)
    plus = robust_loss_rho(squared + step, weights, 0.05)
    minus = robust_loss_rho(squared - step, weights, 0.05)
    np.testing.assert_allclose(rho[1:], (plus[:2] - minus[:2]) / (2 * step), rtol=1e-8)


def test_prior_and_data_share_the_fit_and_sampling_target() -> None:
    problem = _plateau_problem()
    assert problem.scale_prior_center is not None
    unit = encode_physical_vector(problem, {"instrument.scale": 1.5})
    evaluation = evaluate_model(problem, unit)
    assert evaluation.valid
    assert problem_log_probability(problem, unit) == pytest.approx(-evaluation.objective * 1200 / 2)


def test_constrained_scale_prior_jacobian_uses_the_complete_chain() -> None:
    problem = _plateau_problem(constrained=True)
    assert problem.scale_prior_center is not None
    unit = encode_physical_vector(problem, {"component.0.thickness_a": 30.0})
    index = next(i for i, variable in enumerate(problem.variables) if variable.name == "component.0.thickness_a")
    plus, minus = unit.copy(), unit.copy()
    plus[index] += 1e-6
    minus[index] -= 1e-6
    finite_difference = (least_squares_residual(problem, plus)[-1] - least_squares_residual(problem, minus)[-1]) / 2e-6
    _, jacobian = least_squares_system(problem, unit)
    assert abs(finite_difference) > 1
    assert jacobian[-1, index] == pytest.approx(finite_difference, rel=1e-7)


def test_constrained_scale_objective_gradient_matches_finite_difference() -> None:
    problem = _plateau_problem(constrained=True)
    unit = encode_physical_vector(problem, {"component.0.thickness_a": 30.0})
    index = next(i for i, variable in enumerate(problem.variables) if variable.name == "component.0.thickness_a")
    plus, minus = unit.copy(), unit.copy()
    plus[index] += 1e-6
    minus[index] -= 1e-6
    expected = (evaluate_model(problem, plus).objective - evaluate_model(problem, minus).objective) / 2e-6
    assert objective_gradient(problem, unit)[index] == pytest.approx(expected, rel=1e-6)


def test_coarse_grid_freezes_complete_data_prior_evidence() -> None:
    full = _plateau_problem()
    assert full.scale_prior_center is not None
    coarse = compile_coarse_problem(full)
    assert coarse.data.fit_mask.size < full.data.fit_mask.size
    assert coarse.scale_prior_center == full.scale_prior_center
    assert coarse.scale_prior_tau_decades == full.scale_prior_tau_decades
    assert coarse.scale_prior_reason == full.scale_prior_reason
    assert coarse.objective_point_count == 1200
    assert np.sum(coarse.sampling_multipliers[coarse.data.fit_mask]) == pytest.approx(1200)


def test_current_project_declares_only_the_v2_algorithm() -> None:
    project = XrrProject.new((), master_seed=5)
    assert project.schema_version == 5
    assert project.algorithm_version == "xrr-fit-v2-poisson-5"
    assert project.fit_config.objective_version == "2"


@pytest.mark.parametrize("schema", [1, 2])
def test_old_project_schemas_are_explicitly_rejected(schema: int) -> None:
    payload = project_to_dict(XrrProject.new((), master_seed=5))
    payload["schema_version"] = schema
    with pytest.raises(ProjectVersionError, match="unsupported project schema"):
        project_from_dict(payload)


def test_v2_project_rejects_the_old_objective_identity() -> None:
    payload = project_to_dict(XrrProject.new((), master_seed=5))
    payload["fit_config"]["objective_version"] = "1"
    with pytest.raises(ProjectSchemaError, match="unsupported objective"):
        project_from_dict(payload)


def test_joint_solver_and_scalar_target_share_v2_data_and_unbalanced_priors() -> None:
    from tests.unit.fit.joint_evaluation_cases import _joint

    from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector, joint_least_squares_loss

    problem = _joint(scale_prior=True)
    unit = np.full(len(problem.global_variables), 0.54)
    evaluation = evaluate_joint_vector(problem, unit)
    count = sum(member.objective_point_count for member in problem.problems)
    solver_total = np.sum(joint_least_squares_loss(problem)(evaluation.residuals**2)[0]) / 2
    assert solver_total / count == pytest.approx(evaluation.objective, rel=1e-12)


def test_current_project_rejects_the_previous_poisson_refinement_identity() -> None:
    payload = project_to_dict(XrrProject.new((), master_seed=5))
    payload["algorithm_version"] = "xrr-fit-v2-poisson-1"
    with pytest.raises(ProjectSchemaError, match="unsupported algorithm_version"):
        project_from_dict(payload)
