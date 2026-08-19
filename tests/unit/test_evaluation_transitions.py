from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterDefinition, ParameterSetting
from xrr_fitter.model.structure import InterfaceTransition, LayerSpec, StructureSpec, TransitionBranch
from xrr_fitter.physics.geometry import expand_geometry, expand_structure_with_jacobian
from xrr_fitter.physics.stack import expand_structure, rebuild_structure


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


def transition_problem() -> FitEvaluationContext:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    layer = replace(
        film,
        thickness_a=30.0,
        roughness_a=0.0,
        transition=InterfaceTransition(
            branches=(TransitionBranch(kind="erf", weight=1.0, thickness_a=12.0),),
            microslab_max_a=3.0,
        ),
    )
    structure = StructureSpec(
        base.fronting,
        (layer,),
        base.backing,
        backing_roughness_a=3.0,
    )
    config = replace(FitConfig.fast(master_seed=91), scale_prior_enabled=False)
    initial = compile_fit_problem(
        prepared_data(size=48),
        structure,
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    bounds = {"component.0.thickness_a": (20.0, 45.0)}
    settings = tuple(_transition_setting(definition, bounds) for definition in initial.parameter_definitions)
    return compile_fit_problem(
        initial.data,
        structure,
        initial.instrument,
        config,
        settings,
    )


def _transition_setting(
    definition: ParameterDefinition,
    bounds: dict[str, tuple[float, float]],
) -> ParameterSetting:
    """Lock every roughness to zero and pin thickness to a log-symmetric range."""
    if definition.name.endswith("roughness_a"):
        return ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial,
            definition.initial,
            locked=True,
        )
    lower, upper = bounds.get(definition.name, (definition.lower, definition.upper))
    return ParameterSetting(
        definition.name,
        definition.initial,
        lower,
        upper,
        locked=definition.locked,
    )


@pytest.mark.parametrize(
    "name",
    ["component.0.thickness_a", "component.0.density_scale"],
)
def test_transition_jacobian_matches_finite_differences(name: str) -> None:
    problem = transition_problem()
    index = next(position for position, coordinate in enumerate(problem.variables) if coordinate.name == name)
    unit = np.full(len(problem.variables), 0.5)
    stack = evaluation.expanded_structure_jacobian(problem, unit)
    assert stack.stack.thickness_a.size == 7

    step = 1e-6
    forward = unit.copy()
    forward[index] += step
    backward = unit.copy()
    backward[index] -= step
    high = evaluation.expanded_structure_jacobian(problem, forward).stack
    low = evaluation.expanded_structure_jacobian(problem, backward).stack

    np.testing.assert_allclose(
        stack.thickness_jacobian[:, index],
        (high.thickness_a - low.thickness_a) / (2.0 * step),
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        stack.sld_jacobian[:, index],
        (high.sld_a2 - low.sld_a2) / (2.0 * step),
        rtol=1e-6,
        atol=1e-16,
    )


def test_transition_expansion_aligns_across_all_three_paths() -> None:
    problem = transition_problem()
    unit = np.full(len(problem.variables), 0.5)
    expected_media = 7  # fronting + four transition slabs + layer body + backing
    values, value_jacobians = evaluation.values_and_jacobians(problem, unit)
    rebuilt = rebuild_structure(problem.structure, values)
    wavelength = problem.data.beam.effective_wavelength_a

    primal = expand_structure(rebuilt, wavelength)
    differentiable = expand_structure_with_jacobian(
        problem.structure,
        values,
        value_jacobians,
        wavelength,
        len(problem.variables),
    )
    geometry = expand_geometry(rebuilt, len(problem.variables), value_jacobians)

    assert primal.thickness_a.size == expected_media
    np.testing.assert_array_equal(primal.thickness_a, differentiable.stack.thickness_a)
    np.testing.assert_array_equal(primal.thickness_a, geometry.thickness_a)
    assert differentiable.thickness_jacobian.shape[0] == expected_media
    assert geometry.thickness_jacobian is not None
    assert geometry.thickness_jacobian.shape[0] == expected_media
    assert len(geometry.interface_names) == primal.thickness_a.size - 1


def test_angle_layout_and_analytic_path_reject_incident_angles_above_ninety() -> None:
    data = prepared_data(
        size=48,
        two_theta_deg=np.r_[np.linspace(0.1, 3.9, 40), np.linspace(180.1, 180.5, 8)],
    )
    data = replace(
        data,
        validation_mask=np.ones(48, dtype=bool),
        fit_mask=np.ones(48, dtype=bool),
        fit_ready=True,
    )
    problem = compile_fit_problem(
        data,
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=32), scale_prior_enabled=False),
    )
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical["instrument.angle_offset_deg"] = 0.1
    unit = encode_physical_vector(problem, physical)

    _theta, model_mask, _indices, _qz = evaluation._angle_layout(
        problem,
        {"instrument.angle_offset_deg": physical["instrument.angle_offset_deg"]},
    )

    assert np.all(model_mask[:40])
    assert not np.any(model_mask[40:])
    with pytest.raises(ValueError, match="differentiate nonpositive"):
        evaluation.evaluate_model_jacobian(problem, unit)
