"""Compile-time locks on roughness axes owned by an interface transition.

A transition already sets the interface width by microslab blending, so leaving
the Névot-Croce roughness free would broaden the same interface twice. The
compiled snapshot must therefore pin that axis at zero, refuse settings that
reopen it, and carry the lock through staged recompilation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.local_search import local_jacobian, solve_local
from xrr_fitter.fit.problem import compile_fit_problem, compile_stage_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterDefinition, ParameterSetting
from xrr_fitter.model.structure import (
    InterfaceTransition,
    LayerSpec,
    StructureSpec,
    TransitionBranch,
)


def _problem(structure: StructureSpec, settings: tuple[ParameterSetting, ...] = ()):
    return compile_fit_problem(
        prepared_data(size=72),
        structure,
        InstrumentSpec(footprint_mode="fit"),
        replace(FitConfig.fast(master_seed=11), scale_prior_enabled=False),
        settings,
    )


def _transition_structure() -> StructureSpec:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    graded = replace(
        film,
        roughness_a=0.0,
        transition=InterfaceTransition((TransitionBranch("erf", 1.0, 8.0),), microslab_max_a=2.0),
    )
    return replace(base, components=(graded,))


def _definition(problem, name: str) -> ParameterDefinition:
    return next(item for item in problem.parameter_definitions if item.name == name)


def _initial_values(problem) -> dict[str, float]:
    return {item.name: item.initial for item in problem.parameter_definitions}


def test_transition_layer_roughness_is_compiled_as_locked_zero() -> None:
    problem = _problem(_transition_structure())

    definition = _definition(problem, "component.0.roughness_a")

    assert definition.locked is True
    assert definition.initial == 0.0
    assert definition.lower == 0.0
    assert definition.upper == 0.0


def test_transition_layer_roughness_is_not_a_free_variable() -> None:
    problem = _problem(_transition_structure())

    names = {coordinate.name for coordinate in problem.variables}

    assert "component.0.roughness_a" not in names


def test_layer_without_transition_keeps_free_roughness() -> None:
    problem = _problem(simple_structure())

    definition = _definition(problem, "component.0.roughness_a")
    thickness = _definition(problem, "component.0.thickness_a")

    assert definition.locked is False
    assert definition.lower == 0.0
    assert definition.upper == max(50.0, 0.49 * thickness.initial)


def test_transition_thickness_lower_bound_covers_declared_width() -> None:
    problem = _problem(_transition_structure())

    definition = _definition(problem, "component.0.thickness_a")

    assert definition.lower == 8.0


def test_transition_thickness_setting_cannot_reopen_values_below_width() -> None:
    settings = (ParameterSetting("component.0.thickness_a", 20.0, 2.0, 45.0, locked=False),)

    with pytest.raises(ValueError, match="过渡.*厚度"):
        _problem(_transition_structure(), settings)


def _transition_lower_bound_unit(problem) -> np.ndarray:
    unit = np.array(encode_physical_vector(problem, {}), copy=True)
    index = next(
        index for index, coordinate in enumerate(problem.variables) if coordinate.name == "component.0.thickness_a"
    )
    unit[index] = 0.0
    return unit


def test_transition_jacobian_accepts_thickness_at_declared_width() -> None:
    problem = _problem(_transition_structure())
    unit = _transition_lower_bound_unit(problem)

    jacobian = local_jacobian(problem, unit)

    assert jacobian.shape == (np.count_nonzero(problem.data.fit_mask), len(problem.variables))
    assert np.all(np.isfinite(jacobian))


def test_transition_local_solver_accepts_thickness_at_declared_width() -> None:
    problem = _problem(_transition_structure())
    unit = _transition_lower_bound_unit(problem)

    result = solve_local(problem, unit, max_nfev=2)

    assert result.evaluation.valid


def test_unlocking_transition_roughness_is_rejected_at_compile_time() -> None:
    settings = (ParameterSetting("component.0.roughness_a", 3.0, 0.0, 10.0, locked=False),)

    with pytest.raises(ValueError, match="过渡"):
        _problem(_transition_structure(), settings)


def test_locking_transition_roughness_at_nonzero_is_rejected() -> None:
    settings = (ParameterSetting("component.0.roughness_a", 3.0, 3.0, 3.0, locked=True),)

    with pytest.raises(ValueError, match="过渡"):
        _problem(_transition_structure(), settings)


@pytest.mark.parametrize("stage", ("B", "D", "E"))
def test_stage_compilation_preserves_the_transition_lock(stage: str) -> None:
    problem = _problem(_transition_structure())

    staged = compile_stage_problem(problem, stage, _initial_values(problem))

    definition = _definition(staged, "component.0.roughness_a")
    assert definition.locked is True
    assert definition.initial == 0.0
