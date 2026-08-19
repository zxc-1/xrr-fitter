from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import MAX_EXPANDED_SLABS, GradientLayerSpec


def _variable_index(problem: object, name: str) -> int:
    return next(index for index, coordinate in enumerate(problem.variables) if coordinate.name == name)


def test_gradient_bounds_keep_every_candidate_within_expanded_slab_budget() -> None:
    base = simple_structure()
    structure = replace(
        base,
        components=(
            GradientLayerSpec(
                "ramp",
                1e-5 + 1e-7j,
                3e-5 + 3e-7j,
                20.0,
                microslab_max_a=0.1,
            ),
        ),
        backing_roughness_a=0.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=40),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(18), scale_prior_enabled=False),
    )
    unit = np.array(encode_physical_vector(problem, {}), copy=True)
    thickness_index = _variable_index(problem, "component.0.thickness_a")
    unit[thickness_index] = 1.0

    result = evaluate_vector(problem, unit)

    assert result.valid is True
    assert result.expanded_stack is not None
    assert result.expanded_stack.thickness_a.size <= MAX_EXPANDED_SLABS + 2


def test_gradient_topology_and_reflectivity_are_continuous_across_ceil_threshold() -> None:
    base = simple_structure()
    structure = replace(
        base,
        components=(
            GradientLayerSpec(
                "ramp",
                1e-5 + 1e-7j,
                3e-5 + 3e-7j,
                20.0,
                microslab_max_a=10.0,
            ),
        ),
        backing_roughness_a=0.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=48),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(181), scale_prior_enabled=False),
    )

    stacks = []
    models = []
    for thickness in (20.0 - 1e-8, 20.0 + 1e-8):
        unit = encode_physical_vector(problem, {"component.0.thickness_a": thickness})
        stacks.append(evaluation.expanded_structure_jacobian(problem, unit).stack)
        result = evaluation.evaluate_model(problem, unit)
        assert result.valid
        models.append(result.model_normalized)

    assert stacks[0].thickness_a.shape == stacks[1].thickness_a.shape
    np.testing.assert_allclose(stacks[0].sld_a2, stacks[1].sld_a2, rtol=0.0, atol=0.0)
    assert np.nanmax(np.abs(models[1] - models[0])) < 1e-8


def test_multiple_gradient_bounds_share_the_global_slab_budget() -> None:
    base = simple_structure()
    gradients = tuple(
        GradientLayerSpec(
            f"ramp-{index}",
            1e-5 + 1e-7j,
            3e-5 + 3e-7j,
            100.0,
            microslab_max_a=0.1,
        )
        for index in range(2)
    )
    problem = compile_fit_problem(
        prepared_data(size=40),
        replace(base, components=gradients, backing_roughness_a=0.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(182), scale_prior_enabled=False),
    )
    definitions = {definition.name: definition for definition in problem.parameter_definitions}
    total = sum(
        int(np.ceil(definitions[f"component.{index}.thickness_a"].upper / gradient.microslab_max_a))
        for index, gradient in enumerate(gradients)
    )

    assert total <= MAX_EXPANDED_SLABS


def test_gradient_roughness_uses_parent_thickness_limit_and_tangent() -> None:
    base = simple_structure()
    structure = replace(
        base,
        components=(
            GradientLayerSpec(
                "ramp",
                1e-5 + 1e-7j,
                3e-5 + 3e-7j,
                20.0,
                roughness_a=3.0,
                microslab_max_a=2.0,
            ),
        ),
        backing_roughness_a=3.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=48),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=18), scale_prior_enabled=False),
    )

    unit = np.array(encode_physical_vector(problem, {}), copy=True)
    dynamic = evaluation.roughness_dynamic_uppers(problem, unit)
    values, jacobians = evaluation.values_and_jacobians(problem, unit)
    expected_upper = np.nextafter(0.49 * 20.0, 0.0)

    assert dynamic["component.0.roughness_a"] == pytest.approx(expected_upper)
    assert dynamic["backing.roughness_a"] == pytest.approx(expected_upper)
    assert values["component.0.roughness_a"] == pytest.approx(3.0)
    assert values["backing.roughness_a"] == pytest.approx(3.0)

    thickness_index = _variable_index(problem, "component.0.thickness_a")
    step = 1e-6
    forward = unit.copy()
    backward = unit.copy()
    forward[thickness_index] += step
    backward[thickness_index] -= step
    high = evaluation.values_by_name(problem, forward)
    low = evaluation.values_by_name(problem, backward)
    for name in ("component.0.roughness_a", "backing.roughness_a"):
        finite_difference = (high[name] - low[name]) / (2.0 * step)
        assert jacobians[name][thickness_index] == pytest.approx(finite_difference)
