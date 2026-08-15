"""Shared builders for cross-dataset expression-constraint tests."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterReference,
    ParameterSetting,
)

SHARED_NAME = "component.0.density_scale"


def local_problem(*, seed: int, size: int, scale_prior: bool = False):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=scale_prior,
    )
    base = compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none", instrument_id="shared-lab"),
        config,
    )
    free_names = {SHARED_NAME, "instrument.scale"} if scale_prior else {SHARED_NAME}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name in free_names else definition.initial,
            definition.upper if definition.name in free_names else definition.initial,
            locked=definition.name not in free_names,
        )
        for definition in base.parameter_definitions
    )
    compiled = compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        settings,
    )
    return replace(compiled, scale_prior_center=1.0) if scale_prior else compiled


def cross_constraint_joint(*, multiplier: float = 1.0):
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    rule = ConstraintRule(
        ParameterReference("right", SHARED_NAME),
        ConstraintNode(
            "mul",
            operands=(
                ConstraintNode(
                    "ref",
                    reference=ParameterReference("left", SHARED_NAME),
                ),
                ConstraintNode("const", value=multiplier),
            ),
        ),
    )
    return joint_api.compile_joint_problem(
        ("left", "right"),
        (
            local_problem(seed=854, size=40),
            local_problem(seed=854, size=52),
        ),
        (),
        (rule,),
    )


def cross_constraint_chain_joint():
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    left = local_problem(seed=856, size=40, scale_prior=True)
    right = local_problem(seed=856, size=52, scale_prior=True)
    cross = ConstraintRule(
        ParameterReference("right", SHARED_NAME),
        ConstraintNode(
            "ref",
            reference=ParameterReference("left", SHARED_NAME),
        ),
    )
    dependent = ConstraintRule(
        ParameterReference("right", "instrument.scale"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("right", SHARED_NAME),
        ),
    )
    right = compile_fit_problem(
        right.data,
        right.structure,
        right.instrument,
        right.config,
        constraint_rules=(dependent,),
    )
    return joint_api.compile_joint_problem(
        ("left", "right"),
        (left, right),
        (),
        (cross, dependent),
    )


def _roughness_problem(*, seed: int, size: int):
    base = local_problem(seed=seed, size=size)
    free_bounds = {
        "component.0.thickness_a": (10.0, 30.0),
        "component.0.roughness_a": (0.0, 9.0),
    }
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            free_bounds.get(definition.name, (definition.initial, definition.initial))[0],
            free_bounds.get(definition.name, (definition.initial, definition.initial))[1],
            locked=definition.name not in free_bounds,
        )
        for definition in base.parameter_definitions
    )
    return compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        settings,
    )


def cross_roughness_constraint_joint():
    joint_api = import_module("xrr_fitter.fit.joint_problem")
    rule = ConstraintRule(
        ParameterReference("right", "component.0.roughness_a"),
        ConstraintNode(
            "ref",
            reference=ParameterReference("left", "component.0.roughness_a"),
        ),
    )
    return joint_api.compile_joint_problem(
        ("left", "right"),
        (
            _roughness_problem(seed=857, size=40),
            _roughness_problem(seed=857, size=52),
        ),
        (),
        (rule,),
    )


__all__ = [
    "SHARED_NAME",
    "cross_constraint_chain_joint",
    "cross_constraint_joint",
    "cross_roughness_constraint_joint",
    "local_problem",
]
