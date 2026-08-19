from __future__ import annotations

import warnings
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, ModelEvaluation, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ParameterReference,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.structure import LayerSpec

SHARED_NAME = "component.0.density_scale"

TIE_THICKNESS_NAME = "component.0.thickness_a"

TIE_ROUGHNESS_NAME = "component.1.roughness_a"


def _problem(*, seed: int, size: int, scale_prior: bool = False):
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


def _joint(*, scale_prior: bool = False):
    api = import_module("xrr_fitter.fit.joint_problem")
    rule = SharingRule(
        "film-thickness",
        (
            ParameterReference("left", SHARED_NAME),
            ParameterReference("right", SHARED_NAME),
        ),
    )
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _problem(seed=853, size=40, scale_prior=scale_prior),
            _problem(seed=853, size=52, scale_prior=scale_prior),
        ),
        (rule,),
    )


def _tie_problem(*, seed: int, size: int):
    base_structure = simple_structure()
    film = base_structure.components[0]
    assert isinstance(film, LayerSpec)
    structure = replace(
        base_structure,
        components=(
            replace(film, name="upper", thickness_a=20.0, roughness_a=1.0),
            replace(film, name="lower", thickness_a=20.0, roughness_a=2.0),
        ),
    )
    config = replace(FitConfig.fast(seed), scale_prior_enabled=False)
    base = compile_fit_problem(
        prepared_data(size=size),
        structure,
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    settings = []
    for definition in base.parameter_definitions:
        if definition.name == TIE_THICKNESS_NAME:
            settings.append(ParameterSetting(definition.name, 20.0, 10.0, 40.0))
        elif definition.name == TIE_ROUGHNESS_NAME:
            settings.append(ParameterSetting(definition.name, 2.0, 0.0, 20.0))
        else:
            settings.append(
                ParameterSetting(
                    definition.name,
                    definition.initial,
                    definition.initial,
                    definition.initial,
                    locked=True,
                )
            )
    return compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        tuple(settings),
    )


def _tie_joint():
    api = import_module("xrr_fitter.fit.joint_problem")
    rules = tuple(
        SharingRule(
            sharing_key,
            (
                ParameterReference("left", parameter_name),
                ParameterReference("right", parameter_name),
            ),
        )
        for sharing_key, parameter_name in (
            ("shared-thickness", TIE_THICKNESS_NAME),
            ("shared-roughness", TIE_ROUGHNESS_NAME),
        )
    )
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _tie_problem(seed=859, size=40),
            _tie_problem(seed=859, size=52),
        ),
        rules,
    )


def _unequal_roughness_problem(*, thickness_a: float, seed: int, size: int):
    structure = simple_structure()
    film = structure.components[0]
    assert isinstance(film, LayerSpec)
    structure = replace(
        structure,
        components=(replace(film, thickness_a=thickness_a, roughness_a=3.0),),
    )
    base = compile_fit_problem(
        prepared_data(size=size),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(seed), scale_prior_enabled=False),
    )
    free_names = {"component.0.thickness_a", "component.0.roughness_a"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            (thickness_a * 0.5 if definition.name == "component.0.thickness_a" else definition.lower)
            if definition.name in free_names
            else definition.initial,
            (thickness_a * 1.5 if definition.name == "component.0.thickness_a" else definition.upper)
            if definition.name in free_names
            else definition.initial,
            locked=definition.name not in free_names,
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


def _unequal_roughness_joint():
    api = import_module("xrr_fitter.fit.joint_problem")
    rule = SharingRule(
        "shared-physical-roughness",
        (
            ParameterReference("left", "component.0.roughness_a"),
            ParameterReference("right", "component.0.roughness_a"),
        ),
    )
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _unequal_roughness_problem(thickness_a=100.0, seed=863, size=40),
            _unequal_roughness_problem(thickness_a=20.0, seed=863, size=52),
        ),
        (rule,),
    )


def _evaluation(problem, *, objective: float, residual: float, valid: bool = True):
    fit_count = int(np.count_nonzero(problem.data.fit_mask))
    qz = problem.data.qz_a_inv
    return ModelEvaluation(
        valid=valid,
        reason="evaluated" if valid else "physical_constraint",
        parameters=(),
        qz_a_inv=qz,
        model_normalized=np.ones_like(qz),
        fit_log_residuals_decades=np.full(fit_count, residual),
        fit_weighted_residuals=np.full(fit_count, residual),
        objective=objective,
        expanded_stack=None,
        diagnostics=(),
    )


__all__ = [
    "FitConfig",
    "InstrumentSpec",
    "LayerSpec",
    "ModelEvaluation",
    "ParameterReference",
    "ParameterSetting",
    "SHARED_NAME",
    "SearchBudget",
    "SharingRule",
    "SimpleNamespace",
    "TIE_ROUGHNESS_NAME",
    "TIE_THICKNESS_NAME",
    "_evaluation",
    "_joint",
    "_problem",
    "_tie_joint",
    "_tie_problem",
    "_unequal_roughness_joint",
    "_unequal_roughness_problem",
    "compile_fit_problem",
    "encode_physical_vector",
    "evaluate_vector",
    "import_module",
    "np",
    "prepared_data",
    "pytest",
    "replace",
    "simple_structure",
    "warnings",
]
