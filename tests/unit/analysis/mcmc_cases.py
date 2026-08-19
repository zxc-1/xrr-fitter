from __future__ import annotations

import warnings
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector, evaluate_model, values_by_name
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.analysis import EnsembleSamples, McmcConfig
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ParameterPrior,
    ParameterSetting,
    PriorSpec,
)
from xrr_fitter.model.structure import GradientLayerSpec


def _api():
    return import_module("xrr_fitter.analysis.mcmc")


def run_affine_invariant(*args, **kwargs):
    return _api().run_affine_invariant(*args, **kwargs)


def run_problem_mcmc(*args, **kwargs):
    return _api().run_problem_mcmc(*args, **kwargs)


def _problem(*targets: str, scale_prior: bool = False):
    initial = compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(
            FitConfig.fast(929),
            scale_prior_enabled=scale_prior,
            scale_prior_tau_decades=0.2,
        ),
    )
    selected = set(targets or ("component.0.thickness_a", "component.0.density_scale"))
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in selected,
        )
        for definition in initial.parameter_definitions
    )
    problem = compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        settings,
    )
    if scale_prior:
        problem = replace(problem, scale_prior_center=1.1, scale_prior_reason=None)
    return problem


def _candidate(problem, candidate_id: str = "mcmc-center"):
    unit = encode_physical_vector(problem, {})
    return candidate_from_evaluation(
        problem,
        unit,
        evaluate_model(problem, unit),
        candidate_id=candidate_id,
        seed_index=0,
        stop_reason="test center",
        nfev=1,
    )


def _inject_priors(problem, priors):
    definitions = tuple(
        replace(definition, prior=priors[definition.name]) if definition.name in priors else definition
        for definition in problem.parameter_definitions
    )
    return replace(problem, parameter_definitions=definitions)


def _center_ensemble(problem, candidate, config):
    dimension = len(problem.variables)
    ensemble = EnsembleSamples(
        np.broadcast_to(candidate.unit_vector, (2, config.walkers, dimension)).copy(),
        np.zeros((2, config.walkers)),
        np.full(config.walkers, 0.5),
        np.ones(dimension),
        np.full(dimension, 200.0),
    )
    return lambda *args, **kwargs: ensemble


__all__ = [
    "EnsembleSamples",
    "FitConfig",
    "GradientLayerSpec",
    "InstrumentSpec",
    "McmcConfig",
    "ParameterPrior",
    "ParameterSetting",
    "PriorSpec",
    "SimpleNamespace",
    "_api",
    "_candidate",
    "_center_ensemble",
    "_inject_priors",
    "_problem",
    "candidate_from_evaluation",
    "compile_fit_problem",
    "encode_physical_vector",
    "evaluate_model",
    "import_module",
    "np",
    "prepared_data",
    "pytest",
    "replace",
    "run_affine_invariant",
    "run_problem_mcmc",
    "simple_structure",
    "values_by_name",
    "warnings",
]
