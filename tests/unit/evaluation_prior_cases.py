from __future__ import annotations

import warnings
from dataclasses import replace
from math import erf, log, sqrt
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import PriorSpec, _prior_center


def _prior_problem() -> FitEvaluationContext:
    return compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=19), scale_prior_enabled=False),
    )


__all__ = [
    "FitConfig",
    "FitEvaluationContext",
    "InstrumentSpec",
    "PriorSpec",
    "SimpleNamespace",
    "_prior_center",
    "_prior_problem",
    "compile_fit_problem",
    "erf",
    "evaluation",
    "log",
    "np",
    "prepared_data",
    "pytest",
    "replace",
    "simple_structure",
    "sqrt",
    "warnings",
]
