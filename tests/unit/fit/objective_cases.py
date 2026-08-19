from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation_module
from xrr_fitter.evaluation import (
    assign_fit_regions,
    encode_physical_vector,
    log_residuals,
    region_weights,
    robust_log_cost,
    scale_prior_penalty,
)
from xrr_fitter.fit.objective import evaluate_jacobian, evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec


def _problem(*, size: int = 64):
    config = replace(FitConfig.fast(master_seed=7), scale_prior_enabled=False)
    return compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )


def _reliable_plateau_data():
    theta_deg = np.linspace(0.01, 2.0, 600)
    intensity = 0.82 / (1.0 + np.exp((theta_deg - 0.35) / 0.006)) + 1e-8
    return prepared_data(
        size=theta_deg.size,
        two_theta_deg=2.0 * theta_deg,
        intensity_raw=intensity,
    )


def _richardson(problem, unit: np.ndarray) -> np.ndarray:
    def residual(value: np.ndarray) -> np.ndarray:
        result = evaluate_vector(problem, value)
        assert result.valid
        return result.fit_log_residuals_decades

    output = np.empty((np.count_nonzero(problem.data.fit_mask), unit.size))
    step = 5e-5
    for index in range(unit.size):
        coarse_plus = unit.copy()
        coarse_minus = unit.copy()
        fine_plus = unit.copy()
        fine_minus = unit.copy()
        coarse_plus[index] += step
        coarse_minus[index] -= step
        fine_plus[index] += step / 2.0
        fine_minus[index] -= step / 2.0
        coarse = (residual(coarse_plus) - residual(coarse_minus)) / (2.0 * step)
        fine = (residual(fine_plus) - residual(fine_minus)) / step
        output[:, index] = (4.0 * fine - coarse) / 3.0
    return output


__all__ = [
    "FitConfig",
    "InstrumentSpec",
    "LayerSpec",
    "PeriodicBlock",
    "StructureSpec",
    "_problem",
    "_reliable_plateau_data",
    "_richardson",
    "assign_fit_regions",
    "compile_fit_problem",
    "encode_physical_vector",
    "evaluate_jacobian",
    "evaluate_vector",
    "evaluation_module",
    "log_residuals",
    "np",
    "prepared_data",
    "pytest",
    "region_weights",
    "replace",
    "robust_log_cost",
    "scale_prior_penalty",
    "simple_structure",
    "warnings",
]
