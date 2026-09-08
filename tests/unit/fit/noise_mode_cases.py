"""Real compiled problems shared by noise-objective contract tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec


def _config(mode: str):
    config = FitConfig.fast(17)
    return replace(config, noise_model=mode, scale_prior_enabled=False)


def _data(mode: str, size: int = 80):
    data = prepared_data(size=size)
    observed = np.linspace(-0.1, 0.4, size) if mode == "gaussian" else np.arange(size, dtype=float)
    normalization = 100.0 if mode == "poisson" else 1.0
    if mode == "robust_log":
        observed = np.linspace(0.01, 0.4, size)
    return replace(
        data,
        intensity_raw=observed,
        intensity_normalized=observed / normalization,
        normalization=normalization,
        intensity_sigma_raw=np.full(size, 0.02),
        intensity_sigma_normalized=np.full(size, 0.02 / normalization),
    )


def _problem(mode: str, data=None, *, size: int = 80):
    return compile_fit_problem(
        _data(mode, size) if data is None else data,
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        _config(mode),
    )


def _unit(problem):
    return encode_physical_vector(problem, {"instrument.scale": 0.7, "component.0.thickness_a": 35.0})
