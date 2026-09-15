from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.unit.fit.noise_mode_cases import _config, _problem

from xrr_fitter.evaluation import encode_physical_vector, evaluate_model
from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.pipeline import FitSearchRequest, run_fit_search
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.parameters import ParameterFreedom, ParameterReference, ParameterSetting, SharingRule


def _scale_problem(mode: str, size: int):
    baseline = _problem("robust_log", size=size)
    truth = evaluate_model(baseline, encode_physical_vector(baseline, {"instrument.scale": 0.7}))
    normalization = 50000.0 if mode == "poisson" else 1.0
    observed = np.round(truth.model_normalized * normalization) if mode == "poisson" else truth.model_normalized.copy()
    if mode == "gaussian":
        observed[-2:] = -1e-4
    data = replace(
        baseline.data,
        intensity_raw=observed,
        intensity_normalized=observed / normalization,
        normalization=normalization,
        intensity_sigma_raw=np.full(size, 0.01 * normalization),
        intensity_sigma_normalized=np.full(size, 0.01),
    )
    settings = tuple(
        ParameterSetting(definition.name, 0.5, 0.1, 1.5)
        if definition.name == "instrument.scale"
        else ParameterSetting(
            definition.name, definition.initial, definition.lower, definition.upper, freedom=ParameterFreedom.FIXED
        )
        for definition in baseline.parameter_definitions
    )
    config = _config(mode)
    config = replace(config, budget=replace(config.budget, short_de_maxiter=0, full_de_maxiter=0), local_workers=1)
    return compile_fit_problem(data, baseline.structure, baseline.instrument, config, settings)


def _assert_recovered(search, mode: str) -> None:
    best = search.best_candidate
    assert best is not None and best.valid
    scale = next(parameter.value for parameter in best.parameters if parameter.name == "instrument.scale")
    assert scale == pytest.approx(0.7, abs=0.002)
    assert best.noise_model == mode
    assert np.all(np.isfinite(best.residuals))


@pytest.mark.parametrize("mode", ["gaussian", "poisson"])
def test_likelihood_single_search_recovers_and_resumes(mode: str) -> None:
    problem = _scale_problem(mode, 80)
    checkpoints = []
    search = run_fit_search(FitSearchRequest("single", problem), checkpoint=checkpoints.append)
    _assert_recovered(search, mode)
    resumed = run_fit_search(FitSearchRequest("single", problem, checkpoints[-1]))
    np.testing.assert_array_equal(resumed.best_candidate.unit_vector, search.best_candidate.unit_vector)


@pytest.mark.parametrize("mode", ["gaussian", "poisson"])
def test_likelihood_joint_search_recovers_shared_scale_and_resumes(mode: str) -> None:
    problems = (_scale_problem(mode, 80), _scale_problem(mode, 160))
    rules = (
        SharingRule(
            "scale", (ParameterReference("small", "instrument.scale"), ParameterReference("large", "instrument.scale"))
        ),
    )
    problem = compile_joint_problem(("small", "large"), problems, rules)
    checkpoints = []
    searches = run_joint_fit(JointFitRequest(problem), checkpoint=checkpoints.append)
    for search in searches:
        _assert_recovered(search, mode)
    resumed = run_joint_fit(JointFitRequest(problem, resume_checkpoints=checkpoints[-1]))
    for original, restored in zip(searches, resumed, strict=True):
        np.testing.assert_array_equal(restored.best_candidate.unit_vector, original.best_candidate.unit_vector)
