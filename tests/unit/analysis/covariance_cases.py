"""Known-noise, scale-only physical models for uncertainty calibration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector, evaluate_model
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.joint_pipeline import _project_candidate, _summary
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.joint_solvers import solve_joint
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitSearchResult
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterFreedom, ParameterReference, ParameterSetting, SharingRule


def scale_problem(mode="robust_log", *, repeats=1, seed=17, systematic=False):
    # Keep the import axis strictly increasing while repeating the same design
    # to numerical accuracy; every row is a separate generated observation.
    angles = np.repeat(np.linspace(0.1, 3.2, 80), repeats) + np.tile(np.arange(repeats) * 1e-12, 80)
    data = prepared_data(size=angles.size, two_theta_deg=angles)
    data = replace(
        data,
        normalization=1000.0 if mode == "poisson" else 1.0,
        intensity_sigma_normalized=np.full(angles.size, 0.02),
        intensity_sigma_raw=np.full(angles.size, 0.02),
    )
    config = replace(FitConfig.fast(17), scale_prior_enabled=False, noise_model="robust_log")
    initial = compile_fit_problem(data, simple_structure(), InstrumentSpec(footprint_mode="none"), config)
    settings = tuple(
        ParameterSetting(
            item.name,
            item.initial,
            item.lower,
            item.upper,
            freedom=ParameterFreedom.from_locked(item.name != "instrument.scale"),
        )
        for item in initial.parameter_definitions
    )
    problem = compile_fit_problem(data, initial.structure, initial.instrument, config, settings)
    unit = encode_physical_vector(problem, {"instrument.scale": 0.5})
    prediction = evaluate_model(problem, unit).model_normalized
    rng = np.random.default_rng(seed)
    if mode == "gaussian":
        observed = prediction + rng.normal(0, 0.02, prediction.size)
    elif mode == "poisson":
        observed = rng.poisson(data.normalization * prediction) / data.normalization
    else:
        noise = rng.normal(0, 0.04, prediction.size)
        if systematic:
            noise += 0.18 * np.sin(np.linspace(0, 5 * np.pi, prediction.size))
        observed = (prediction + data.r_floor) * 10**noise - data.r_floor
    data = replace(data, intensity_raw=observed * data.normalization, intensity_normalized=observed)
    return compile_fit_problem(data, problem.structure, problem.instrument, replace(config, noise_model=mode), settings)


def scale_candidate(problem, *, scale=0.5, candidate_id="E-0"):
    unit = encode_physical_vector(problem, {"instrument.scale": scale})
    return candidate_from_evaluation(
        problem,
        unit,
        evaluate_model(problem, unit),
        candidate_id=candidate_id,
        seed_index=0,
        stop_reason="converged",
        nfev=1,
    )


def joint_scale_searches(mode="robust_log", *, systematic=False, repeats=1):
    problems = (
        scale_problem(mode, seed=17, systematic=systematic),
        scale_problem(mode, seed=18, systematic=systematic, repeats=repeats),
    )
    ids = ("small", "large")
    rule = SharingRule("shared-scale", tuple(ParameterReference(name, "instrument.scale") for name in ids))
    joint = compile_joint_problem(ids, problems, (rule,))
    rows = []
    for index, scale in enumerate((0.46, 0.49, 0.51, 0.54)):
        unit = encode_physical_vector(joint.problems[0], {"instrument.scale": scale})
        solved = solve_joint(joint, unit, 80, None)
        rows.append(_project_candidate(joint, solved, f"E-{index}", index))
    summary = _summary("E", tuple(rows))
    searches = []
    for local, candidates in zip(joint.problems, zip(*rows, strict=True), strict=True):
        best = min(range(len(candidates)), key=lambda index: candidates[index].ranking_objective)
        searches.append(
            FitSearchResult(
                local.parameter_definitions,
                candidates,
                best,
                (),
                (17, 18, 19, 20),
                (summary,),
                local.region_labels,
                local.weights,
            )
        )
    return joint, tuple(searches)
