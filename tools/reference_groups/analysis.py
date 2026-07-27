"""Replay frozen R22 uncertainty evidence through current R23 production."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
from pathlib import PurePosixPath

import numpy as np

from xrr_fitter.analysis.bootstrap import bootstrap_problem_local
from xrr_fitter.analysis.classification import classify_result
from xrr_fitter.analysis.mcmc import run_problem_mcmc
from xrr_fitter.analysis.profiles import build_problem_profile
from xrr_fitter.analysis.report import build_uncertainty_report
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.io.project_codec import project_from_bytes
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.analysis import McmcConfig
from xrr_fitter.model.data import with_fit_mask
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.parameters import ParameterSetting


ARTIFACTS = ("golden/analysis.json", "golden/analysis.npz")
INPUTS = {
    "mo-si-periodic-data": (
        "bundled-example-data",
        "xrr_fitter/examples/mo-si-periodic.xy",
        65125,
        "5bcdf3669698c4482e409b65fca794e500c41924953bc4f12dfe1aeee5d3bd70",
    ),
    "mo-si-periodic-project": (
        "bundled-example-project",
        "xrr_fitter/examples/mo-si-periodic.xrrproj.json",
        29298,
        "613e86c22605b111ceb57fd6b3a63f93e3a330cfac65cc18d37b2f1a5c2407ee",
    ),
}
INPUT_ORDER = tuple(INPUTS)
SEEDS = (3001, 3002, 3003)
CONFIGURATION = {
    "bootstrap_sample_count": 8,
    "case": "mo-si-periodic",
    "mcmc": {
        "burn_in": 4,
        "production_steps": 8,
        "thin": 2,
        "walkers": 6,
    },
    "profile_parameter": "component.0.period_a",
    "targets": [
        "component.0.layer.0.thickness_a",
        "component.0.layer.1.thickness_a",
    ],
}


def _expected_input(value: object) -> tuple[str, str, int, str]:
    try:
        return INPUTS[value.input_id]
    except (AttributeError, KeyError) as error:
        raise ValueError("analysis input identity drift") from error


def _validate_input(value: object) -> bytes:
    input_class, path, size, digest = _expected_input(value)
    if value.input_class != input_class or value.path != path:
        raise ValueError("analysis input identity drift")
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("analysis input content must be bytes")
    if value.size != size or value.sha256 != digest:
        raise ValueError("analysis input size or hash drift")
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("analysis input content or hash drift")
    return content


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "analysis":
        raise ValueError("analysis group drift")
    if tuple(context.artifacts) != ARTIFACTS:
        raise ValueError("analysis artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("analysis configuration drift")
    if tuple(context.seeds) != SEEDS:
        raise ValueError("analysis seed drift")
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("analysis input set or order drift")
    return {value.input_id: _validate_input(value) for value in inputs}


def _problem(contents: dict[str, bytes]):
    project = project_from_bytes(contents["mo-si-periodic-project"])
    if len(project.datasets) != 1:
        raise ValueError("analysis project dataset drift")
    dataset = project.datasets[0]
    if dataset.dataset_id != "mo-si-periodic" or dataset.structure is None:
        raise ValueError("analysis project identity drift")
    data = read_xy_bytes(
        contents["mo-si-periodic-data"],
        source_path=PurePosixPath(INPUTS["mo-si-periodic-data"][1]),
        beam=dataset.beam,
        import_angle_offset_deg=dataset.import_angle_offset_deg,
        column_mapping=dataset.column_mapping,
    )
    data = with_fit_mask(data, np.asarray(dataset.fit_mask, dtype=np.bool_))
    base = compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        project.fit_config,
        dataset.parameter_settings,
    )
    targets = set(CONFIGURATION["targets"])
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in targets,
        )
        for definition in base.parameter_definitions
    )
    config = replace(
        FitConfig.fast(SEEDS[0]),
        budget=SearchBudget(0, 0, 5, 1, CONFIGURATION["bootstrap_sample_count"]),
        local_workers=1,
    )
    return compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        config,
        settings,
    )


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    return value


def _artifacts(problem: object) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    center = encode_physical_vector(problem, {})
    evaluation = evaluate_vector(problem, center)
    candidate = candidate_from_evaluation(
        problem,
        center,
        evaluation,
        candidate_id="analysis-initial",
        seed_index=0,
        stop_reason="reference initial evaluation",
        nfev=1,
    )
    profile = build_problem_profile(
        problem,
        center,
        CONFIGURATION["profile_parameter"],
    )
    bootstrap = bootstrap_problem_local(
        problem,
        candidate,
        sample_count=CONFIGURATION["bootstrap_sample_count"],
        child_seed=SEEDS[1],
    )
    mcmc_values = CONFIGURATION["mcmc"]
    mcmc = run_problem_mcmc(
        problem,
        candidate,
        McmcConfig(**mcmc_values),
        child_seed=SEEDS[2],
    )
    report = build_uncertainty_report(
        problem,
        (candidate,),
        profile_names=(),
        bootstrap=bootstrap,
    )
    report = replace(report, mcmc=mcmc)
    summary = {
        "schema": "xrr-r22-analysis-reference-v1",
        "case": CONFIGURATION["case"],
        "binary_profile": {
            "name": profile.name,
            "lower_closed": profile.lower_closed,
            "upper_closed": profile.upper_closed,
        },
        "bootstrap": {
            "parameter_names": list(bootstrap.parameter_names),
            "intervals": _json_value(bootstrap.intervals),
            "failure_rate": bootstrap.failure_rate,
        },
        "mcmc": {
            "config": _json_value(mcmc.config),
            "child_seed": mcmc.child_seed,
            "parameter_names": list(mcmc.parameter_names),
            "boundary_hits": list(mcmc.boundary_hits),
            "label": mcmc.label,
            "warnings": list(mcmc.warnings),
            "candidate_id": mcmc.candidate_id,
        },
        "report": {
            "candidate_id": report.candidate_id,
            "correlation_names": list(report.correlation_names),
            "strong_correlations": _json_value(report.strong_correlations),
            "systematic_residual": report.systematic_residual,
            "residual_autocorrelation": report.residual_autocorrelation,
            "boundary_hits": list(report.boundary_hits),
            "diagnostics": _json_value(report.diagnostics),
            "bootstrap_intervals": _json_value(report.bootstrap_intervals),
            "bootstrap_failure_rate": report.bootstrap_failure_rate,
        },
        "classification": _json_value(classify_result(problem, (candidate,), report)),
    }
    arrays = {
        "binary_profile_objectives": profile.objectives,
        "binary_profile_values": profile.values,
        "bootstrap_samples": bootstrap.samples,
        "mcmc_acceptance_fraction": mcmc.acceptance_fraction,
        "mcmc_effective_sample_size": mcmc.effective_sample_size,
        "mcmc_log_probability": mcmc.log_probability,
        "mcmc_samples_physical": mcmc.samples_physical,
        "mcmc_split_rhat": mcmc.split_rhat,
        "report_correlation_matrix": report.correlation_matrix,
    }
    return summary, arrays


def replay(context: object) -> dict[str, object]:
    """Build strict normalized analysis artifacts from declared inputs."""
    summary, arrays = _artifacts(_problem(_validate_context(context)))
    return {ARTIFACTS[0]: summary, ARTIFACTS[1]: arrays}
