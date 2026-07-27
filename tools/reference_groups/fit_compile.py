"""Replay frozen R22 fit-compilation cases through current R23 production."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import PurePosixPath

import numpy as np

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_jacobian, evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.io.project_codec import project_from_bytes
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.data import with_fit_mask
from xrr_fitter.model.fitting import FitEvaluationContext


ARTIFACTS = ("golden/fit_compile.json", "golden/fit_compile.npz")
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
    "single-layer-data": (
        "bundled-example-data",
        "xrr_fitter/examples/single-layer.xy",
        43223,
        "85729258067ff1c953257f6e784b6ec5a5c9e175e92f449ae0bc04680c1e42ea",
    ),
    "single-layer-project": (
        "bundled-example-project",
        "xrr_fitter/examples/single-layer.xrrproj.json",
        20247,
        "c2aae5beca68b95d5dd0f06659fdf73c7ddc8921aa46e76bda7e7d2cae35fa65",
    ),
}
INPUT_ORDER = tuple(INPUTS)
CONFIGURATION = {
    "cases": ["single-layer", "mo-si-periodic"],
    "operations": [
        "compile_fit_problem",
        "evaluate_vector",
        "evaluate_jacobian",
    ],
}
PREFIXES = {"single-layer": "single", "mo-si-periodic": "mo_si"}


def _validate_input(value: object) -> bytes:
    try:
        input_class, path, size, digest = INPUTS[value.input_id]
    except (AttributeError, KeyError) as error:
        raise ValueError("fit_compile input identity drift") from error
    if value.input_class != input_class or value.path != path:
        raise ValueError("fit_compile input identity drift")
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("fit_compile input content must be bytes")
    if value.size != size or value.sha256 != digest:
        raise ValueError("fit_compile input size or hash drift")
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("fit_compile input content or hash drift")
    return content


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "fit_compile":
        raise ValueError("fit_compile group drift")
    if tuple(context.artifacts) != ARTIFACTS:
        raise ValueError("fit_compile artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("fit_compile configuration drift")
    if tuple(context.seeds):
        raise ValueError("fit_compile seed drift")
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("fit_compile input set or order drift")
    return {value.input_id: _validate_input(value) for value in inputs}


def _compile_case(contents: dict[str, bytes], stem: str) -> FitEvaluationContext:
    project = project_from_bytes(contents[f"{stem}-project"])
    if len(project.datasets) != 1:
        raise ValueError(f"fit_compile project dataset drift: {stem}")
    dataset = project.datasets[0]
    if dataset.dataset_id != stem or dataset.structure is None:
        raise ValueError(f"fit_compile project identity drift: {stem}")
    input_id = f"{stem}-data"
    data = read_xy_bytes(
        contents[input_id],
        source_path=PurePosixPath(INPUTS[input_id][1]),
        beam=dataset.beam,
        import_angle_offset_deg=dataset.import_angle_offset_deg,
        column_mapping=dataset.column_mapping,
    )
    data = with_fit_mask(data, np.asarray(dataset.fit_mask, dtype=np.bool_))
    return compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        project.fit_config,
        dataset.parameter_settings,
    )


def _region_counts(problem: FitEvaluationContext) -> dict[str, int]:
    labels, counts = np.unique(
        problem.region_labels[problem.data.fit_mask],
        return_counts=True,
    )
    return {
        str(int(label)): int(count)
        for label, count in zip(labels, counts, strict=True)
    }


def _definition_bounds(problem: FitEvaluationContext) -> list[dict[str, object]]:
    return [
        {
            "name": definition.name,
            "initial": definition.initial,
            "lower": definition.lower,
            "upper": definition.upper,
            "transform": definition.transform,
            "locked": definition.locked,
        }
        for definition in problem.parameter_definitions
    ]


def _jacobian_summary(jacobian: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(jacobian.shape),
        "finite": bool(np.all(np.isfinite(jacobian))),
        "frobenius_norm": float(np.linalg.norm(jacobian)),
        "max_abs": float(np.max(np.abs(jacobian))),
    }


def _case_artifacts(
    problem: FitEvaluationContext,
    prefix: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    initial_unit = encode_physical_vector(problem, {})
    initial = evaluate_vector(problem, initial_unit)
    jacobian = evaluate_jacobian(problem, initial_unit)
    definitions = [asdict(definition) for definition in problem.parameter_definitions]
    summary = {
        "parameter_definitions": definitions,
        "variables": [asdict(variable) for variable in problem.variables],
        "parameter_order": [variable.name for variable in problem.variables],
        "bounds": _definition_bounds(problem),
        "region_counts": _region_counts(problem),
        "weight_sum": float(np.sum(problem.weights)),
        "scale_prior_center": problem.scale_prior_center,
        "scale_prior_tau_decades": problem.scale_prior_tau_decades,
        "scale_prior_reason": problem.scale_prior_reason,
        "warnings": list(problem.warnings),
        "initial_evaluation": {
            "valid": initial.valid,
            "reason": initial.reason,
            "objective": initial.objective,
            "diagnostics": [asdict(value) for value in initial.diagnostics],
        },
        "jacobian": _jacobian_summary(jacobian),
    }
    arrays = {
        f"{prefix}_initial_model": initial.model_normalized,
        f"{prefix}_initial_residual": initial.fit_log_residuals_decades,
        f"{prefix}_initial_unit": initial_unit,
        f"{prefix}_initial_weighted_residual": initial.fit_weighted_residuals,
        f"{prefix}_jacobian": jacobian,
        f"{prefix}_qz_a_inv": initial.qz_a_inv,
        f"{prefix}_region_labels": problem.region_labels,
        f"{prefix}_weights": problem.weights,
    }
    return summary, arrays


def replay(context: object) -> dict[str, object]:
    """Build normalized fit compilation artifacts from declared replay inputs."""
    contents = _validate_context(context)
    summaries: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    for stem in CONFIGURATION["cases"]:
        summary, case_arrays = _case_artifacts(
            _compile_case(contents, stem),
            PREFIXES[stem],
        )
        summaries[stem] = summary
        arrays.update(case_arrays)
    return {
        ARTIFACTS[0]: {
            "schema": "xrr-r22-fit-compile-reference-v1",
            "cases": summaries,
        },
        ARTIFACTS[1]: arrays,
    }
