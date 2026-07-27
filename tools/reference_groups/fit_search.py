"""Replay the frozen staged search and Task 8 analysis through R23."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
from pathlib import PurePosixPath

import numpy as np

from xrr_fitter.analysis.profiles import recover_profile_basin
from xrr_fitter.analysis.report import analyze_search_result
from xrr_fitter.fit.pipeline import (
    FitSearchRequest,
    continue_profile_basin,
    run_fit_search,
)
from xrr_fitter.fit.checkpoint import build_checkpoint
from xrr_fitter.io.project_codec import project_from_bytes
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.data import with_fit_mask
from xrr_fitter.model.fitting import (
    FitCheckpoint,
    FitConfig,
    FitProgress,
    FitSearchResult,
    SearchBudget,
)
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.fit.problem import compile_fit_problem


ARTIFACTS = ("golden/fit_search.json", "golden/fit_search.npz")
INPUTS = {
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
SEEDS = (20260723,)
CONFIGURATION = {
    "budget": {
        "bootstrap_samples": 8,
        "full_de_maxiter": 0,
        "local_min_nfev": 5,
        "local_nfev_per_parameter": 1,
        "short_de_maxiter": 0,
    },
    "case": "single-layer",
    "local_workers": 1,
    "master_seed": 20260723,
    "resume_checkpoint_stage": "D",
    "targets": ["component.0.thickness_a"],
}


def _expected_input(value: object) -> tuple[str, str, int, str]:
    try:
        return INPUTS[value.input_id]
    except (AttributeError, KeyError) as error:
        raise ValueError("fit_search input identity drift") from error


def _validate_input(value: object) -> bytes:
    input_class, path, size, digest = _expected_input(value)
    if value.input_class != input_class or value.path != path:
        raise ValueError("fit_search input identity drift")
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("fit_search input content must be bytes")
    if value.size != size or value.sha256 != digest:
        raise ValueError("fit_search input size or hash drift")
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("fit_search input content or hash drift")
    return content


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "fit_search":
        raise ValueError("fit_search group drift")
    if tuple(context.artifacts) != ARTIFACTS:
        raise ValueError("fit_search artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("fit_search configuration drift")
    if tuple(context.seeds) != SEEDS:
        raise ValueError("fit_search seed drift")
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("fit_search input set or order drift")
    return {value.input_id: _validate_input(value) for value in inputs}


def _problem(contents: dict[str, bytes]):
    project = project_from_bytes(contents["single-layer-project"])
    if len(project.datasets) != 1:
        raise ValueError("fit_search project dataset drift")
    dataset = project.datasets[0]
    if dataset.dataset_id != "single-layer" or dataset.structure is None:
        raise ValueError("fit_search project identity drift")
    data = read_xy_bytes(
        contents["single-layer-data"],
        source_path=PurePosixPath(INPUTS["single-layer-data"][1]),
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
        FitConfig.fast(CONFIGURATION["master_seed"]),
        budget=SearchBudget(**CONFIGURATION["budget"]),
        local_workers=CONFIGURATION["local_workers"],
    )
    problem = compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        config,
        settings,
    )
    return problem, data.source_sha256


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


def _candidate(candidate: object) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "seed_index": candidate.seed_index,
        "unit_vector": candidate.unit_vector.tolist(),
        "parameters": _json_value(candidate.parameters),
        "objective": candidate.objective,
        "valid": candidate.valid,
        "stop_reason": candidate.stop_reason,
        "nfev": candidate.nfev,
        "ranking_objective": candidate.ranking_objective,
        "diagnostics": _json_value(candidate.diagnostics),
    }


def _result_summary(result: FitResult) -> dict[str, object]:
    return {
        "parameter_definitions": _json_value(result.parameter_definitions),
        "candidate_order": [candidate.candidate_id for candidate in result.candidates],
        "candidates": [_candidate(candidate) for candidate in result.candidates],
        "best_index": result.best_index,
        "confidence": _json_value(result.confidence),
        "warnings": list(result.warnings),
        "child_seeds": list(result.child_seeds),
        "stage_summaries": _json_value(result.stage_summaries),
        "classification_evidence": list(result.classification_evidence),
    }


def _checkpoint_summary(checkpoint: FitCheckpoint) -> dict[str, object]:
    return {
        "stage": checkpoint.stage,
        "data_sha256": checkpoint.data_sha256,
        "structure_fingerprint": checkpoint.structure_fingerprint,
        "instrument_fingerprint": checkpoint.instrument_fingerprint,
        "config_fingerprint": checkpoint.config_fingerprint,
        "parameter_settings_fingerprint": checkpoint.parameter_settings_fingerprint,
        "joint_layout_fingerprint": checkpoint.joint_layout_fingerprint,
        "candidate_order": [candidate.candidate_id for candidate in checkpoint.candidates],
        "candidates": [_candidate(candidate) for candidate in checkpoint.candidates],
        "child_seeds": list(checkpoint.child_seeds),
        "runtime_warnings": list(checkpoint.runtime_warnings),
        "stage_summaries": _json_value(checkpoint.stage_summaries),
    }


def _analyze(
    problem: object,
    search: FitSearchResult,
    dataset_id: str,
    progress: list[FitProgress],
    checkpoint: Callable[[FitCheckpoint], None] | None = None,
) -> FitResult:
    best = search.best_candidate
    if best is None:
        raise ValueError("fit_search has no Stage-E winner")
    decision = recover_profile_basin(problem, best)
    if decision is not None:
        search = continue_profile_basin(
            problem,
            search,
            decision.unit_vector,
            parameter_name=decision.parameter_name,
            checkpoint=checkpoint,
        )
    return analyze_search_result(
        problem,
        search,
        dataset_id=dataset_id,
        progress=progress.append,
    )


def _stage_e_checkpoint_publisher(
    checkpoints: list[FitCheckpoint],
) -> Callable[[FitCheckpoint], None]:
    def publish(checkpoint: FitCheckpoint) -> None:
        if not checkpoints or checkpoints[-1].stage != "E":
            raise ValueError("fit_search did not publish its Stage-E checkpoint")
        checkpoints[-1] = checkpoint

    return publish


def _publish_analyzed_checkpoint(
    problem: object,
    checkpoints: list[FitCheckpoint],
    result: FitResult,
) -> None:
    if not checkpoints or checkpoints[-1].stage != "E":
        raise ValueError("fit_search did not publish its Stage-E checkpoint")
    previous = checkpoints[-1]
    summaries = tuple(
        summary for summary in result.stage_summaries if summary.stage != "uncertainty"
    )
    checkpoints[-1] = build_checkpoint(
        problem,
        stage="E",
        candidates=result.candidates,
        child_seeds=previous.child_seeds,
        runtime_warnings=previous.runtime_warnings,
        stage_summaries=summaries,
    )


def _arrays(result: FitResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "region_labels": result.region_labels,
        "region_weights": result.region_weights,
    }
    for index, candidate in enumerate(result.candidates):
        prefix = f"candidate_{index:02d}"
        arrays[f"{prefix}_log_residuals"] = candidate.log_residuals_decades
        arrays[f"{prefix}_model"] = candidate.model_normalized
        arrays[f"{prefix}_qz_a_inv"] = candidate.qz_a_inv
        arrays[f"{prefix}_sld_depth"] = candidate.sld_depth_a
        arrays[f"{prefix}_sld_imag"] = candidate.sld_profile_a2.imag
        arrays[f"{prefix}_sld_real"] = candidate.sld_profile_a2.real
        arrays[f"{prefix}_unit"] = candidate.unit_vector
        arrays[f"{prefix}_weighted_residuals"] = candidate.weighted_residuals
    return arrays


def _results_equivalent(
    result: FitResult,
    resumed: FitResult,
    result_summary: dict[str, object],
    resumed_summary: dict[str, object],
) -> bool:
    arrays_match = all(
        np.array_equal(first.model_normalized, second.model_normalized, equal_nan=True)
        and np.array_equal(first.unit_vector, second.unit_vector, equal_nan=True)
        for first, second in zip(result.candidates, resumed.candidates, strict=True)
    )
    return result_summary == resumed_summary and arrays_match


def replay(context: object) -> dict[str, object]:
    """Build strict fresh/resume search evidence from declared replay inputs."""
    problem, dataset_id = _problem(_validate_context(context))
    progress: list[FitProgress] = []
    checkpoints: list[FitCheckpoint] = []
    search = run_fit_search(
        FitSearchRequest(dataset_id, problem),
        progress=progress.append,
        checkpoint=checkpoints.append,
    )
    stage_d = next(value for value in checkpoints if value.stage == "D")
    result = _analyze(
        problem,
        search,
        dataset_id,
        progress,
        _stage_e_checkpoint_publisher(checkpoints),
    )
    _publish_analyzed_checkpoint(problem, checkpoints, result)

    resumed_checkpoints: list[FitCheckpoint] = []
    resumed_search = run_fit_search(
        FitSearchRequest(dataset_id, problem, stage_d),
        checkpoint=resumed_checkpoints.append,
    )
    resumed_progress: list[FitProgress] = []
    resumed = _analyze(
        problem,
        resumed_search,
        dataset_id,
        resumed_progress,
        _stage_e_checkpoint_publisher(resumed_checkpoints),
    )
    _publish_analyzed_checkpoint(problem, resumed_checkpoints, resumed)

    result_summary = _result_summary(result)
    resumed_summary = _result_summary(resumed)
    equivalent = _results_equivalent(result, resumed, result_summary, resumed_summary)
    if not equivalent:
        raise ValueError("R23 Stage-D resume result drift")
    summary = {
        "schema": "xrr-r22-fit-search-reference-v1",
        "case": CONFIGURATION["case"],
        "progress": [_json_value(value) for value in progress],
        "checkpoints": [_checkpoint_summary(value) for value in checkpoints],
        "resumed_checkpoints": [
            _checkpoint_summary(value) for value in resumed_checkpoints
        ],
        "resume_equivalent": equivalent,
        "final_result": result_summary,
    }
    return {ARTIFACTS[0]: summary, ARTIFACTS[1]: _arrays(result)}
