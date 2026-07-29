"""Replay the frozen R22 service workflows through the R23 public operations.

The adapter snapshots domain values and normalized export bytes without
depending on test support or the committed expected artifact.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

import numpy as np

from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.io import export_run
from xrr_fitter.io.project_codec import project_from_bytes
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.data import with_fit_mask
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting, SharingRule
from xrr_fitter.model.project import ProjectUiState, XrrProject
from xrr_fitter.services import exports
from xrr_fitter.services.datasets import add_dataset
from xrr_fitter.services.fitting import fit_project
from xrr_fitter.services.projects import clear_fit_results, inspect_sources, select_candidate


ARTIFACT = "golden/services.json"
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
SEEDS = (20260724, 20260725)
CONFIGURATION = {
    "case": "single-layer",
    "dataset_ids": ["service-a", "service-b"],
    "independent_target": "component.0.thickness_a",
    "joint_target": "component.0.density_scale",
    "joint_sharing_key": "material.Mo.density_scale",
    "fit_budget": {
        "short_de_maxiter": 0,
        "full_de_maxiter": 0,
        "local_min_nfev": 5,
        "local_nfev_per_parameter": 1,
        "bootstrap_samples": 8,
    },
    "export_timestamp": "20260726T120000Z",
    "export_token": "5eedc0de",
    "xlsx_excluded_members": ["docProps/core.xml"],
    "input_root_token": "${R22_INPUT_ROOT}",
    "operations": [
        "dataset_id_for",
        "auto_fit_independent",
        "auto_fit_joint",
        "validate_sharing_rules",
        "invalidate_changed_results",
        "export_project_result",
    ],
}
DATASET_EXPORT_ORDER = (
    "fit_result.xlsx",
    "fit_result.json",
    "fit_overview.png",
    "sld_profile.png",
    "residuals.png",
    "run_log.txt",
)
ROOT_EXPORT_ORDER = (
    "compatibility_summary.xlsx",
    "batch_summary.xlsx",
    "parameter_trends.png",
)


def _expected_input(value: object) -> tuple[str, str, int, str]:
    try:
        return INPUTS[value.input_id]
    except (AttributeError, KeyError) as error:
        raise ValueError("services input identity drift") from error


def _validate_input(value: object) -> bytes:
    input_class, path, size, digest = _expected_input(value)
    if value.input_class != input_class or value.path != path:
        raise ValueError("services input identity drift")
    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("services input content must be bytes")
    if value.size != size or value.sha256 != digest:
        raise ValueError("services input size or hash drift")
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("services input content or hash drift")
    return content


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "services":
        raise ValueError("services group drift")
    if tuple(context.artifacts) != (ARTIFACT,):
        raise ValueError("services artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("services configuration drift")
    if tuple(context.seeds) != SEEDS:
        raise ValueError("services seed drift")
    inputs = tuple(context.inputs)
    if tuple(value.input_id for value in inputs) != INPUT_ORDER:
        raise ValueError("services input set or order drift")
    return {value.input_id: _validate_input(value) for value in inputs}


def _materialize_inputs(root: Path, contents: dict[str, bytes]) -> Path:
    examples = root / "xrr_fitter" / "examples"
    examples.mkdir(parents=True)
    for input_id, content in contents.items():
        path = root.joinpath(*PurePosixPath(INPUTS[input_id][1]).parts)
        path.write_bytes(content)
    return examples


def _single_case(contents: dict[str, bytes], examples: Path):
    project = project_from_bytes(contents["single-layer-project"])
    if len(project.datasets) != 1 or project.datasets[0].dataset_id != "single-layer":
        raise ValueError("services single-layer project drift")
    dataset = project.datasets[0]
    if dataset.structure is None:
        raise ValueError("services single-layer structure is missing")
    data = read_xy_bytes(
        contents["single-layer-data"],
        source_path=examples / "single-layer.xy",
        beam=dataset.beam,
        import_angle_offset_deg=dataset.import_angle_offset_deg,
        column_mapping=dataset.column_mapping,
    )
    data = with_fit_mask(data, np.asarray(dataset.fit_mask, dtype=np.bool_))
    if data.source_sha256 != dataset.source_sha256:
        raise ValueError("services single-layer source drift")
    return project, dataset, data


def _fit_settings(project, dataset, data, target: str, master_seed: int):
    base = compile_fit_problem(
        data,
        dataset.structure,
        dataset.instrument,
        project.fit_config,
        dataset.parameter_settings,
    )
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name != target,
        )
        for definition in base.parameter_definitions
    )
    config = replace(
        FitConfig.fast(master_seed),
        budget=SearchBudget(**CONFIGURATION["fit_budget"]),
        local_workers=1,
    )
    return config, settings


def _service_project(
    case: tuple[object, object, object],
    examples: Path,
    master_seed: int,
    target: str,
    mode: str,
) -> XrrProject:
    source_project, source_dataset, data = case
    config, settings = _fit_settings(
        source_project,
        source_dataset,
        data,
        target,
        master_seed,
    )
    identifiers = tuple(CONFIGURATION["dataset_ids"])
    datasets = tuple(
        replace(source_dataset, dataset_id=dataset_id, parameter_settings=settings)
        for dataset_id in identifiers
    )
    sharing = ()
    if mode == "joint":
        sharing = (
            SharingRule(
                CONFIGURATION["joint_sharing_key"],
                tuple(ParameterReference(dataset_id, target) for dataset_id in identifiers),
            ),
        )
    return replace(
        XrrProject.new(datasets, master_seed=master_seed),
        fit_config=config,
        batch_mode=mode,
        sharing_rules=sharing,
        ui_state=ProjectUiState(active_dataset_id=identifiers[0]),
        base_directory=str(examples),
    )


def _dataset_id_allocation(data: bytes, examples: Path, instrument) -> list[str]:
    source = examples / "sample.xy"
    source.write_bytes(data)
    project = XrrProject.new((), master_seed=0)
    for _index in range(3):
        project = add_dataset(project, source, instrument)
    return [dataset.dataset_id for dataset in project.datasets]


def _json_mapping(value: dict[object, object]) -> dict[str, object]:
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_sequence(value: tuple[object, ...] | list[object]) -> list[object]:
    return [_json_value(item) for item in value]


def _json_dataclass(value: object) -> dict[str, object]:
    return {
        field.name: _json_value(getattr(value, field.name))
        for field in fields(value)
    }


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return _json_mapping(value)
    if isinstance(value, (tuple, list)):
        return _json_sequence(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_dataclass(value)
    return value


def _candidate(candidate: object) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "seed_index": int(candidate.seed_index),
        "unit_vector": candidate.unit_vector.tolist(),
        "parameters": _json_value(candidate.parameters),
        "objective": float(candidate.objective),
        "valid": bool(candidate.valid),
        "stop_reason": candidate.stop_reason,
        "nfev": int(candidate.nfev),
        "ranking_objective": (
            None
            if candidate.ranking_objective is None
            else float(candidate.ranking_objective)
        ),
        "diagnostics": _json_value(candidate.diagnostics),
    }


def _fit_result_summary(result: object) -> dict[str, object]:
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


def _checkpoint_project(project: XrrProject) -> dict[str, object]:
    datasets = []
    for dataset in project.datasets:
        checkpoint = dataset.checkpoint
        datasets.append(
            {
                "dataset_id": dataset.dataset_id,
                "stage": None if checkpoint is None else checkpoint.stage,
                "candidate_order": (
                    []
                    if checkpoint is None
                    else [candidate.candidate_id for candidate in checkpoint.candidates]
                ),
                "child_seeds": [] if checkpoint is None else list(checkpoint.child_seeds),
            }
        )
    return {"datasets": datasets}


def _project_fit_summary(result, progress: list[object], checkpoints: list[XrrProject]):
    datasets = []
    for item in result.datasets:
        fit_result = item.fit_result
        if fit_result.best_index is None:
            raise ValueError(f"services fit has no winner: {item.dataset_id}")
        datasets.append(
            {
                "dataset_id": item.dataset_id,
                "best_candidate_id": fit_result.best_candidate.candidate_id,
                "fit_result": _fit_result_summary(fit_result),
            }
        )
    return {
        "mode": result.mode,
        "cancelled": bool(result.cancelled),
        "warnings": list(result.warnings),
        "dataset_order": [item.dataset_id for item in result.datasets],
        "sharing_rules": _json_value(result.updated_project.sharing_rules),
        "progress": [_json_value(item) for item in progress],
        "checkpoints": [_checkpoint_project(project) for project in checkpoints],
        "datasets": datasets,
    }


def _fit_workflow(project: XrrProject):
    progress: list[object] = []
    checkpoints: list[XrrProject] = []
    result = fit_project(
        project,
        progress_callback=progress.append,
        checkpoint_callback=checkpoints.append,
    )
    fitted = result.updated_project
    for item in result.datasets:
        candidate = item.fit_result.best_candidate
        if candidate is None:
            raise ValueError(f"services fit did not complete: {item.dataset_id}")
        fitted = select_candidate(fitted, item.dataset_id, candidate.candidate_id)
    return result, fitted, progress, checkpoints


def _invalidation_summary(project: XrrProject) -> dict[str, object]:
    dataset = project.datasets[0]
    source = Path(project.base_directory) / dataset.source_path
    with tempfile.TemporaryDirectory(prefix="xrr-r23-services-invalidation-") as directory:
        isolated_source = Path(directory) / source.name
        isolated_source.write_bytes(source.read_bytes())
        isolated_dataset = replace(dataset, source_path=isolated_source.name)
        selected = tuple(
            item
            for item in project.ui_state.selected_candidate_ids
            if item[0] == isolated_dataset.dataset_id
        )
        isolated = replace(
            project,
            datasets=(isolated_dataset,),
            base_directory=directory,
            ui_state=replace(
                project.ui_state,
                active_dataset_id=isolated_dataset.dataset_id,
                selected_candidate_ids=selected,
            ),
        )
        isolated_source.write_bytes(isolated_source.read_bytes() + b"# reference drift\n")
        validation = inspect_sources(isolated)
        invalidated = clear_fit_results(isolated, (isolated_dataset.dataset_id,))
    if validation.valid or invalidated.datasets[0].last_valid_result is not None:
        raise ValueError("services source drift did not invalidate the fitted result")
    record = validation.datasets[0]
    return {
        "before": {
            "has_result": isolated.datasets[0].last_valid_result is not None,
            "has_checkpoint": isolated.datasets[0].checkpoint is not None,
            "selected_candidate_ids": _json_value(isolated.ui_state.selected_candidate_ids),
        },
        "validation": {
            "dataset_id": record.dataset_id,
            "status": record.status.value,
            "expected_sha256": record.expected_sha256,
            "actual_sha256": record.actual_sha256,
            "message": record.message,
        },
        "after": {
            "has_result": invalidated.datasets[0].last_valid_result is not None,
            "has_checkpoint": invalidated.datasets[0].checkpoint is not None,
            "selected_candidate_ids": _json_value(
                invalidated.ui_state.selected_candidate_ids
            ),
        },
        "declarations_preserved": {
            "source_path": (
                invalidated.datasets[0].source_path == isolated.datasets[0].source_path
            ),
            "source_sha256": (
                invalidated.datasets[0].source_sha256 == isolated.datasets[0].source_sha256
            ),
        },
    }


def _canonical(value: object) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _record(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _normalize_export_paths(content: bytes, project: XrrProject) -> bytes:
    base = Path(project.base_directory)
    roots = sorted({str(base), str(base.resolve())}, key=len, reverse=True)
    normalized = content
    for root in roots:
        normalized = normalized.replace(
            root.encode("utf-8"),
            CONFIGURATION["input_root_token"].encode("utf-8"),
        )
    return normalized


def _normalized_export_file(path: Path, relative: str, project: XrrProject):
    original = path.read_bytes()
    content = _normalize_export_paths(original, project)
    record = _record(relative, content)
    if content != original:
        record["normalization"] = "input-root-token-v1"
    return record


def _normalized_xlsx(path: Path, relative: str, project: XrrProject):
    excluded = tuple(CONFIGURATION["xlsx_excluded_members"])
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not set(excluded) <= set(names):
            raise ValueError(f"services XLSX member drift: {relative}")
        members = [
            _record(name, _normalize_export_paths(archive.read(name), project))
            for name in sorted(names)
            if name not in excluded
        ]
    return {
        "path": relative,
        "normalization": "zip-members-and-input-root-v1",
        "excluded_members": list(excluded),
        "members": members,
        "sha256": hashlib.sha256(_canonical(members)).hexdigest(),
    }


def _ordered_paths(records, order: tuple[str, ...]) -> list[str]:
    by_name = {Path(record.path).name: record.path for record in records}
    if set(by_name) != set(order):
        raise ValueError("services export member set drift")
    return [by_name[name] for name in order]


def _publish_fixed_export(project: XrrProject, output: Path):
    original_publish = exports.publish_export_run
    original_token = export_run.secrets.token_hex

    def publish(output_dir, datasets, root_files):
        return original_publish(
            output_dir,
            datasets,
            root_files,
            run_timestamp=CONFIGURATION["export_timestamp"],
        )

    exports.publish_export_run = publish
    export_run.secrets.token_hex = lambda _size: CONFIGURATION["export_token"]
    try:
        return exports.export_result(project, output)
    finally:
        exports.publish_export_run = original_publish
        export_run.secrets.token_hex = original_token


def _export_summary(project: XrrProject) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="xrr-r23-services-export-") as directory:
        output = Path(directory)
        manifest = _publish_fixed_export(project, output)
        run = manifest.run_directory
        root_files = _ordered_paths(manifest.root_files, ROOT_EXPORT_ORDER)
        datasets = [
            {
                "dataset_id": item.dataset_id,
                "directory": item.directory,
                "files": _ordered_paths(item.files, DATASET_EXPORT_ORDER),
            }
            for item in manifest.datasets
        ]
        artifacts = []
        for record in manifest.files:
            relative = record.path
            path = run / relative
            if path.suffix == ".xlsx":
                artifacts.append(_normalized_xlsx(path, relative, project))
            else:
                artifacts.append(_normalized_export_file(path, relative, project))
        if tuple(output.glob(".partial-*")):
            raise ValueError("services export left a partial directory")
        return {
            "timestamp": CONFIGURATION["export_timestamp"],
            "token": CONFIGURATION["export_token"],
            "manifest": {
                "run_directory": run.name,
                "root_files": root_files,
                "datasets": datasets,
            },
            "artifacts": sorted(artifacts, key=lambda item: str(item["path"])),
        }


def replay(context: object) -> dict[str, object]:
    """Build the exact normalized service artifact from declared replay inputs."""
    contents = _validate_context(context)
    with tempfile.TemporaryDirectory(prefix="xrr-r23-services-reference-") as directory:
        examples = _materialize_inputs(Path(directory), contents)
        case = _single_case(contents, examples)
        allocation = _dataset_id_allocation(
            contents["single-layer-data"], examples, case[1].instrument
        )
        independent_project = _service_project(
            case,
            examples,
            SEEDS[0],
            CONFIGURATION["independent_target"],
            "independent",
        )
        independent, fitted, progress, checkpoints = _fit_workflow(
            independent_project
        )
        joint_project = _service_project(
            case,
            examples,
            SEEDS[1],
            CONFIGURATION["joint_target"],
            "joint",
        )
        joint, _joint_fitted, joint_progress, joint_checkpoints = _fit_workflow(
            joint_project
        )
        value = {
            "schema": "xrr-r22-services-reference-v1",
            "dataset_id_allocation": allocation,
            "independent_workflow": _project_fit_summary(
                independent, progress, checkpoints
            ),
            "joint_workflow": _project_fit_summary(
                joint, joint_progress, joint_checkpoints
            ),
            "invalidation": _invalidation_summary(fitted),
            "export": _export_summary(fitted),
        }
    return {ARTIFACT: value}
