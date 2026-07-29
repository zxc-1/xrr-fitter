"""Deterministic JSON, CSV, workbook, and log export serialization.

The service layer will validate live project sources and select the exported
result. This module begins after that orchestration boundary: one immutable
``DatasetExportData`` binds the persisted dataset, its prepared source data,
the explicitly selected candidate, and the already allocated directory mapping.

The project and fit-result portions of JSON always pass through the existing
authoritative codecs. Prepared rows and selected-model arrays are added as
explicit export provenance; nonfinite display gaps become JSON null while
standard JSON NaN and Infinity tokens remain forbidden.

Workbook construction uses fixed sheet and column order, fixed document
metadata, and disabled string-to-formula and string-to-URL conversion. The
same structured values used by JSON feed compact legal JSON workbook cells.
This keeps hostile-looking dataset names literal and makes identical inputs
produce identical XLSX bytes.

Compatibility and batch workbooks retain project dataset order. CSV uses a
fixed UTF-8/LF dialect. The text log records warnings, child seeds, stage stop
evidence, and diagnostic point identities without performing any filesystem
operation or changing model values.

Multi-dataset serializers require a complete context set bound to one project
object. They restore persisted project order before building tables, so caller
iteration order cannot affect output. Oxide warning correlation arrives as a
service-owned conclusion; this layer never guesses material or rule-version
equivalence from persisted decisions.

Raw workbook columns deliberately combine rows with different cardinalities.
Derived points and merged source groups occupy their aligned leading rows,
while source scalar metadata occupies only the first row. Original raw text
and parse status retain the full source-row cardinality independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO, StringIO
import json
from math import isfinite
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from xrr_fitter.io.codec_results import fit_result_to_dict
from xrr_fitter.io.project_codec import project_to_dict
from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitCandidate
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.project import DatasetProject, XrrProject


WORKBOOK_CREATED = datetime(2000, 1, 1)
WORKBOOK_OPTIONS = {
    "strings_to_formulas": False,
    "strings_to_urls": False,
    "strings_to_numbers": False,
}


def _mapping_pairs(values: object) -> tuple[tuple[str, str], ...]:
    mapping = tuple(tuple(value) for value in values)
    if any(len(value) != 2 or not value[0] or not value[1] for value in mapping):
        raise ValueError("directory_mapping must contain nonempty pairs")
    return mapping


def _validate_mapping_uniqueness(mapping: tuple[tuple[str, str], ...]) -> None:
    identifiers = tuple(value[0] for value in mapping)
    directories = tuple(value[1] for value in mapping)
    if len(identifiers) != len(set(identifiers)) or len(directories) != len(set(directories)):
        raise ValueError("directory_mapping values must be unique")


def _direct_child(directory: str) -> bool:
    path = PurePosixPath(directory)
    return path.name == directory and directory not in {".", ".."} and "\\" not in directory


def _directory_mapping(values: object) -> tuple[tuple[str, str], ...]:
    mapping = _mapping_pairs(values)
    _validate_mapping_uniqueness(mapping)
    if any(not _direct_child(value[1]) for value in mapping):
        raise ValueError("dataset directories must be direct relative children")
    return mapping


def _validate_export_roots(context: DatasetExportData) -> None:
    if not isinstance(context.project, XrrProject):
        raise TypeError("project must be an XrrProject")
    if not isinstance(context.dataset, DatasetProject):
        raise TypeError("dataset must be a DatasetProject")
    if not isinstance(context.data, PreparedData):
        raise TypeError("data must be PreparedData")
    if not any(value is context.dataset for value in context.project.datasets):
        raise ValueError("dataset must belong to project")


def _validate_data_identity(context: DatasetExportData) -> None:
    if context.data.source_sha256 != context.dataset.source_sha256:
        raise ValueError("prepared data source hash does not match dataset")
    if context.data.beam != context.dataset.beam:
        raise ValueError("prepared data beam does not match dataset")
    observed = tuple(bool(value) for value in context.data.fit_mask)
    if observed != context.dataset.fit_mask:
        raise ValueError("prepared data fit mask does not match dataset")


def _validate_mapping_order(
    context: DatasetExportData,
    mapping: tuple[tuple[str, str], ...],
) -> None:
    expected = tuple(value.dataset_id for value in context.project.datasets)
    if tuple(value[0] for value in mapping) != expected:
        raise ValueError("directory_mapping must follow project dataset order")


def _validate_selected_axis(context: DatasetExportData) -> None:
    if context.selected.qz_a_inv.size != context.data.two_theta_deg.size:
        raise ValueError("selected candidate arrays do not align with prepared data")


def _validate_selected_ownership(context: DatasetExportData) -> None:
    if not isinstance(context.selected, FitCandidate):
        raise TypeError("selected must be a FitCandidate")
    if not any(value is context.selected for value in context.result.candidates):
        raise ValueError("selected candidate must belong to persisted result")


@dataclass(frozen=True, slots=True)
class ExportReplayIdentity:
    """Service-owned seed branches needed to replay an exported fit."""

    service_seed_tree_version: int
    independent_root_child: int
    joint_root_child: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class DatasetExportData:
    """Validated model, prepared data, and service-owned export conclusions."""

    project: XrrProject
    dataset: DatasetProject
    data: PreparedData
    directory_mapping: tuple[tuple[str, str], ...]
    selected: FitCandidate
    replay_identity: ExportReplayIdentity
    matching_surface_oxide_rejection: bool

    def __post_init__(self) -> None:
        _validate_export_roots(self)
        _validate_data_identity(self)
        mapping = _directory_mapping(self.directory_mapping)
        _validate_mapping_order(self, mapping)
        _validate_selected_ownership(self)
        _validate_selected_axis(self)
        if not isinstance(self.replay_identity, ExportReplayIdentity):
            raise TypeError("replay_identity must be an ExportReplayIdentity")
        if not isinstance(self.matching_surface_oxide_rejection, bool):
            raise TypeError("matching_surface_oxide_rejection must be bool")
        object.__setattr__(self, "directory_mapping", mapping)

    @property
    def result(self) -> FitResult:
        result = self.dataset.last_valid_result
        if result is None:
            raise ValueError("dataset has no fit result")
        return result


def _project_dataset_document(context: DatasetExportData) -> dict[str, Any]:
    document = project_to_dict(context.project)
    index = next(
        index
        for index, dataset in enumerate(context.project.datasets)
        if dataset is context.dataset
    )
    return document["datasets"][index]


def _export_project_document(project: XrrProject) -> dict[str, Any]:
    document = project_to_dict(project)
    for dataset in document["datasets"]:
        dataset.pop("display_name", None)
    return document


def _finite_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _array_values(value: np.ndarray | None) -> list[object] | None:
    if value is None:
        return None
    return [_finite_scalar(item) for item in np.asarray(value).tolist()]


def _diagnostics(value: object) -> list[dict[str, object]]:
    return [
        {
            "code": item.code,
            "message": item.message,
            "point_indices": list(item.point_indices),
        }
        for item in value
    ]


def _raw_data_payload(
    context: DatasetExportData,
    document: dict[str, Any],
) -> dict[str, object]:
    data = context.data
    return {
        "source_path": str(data.source_path),
        "source_sha256": data.source_sha256,
        "raw_rows": list(data.raw_rows),
        "raw_parse_status": list(data.raw_parse_status),
        "source_row_groups": [list(value) for value in data.source_row_groups],
        "beam_kind": data.beam.kind,
        "import_angle_offset_deg": data.import_angle_offset_deg,
        "column_mapping": document["column_mapping"],
        "two_theta_deg": _array_values(data.two_theta_deg),
        "intensity_raw": _array_values(data.intensity_raw),
        "intensity_sigma_raw": _array_values(data.intensity_sigma_raw),
        "resolution_raw": _array_values(data.resolution_raw),
        "qz_a_inv": _array_values(data.qz_a_inv),
        "intensity_normalized": _array_values(data.intensity_normalized),
        "intensity_sigma_normalized": _array_values(
            data.intensity_sigma_normalized
        ),
        "sigma_q_a_inv": _array_values(data.sigma_q_a_inv),
        "validation_mask": [bool(value) for value in data.validation_mask],
        "fit_mask": list(context.dataset.fit_mask),
        "normalization": data.normalization,
        "r_floor": data.r_floor,
        "fit_ready": data.fit_ready,
        "warnings": list(data.warnings),
    }


def _model_residuals_payload(context: DatasetExportData) -> dict[str, object]:
    selected = context.selected
    return {
        "qz_a_inv": _array_values(selected.qz_a_inv),
        "model_normalized": _array_values(selected.model_normalized),
        "log_residuals_decades": _array_values(selected.log_residuals_decades),
        "weighted_residuals": _array_values(selected.weighted_residuals),
        "candidate_id": selected.candidate_id,
    }


def _run_info_payload(
    context: DatasetExportData,
    document: dict[str, Any],
) -> dict[str, object]:
    result = context.result
    identity = context.replay_identity
    project_document = project_to_dict(context.project)
    config = context.project.fit_config
    fitted_instrument = {
        value.name: value.value
        for value in context.selected.parameters
        if value.name.startswith("instrument.")
    }
    mcmc = result.uncertainty.mcmc if result.uncertainty is not None else None
    return {
        "schema_version": context.project.schema_version,
        "algorithm_version": context.project.algorithm_version,
        "fit_config": project_document["fit_config"],
        "project_master_seed": context.project.master_seed,
        "service_seed_tree_version": identity.service_seed_tree_version,
        "independent_root_child": identity.independent_root_child,
        "joint_root_child": identity.joint_root_child,
        "optimizer_child_seeds": list(result.child_seeds),
        "mcmc_child_seed": None if mcmc is None else mcmc.child_seed,
        "selected_candidate_id": context.selected.candidate_id,
        "fitted_instrument_parameters": fitted_instrument,
        "dataset_id": context.dataset.dataset_id,
        "source_path": context.dataset.source_path,
        "source_sha256": context.dataset.source_sha256,
        "beam": document["beam"],
        "instrument": document["instrument"],
        "scale_prior": document["scale_prior"],
        "structure_evidence": document["structure_evidence"],
        "oxide_decisions": document["oxide_decisions"],
        "confidence": result.confidence.value,
        "warnings": list(result.warnings),
        "fringe_screen_threshold_version": config.fringe_screen_threshold_version,
        "budget_reclaim_threshold_version": config.budget_reclaim_threshold_version,
        "downsample_rule_version": config.downsample_rule_version,
        "jacobian_version": config.jacobian_version,
        "dataset_directory": dict(context.directory_mapping)[context.dataset.dataset_id],
        "dataset_directory_mapping": dict(context.directory_mapping),
    }


def _candidate_views(result: FitResult) -> tuple[dict[str, object], list[dict[str, object]]]:
    encoded = fit_result_to_dict(result)
    if encoded is None:
        raise ValueError("fit result is required for export")
    candidates = [
        {**item, "archived": source.stop_reason == "early_eliminated"}
        for item, source in zip(encoded["candidates"], result.candidates, strict=True)
    ]
    return encoded, candidates


def _convergence_payload(result: FitResult) -> dict[str, object]:
    candidates = tuple(
        candidate
        for candidate in result.candidates
        if candidate.valid
        and isfinite(candidate.objective)
        and candidate.stop_reason != "early_eliminated"
    )
    return {
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "objectives": [candidate.objective for candidate in candidates],
    }


def dataset_payload(context: DatasetExportData) -> dict[str, object]:
    """Build the complete strict-JSON payload for one selected dataset."""
    if not isinstance(context, DatasetExportData):
        raise TypeError("context must be DatasetExportData")
    project_document = _export_project_document(context.project)
    dataset_document = _project_dataset_document(context)
    fit_result, candidates = _candidate_views(context.result)
    return {
        "dataset_id": context.dataset.dataset_id,
        "source_path": context.dataset.source_path,
        "source_sha256": context.dataset.source_sha256,
        "beam": dataset_document["beam"],
        "instrument": dataset_document["instrument"],
        "scale_prior": dataset_document["scale_prior"],
        "structure_evidence": dataset_document["structure_evidence"],
        "oxide_decisions": dataset_document["oxide_decisions"],
        "raw_data": _raw_data_payload(context, dataset_document),
        "model_residuals": _model_residuals_payload(context),
        "fit_result": fit_result,
        "project": project_document,
        "candidates": candidates,
        "convergence": _convergence_payload(context.result),
        "run_info": _run_info_payload(context, dataset_document),
    }


def _strict_json(value: object, *, pretty: bool) -> str:
    options = {
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if pretty:
        return json.dumps(value, indent=2, **options)
    return json.dumps(value, **options)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def json_text(value: object) -> str:
    """Encode one legal structured workbook cell."""
    return _strict_json(value, pretty=False)


def dataset_json_bytes(context: DatasetExportData) -> bytes:
    """Serialize complete dataset provenance as deterministic UTF-8 JSON."""
    return (_strict_json(dataset_payload(context), pretty=True) + "\n").encode("utf-8")


def _parameter_frame(context: DatasetExportData) -> pd.DataFrame:
    columns = ["parameter_name", "value", "lower", "upper"]
    rows = [
        {
            "parameter_name": value.name,
            "value": value.value,
            "lower": value.lower,
            "upper": value.upper,
        }
        for value in context.selected.parameters
    ]
    return pd.DataFrame(rows, columns=columns)


def _dataset_parameter_frame(context: DatasetExportData) -> pd.DataFrame:
    columns = [
        "name",
        "display_name",
        "category",
        "value",
        "lower",
        "upper",
        "unit",
        "transform",
        "locked",
        "integer",
        "expert_only",
        "sharing_key",
        "selected_candidate_id",
    ]
    values = {value.name: value for value in context.selected.parameters}
    rows = []
    for definition in context.result.parameter_definitions:
        fitted = values.get(definition.name)
        rows.append(
            {
                "name": definition.name,
                "display_name": definition.display_name,
                "category": definition.category,
                "value": None if fitted is None else fitted.value,
                "lower": definition.lower if fitted is None else fitted.lower,
                "upper": definition.upper if fitted is None else fitted.upper,
                "unit": definition.unit,
                "transform": definition.transform,
                "locked": definition.locked,
                "integer": definition.integer,
                "expert_only": definition.expert_only,
                "sharing_key": definition.sharing_key,
                "selected_candidate_id": context.selected.candidate_id,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _candidate_frame(context: DatasetExportData) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "seed_index",
        "objective",
        "ranking_objective",
        "valid",
        "archived",
        "stop_reason",
        "nfev",
        "parameters_json",
        "diagnostics_json",
    ]
    rows = [
        {
            "candidate_id": value.candidate_id,
            "seed_index": value.seed_index,
            "objective": _finite_scalar(value.objective),
            "ranking_objective": _finite_scalar(value.ranking_objective),
            "valid": value.valid,
            "archived": value.stop_reason == "early_eliminated",
            "stop_reason": value.stop_reason,
            "nfev": value.nfev,
            "parameters_json": json_text(
                {item.name: item.value for item in value.parameters}
            ),
            "diagnostics_json": json_text(_diagnostics(value.diagnostics)),
        }
        for value in context.result.candidates
    ]
    return pd.DataFrame(rows, columns=columns)


def _optional_series(value: np.ndarray | None, size: int) -> pd.Series:
    values = np.full(size, np.nan, dtype=float) if value is None else value
    return pd.Series(values)


def _raw_data_frame(context: DatasetExportData) -> pd.DataFrame:
    data = context.data
    size = data.two_theta_deg.size
    document = _project_dataset_document(context)
    columns: dict[str, pd.Series] = {
        "row_index": pd.Series(np.arange(size, dtype=int)),
        "two_theta_deg": pd.Series(data.two_theta_deg),
        "intensity_raw": pd.Series(data.intensity_raw),
        "intensity_sigma_raw": _optional_series(data.intensity_sigma_raw, size),
        "resolution_raw": _optional_series(data.resolution_raw, size),
        "qz_a_inv": pd.Series(data.qz_a_inv),
        "intensity_normalized": pd.Series(data.intensity_normalized),
        "intensity_sigma_normalized": _optional_series(
            data.intensity_sigma_normalized,
            size,
        ),
        "sigma_q_a_inv": _optional_series(data.sigma_q_a_inv, size),
        "validation_included": pd.Series(data.validation_mask),
        "fit_included": pd.Series(context.dataset.fit_mask),
        "source_row_group_json": pd.Series(
            tuple(json_text(list(value)) for value in data.source_row_groups),
            dtype=object,
        ),
        "raw_row_index": pd.Series(np.arange(len(data.raw_rows), dtype=int)),
        "raw_row_text": pd.Series(data.raw_rows, dtype=object),
        "raw_parse_status": pd.Series(data.raw_parse_status, dtype=object),
        "source_path": pd.Series((str(data.source_path),), dtype=object),
        "source_sha256": pd.Series((data.source_sha256,), dtype=object),
        "beam_json": pd.Series((json_text(document["beam"]),), dtype=object),
        "import_angle_offset_deg": pd.Series((data.import_angle_offset_deg,)),
        "column_mapping_json": pd.Series(
            (json_text(document["column_mapping"]),),
            dtype=object,
        ),
        "normalization": pd.Series((data.normalization,)),
        "r_floor": pd.Series((data.r_floor,)),
        "fit_ready": pd.Series((data.fit_ready,)),
        "warnings_json": pd.Series((json_text(list(data.warnings)),), dtype=object),
    }
    return pd.DataFrame(columns)


def _model_frame(context: DatasetExportData) -> pd.DataFrame:
    selected = context.selected
    size = context.data.two_theta_deg.size
    return pd.DataFrame(
        {
            "row_index": np.arange(size, dtype=int),
            "two_theta_deg": context.data.two_theta_deg,
            "qz_a_inv": selected.qz_a_inv,
            "model_normalized": selected.model_normalized,
            "log_residuals_decades": selected.log_residuals_decades,
            "weighted_residuals": selected.weighted_residuals,
            "fit_included": context.dataset.fit_mask,
        }
    )


def _correlation_frame(context: DatasetExportData) -> pd.DataFrame:
    report = context.result.uncertainty
    if report is None:
        return pd.DataFrame(columns=["parameter"])
    rows = []
    for index, name in enumerate(report.correlation_names):
        row = {"parameter": name}
        row.update(
            dict(
                zip(
                    report.correlation_names,
                    report.correlation_matrix[index],
                    strict=True,
                )
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _profiles_frame(context: DatasetExportData) -> pd.DataFrame:
    columns = ["name", "value", "objective", "lower_closed", "upper_closed"]
    report = context.result.uncertainty
    if report is None:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "name": profile.name,
            "value": value,
            "objective": _finite_scalar(objective),
            "lower_closed": profile.lower_closed,
            "upper_closed": profile.upper_closed,
        }
        for profile in report.profiles
        for value, objective in zip(
            profile.values,
            profile.objectives,
            strict=True,
        )
    ]
    return pd.DataFrame(rows, columns=columns)


def _run_info_frame(context: DatasetExportData) -> pd.DataFrame:
    info = _run_info_payload(context, _project_dataset_document(context))
    row = {
        "dataset_id": info["dataset_id"],
        "dataset_directory": info["dataset_directory"],
        "dataset_directory_mapping": json_text(info["dataset_directory_mapping"]),
        "source_path": info["source_path"],
        "source_sha256": info["source_sha256"],
        "schema_version": info["schema_version"],
        "algorithm_version": info["algorithm_version"],
        "project_master_seed": info["project_master_seed"],
        "service_seed_tree_version": info["service_seed_tree_version"],
        "independent_root_child": info["independent_root_child"],
        "joint_root_child": info["joint_root_child"],
        "optimizer_child_seeds": json_text(info["optimizer_child_seeds"]),
        "mcmc_child_seed": info["mcmc_child_seed"],
        "selected_candidate_id": info["selected_candidate_id"],
        "fitted_instrument_parameters": json_text(
            info["fitted_instrument_parameters"]
        ),
        "confidence": info["confidence"],
        "candidate_count": len(context.result.candidates),
        "warnings": json_text(info["warnings"]),
        "beam": json_text(info["beam"]),
        "instrument": json_text(info["instrument"]),
        "scale_prior": json_text(info["scale_prior"]),
        "structure_evidence": json_text(info["structure_evidence"]),
        "oxide_decisions": json_text(info["oxide_decisions"]),
        "fringe_screen_threshold_version": info["fringe_screen_threshold_version"],
        "budget_reclaim_threshold_version": info["budget_reclaim_threshold_version"],
        "downsample_rule_version": info["downsample_rule_version"],
        "jacobian_version": info["jacobian_version"],
    }
    return pd.DataFrame([row])


def _workbook_bytes(frames: tuple[tuple[str, pd.DataFrame], ...]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(
        buffer,
        engine="xlsxwriter",
        engine_kwargs={"options": WORKBOOK_OPTIONS},
    ) as writer:
        writer.book.set_properties(
            {
                "title": "XRR deterministic export",
                "author": "xrr-fitter",
                "created": WORKBOOK_CREATED,
            }
        )
        for sheet_name, frame in frames:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()


def dataset_workbook_bytes(context: DatasetExportData) -> bytes:
    """Serialize the stable seven-sheet per-dataset workbook."""
    frames = (
        ("Parameters", _dataset_parameter_frame(context)),
        ("Candidates", _candidate_frame(context)),
        ("RawData", _raw_data_frame(context)),
        ("ModelResiduals", _model_frame(context)),
        ("Correlation", _correlation_frame(context)),
        ("Profiles", _profiles_frame(context)),
        ("RunInfo", _run_info_frame(context)),
    )
    return _workbook_bytes(frames)


def parameters_csv_bytes(context: DatasetExportData) -> bytes:
    """Serialize selected parameters with a stable UTF-8/LF dialect."""
    stream = StringIO(newline="")
    _parameter_frame(context).to_csv(
        stream,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return stream.getvalue().encode("utf-8")


def _summary_frame(contexts: tuple[DatasetExportData, ...]) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "confidence",
        "objective",
        "selected_candidate_id",
        "warnings",
    ]
    rows = [
        {
            "dataset_id": context.dataset.dataset_id,
            "confidence": context.result.confidence.value,
            "objective": context.selected.objective,
            "selected_candidate_id": context.selected.candidate_id,
            "warnings": json_text(context.result.warnings),
        }
        for context in contexts
    ]
    return pd.DataFrame(rows, columns=columns)


def _parameters_nm_frame(contexts: tuple[DatasetExportData, ...]) -> pd.DataFrame:
    rows = []
    suffixes = (".thickness_a", ".roughness_a", ".microslab_max_a")
    for context in contexts:
        rows.extend(
            {
                "dataset_id": context.dataset.dataset_id,
                "parameter_name": value.name,
                "value_angstrom": value.value,
                "value_nm": value.value / 10.0,
            }
            for value in context.selected.parameters
            if value.name.endswith(suffixes)
        )
    return pd.DataFrame(
        rows,
        columns=["dataset_id", "parameter_name", "value_angstrom", "value_nm"],
    )


def _curves_frame(contexts: tuple[DatasetExportData, ...]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "dataset_id": [context.dataset.dataset_id]
                * context.data.two_theta_deg.size,
                "two_theta_deg": context.data.two_theta_deg,
                "qz_a_inv": context.selected.qz_a_inv,
                "intensity_raw": context.data.intensity_raw,
                "intensity_normalized": context.data.intensity_normalized,
                "model_normalized": context.selected.model_normalized,
                "fit_included": context.dataset.fit_mask,
            }
        )
        for context in contexts
    ]
    return pd.concat(frames, ignore_index=True)


def _contexts(values: object) -> tuple[DatasetExportData, ...]:
    contexts = tuple(values)
    if not contexts:
        raise TypeError("contexts must contain DatasetExportData values")
    first = contexts[0]
    if not isinstance(first, DatasetExportData):
        raise TypeError("contexts must contain DatasetExportData values")
    project = first.project
    by_dataset = {id(first.dataset): first}
    for context in contexts[1:]:
        if not isinstance(context, DatasetExportData):
            raise TypeError("contexts must contain DatasetExportData values")
        if context.project is not project:
            raise ValueError("contexts must belong to the same project object")
        by_dataset[id(context.dataset)] = context
    expected = tuple(map(id, project.datasets))
    if len(by_dataset) != len(contexts) or set(by_dataset) != set(expected):
        raise ValueError("contexts must cover every project dataset exactly once")
    return tuple(map(by_dataset.__getitem__, expected))


def compatibility_workbook_bytes(contexts: object) -> bytes:
    """Serialize R22-compatible summary, nanometer, and curve sheets."""
    values = _contexts(contexts)
    return _workbook_bytes(
        (
            ("Summary", _summary_frame(values)),
            ("Parameters_nm", _parameters_nm_frame(values)),
            ("Curves", _curves_frame(values)),
        )
    )


def batch_workbook_bytes(contexts: object) -> bytes:
    """Serialize deterministic multi-dataset summary and parameter sheets."""
    values = _contexts(contexts)
    columns = ["dataset_id", "parameter_name", "value", "lower", "upper"]
    rows = [
        {
            "dataset_id": context.dataset.dataset_id,
            "parameter_name": parameter.name,
            "value": parameter.value,
            "lower": parameter.lower,
            "upper": parameter.upper,
        }
        for context in values
        for parameter in context.selected.parameters
    ]
    parameters = pd.DataFrame(rows, columns=columns)
    return _workbook_bytes(
        (("Summary", _summary_frame(values)), ("Parameters", parameters))
    )


def _diagnostic_qz_range(
    context: DatasetExportData,
    indices: tuple[int, ...],
) -> str:
    size = context.data.qz_a_inv.size
    valid = tuple(index for index in indices if 0 <= index < size)
    if not valid:
        return "[]"
    qz = context.data.qz_a_inv[list(valid)]
    return f"[{float(np.min(qz)):.12g},{float(np.max(qz)):.12g}]"


def _diagnostic_line(
    context: DatasetExportData,
    diagnostic: PhysicsDiagnostic,
) -> str:
    indices = _compact_json(diagnostic.point_indices)
    qz_range = _diagnostic_qz_range(context, diagnostic.point_indices)
    return (
        f"{diagnostic.code}: {diagnostic.message}; "
        f"full_data_indices={indices}; qz_a_inv_range={qz_range}"
    )


def _persisted_diagnostics(
    context: DatasetExportData,
) -> tuple[PhysicsDiagnostic, ...]:
    uncertainty = context.result.uncertainty
    if uncertainty is None:
        return context.selected.diagnostics
    return (*context.selected.diagnostics, *uncertainty.diagnostics)


def _rejected_surface_oxide(
    context: DatasetExportData,
    diagnostics: tuple[PhysicsDiagnostic, ...],
) -> bool:
    residual = any(
        value.code == "surface_thin_layer_residual" for value in diagnostics
    )
    return context.matching_surface_oxide_rejection and residual


def run_log_bytes(context: DatasetExportData) -> bytes:
    """Serialize stable warnings, seed lineage, stages, and diagnostics."""
    result = context.result
    identity = context.replay_identity
    mcmc = result.uncertainty.mcmc if result.uncertainty is not None else None
    lines = [
        f"dataset_id: {context.dataset.dataset_id}",
        f"confidence: {result.confidence.value}",
        f"candidate_count: {len(result.candidates)}",
        f"project_master_seed: {context.project.master_seed}",
        f"service_seed_tree_version: {identity.service_seed_tree_version}",
        f"independent_root_child: {identity.independent_root_child}",
        f"joint_root_child: {identity.joint_root_child}",
        f"optimizer_child_seeds: {_compact_json(result.child_seeds)}",
        f"mcmc_child_seed: {None if mcmc is None else mcmc.child_seed}",
    ]
    lines.extend(f"warning: {value}" for value in result.warnings)
    lines.extend(
        "stage "
        f"{stage.stage}: candidate_ids={_compact_json(stage.candidate_ids)}; "
        f"best_objective={stage.best_objective}; total_nfev={stage.total_nfev}; "
        f"stop_reasons={_compact_json(stage.stop_reasons)}"
        for stage in result.stage_summaries
    )
    diagnostics = _persisted_diagnostics(context)
    lines.extend(_diagnostic_line(context, value) for value in diagnostics)
    if _rejected_surface_oxide(context, diagnostics):
        lines.append("疑似缺失自然氧化层（此前已拒绝建议）")
    return ("\n".join(lines) + "\n").encode("utf-8")
