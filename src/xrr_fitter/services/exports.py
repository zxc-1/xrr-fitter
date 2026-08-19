"""Source-checked construction and atomic publication of result exports."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from xrr_fitter.io.export_plots import (
    fit_overview_png,
    parameter_trends_png,
    residuals_png,
    sld_profile_png,
)
from xrr_fitter.io.export_run import (
    ArtifactPayload,
    DatasetArtifacts,
    _dataset_directory,
    publish_export_run,
)
from xrr_fitter.io.export_tables import (
    PROJECT_SNAPSHOT_PATH,
    DatasetExportData,
    ExportReplayIdentity,
    batch_workbook_bytes,
    compatibility_workbook_bytes,
    dataset_json_bytes,
    dataset_workbook_bytes,
    run_log_bytes,
)
from xrr_fitter.io.orso import orso_bytes
from xrr_fitter.io.project_codec import project_to_bytes
from xrr_fitter.io.source import resolve_source_path
from xrr_fitter.model.export import ExportFileRecord, ExportManifest
from xrr_fitter.model.operations import ProjectFitResult
from xrr_fitter.model.project import DatasetProject, XrrProject
from xrr_fitter.services.datasets import _prepared_current, service_seed_branches
from xrr_fitter.services.projects import inspect_sources
from xrr_fitter.services.structures import suggest_oxide_layers


def load_export_data(project: XrrProject, dataset: DatasetProject):
    """Restore prepared source data through persisted import declarations."""
    return _prepared_current(project, dataset)


def _project(value: XrrProject | ProjectFitResult) -> XrrProject:
    if isinstance(value, ProjectFitResult):
        return value.updated_project
    if not isinstance(value, XrrProject):
        raise TypeError("result must be an XrrProject or ProjectFitResult")
    return value


def _require_current_sources(project: XrrProject) -> None:
    validation = inspect_sources(project)
    if validation.valid:
        return
    if validation.issues:
        raise ValueError(validation.issues[0].message)
    record = next((item for item in validation.datasets if item.status.value != "ok"), None)
    message = "source validation failed" if record is None else record.message
    raise ValueError(message)


def _selected_ids(project: XrrProject) -> dict[str, str]:
    return dict(project.ui_state.selected_candidate_ids)


def _selected_candidate(dataset: DatasetProject, selected_id: str | None):
    result = dataset.last_valid_result
    if result is None:
        raise ValueError(f"dataset {dataset.dataset_id} has no fit result")
    if selected_id is None:
        candidate = result.best_candidate
    else:
        candidate = next(
            (item for item in result.candidates if item.candidate_id == selected_id),
            None,
        )
    if candidate is None:
        raise ValueError(f"dataset {dataset.dataset_id} has no selected candidate")
    return candidate


def _matching_surface_rejection(dataset: DatasetProject) -> bool:
    if dataset.structure is None:
        return False
    suggestions = suggest_oxide_layers(dataset.structure)
    identities = {
        (
            suggestion.base_material,
            suggestion.oxide_material.formula,
            suggestion.location,
            suggestion.oxide_table_version,
        )
        for suggestion in suggestions
        if suggestion.location == "surface"
    }
    return any(
        not decision.accepted
        and (
            decision.base_material,
            decision.oxide_material,
            decision.location,
            decision.oxide_table_version,
        )
        in identities
        for decision in dataset.oxide_decisions
    )


def _contexts(
    project: XrrProject,
    project_reference: ExportFileRecord,
) -> tuple[DatasetExportData, ...]:
    mapping = tuple(
        (dataset.dataset_id, _dataset_directory(order, dataset.dataset_id))
        for order, dataset in enumerate(project.datasets, start=1)
    )
    independent, joint, _mcmc = service_seed_branches(project)
    selected = _selected_ids(project)
    return tuple(
        DatasetExportData(
            project=project,
            dataset=dataset,
            data=load_export_data(project, dataset),
            directory_mapping=mapping,
            selected=_selected_candidate(dataset, selected.get(dataset.dataset_id)),
            replay_identity=ExportReplayIdentity(
                1,
                independent[dataset.dataset_id],
                joint,
            ),
            matching_surface_oxide_rejection=_matching_surface_rejection(dataset),
            project_reference=project_reference,
        )
        for dataset in project.datasets
    )


def _snapshot_project(project: XrrProject) -> XrrProject:
    datasets = tuple(
        replace(
            dataset,
            source_path=str(resolve_source_path(project, dataset).resolve()),
        )
        for dataset in project.datasets
    )
    return replace(project, datasets=datasets, base_directory=None)


def _dataset_artifacts(context: DatasetExportData, *, include_ort: bool) -> DatasetArtifacts:
    files = [
        ArtifactPayload("fit_result.xlsx", dataset_workbook_bytes(context)),
        ArtifactPayload("fit_result.json", dataset_json_bytes(context)),
        ArtifactPayload("fit_overview.png", fit_overview_png(context)),
        ArtifactPayload("sld_profile.png", sld_profile_png(context)),
        ArtifactPayload("residuals.png", residuals_png(context)),
        ArtifactPayload("run_log.txt", run_log_bytes(context)),
    ]
    if include_ort:
        # 架构门禁禁止 ``services.exports`` 依赖 ``analysis`` 或 numpy，协方差矩阵改由
        # model 层 ``UncertaintyReport.covariance`` 派生（修正 9 的合规落点），服务层仅读取
        # 并透传，缺逐参数 sigma 时为 ``None``，导出即记录缺席原因。
        report = context.selected_uncertainty
        covariance = None if report is None else report.covariance
        files.append(ArtifactPayload("fit_result.ort", orso_bytes(context, covariance=covariance)))
    return DatasetArtifacts(context.dataset.dataset_id, tuple(files))


def _root_artifacts(
    contexts: tuple[DatasetExportData, ...],
) -> tuple[ArtifactPayload, ...]:
    values = [
        ArtifactPayload(
            "compatibility_summary.xlsx",
            compatibility_workbook_bytes(contexts),
        )
    ]
    if len(contexts) > 1:
        values.extend(
            (
                ArtifactPayload("batch_summary.xlsx", batch_workbook_bytes(contexts)),
                ArtifactPayload("parameter_trends.png", parameter_trends_png(contexts)),
            )
        )
    return tuple(values)


def export_result(
    result: XrrProject | ProjectFitResult,
    output_dir: str | Path,
    *,
    include_ort: bool = False,
) -> ExportManifest:
    """Validate, serialize, then atomically publish one complete export run.

    ``include_ort`` opts each dataset directory into an additional
    ``fit_result.ort`` artifact; left ``False`` the published tree is byte-for-byte
    identical to a run without ORSO support.
    """
    project = _project(result)
    if not project.datasets:
        raise ValueError("project has no datasets")
    _require_current_sources(project)
    snapshot = project_to_bytes(_snapshot_project(project))
    project_reference = ExportFileRecord(
        PROJECT_SNAPSHOT_PATH,
        len(snapshot),
        sha256(snapshot).hexdigest(),
    )
    contexts = _contexts(project, project_reference)
    datasets = tuple(_dataset_artifacts(context, include_ort=include_ort) for context in contexts)
    root_files = (
        ArtifactPayload(PROJECT_SNAPSHOT_PATH, snapshot),
        *_root_artifacts(contexts),
    )
    return publish_export_run(output_dir, datasets, root_files)
