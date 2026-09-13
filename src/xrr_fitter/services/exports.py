"""Source-checked construction and atomic publication of result exports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from xrr_fitter.io.export_log import run_log_bytes
from xrr_fitter.io.export_plots import (
    fit_overview_png,
    parameter_trends_png,
    residuals_png,
    sld_profile_png,
    sld_profile_svg,
)
from xrr_fitter.io.export_run import (
    ArtifactProducer,
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
    parameters_csv_bytes,
)
from xrr_fitter.io.orso import orso_bytes
from xrr_fitter.io.project_codec import project_to_bytes
from xrr_fitter.io.source import resolve_source_path
from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.export import (
    DEFAULT_FORMATS,
    ExportFileRecord,
    ExportFormat,
    ExportManifest,
    ExportPlan,
    normalize_export_formats,
)
from xrr_fitter.model.fitting import FitCandidate
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


def _dataset_artifacts(
    context: DatasetExportData,
    *,
    formats: frozenset[ExportFormat],
) -> DatasetArtifacts:
    files: list[ArtifactProducer] = []
    if ExportFormat.XLSX in formats:
        files.append(ArtifactProducer("fit_result.xlsx", lambda: dataset_workbook_bytes(context)))
    if ExportFormat.JSON in formats:
        files.append(ArtifactProducer("fit_result.json", lambda: dataset_json_bytes(context)))
    if ExportFormat.PNG in formats:
        files.extend(
            (
                ArtifactProducer("fit_overview.png", lambda: fit_overview_png(context)),
                ArtifactProducer("sld_profile.png", lambda: sld_profile_png(context)),
                ArtifactProducer("residuals.png", lambda: residuals_png(context)),
            )
        )
    # 运行日志不挂在任何格式上：它记的是这次发布本身（用了哪些源、哪个候选解），换一种
    # 渲染格式并不改变这件事，选不选格式也不该关掉这条记录。
    files.append(ArtifactProducer("run_log.txt", lambda: run_log_bytes(context)))
    if ExportFormat.CSV in formats:
        files.append(ArtifactProducer("parameters.csv", lambda: parameters_csv_bytes(context)))
    if ExportFormat.SVG in formats:
        files.append(ArtifactProducer("sld_profile.svg", lambda: sld_profile_svg(context)))
    if ExportFormat.ORT in formats:
        # 架构门禁禁止 ``services.exports`` 依赖 ``analysis`` 或 numpy，协方差矩阵改由
        # model 层 ``UncertaintyReport.covariance`` 派生（修正 9 的合规落点），服务层仅读取
        # 并透传，缺逐参数 sigma 时为 ``None``，导出即记录缺席原因。
        def render_orso() -> bytes:
            report = context.selected_uncertainty
            covariance = None if report is None else report.covariance
            return orso_bytes(context, covariance=covariance)

        files.append(ArtifactProducer("fit_result.ort", render_orso))
    return DatasetArtifacts(context.dataset.dataset_id, tuple(files))


def _root_artifacts(
    contexts: tuple[DatasetExportData, ...],
    *,
    formats: frozenset[ExportFormat],
) -> tuple[ArtifactProducer, ...]:
    values: list[ArtifactProducer] = []
    if ExportFormat.XLSX in formats:
        values.append(
            ArtifactProducer(
                "compatibility_summary.xlsx",
                lambda: compatibility_workbook_bytes(contexts),
            )
        )
    batch = len(contexts) > 1
    if batch and ExportFormat.XLSX in formats:
        values.append(ArtifactProducer("batch_summary.xlsx", lambda: batch_workbook_bytes(contexts)))
    if batch and ExportFormat.PNG in formats:
        values.append(ArtifactProducer("parameter_trends.png", lambda: parameter_trends_png(contexts)))
    return tuple(values)


def _run_stages(results: tuple[FitResult, ...]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for result in results:
        for summary in result.stage_summaries:
            ordered.setdefault(summary.stage, None)
    return tuple(ordered)


def _weakest_confidence(results: tuple[FitResult, ...]) -> ConfidenceClass:
    # ``ConfidenceClass`` declares its members from strongest to weakest, so the
    # declaration position is the severity order; a batch is only as trustworthy
    # as its least trustworthy published dataset.
    severity = tuple(ConfidenceClass)
    return max((result.confidence for result in results), key=severity.index)


def _replayable(selected: tuple[FitCandidate, ...]) -> bool:
    # ``fit.stages`` hands out ``seed_index`` by ``enumerate(child_seeds)``, and
    # Stage-B archives deliberately overwrite it with -1. Publishing such a
    # candidate leaves no seed to replay the run from.
    return all(candidate.seed_index >= 0 for candidate in selected)


def _described(
    manifest: ExportManifest,
    project: XrrProject,
    contexts: tuple[DatasetExportData, ...],
) -> ExportManifest:
    """Describe the run that produced an already published manifest.

    Publication cannot know these conclusions, and they deliberately stay out of
    the on-disk ``export_manifest.json``: the published bytes remain unchanged
    while the returned record explains where they came from.
    """
    return replace(
        manifest,
        mode=project.batch_mode,
        stages=_run_stages(tuple(context.result for context in contexts)),
        confidence=_weakest_confidence(tuple(context.result for context in contexts)),
        reproducible=_replayable(tuple(context.selected for context in contexts)),
    )


def describe_export_plan(result: XrrProject | ProjectFitResult) -> ExportPlan:
    """State the conclusions an export run would reach, before anything is published.

    The dialog has to preview ``mode``/``stages``/``confidence``/``reproducible``
    while the user is still choosing formats and a destination. Recomputing them in
    the GUI would fork the derivation, so this reaches the same three helpers that
    ``_described`` uses on the way out. Every value comes from the project itself:
    no source data is loaded and no bytes are written.
    """
    project = _project(result)
    if not project.datasets:
        raise ValueError("project has no datasets")
    selected_ids = _selected_ids(project)
    results: list[FitResult] = []
    selected: list[FitCandidate] = []
    for dataset in project.datasets:
        fit_result = dataset.last_valid_result
        if fit_result is None:
            raise ValueError(f"dataset {dataset.dataset_id} has no fit result")
        results.append(fit_result)
        selected.append(_selected_candidate(dataset, selected_ids.get(dataset.dataset_id)))
    return ExportPlan(
        dataset_ids=tuple(dataset.dataset_id for dataset in project.datasets),
        mode=project.batch_mode,
        stages=_run_stages(tuple(results)),
        confidence=_weakest_confidence(tuple(results)),
        reproducible=_replayable(tuple(selected)),
    )


def export_result(
    result: XrrProject | ProjectFitResult,
    output_dir: str | Path,
    *,
    formats: Sequence[ExportFormat] = DEFAULT_FORMATS,
) -> ExportManifest:
    """Validate, serialize, then atomically publish one complete export run.

    ``formats`` selects which renderings the run contains; it is a set, so the order it
    is given in does not reach the published tree. Left at :data:`DEFAULT_FORMATS` the
    tree is byte-for-byte identical to a run from before ORT, CSV and SVG existed.
    Provenance -- ``run_log.txt``, the project snapshot, the manifest -- is published
    whatever the selection, because it records the publication rather than a rendering
    of the result.
    """
    selected = normalize_export_formats(formats)
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
    datasets = tuple(_dataset_artifacts(context, formats=selected) for context in contexts)
    root_files = (
        ArtifactProducer(PROJECT_SNAPSHOT_PATH, lambda: snapshot),
        *_root_artifacts(contexts, formats=selected),
    )
    manifest = publish_export_run(output_dir, datasets, root_files)
    return _described(manifest, project, contexts)
