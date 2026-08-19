"""API workflow execution and atomic candidate publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import xrr_fitter.api as api
from tests.support.approved_workflow_model import (
    CANDIDATE_SCHEMA,
    CASE_SPECS,
    NOT_RUN,
    OPERATIONS,
    ROOT,
    CaseRecord,
    CaseSpec,
    FileRecord,
    RunRecord,
    candidate_environment,
    canonical_json_bytes,
    configuration_sha256,
    file_record,
    file_value,
    normalize_source_paths,
    owner_source,
    workflow_contract_sha256,
)

EnvironmentBuilder = Callable[[], dict[str, object]]
RunExecutor = Callable[[CaseSpec, Path, int, int, Path], RunRecord]


def _prepared_project(spec: CaseSpec, source: Path, seed: int):
    project = api.load_project(ROOT / "examples" / spec.template_name)
    project = replace(project, fit_config=replace(project.fit_config, master_seed=seed))
    dataset_id = project.datasets[0].dataset_id
    preview = api.preview_source_update(project, dataset_id, source)
    return api.accept_source_update(project, preview)


def _fitted_project(spec: CaseSpec, source: Path, seed: int):
    project = _prepared_project(spec, source, seed)
    readiness = api.preflight_fit(project)
    if not readiness.ready:
        raise ValueError(f"approved workflow is not fit-ready: {spec.case_id}: {readiness.reason}")
    result = api.fit_project(project)
    if result.cancelled or len(result.datasets) != 1:
        raise ValueError(f"approved workflow did not complete: {spec.case_id}")
    fit_result = result.datasets[0].fit_result
    best = fit_result.best_candidate
    if best is None:
        raise ValueError(f"approved workflow produced no candidate: {spec.case_id}")
    return api.select_candidate(result.updated_project, result.datasets[0].dataset_id, best.candidate_id)


def _published_export(project, output: Path) -> Path:
    publication = output / "publication"
    manifest = api.export_result(project, publication)
    target = output / "exports"
    manifest.run_directory.rename(target)
    publication.rmdir()
    return target


def _evidence_records(
    export_root: Path,
    logical_prefix: str,
) -> tuple[tuple[FileRecord, ...], tuple[FileRecord, ...]]:
    records = tuple(
        file_record(path, f"{logical_prefix}/exports/{path.relative_to(export_root).as_posix()}")
        for path in export_root.rglob("*")
        if path.is_file()
    )
    exports = tuple(sorted((item for item in records if not item.path.endswith(".png")), key=lambda item: item.path))
    plots = tuple(sorted((item for item in records if item.path.endswith(".png")), key=lambda item: item.path))
    if not exports or not plots:
        raise ValueError("approved workflow must publish exports and plots")
    return exports, plots


def execute_run(
    spec: CaseSpec,
    source: Path,
    seed: int,
    ordinal: int,
    staging: Path,
) -> RunRecord:
    run_root = staging / spec.case_id / f"run-{ordinal}"
    run_root.mkdir(parents=True)
    project = _fitted_project(spec, source, seed)
    project_path = run_root / "project.xrrproj.json"
    api.save_project(project, project_path)
    reopened = api.load_project(project_path)
    export_root = _published_export(reopened, run_root)
    result_files = tuple(export_root.rglob("fit_result.json"))
    if len(result_files) != 1:
        raise ValueError("approved workflow must publish exactly one fit_result.json")
    normalized = json.loads(result_files[0].read_text(encoding="utf-8"))
    normalized = normalize_source_paths(normalized, spec.source_name)
    canonical_json_bytes(normalized)
    prefix = f"runs/{spec.case_id}/run-{ordinal}"
    exports, plots = _evidence_records(export_root, prefix)
    return RunRecord(
        ordinal,
        seed,
        file_record(project_path, f"{prefix}/project.xrrproj.json"),
        exports,
        plots,
        normalized,
    )


def _observation(run: RunRecord) -> dict[str, object]:
    info = run.normalized_result.get("run_info", run.normalized_result)
    if not isinstance(info, dict):
        raise ValueError("normalized run_info must be an object")
    return {
        "ordinal": run.ordinal,
        "seed": run.seed,
        "confidence": info.get("confidence"),
        "warnings": info.get("warnings", []),
        "selected_candidate_id": info.get("selected_candidate_id"),
    }


def _capture_case(
    spec: CaseSpec,
    root: Path,
    staging: Path,
    run_executor: RunExecutor,
) -> CaseRecord:
    source_path = owner_source(root, spec)
    source = file_record(source_path, spec.source_name)
    seeds = (spec.repeated_seed, spec.repeated_seed, spec.repeated_seed, spec.fresh_seed)
    runs = tuple(
        run_executor(
            spec=spec,
            source=source_path,
            seed=seed,
            ordinal=ordinal,
            staging=staging,
        )
        for ordinal, seed in enumerate(seeds, start=1)
    )
    if not (runs[0].normalized_result == runs[1].normalized_result == runs[2].normalized_result):
        raise ValueError(f"same-seed workflow is not reproducible: {spec.case_id}")
    normalized = {
        "status": NOT_RUN,
        "same_seed_reproducible": True,
        "owner_review_required": True,
        "observations": [_observation(run) for run in runs],
    }
    conclusion = f"owner review required for {spec.case_id}; no domain verdict recorded"
    return CaseRecord(
        spec.case_id,
        source,
        configuration_sha256(spec, source),
        OPERATIONS,
        runs,
        normalized,
        conclusion,
    )


def _run_value(run: RunRecord) -> dict[str, object]:
    return {
        "ordinal": run.ordinal,
        "seed": run.seed,
        "project": file_value(run.project),
        "exports": [file_value(item) for item in run.exports],
        "plots": [file_value(item) for item in run.plots],
        "normalized_result": run.normalized_result,
    }


def _case_value(case: CaseRecord) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "source": file_value(case.source),
        "configuration_sha256": case.configuration_sha256,
        "operations": list(case.operations),
        "runs": [_run_value(item) for item in case.runs],
        "normalized_result": case.normalized_result,
        "conclusion": case.conclusion,
    }


def _candidate_value(
    records: tuple[CaseRecord, ...],
    environment_builder: EnvironmentBuilder,
) -> dict[str, object]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "environment": environment_builder(),
        "workflow_contract_sha256": workflow_contract_sha256(),
        "cases": [_case_value(item) for item in records],
    }


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


def _candidate_temp(report: Path, content: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".candidate-", suffix=".tmp", dir=report)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _remove(Path(name))
        raise
    return Path(name)


def _publish_capture(report: Path, staging: Path, content: bytes) -> None:
    runs = report / "runs"
    candidate = report / "approved-data-candidate.json"
    temporary = _candidate_temp(report, content)
    published_runs = False
    try:
        staging.rename(runs)
        published_runs = True
        os.link(temporary, candidate)
        temporary.unlink()
    except BaseException:
        _remove(temporary)
        if published_runs:
            _remove(runs)
        raise


def _prepare_report(report: Path) -> None:
    if report.is_symlink():
        raise ValueError("approved report directory must not be a symlink")
    report.mkdir(parents=True, exist_ok=True)
    for name in ("runs", "approved-data-candidate.json"):
        if os.path.lexists(report / name):
            raise FileExistsError(f"approved report output already exists: {name}")


def run_api_acceptance(
    owner_root: str | Path,
    report_dir: str | Path,
    *,
    environment_builder: EnvironmentBuilder = candidate_environment,
    run_executor: RunExecutor = execute_run,
) -> tuple[CaseRecord, ...]:
    root = Path(owner_root)
    report = Path(report_dir)
    for spec in CASE_SPECS:
        owner_source(root, spec)
    _prepare_report(report)
    staging = Path(tempfile.mkdtemp(prefix=".capture-", dir=report))
    try:
        records = tuple(_capture_case(spec, root, staging, run_executor) for spec in CASE_SPECS)
        content = canonical_json_bytes(_candidate_value(records, environment_builder))
        _publish_capture(report, staging, content)
        return records
    except BaseException:
        _remove(staging)
        raise
