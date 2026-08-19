"""Dataset lookup, source-path resolution, and raw-byte validation."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from xrr_fitter.model.project import (
    DatasetProject,
    DatasetSourceValidation,
    ProjectValidation,
    SourceStatus,
    XrrProject,
)


def dataset_index(project: XrrProject, dataset_id: str) -> int:
    """Return the unique persisted position for a dataset ID."""
    matches = tuple(index for index, dataset in enumerate(project.datasets) if dataset.dataset_id == dataset_id)
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate dataset: {dataset_id}")
    return matches[0]


def dataset_by_id(project: XrrProject, dataset_id: str) -> DatasetProject:
    """Return the unique dataset value for a stable persisted ID."""
    return project.datasets[dataset_index(project, dataset_id)]


def resolve_source_path(project: XrrProject, dataset: DatasetProject) -> Path:
    """Resolve a source for I/O without changing its persisted declaration."""
    source = Path(dataset.source_path)
    if source.is_absolute():
        return source
    if not project.base_directory:
        raise ValueError("base_directory is required for relative source paths")
    return Path(project.base_directory) / source


def _record(
    dataset: DatasetProject,
    source: Path,
    status: SourceStatus,
    actual_sha256: str | None,
    message: str,
) -> DatasetSourceValidation:
    return DatasetSourceValidation(
        dataset_id=dataset.dataset_id,
        status=status,
        expected_sha256=dataset.source_sha256,
        actual_sha256=actual_sha256,
        message=message,
        source_identity=str(source.absolute()),
    )


def validate_source(
    project: XrrProject,
    dataset: DatasetProject,
) -> DatasetSourceValidation:
    """Hash current raw bytes and classify availability without parsing them."""
    source = resolve_source_path(project, dataset)
    try:
        actual = sha256(source.read_bytes()).hexdigest()
    except FileNotFoundError:
        return _record(
            dataset,
            source,
            SourceStatus.MISSING,
            None,
            f"数据集 {dataset.dataset_id} 的源文件不存在: {source}",
        )
    except OSError as error:
        return _record(
            dataset,
            source,
            SourceStatus.UNREADABLE,
            None,
            f"数据集 {dataset.dataset_id} 的源文件不可读取: {error}",
        )
    status = SourceStatus.OK if actual == dataset.source_sha256 else SourceStatus.HASH_MISMATCH
    message = (
        f"数据集 {dataset.dataset_id} 的源文件校验通过"
        if status is SourceStatus.OK
        else f"数据集 {dataset.dataset_id} 的源文件已变化"
    )
    return _record(dataset, source, status, actual, message)


def validate_sources(project: XrrProject) -> ProjectValidation:
    """Return ordered source observations without mutating project state."""
    return ProjectValidation(tuple(validate_source(project, dataset) for dataset in project.datasets))
