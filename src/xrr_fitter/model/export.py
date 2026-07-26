"""Immutable records for deterministic export publication manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def _relative(value: str, field: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a normalized relative POSIX path")
    return value


def _sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("file SHA-256 must be lowercase hexadecimal")


def _typed_records(values: object, expected: type, field: str) -> tuple:
    records = tuple(values)
    if any(not isinstance(value, expected) for value in records):
        raise TypeError(f"{field} contain invalid values")
    return records


def _sorted_unique_paths(records: tuple[ExportFileRecord, ...]) -> None:
    paths = tuple(value.path for value in records)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("dataset file records must be sorted and unique")


def _run_directory(value: Path) -> Path:
    directory = Path(value)
    if not directory.name or directory in {Path("."), Path("..")}:
        raise ValueError("run_directory must identify a directory")
    return directory


def _validate_dataset_ids(datasets: tuple[DatasetExportManifest, ...]) -> None:
    identifiers = tuple(value.dataset_id for value in datasets)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset export IDs must be unique")


def _validate_manifest_paths(
    datasets: tuple[DatasetExportManifest, ...],
    root_files: tuple[ExportFileRecord, ...],
) -> None:
    paths = tuple(value.path for value in root_files) + tuple(
        record.path for dataset in datasets for record in dataset.files
    )
    if len(paths) != len(set(paths)):
        raise ValueError("export manifest contains duplicate file paths")


@dataclass(frozen=True, slots=True)
class ExportFileRecord:
    """One published file bound to its relative path, size, and digest."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _relative(self.path, "file path")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size <= 0:
            raise ValueError("file size must be a positive integer")
        _sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class DatasetExportManifest:
    """Sorted publication records owned by one dataset directory."""

    dataset_id: str
    directory: str
    files: tuple[ExportFileRecord, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        _relative(self.directory, "dataset directory")
        files = _typed_records(self.files, ExportFileRecord, "dataset files")
        _sorted_unique_paths(files)
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Complete immutable record of one atomically published export run."""

    run_directory: Path
    datasets: tuple[DatasetExportManifest, ...]
    root_files: tuple[ExportFileRecord, ...]

    def __post_init__(self) -> None:
        run_directory = _run_directory(self.run_directory)
        datasets = _typed_records(self.datasets, DatasetExportManifest, "datasets")
        root_files = _typed_records(self.root_files, ExportFileRecord, "root_files")
        _validate_dataset_ids(datasets)
        _validate_manifest_paths(datasets, root_files)
        object.__setattr__(self, "run_directory", run_directory)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "root_files", root_files)

    @property
    def files(self) -> tuple[ExportFileRecord, ...]:
        return self.root_files + tuple(
            record for dataset in self.datasets for record in dataset.files
        )
