"""Immutable records for deterministic export publication manifests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from xrr_fitter.model.analysis import ConfidenceClass

EXPORT_MODES = ("independent", "joint")


class ExportFormat(StrEnum):
    """One serialization a publication may contain, named by its file extension.

    A format is a choice of *rendering*, not of content: XLSX, JSON and PNG carry the
    same fit result three ways, so dropping one drops that rendering and nothing the
    others still say. Format-neutral provenance -- the run log, the project snapshot,
    the manifest -- is deliberately outside this enum: no format selection can switch
    off the record of what was published.
    """

    XLSX = "xlsx"
    JSON = "json"
    PNG = "png"
    ORT = "ort"
    CSV = "csv"
    SVG = "svg"

    def __repr__(self) -> str:
        """Name the member instead of spelling out ``<ExportFormat.XLSX: 'xlsx'>``.

        ``inspect.signature`` renders a default by its ``repr``, and the default format
        set is pinned in the public-API golden. Enum's own ``repr`` would put the value
        twice into that golden line and make it unreadable at exactly the place a
        reader goes to learn what a default publication contains.
        """
        return f"{type(self).__name__}.{self.name}"


# 默认发布集：不含 ORT/CSV/SVG，于是默认那棵树与这三种可选格式存在之前逐位一致。
DEFAULT_FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat.XLSX,
    ExportFormat.JSON,
    ExportFormat.PNG,
)


def normalize_export_formats(values: Iterable[ExportFormat | str]) -> frozenset[ExportFormat]:
    """Coerce a requested format set, rejecting an empty one and any unknown name.

    An empty set would publish a tree with provenance and no result, which no caller
    can have meant; a misspelled name would silently drop the format the caller asked
    for. Both were unreachable while the parameter was a row of booleans, so they are
    checked here rather than left to the artifact builders.
    """
    formats = frozenset(ExportFormat(value) for value in values)
    if not formats:
        raise ValueError("export requires at least one export format")
    return formats


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


def _plan_dataset_ids(values: object) -> tuple[str, ...]:
    identifiers = tuple(values)
    if not identifiers:
        raise ValueError("a plan must describe at least one dataset")
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("dataset IDs must be nonempty names")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset IDs must be unique")
    return identifiers


def _export_mode(value: str) -> str:
    if value not in EXPORT_MODES:
        raise ValueError(f"unsupported export mode: {value}")
    return value


def _stage_names(values: object) -> tuple[str, ...]:
    # The descriptor names each stage the search actually reached, once, in the
    # order it was first reached. Repetition is normalized by the producer, so a
    # duplicate here means the caller lost the stage history's ordering.
    stages = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in stages):
        raise ValueError("stages must be nonempty names")
    if len(stages) != len(set(stages)):
        raise ValueError("stages must not repeat")
    return stages


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
    """Complete immutable record of one atomically published export run.

    The file records describe *what* was published; ``mode`` through
    ``reproducible`` describe the run that produced it, so a reader does not
    have to re-derive them from the project snapshot. Publication itself cannot
    know them, so each defaults to its weakest reading: an unpopulated
    descriptor never claims more than the run earned.
    """

    run_directory: Path
    datasets: tuple[DatasetExportManifest, ...]
    root_files: tuple[ExportFileRecord, ...]
    mode: str = "independent"
    stages: tuple[str, ...] = ()
    confidence: ConfidenceClass = ConfidenceClass.UNTRUSTED
    reproducible: bool = False

    def __post_init__(self) -> None:
        run_directory = _run_directory(self.run_directory)
        datasets = _typed_records(self.datasets, DatasetExportManifest, "datasets")
        root_files = _typed_records(self.root_files, ExportFileRecord, "root_files")
        _validate_dataset_ids(datasets)
        _validate_manifest_paths(datasets, root_files)
        stages = _stage_names(self.stages)
        if not isinstance(self.confidence, ConfidenceClass):
            raise TypeError("confidence must be ConfidenceClass")
        if not isinstance(self.reproducible, bool):
            raise TypeError("reproducible must be a bool")
        object.__setattr__(self, "run_directory", run_directory)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "root_files", root_files)
        object.__setattr__(self, "mode", _export_mode(self.mode))
        object.__setattr__(self, "stages", stages)

    @property
    def files(self) -> tuple[ExportFileRecord, ...]:
        return self.root_files + tuple(record for dataset in self.datasets for record in dataset.files)


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """What an export run *would* conclude, stated before anything is published.

    The dialog has to show ``mode`` through ``reproducible`` while the user is
    still choosing a destination, so the same descriptors an ``ExportManifest``
    reports on the way out are answered here on the way in. A plan owns no
    ``run_directory`` and no file records: there is no directory yet and nothing
    has been written, so the type makes it impossible to mistake a plan for a
    publication. The defaults match ``ExportManifest``'s weakest reading.
    """

    dataset_ids: tuple[str, ...]
    mode: str = "independent"
    stages: tuple[str, ...] = ()
    confidence: ConfidenceClass = ConfidenceClass.UNTRUSTED
    reproducible: bool = False

    def __post_init__(self) -> None:
        dataset_ids = _plan_dataset_ids(self.dataset_ids)
        stages = _stage_names(self.stages)
        if not isinstance(self.confidence, ConfidenceClass):
            raise TypeError("confidence must be ConfidenceClass")
        if not isinstance(self.reproducible, bool):
            raise TypeError("reproducible must be a bool")
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "mode", _export_mode(self.mode))
        object.__setattr__(self, "stages", stages)
