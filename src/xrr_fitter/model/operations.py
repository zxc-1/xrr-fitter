"""Cross-dataset result, readiness, and worker event value records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.automation import ImportFailure
from xrr_fitter.model.fitting import FitProgress
from xrr_fitter.model.project import XrrProject


def _nonempty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


def _dataset_results(values: object) -> tuple[DatasetFitResult, ...]:
    datasets = tuple(values)
    if any(not isinstance(value, DatasetFitResult) for value in datasets):
        raise TypeError("datasets must contain DatasetFitResult values")
    identifiers = tuple(value.dataset_id for value in datasets)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("project fit dataset IDs must be unique")
    return datasets


@dataclass(frozen=True, slots=True)
class DatasetFitResult:
    """One analyzed result bound to its stable project dataset ID."""

    dataset_id: str
    fit_result: FitResult

    def __post_init__(self) -> None:
        _nonempty(self.dataset_id, "dataset_id")
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be FitResult")


@dataclass(frozen=True, slots=True)
class ProjectFitResult:
    """Published result of an independent or joint project fit."""

    mode: str
    datasets: tuple[DatasetFitResult, ...]
    warnings: tuple[str, ...]
    updated_project: XrrProject
    cancelled: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"independent", "joint"}:
            raise ValueError("mode must be independent or joint")
        datasets = _dataset_results(self.datasets)
        if not isinstance(self.updated_project, XrrProject):
            raise TypeError("updated_project must be XrrProject")
        if not isinstance(self.cancelled, bool):
            raise TypeError("cancelled must be bool")
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class ProjectImportResult:
    updated_project: XrrProject
    import_batch_id: str
    imported_dataset_ids: tuple[str, ...]
    failures: tuple[ImportFailure, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.updated_project, XrrProject):
            raise TypeError("updated_project must be XrrProject")
        if not self.import_batch_id.strip():
            raise ValueError("import_batch_id must not be empty")
        identifiers = tuple(self.imported_dataset_ids)
        invalid = any(not value for value in identifiers)
        if len(identifiers) != len(set(identifiers)) or invalid:
            raise ValueError("imported_dataset_ids must be unique and nonempty")
        failures = tuple(self.failures)
        if any(not isinstance(value, ImportFailure) for value in failures):
            raise TypeError("failures must contain ImportFailure values")
        object.__setattr__(self, "imported_dataset_ids", identifiers)
        object.__setattr__(self, "failures", failures)


@dataclass(frozen=True, slots=True)
class OperationError:
    """Serializable exception identity, message, and formatted traceback."""

    exception_type: str
    message: str
    traceback: str

    def __post_init__(self) -> None:
        _nonempty(self.exception_type, "exception_type")
        _nonempty(self.message, "message")
        _nonempty(self.traceback, "traceback")


OperationKind = Literal[
    "progress",
    "checkpoint",
    "fit_result",
    "mcmc_result",
    "cancelled",
    "error",
    "stopped",
]

EVENT_PAYLOAD_FIELDS = {
    "progress": "progress",
    "checkpoint": "checkpoint",
    "fit_result": "fit_result",
    "mcmc_result": "mcmc_result",
    "cancelled": "cancellation",
    "error": "error",
    "stopped": None,
}


def _event_sequence(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("sequence must be a nonnegative integer")


def _event_payloads(event: OperationEvent) -> dict[str, object]:
    return {
        "progress": event.progress,
        "checkpoint": event.checkpoint,
        "fit_result": event.fit_result,
        "mcmc_result": event.mcmc_result,
        "cancellation": event.cancellation,
        "error": event.error,
    }


def _validate_event_tag(kind: str, present: tuple[str, ...]) -> None:
    if kind not in EVENT_PAYLOAD_FIELDS:
        raise ValueError(f"unsupported operation event kind: {kind}")
    required = EVENT_PAYLOAD_FIELDS[kind]
    if required is None and present:
        raise ValueError("stopped event payload fields must all be None")
    if required is not None and not present:
        raise ValueError("event must have exactly one matching payload")
    if required is not None and present != (required,):
        raise ValueError("event payload does not match kind")


def _validate_event_types(event: OperationEvent) -> None:
    expected_types = (
        (event.progress, FitProgress, "progress"),
        (event.checkpoint, XrrProject, "checkpoint"),
        (event.fit_result, ProjectFitResult, "fit_result"),
        (event.mcmc_result, XrrProject, "mcmc_result"),
        (event.error, OperationError, "error"),
    )
    for value, expected, field in expected_types:
        if value is not None and not isinstance(value, expected):
            raise TypeError(f"{field} payload has an invalid type")
    if event.cancellation is not None:
        _nonempty(event.cancellation, "cancellation reason")


@dataclass(frozen=True, slots=True)
class OperationEvent:
    """Exactly one tagged worker payload with a monotonic sequence number."""

    sequence: int
    kind: OperationKind
    progress: FitProgress | None = None
    checkpoint: XrrProject | None = None
    fit_result: ProjectFitResult | None = None
    mcmc_result: XrrProject | None = None
    cancellation: str | None = None
    error: OperationError | None = None

    def __post_init__(self) -> None:
        _event_sequence(self.sequence)
        payloads = _event_payloads(self)
        present = tuple(name for name, value in payloads.items() if value is not None)
        _validate_event_tag(self.kind, present)
        _validate_event_types(self)


@dataclass(frozen=True, slots=True)
class FitReadiness:
    """User-facing preflight readiness and its display message."""

    ready: bool
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool):
            raise TypeError("ready must be bool")
        if not self.message.strip():
            raise ValueError("message must not be empty")
