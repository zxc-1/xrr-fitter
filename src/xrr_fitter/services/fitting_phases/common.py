"""Immutable fitting service value contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.fitting import FitEvaluationContext, FitProgress
from xrr_fitter.model.project import DatasetProject, XrrProject

ProgressCallback = Callable[[FitProgress], None]
CheckpointCallback = Callable[[XrrProject], None]
CancellationProbe = Callable[[], bool]


def _validate_automatic_result(
    fit_result: FitResult,
    passed: bool,
    reason: str | None,
) -> None:
    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be FitResult")
    if not isinstance(passed, bool):
        raise TypeError("passed must be bool")
    if passed and reason is not None:
        raise ValueError("passed result must not have a reason")
    if not passed and not reason:
        raise ValueError("failed quality decision requires a reason")


@dataclass(frozen=True, slots=True)
class PreparedDatasetFit:
    """One source-checked, compiled dataset ready for service execution."""

    dataset_id: str
    dataset_index: int
    updated_dataset: DatasetProject
    problem: FitEvaluationContext


@dataclass(frozen=True, slots=True)
class AutomaticPreparedResult:
    """Automatic fit output plus the quality gate that owns publication."""

    prepared: PreparedDatasetFit
    fit_result: FitResult
    passed: bool
    reason: str | None

    def __post_init__(self) -> None:
        _validate_automatic_result(self.fit_result, self.passed, self.reason)
