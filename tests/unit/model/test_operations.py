from __future__ import annotations

from dataclasses import fields

import pytest

from tests.support.model_cases import final_fit_result, fit_result, project
from xrr_fitter.model.operations import (
    DatasetFitResult,
    FitReadiness,
    OperationError,
    OperationEvent,
    ProjectFitResult,
)


def _project_result() -> ProjectFitResult:
    current = project()
    return ProjectFitResult(
        mode="independent",
        datasets=(DatasetFitResult("curve", final_fit_result()),),
        warnings=(),
        updated_project=current,
    )


def test_project_fit_result_schema_is_immutable_and_identity_checked() -> None:
    result = _project_result()

    assert [field.name for field in fields(ProjectFitResult)] == [
        "mode",
        "datasets",
        "warnings",
        "updated_project",
        "cancelled",
    ]
    assert result.datasets[0].dataset_id == "curve"
    with pytest.raises(ValueError, match="mode"):
        ProjectFitResult("parallel", result.datasets, (), result.updated_project)


def test_operation_error_has_exact_serializable_schema() -> None:
    value = OperationError("ValueError", "invalid", "traceback text")

    assert [field.name for field in fields(OperationError)] == [
        "exception_type",
        "message",
        "traceback",
    ]
    assert value.exception_type == "ValueError"
    with pytest.raises(ValueError, match="exception_type"):
        OperationError("", "invalid", "traceback text")


@pytest.mark.parametrize(
    "kind",
    ["progress", "checkpoint", "fit_result", "mcmc_result", "cancelled", "error", "stopped"],
)
def test_operation_event_enforces_tagged_payload_schema(kind: str) -> None:
    from xrr_fitter.model.fitting import FitProgress

    payloads = {
        "progress": {"progress": FitProgress("curve", "stage-a", 1, 2, 3.0, "running")},
        "checkpoint": {"checkpoint": project()},
        "fit_result": {"fit_result": _project_result()},
        "mcmc_result": {"mcmc_result": project()},
        "cancelled": {"cancellation": "requested"},
        "error": {"error": OperationError("ValueError", "bad", "trace")},
        "stopped": {},
    }
    payload = payloads[kind]
    event = OperationEvent(sequence=0, kind=kind, **payload)

    assert event.kind == kind


def test_operation_event_rejects_wrong_or_multiple_payloads() -> None:
    result = _project_result()
    with pytest.raises(ValueError, match="exactly one"):
        OperationEvent(0, "fit_result")
    with pytest.raises(ValueError, match="payload"):
        OperationEvent(0, "fit_result", fit_result=result, cancellation="also set")
    with pytest.raises(ValueError, match="sequence"):
        OperationEvent(-1, "stopped")


def test_dataset_fit_result_requires_analyzed_final_result() -> None:
    with pytest.raises(TypeError, match="FitResult"):
        DatasetFitResult("curve", fit_result())


def test_fit_readiness_preserves_public_ready_message_schema() -> None:
    value = FitReadiness(False, "Structure required")

    assert [field.name for field in fields(FitReadiness)] == ["ready", "message"]
    assert value.message == "Structure required"
    with pytest.raises(ValueError, match="message"):
        FitReadiness(True, "")
