"""Public post-delivery owner-data acceptance helpers."""

from __future__ import annotations

from pathlib import Path

from tests.support.approved_workflow_capture import (
    execute_run as _execute_run,
)
from tests.support.approved_workflow_capture import (
    run_api_acceptance as _run_api_acceptance,
)
from tests.support.approved_workflow_gui import (
    verify_gui_acceptance as _run_gui_acceptance,
)
from tests.support.approved_workflow_gui import (
    verify_gui_case as _verify_gui_case,
)
from tests.support.approved_workflow_model import (
    CaseRecord,
    CaseSpec,
    FileRecord,
    GuiResult,
    RunRecord,
    canonical_json_bytes,
)
from tests.support.approved_workflow_model import (
    candidate_environment as _candidate_environment,
)

__all__ = [
    "CaseRecord",
    "CaseSpec",
    "FileRecord",
    "GuiResult",
    "RunRecord",
    "canonical_json_bytes",
    "run_api_acceptance",
    "verify_gui_acceptance",
]


def run_api_acceptance(
    owner_root: str | Path,
    report_dir: str | Path,
) -> tuple[CaseRecord, ...]:
    return _run_api_acceptance(
        owner_root,
        report_dir,
        environment_builder=_candidate_environment,
        run_executor=_execute_run,
    )


def verify_gui_acceptance(
    owner_root: str | Path,
    report_dir: str | Path,
) -> tuple[GuiResult, ...]:
    return _run_gui_acceptance(
        owner_root,
        report_dir,
        case_verifier=_verify_gui_case,
    )
