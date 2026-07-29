from __future__ import annotations

import os
from pathlib import Path

from tests.support.approved_workflows import run_api_acceptance


def test_real_data_workflows_produce_four_run_candidate_records() -> None:
    root = Path(os.environ["XRR_APPROVED_DATA_ROOT"])
    report_dir = Path(os.environ["XRR_APPROVED_REPORT_DIR"])

    records = run_api_acceptance(root, report_dir)

    assert tuple(record.case_id for record in records) == (
        "known_single_layer",
        "unstable_multilayer",
        "workable_mo_si_multilayer",
    )
    assert all(tuple(run.ordinal for run in record.runs) == (1, 2, 3, 4) for record in records)
