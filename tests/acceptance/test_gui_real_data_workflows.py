from __future__ import annotations

import os
from pathlib import Path

from tests.support.approved_workflows import verify_gui_acceptance


def test_gui_real_data_workflows_round_trip_owner_projects() -> None:
    root = Path(os.environ["XRR_APPROVED_DATA_ROOT"])
    report_dir = Path(os.environ["XRR_APPROVED_REPORT_DIR"])

    results = verify_gui_acceptance(root, report_dir)

    assert tuple(result.case_id for result in results) == (
        "known_single_layer",
        "unstable_multilayer",
        "workable_mo_si_multilayer",
    )
    assert all(result.project_reopened for result in results)
    assert all(result.exports_verified and result.plots_verified for result in results)
