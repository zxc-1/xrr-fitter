"""Aggregate fit progress must not impersonate per-dataset objective evidence."""

from types import SimpleNamespace

import pytest
from tests.gui.test_fit_progress import _project

import xrr_fitter.api as api
from xrr_fitter.gui.data.panel import DataPanel
from xrr_fitter.gui.document import ProjectDocument


def _progress(*, objectives=None):
    return api.FitProgress(None, "finalizing", 0, 1, 5.0, "finalizing joint fit", dataset_objectives=objectives)


def test_aggregate_frame_has_no_per_dataset_objectives(tmp_path):
    # This projection is pure: no QApplication or widget is needed to verify
    # the identity contract at the worker-to-card boundary.
    owner = SimpleNamespace(document=ProjectDocument(_project(tmp_path, count=2)))

    assert DataPanel._progress_objectives(owner, _progress()) == {}


@pytest.mark.parametrize("existing", (False, True))
def test_aggregate_frame_does_not_create_or_overwrite_member_readings(qtbot, tmp_path, existing):
    panel = DataPanel(ProjectDocument(_project(tmp_path, count=2)))
    qtbot.addWidget(panel)
    panel.set_running(True)
    if existing:
        panel.set_run_progress(_progress(objectives=(("curve-0", 1.0), ("curve-1", 4.0))))
    before = tuple(panel.run_subline(name) for name in ("curve-0", "curve-1"))

    panel.set_run_progress(_progress())

    assert tuple(panel.run_subline(name) for name in ("curve-0", "curve-1")) == before
