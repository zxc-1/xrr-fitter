"""The skip action mirrors search-stage capability rather than job liveness."""

from __future__ import annotations

import pytest
from tests.gui.test_fit_progress import _FakeJob, _panel

import xrr_fitter.api as api


@pytest.fixture
def running_panel(qtbot, tmp_path, monkeypatch):
    panel = _panel(qtbot, tmp_path)
    job = _FakeJob()
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    assert panel.start_fit()
    return panel


def test_skip_is_disabled_before_the_first_search_progress(running_panel) -> None:
    assert not running_panel.skip_button.isEnabled()


@pytest.mark.parametrize(
    "stage", ["A", "B", "C", "D", "E", "basin-recovery", "profile", "bootstrap", "finalizing", "mcmc"]
)
def test_skip_enabled_only_for_search_stages(running_panel, stage) -> None:
    running_panel.controller.progress_changed.emit(api.FitProgress("curve", stage, 0, 1, 1.0, stage))
    assert running_panel.skip_button.isEnabled() == (stage in {"A", "B", "C", "D", "E"})


def test_cancel_disables_skip_even_if_search_progress_arrives(running_panel) -> None:
    progress = api.FitProgress("curve", "C", 0, 1, 1.0, "search")
    running_panel.controller.progress_changed.emit(progress)
    assert running_panel.skip_button.isEnabled()
    running_panel.cancel_button.click()
    running_panel.controller.progress_changed.emit(progress)
    assert not running_panel.skip_button.isEnabled()


def test_stopped_stage_does_not_enable_skip_on_next_run(running_panel) -> None:
    running_panel.controller.progress_changed.emit(api.FitProgress("curve", "E", 0, 1, 1.0, "search"))
    running_panel._project_running_state(False)
    running_panel._project_running_state(True)
    assert not running_panel.skip_button.isEnabled()
