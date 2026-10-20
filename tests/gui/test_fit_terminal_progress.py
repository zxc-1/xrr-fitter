"""Intermediate automatic finalizing frames are not whole-job completion."""

import pytest

import xrr_fitter.api as api
from xrr_fitter.gui.fitting.progress import PROGRESS_RESOLUTION, ProgressView


@pytest.mark.parametrize("members", ((), ("left",), ("left", "right")))
def test_finalizing_progress_frames_never_claim_whole_job_completion(qtbot, members):
    view = ProgressView()
    qtbot.addWidget(view)
    view.set_members(members)
    for owner in members or (None,):
        view.set_progress(api.FitProgress(owner, "finalizing", 1, 1, 1.0, "fast analysis complete"))

    assert view.bar.value() == 990
    assert view.overall_percent() == 99


def test_joint_refinement_after_completed_prefits_remains_nonterminal(qtbot):
    view = ProgressView()
    qtbot.addWidget(view)
    view.set_members(("left", "right"))
    view.set_progress(api.FitProgress("left", "finalizing", 1, 1, 1.0, "fast analysis complete"))
    assert view.bar.value() == 500
    view.set_progress(api.FitProgress("right", "finalizing", 1, 1, 2.0, "fast analysis complete"))
    assert view.bar.value() == 990

    view.set_progress(
        api.FitProgress(
            None,
            "E",
            0,
            4,
            3.0,
            "joint refinement",
            dataset_objectives=(("left", 1.0), ("right", 2.0)),
        )
    )

    assert view.bar.value() == 990
    assert view.overall_percent() == 99


def test_nonterminal_finalizing_does_not_show_zero_remaining_time(qtbot):
    clock = [0.0]
    view = ProgressView(clock=lambda: clock[0])
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("left", "finalizing", 0, 1, 1.0, "analysis"))
    clock[0] = 1.0
    view.set_progress(api.FitProgress("left", "finalizing", 1, 1, 1.0, "analysis complete"))
    clock[0] = 30.0

    view._tick()

    assert view.status_summary().remaining != "00:00"
    assert view.overall_percent() < 100


def test_freeze_is_not_success_and_reset_reopens_the_run(qtbot):
    view = ProgressView()
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress(None, "finalizing", 1, 1, 1.0, "analysis complete"))
    view.freeze()
    assert view.overall_percent() < 100

    view.reset()

    assert view.overall_percent() == 0
    assert view.status_summary().remaining == ""


def test_reset_clears_a_published_completion(qtbot):
    view = ProgressView()
    qtbot.addWidget(view)
    view.mark_completed()
    assert view.bar.value() == PROGRESS_RESOLUTION
    assert view.status_summary().remaining == "00:00"

    view.reset()
    view.set_progress(api.FitProgress(None, "A", 1, 2, 1.0, "new search"))

    assert view.bar.value() < PROGRESS_RESOLUTION
    assert view.status_summary().remaining != "00:00"
    assert view._heartbeat.isActive()
