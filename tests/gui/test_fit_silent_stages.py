"""Long profile calculations must not claim their stale ETA is completion."""

import xrr_fitter.api as api
from xrr_fitter.gui.fitting.progress import ProgressView


def test_stage_boundary_discards_eta_learned_from_faster_search(qtbot):
    times = iter((0.0, 1.0, 2.0))
    view = ProgressView(clock=lambda: next(times))
    qtbot.addWidget(view)
    try:
        view.set_progress(api.FitProgress("a", "E", 1, 4, 1.0, "search"))
        view.set_progress(api.FitProgress("a", "E", 2, 4, 1.0, "search"))
        view.set_progress(api.FitProgress("a", "profile", 0, 8, 1.0, "profile"))

        assert "--:--" in view.meta_remaining.text()
    finally:
        view.freeze()


def test_expired_eta_says_still_computing_instead_of_zero_seconds(qtbot):
    times = iter((0.0, 1.0, 70.0))
    view = ProgressView(clock=lambda: next(times))
    qtbot.addWidget(view)
    try:
        view.set_progress(api.FitProgress("a", "profile", 0, 8, 1.0, "profile"))
        view.set_progress(api.FitProgress("a", "profile", 1, 8, 1.0, "profile"))
        view._tick()

        assert "01:10" in view.meta_elapsed.text()
        assert "仍在计算" in view.meta_remaining.text()
        assert "00:00" not in view.status_summary().remaining
    finally:
        view.freeze()


def test_resume_does_not_use_queued_paused_frames_as_new_eta(qtbot):
    times = iter((0.0, 1.0, 2.0, 30.0, 31.0))
    view = ProgressView(clock=lambda: next(times))
    qtbot.addWidget(view)
    try:
        view.set_progress(api.FitProgress("a", "B", 1, 10, 1.0, "search"))
        view.set_pause_requested(True)
        view.set_progress(api.FitProgress("a", "B", 2, 10, 1.0, "queued"))
        view.set_pause_requested(False)
        view.set_progress(api.FitProgress("a", "B", 3, 10, 1.0, "resumed"))

        assert "--:--" in view.meta_remaining.text()
    finally:
        view.freeze()
