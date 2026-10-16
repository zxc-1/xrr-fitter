"""Long profile calculations must not claim their stale ETA is completion."""

import xrr_fitter.api as api
from xrr_fitter.gui.fitting.progress import ProgressView


def test_a_silent_stage_keeps_counting_down_instead_of_blanking(qtbot):
    """profile 一帧不发能沉默一分钟；这一段的剩余时间照样往下数，而不是一换阶段就换成 --:--。

    换阶段清空估计，是在权重靠猜的年代防「快阶段学来的速度」冒充慢阶段的。权重按实测定下
    之后，前面学到的速度正是这段沉默最好的预报；真的超时了，下一条用例的「仍在计算」接手。
    """
    times = iter((0.0, 10.0, 20.0, 30.0, 31.0, 45.0))
    view = ProgressView(clock=lambda: next(times))
    qtbot.addWidget(view)
    try:
        view.set_progress(api.FitProgress("a", "E", 1, 4, 1.0, "search"))
        view.set_progress(api.FitProgress("a", "E", 2, 4, 1.0, "search"))
        view.set_progress(api.FitProgress("a", "E", 3, 4, 1.0, "search"))
        view.set_progress(api.FitProgress("a", "E", 4, 4, 1.0, "search"))
        view.set_progress(api.FitProgress("a", "profile", 0, 8, 1.0, "profile"))
        view._tick()

        assert "--:--" not in view.meta_remaining.text()
        assert "仍在计算" not in view.meta_remaining.text()
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
