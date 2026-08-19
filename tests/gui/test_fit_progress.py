from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QLabel, QProgressBar, QPushButton

import xrr_fitter.api as api


class _FakeJob:
    def __init__(self, events=()) -> None:
        self.events = tuple(events)
        self.is_running = True
        self.closed = False

    def poll(self):
        events, self.events = self.events, ()
        if any(event.kind == "stopped" for event in events):
            self.is_running = False
        return events

    def cancel(self) -> None:
        pass

    def force_stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path, count=1):
    project = api.new_project()
    air = api.MaterialSpec("Air", None, None, 0.0j)
    silicon = api.MaterialSpec("Si", "Si", 2.329)
    silica = api.MaterialSpec("SiO2", "SiO2", 2.20)
    structure = api.StructureSpec(
        air,
        (api.LayerSpec("film", silica, 40.0),),
        silicon,
    )
    for index in range(count):
        source = _write_curve(tmp_path / f"curve-{index}.xy")
        project = api.add_dataset(project, source, api.InstrumentSpec())
        project = api.set_structure(project, f"curve-{index}", structure)
    return project


def _panel(qtbot, tmp_path, count=1):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    panel = FitPanel(ProjectDocument(_project(tmp_path, count)))
    qtbot.addWidget(panel)
    return panel


def test_progress_view_renders_stage_identity_objective_and_message(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    progress = api.FitProgress("curve", "C", 3, 10, 0.0125, "local search")

    view.set_progress(progress)

    bar = view.findChild(QProgressBar, "fitProgressBar")
    stage = view.findChild(QLabel, "fitProgressStage")
    detail = view.findChild(QLabel, "fitProgressDetail")
    # The bar reports one global scale so switching stages cannot rewind it;
    # the per-stage counts stay visible in the stage label instead.
    assert bar.maximum() == 1000
    assert 0 < bar.value() < 1000
    assert "curve" in stage.text()
    assert "3/10" in detail.text()
    assert "local search" in detail.text()
    assert "0.0125" in detail.text()


def test_fit_panel_controls_running_state_and_publishes_result(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    updated = api.set_expert_mode(original, True)
    result = api.ProjectFitResult("independent", (), (), updated)
    job = _FakeJob(
        (
            api.OperationEvent(0, "fit_result", fit_result=result),
            api.OperationEvent(1, "stopped"),
        )
    )
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    start = panel.findChild(QPushButton, "startFitButton")
    cancel = panel.findChild(QPushButton, "cancelFitButton")

    assert panel.start_fit() is True
    assert start.isEnabled() is False
    assert cancel.isEnabled() is True

    panel.controller.poll_now()

    assert panel.document.project is updated
    assert panel.is_running is False
    assert start.isEnabled() is True
    assert cancel.isEnabled() is False


def test_fit_panel_checkpoint_adopts_project_before_terminal_result(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    checkpoint = api.set_expert_mode(panel.document.project, True)
    job = _FakeJob(
        (
            api.OperationEvent(0, "checkpoint", checkpoint=checkpoint),
            api.OperationEvent(1, "cancelled", cancellation="requested"),
            api.OperationEvent(2, "stopped"),
        )
    )
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    results: list[object] = []
    panel.result_published.connect(results.append)

    panel.start_fit()
    panel.controller.poll_now()

    assert panel.document.project is checkpoint
    assert results == []
    assert panel.status_text() == "已取消：requested，已保存检查点，可从中恢复"


def test_fit_panel_preflight_failure_does_not_start_worker(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    starts: list[object] = []
    monkeypatch.setattr(
        api,
        "preflight_fit",
        lambda _project: api.FitReadiness(False, "结构尚未准备"),
    )
    monkeypatch.setattr(api, "start_fit_job", starts.append)

    assert panel.start_fit() is False
    assert starts == []
    assert panel.status_text() == "结构尚未准备"


def test_batch_mode_selector_is_visible_persisted_and_rejects_one_dataset_joint(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    selector = panel.findChild(QComboBox, "batchModeSelector")

    assert [selector.itemData(index) for index in range(selector.count())] == [
        "independent",
        "joint",
    ]
    before = panel.document.project
    try:
        panel.set_batch_mode("joint")
    except ValueError as error:
        assert "requires at least two datasets" in str(error)
    else:
        raise AssertionError("one-dataset joint mode was accepted")
    assert panel.document.project is before

    joint = _panel(qtbot, tmp_path / "joint", count=2)
    assert joint.set_batch_mode("joint") is True
    assert joint.document.project.batch_mode == "joint"


def test_progress_bar_never_moves_backwards_across_stage_changes(qtbot) -> None:
    """Each stage owns a slice of one fixed range, so the bar only advances."""
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    bar = view.findChild(QProgressBar, "fitProgressBar")

    observed: list[int] = []
    sequence = (
        ("A", 512, 512),
        ("B", 1, 2),
        ("B", 2, 2),
        ("C", 3, 6),
        ("C", 6, 6),
        ("D", 6, 6),
        ("E", 4, 4),
        ("bootstrap", 9, 9),
        ("profile", 9, 9),
        ("finalizing", 1, 1),
    )
    for stage, completed, total in sequence:
        view.set_progress(api.FitProgress("curve", stage, completed, total, 1.0, stage))
        observed.append(bar.value())

    assert observed == sorted(observed), f"bar moved backwards: {observed}"
    assert bar.maximum() == 1000
    assert observed[0] > 0
    assert observed[-1] == 1000


def test_progress_view_reports_stage_ordinal_and_localized_stage_name(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "C", 3, 6, 0.5, "refining"))

    stage = view.findChild(QLabel, "fitProgressStage")
    detail = view.findChild(QLabel, "fitProgressDetail")
    assert "阶段 3/9" in stage.text()
    assert "密度精修" in stage.text()
    assert "3/6" in detail.text()


def test_progress_view_reset_returns_the_bar_to_zero(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "E", 4, 4, 1.0, "seeds"))

    view.reset()

    bar = view.findChild(QProgressBar, "fitProgressBar")
    assert bar.value() == 0
    assert view.findChild(QLabel, "fitProgressStage").text() == "等待开始"


def test_unknown_stage_does_not_move_the_progress_bar(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    bar = view.findChild(QProgressBar, "fitProgressBar")
    view.set_progress(api.FitProgress("curve", "C", 6, 6, 1.0, "refining"))
    before = bar.value()

    view.set_progress(api.FitProgress("curve", "mystery-stage", 1, 1, 1.0, "?"))

    assert bar.value() == before


class _StepClock:
    """Deterministic monotonic clock: each call returns the next scripted value."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def test_progress_view_shows_elapsed_and_estimated_remaining(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(100.0, 110.0))
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "B", 0, 10, 1.0, "search"))
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))

    meta = view.findChild(QLabel, "fitProgressMeta")
    assert "已用 00:10" in meta.text()
    assert "预计剩余" in meta.text()
    assert "--:--" not in meta.text()  # enough progress to extrapolate


def test_progress_view_defers_remaining_estimate_until_enough_progress(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(0.0, 3.0))
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "A", 0, 10, 1.0, "seeding"))
    view.set_progress(api.FitProgress("curve", "A", 1, 10, 1.0, "seeding"))

    meta = view.findChild(QLabel, "fitProgressMeta")
    assert "已用 00:03" in meta.text()
    assert "--:--" in meta.text()  # frac far too small to extrapolate honestly


def test_progress_view_announces_graceful_cancellation(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(0.0, 5.0))
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 3, 10, 1.0, "search"))

    view.mark_cancelling()

    meta = view.findChild(QLabel, "fitProgressMeta")
    assert "正在取消，等待当前种子结束…" in meta.text()


def test_progress_view_reset_clears_elapsed_and_cancel_state(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(100.0, 200.0, 210.0))
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))
    view.mark_cancelling()

    view.reset()

    meta = view.findChild(QLabel, "fitProgressMeta")
    assert meta.text() == ""
    # A fresh run restarts the clock from the next first frame, not the old start.
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))
    assert "正在取消" not in meta.text()
    assert "已用 00:10" in meta.text()  # 210-200, not 210-100


def test_fit_panel_cancel_button_shows_graceful_cancel_feedback(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from PySide6.QtWidgets import QLabel

    panel = _panel(qtbot, tmp_path)
    cancel_calls: list[int] = []
    job = _FakeJob(())
    job.cancel = lambda: cancel_calls.append(1)  # type: ignore[method-assign]
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)

    panel.start_fit()
    cancel = panel.findChild(QPushButton, "cancelFitButton")
    cancel.click()

    meta = panel.progress_view.findChild(QLabel, "fitProgressMeta")
    assert "正在取消，等待当前种子结束…" in meta.text()
    assert cancel_calls == [1]  # the worker is still asked to cancel


def test_progress_detail_shows_stage_local_percent(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "B", 3, 10, 1.0, "search"))

    detail = view.findChild(QLabel, "fitProgressDetail")
    # The stage-local percent gives a slow stage visible motion the global bar
    # cannot show while it is pinned inside that stage's slice.
    assert "3/10 (30%)" in detail.text()


def test_progress_heartbeat_advances_elapsed_between_frames(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    # First frame at t=100 opens the clock; the heartbeat later reads t=140.
    view = ProgressView(clock=_StepClock(100.0, 140.0))
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))

    view._tick()

    meta = view.findChild(QLabel, "fitProgressMeta")
    assert "已用 00:40" in meta.text()  # ticked forward without a new frame


def test_progress_freeze_stops_the_heartbeat(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))
    assert view._heartbeat.isActive()

    view.freeze()

    assert not view._heartbeat.isActive()


def test_remaining_estimator_tracks_recent_rate_not_cumulative_average() -> None:
    from xrr_fitter.gui.fitting.progress import _RemainingEstimator

    estimator = _RemainingEstimator(smoothing=1.0)  # follow the latest rate fully
    # A fast opening burst: 40% of the bar in 4s (rate 0.10/s).
    estimator.update(0.0, 0.0)
    estimator.update(4.0, 0.40)
    # A slow later stage: only 5% more over the next 5s (rate 0.01/s).
    estimator.update(9.0, 0.45)

    remaining = estimator.remaining(9.0)
    # Cumulative average (0.45/9 = 0.05/s) would predict ~11s; the recent slow
    # rate (0.01/s) honestly predicts ~55s for the remaining 55%.
    assert remaining is not None
    assert 50.0 <= remaining <= 60.0


def test_remaining_estimator_counts_down_between_updates() -> None:
    from xrr_fitter.gui.fitting.progress import _RemainingEstimator

    estimator = _RemainingEstimator(smoothing=1.0)
    estimator.update(0.0, 0.0)
    estimator.update(10.0, 0.50)  # 0.05/s → finish projected at t=20

    # No new frame arrived, but wall-clock advanced; remaining must shrink.
    first = estimator.remaining(12.0)
    later = estimator.remaining(16.0)
    assert first is not None and later is not None
    assert later < first
    assert abs(first - 8.0) < 1e-9  # 20 - 12
    assert abs(later - 4.0) < 1e-9  # 20 - 16


def test_remaining_estimator_reset_clears_projection() -> None:
    from xrr_fitter.gui.fitting.progress import _RemainingEstimator

    estimator = _RemainingEstimator()
    estimator.update(0.0, 0.0)
    estimator.update(4.0, 0.40)
    assert estimator.remaining(4.0) is not None

    estimator.reset()

    assert estimator.remaining(4.0) is None


def _preview_progress(value):
    import numpy as np

    qz = np.array([0.1, 0.2, 0.3], dtype=float)
    model = np.array([value, value, value], dtype=float)
    return api.FitProgress(
        "curve",
        "B",
        5,
        10,
        1.0,
        "search",
        preview_qz_a_inv=qz,
        preview_model_normalized=model,
    )


def test_fit_panel_throttles_preview_frames(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    clock = _StepClock(0.0, 0.05, 0.10, 0.30)  # window is 0.2s
    panel = FitPanel(ProjectDocument(_project(tmp_path)), clock=clock)
    qtbot.addWidget(panel)
    emitted: list[object] = []
    panel.preview_available.connect(lambda _qz, model: emitted.append(model[0]))

    panel._project_preview(_preview_progress(1.0))  # t=0.00 → emit
    panel._project_preview(_preview_progress(2.0))  # t=0.05 → dropped
    panel._project_preview(_preview_progress(3.0))  # t=0.10 → dropped
    panel._project_preview(_preview_progress(4.0))  # t=0.30 → emit (past window)

    assert emitted == [1.0, 4.0]


def test_fit_panel_preview_window_resets_on_new_fit(qtbot, tmp_path, monkeypatch) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    clock = _StepClock(0.0, 0.05)
    panel = FitPanel(ProjectDocument(_project(tmp_path)), clock=clock)
    qtbot.addWidget(panel)
    emitted: list[object] = []
    panel.preview_available.connect(lambda _qz, model: emitted.append(model[0]))
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: _FakeJob(()))

    panel._project_preview(_preview_progress(1.0))  # t=0.00 → emit, arms window
    panel.start_fit()  # a fresh run clears the throttle so its first preview shows
    panel._project_preview(_preview_progress(2.0))  # t=0.05 → emit despite being <0.2s later

    assert emitted == [1.0, 2.0]


def test_fit_panel_cancel_without_checkpoint_reports_no_resume(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    # Cancel arrives before any checkpoint, so the user must be told plainly
    # that there is nothing to resume from rather than left guessing.
    job = _FakeJob(
        (
            api.OperationEvent(0, "cancelled", cancellation="requested"),
            api.OperationEvent(1, "stopped"),
        )
    )
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)

    panel.start_fit()
    panel.controller.poll_now()

    assert panel.status_text() == "已取消：requested，本次未产生检查点"


def test_start_button_reads_new_fit_without_checkpoint(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)

    # A fresh project has no checkpoint, so the action stays a plain new fit.
    assert panel.has_resumable_checkpoint() is False
    assert panel.start_button.text() == "开始拟合"


def test_start_button_relabels_to_resume_when_checkpoint_present(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)

    # A resumable checkpoint must be surfaced: a fit silently continues from it,
    # so the button says "继续拟合" and explains the pending resume on hover.
    monkeypatch.setattr(panel, "has_resumable_checkpoint", lambda: True)
    panel._refresh_controls()

    assert panel.start_button.text() == "继续拟合"
    assert "检查点" in panel.start_button.toolTip()


def test_progress_view_shows_joint_layout_banner(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    layout = api.JointFitLayout(
        ("first", "second"),
        (
            api.SharingRule(
                "shared-thickness",
                (
                    api.ParameterReference("first", "component.0.thickness_a"),
                    api.ParameterReference("second", "component.0.thickness_a"),
                ),
            ),
        ),
    )

    view.set_joint_layout(layout)

    banner = view.findChild(QLabel, "fitProgressJointLayout")
    assert banner is not None
    # The banner names the coupled datasets and the shared-parameter groups so a
    # joint run is no longer an anonymous "联合拟合".
    assert "first" in banner.text() and "second" in banner.text()
    assert "shared-thickness" in banner.text()
    assert not banner.isHidden()

    # An independent fit clears the banner so it never lingers with stale names.
    view.set_joint_layout(None)
    assert banner.isHidden()


def test_fit_panel_publishes_joint_layout_on_start(qtbot, tmp_path, monkeypatch) -> None:
    panel = _panel(qtbot, tmp_path, count=2)
    joint = api.set_batch_mode(panel.document.project, "joint")
    panel.document.replace_project(joint)
    job = _FakeJob((api.OperationEvent(0, "stopped"),))
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)

    assert panel.start_fit() is True

    banner = panel.progress_view.findChild(QLabel, "fitProgressJointLayout")
    assert banner is not None
    assert "curve-0" in banner.text() and "curve-1" in banner.text()
    assert not banner.isHidden()
