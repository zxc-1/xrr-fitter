from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QProgressBar, QPushButton

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


def test_set_batch_mode_persists_a_switch_and_rejects_one_dataset_joint(
    qtbot,
    tmp_path,
) -> None:
    """``set_batch_mode`` 是这张卡对外的写口，命令栏的批量段调的就是它。

    控件本身已经不在这张卡上——设计稿把 ``批量 独立|联合`` 画在命令栏，见
    ``test_batch_segment.py``。留在这里的是写口的契约：一个数据集要联合拟合时抛
    ``ValueError`` 且项目一个字不动，两个数据集时返回 ``True``。
    """
    panel = _panel(qtbot, tmp_path)

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


def test_a_finished_stage_says_how_long_it_took(qtbot) -> None:
    """帧④ 的阶段梯子给已完成的行标耗时（`0.8s`、`21s`），不是一句「已完成」。

    九个阶段的真实开销差一个数量级：设计稿里 B 一个人吃掉二十多秒，A 不到一秒。跑完之后
    满屏「已完成／已完成／已完成」，等于把"时间花在哪了"这条信息扔掉——而下次要不要调
    预算、要不要换搜索策略，看的正是这条。

    耗时不需要后端新报字段：这张卡自己就握着时钟，它知道某阶段的首帧何时到、又何时被下
    一阶段顶替。所以这里钉的是"把已经读到的钟面写出来"，而不是行里那些真的没有 API 支撑
    的后端计数（`128 组`、`迭代 342`）。
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    # A 起于 t=0、被 B 顶替于 t=0.8；B 于 t=21.8 交给 C，C 又于 t=30.8 交给 D。
    view = ProgressView(clock=_StepClock(0.0, 0.8, 21.8, 30.8))
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "A", 512, 512, 1.0, "screening"))
    view.set_progress(api.FitProgress("curve", "B", 2, 2, 1.0, "search"))
    view.set_progress(api.FitProgress("curve", "C", 6, 6, 1.0, "refining"))
    view.set_progress(api.FitProgress("curve", "D", 3, 10, 1.0, "polishing"))

    assert view.findChild(QLabel, "fitStageMetric_A").text() == "0.8s"
    assert view.findChild(QLabel, "fitStageMetric_B").text() == "21s"
    # 十分位只在不到一秒时保留：9.0s 那个小数位既不影响判断，又让整列读起来不齐。
    assert view.findChild(QLabel, "fitStageMetric_C").text() == "9s"


def test_a_stage_that_never_ran_is_not_credited_with_zero_seconds(qtbot) -> None:
    """跳过的阶段只说「已完成」，不报一个 `0.0s` 的假读数。

    耗时是从"首帧到被顶替"两次钟面差出来的；一个阶段压根没发过帧，这个差就不存在。此时
    填 `0.0s` 会把"没测到"说成"快到测不出"，比不说更糟。
    """
    from xrr_fitter.gui.fitting.progress import STAGE_METRIC_DONE, ProgressView

    view = ProgressView(clock=_StepClock(0.0, 5.0))
    qtbot.addWidget(view)

    # A 之后直接跳到 C：B 从未发过帧，却会被梯子标成已完成。
    view.set_progress(api.FitProgress("curve", "A", 512, 512, 1.0, "screening"))
    view.set_progress(api.FitProgress("curve", "C", 3, 6, 1.0, "refining"))

    assert view.findChild(QLabel, "fitStageMetric_B").text() == STAGE_METRIC_DONE


def test_progress_view_shows_elapsed_and_estimated_remaining(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(100.0, 110.0))
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "B", 0, 10, 1.0, "search"))
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))

    assert "已用 00:10" in view.findChild(QLabel, "fitProgressMetaElapsed").text()
    assert "预计剩余" in view.findChild(QLabel, "fitProgressMetaRemaining").text()
    assert "--:--" not in view.findChild(QLabel, "fitProgressMetaRemaining").text()


def test_progress_view_reads_out_the_scale_its_card_promises(qtbot) -> None:
    """Frame ④ prints 进度 620 / 1000 in the meta row and leaves the bar wordless.

    The card's subtitle promises a monotonic 0-1000 scale, yet the only number on
    screen was the bar's own centred percentage: nothing ever showed the value
    that scale counts, and the two readings stated one position in two different
    units. The mockup's bar is a bare trough for exactly that reason -- one
    number, in the promised scale, in the row already carrying 已用 and 预计剩余.
    """
    from xrr_fitter.gui.fitting.progress import PROGRESS_RESOLUTION, ProgressView

    view = ProgressView(clock=_StepClock(100.0, 110.0))
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "B", 0, 10, 1.0, "search"))
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))

    bar = view.findChild(QProgressBar, "fitProgressBar")
    # Stage B opens at 0.06 and spans 0.34, so half of it lands on 230/1000.
    assert bar.value() == 230
    assert f"进度 {bar.value()} / {PROGRESS_RESOLUTION}" in view.findChild(QLabel, "fitProgressMetaPosition").text()
    assert not bar.isTextVisible(), "the bar restates the position in percent"


def test_progress_view_publishes_a_one_line_summary_for_the_status_bar(qtbot) -> None:
    """Guided mode hides this card, so the run has to be readable elsewhere.

    ``set_guidance_visible(True)`` hides the whole inspector column, and this card
    lives in it. The status bar is then the only surface a guided user can see a
    run on, which is why the mockup's frame ④ bar carries 阶段 4 / 9 and
    进度 620 / 1000. The summary is published from here rather than recomputed by
    the bar: a second estimator would drift from the one on this card.
    """
    from xrr_fitter.gui.fitting.progress import PROGRESS_RESOLUTION, STAGE_WEIGHTS, ProgressView, RunSummary

    view = ProgressView(clock=_StepClock(100.0, 110.0))
    qtbot.addWidget(view)

    # Nothing has run, so there is no position to report.
    assert view.status_summary() == RunSummary()

    published: list[RunSummary] = []
    view.summary_changed.connect(published.append)
    view.set_progress(api.FitProgress("curve", "B", 0, 10, 1.0, "search"))
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))

    summary = view.status_summary()
    # 三段分开报：设计稿帧④ 把阶段与进度留在弹簧左边、「预计剩余」挪到右边，一句拼好的
    # 话既没法分段加粗也没法分左右。说明文字「阶段 」「预计剩余 」住在状态栏那一段里，
    # 所以这里的 stage 只有序数。
    assert summary.stage == f"2 / {len(STAGE_WEIGHTS)}"
    assert summary.position == f"进度 230 / {PROGRESS_RESOLUTION}"
    assert summary.remaining != ""
    # Every frame republishes, so the bar never lags the card.
    assert published[-1] == summary

    view.reset()
    assert view.status_summary() == RunSummary()
    assert published[-1] == RunSummary()


def test_progress_view_summary_reports_a_cancel_in_flight(qtbot) -> None:
    """A cancelled run keeps counting seeds down, so the bar must not read as live.

    ``mark_cancelling`` already replaces the card's meta row; the status bar shows
    the same wait, otherwise a guided user sees a position that no longer moves
    with no reason given.

    这句话落在进度那一段：那一段本来就是整句白文（``进度 620 / 1000``），而阶段序数与
    预计剩余在等最后一颗种子收尾时都已不再是真话，所以那两段一并留空退场。
    """
    from xrr_fitter.gui.fitting.progress import CANCELLING_TEXT, ProgressView, RunSummary

    view = ProgressView(clock=_StepClock(100.0, 110.0))
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))

    published: list[RunSummary] = []
    view.summary_changed.connect(published.append)
    view.mark_cancelling()

    assert view.status_summary() == RunSummary(position=CANCELLING_TEXT)
    assert published[-1] == RunSummary(position=CANCELLING_TEXT)


def test_progress_view_defers_remaining_estimate_until_enough_progress(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(0.0, 3.0))
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "A", 0, 10, 1.0, "seeding"))
    view.set_progress(api.FitProgress("curve", "A", 1, 10, 1.0, "seeding"))

    assert "已用 00:03" in view.findChild(QLabel, "fitProgressMetaElapsed").text()
    assert "--:--" in view.findChild(QLabel, "fitProgressMetaRemaining").text()


def test_progress_view_announces_graceful_cancellation(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(0.0, 5.0))
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 3, 10, 1.0, "search"))

    view.mark_cancelling()

    assert "正在取消，等待当前种子结束…" in view.findChild(QLabel, "fitProgressMetaElapsed").text()


def test_progress_view_reset_clears_elapsed_and_cancel_state(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView(clock=_StepClock(100.0, 200.0, 210.0))
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))
    view.mark_cancelling()

    view.reset()

    elapsed = view.findChild(QLabel, "fitProgressMetaElapsed")
    assert elapsed.text() == ""
    # A fresh run restarts the clock from the next first frame, not the old start.
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))
    view.set_progress(api.FitProgress("curve", "B", 5, 10, 1.0, "search"))
    assert "正在取消" not in elapsed.text()
    assert "已用 00:10" in elapsed.text()  # 210-200, not 210-100


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

    assert "正在取消，等待当前种子结束…" in panel.progress_view.findChild(QLabel, "fitProgressMetaElapsed").text()
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

    assert "已用 00:40" in view.findChild(QLabel, "fitProgressMetaElapsed").text()


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

    clock = _StepClock(0.0, 0.02, 0.04, 0.10)  # window is 0.05s
    panel = FitPanel(ProjectDocument(_project(tmp_path)), clock=clock)
    qtbot.addWidget(panel)
    emitted: list[object] = []
    panel.preview_available.connect(lambda _qz, model: emitted.append(model[0]))

    panel._project_preview(_preview_progress(1.0))  # t=0.00 → emit
    panel._project_preview(_preview_progress(2.0))  # t=0.02 → dropped
    panel._project_preview(_preview_progress(3.0))  # t=0.04 → dropped
    panel._project_preview(_preview_progress(4.0))  # t=0.10 → emit (past window)

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


def test_the_joint_banner_says_which_side_of_the_split_a_number_is_on(qtbot) -> None:
    """帧④ 的 🔗 横幅：既报共享了什么，也报其余各自独立，并做成设计稿的框。

    联合拟合里一个共享参数只显示一个值，而那个值是几套数据折衷出来的；没进共享规则的
    参数则各数据集各有一份，差异是真信号。只报共享侧，用户看着某个数就无从判断它属于
    哪一侧——把折衷读成真实差异，或者反过来。所以两侧都要点明。

    "其余各自独立"是通则而不是抄设计稿那句「各自独立标度与本底」：独立侧由共享规则的补集
    定义，用户真把标度也共享了，写死的那句就会在屏幕上说假话。
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    view.set_joint_layout(
        api.JointFitLayout(
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
    )

    banner = view.findChild(QLabel, "fitProgressJointLayout")
    assert banner is not None
    assert "独立" in banner.text(), "横幅没有说未共享的参数各自独立"
    assert banner.property("bannerBox") is True, "联合横幅没做成设计稿的框"


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


def test_the_stop_control_says_what_stopping_keeps(qtbot, tmp_path) -> None:
    """帧④ 控制段的第三颗写「⏹ 停止并保留最优」，下面一句说明停了会留下什么。

    「取消」在这一屏读作「作废这一次运行」——而这个命令实际上保留当前最优候选和已经
    跑完的阶段，停完可以直接进结果复核。一个字面意思与行为相反的按钮，跑到一半的人
    多半不敢按，只能等它跑完或强制杀掉。
    """
    panel = _panel(qtbot, tmp_path)

    cancel = panel.findChild(QPushButton, "cancelFitButton")
    help_label = panel.findChild(QLabel, "fitStopHelp")

    assert cancel is not None and cancel.text() == "⏹ 停止并保留最优"
    assert help_label is not None
    assert help_label.text() == "停止后保留当前最优候选，可直接进入结果复核，不丢弃已完成阶段。"


def test_the_refresh_badge_follows_the_poller_when_it_backs_off(qtbot, tmp_path) -> None:
    """轮询退到空闲间隔时，徽标跟着改口。

    长时间没有事件时轮询会从 250 ms 退到 1000 ms；徽标仍写着活跃周期，就成了一句与画面
    对不上的保证——读者正是拿这个周期去分「曲线不动是收敛了还是界面卡住了」。
    """
    panel = _panel(qtbot, tmp_path)
    badge = panel.progress_view.refresh_badge

    panel.controller.poll_interval_changed.emit(200)

    assert badge.text() == "实时刷新 · 每 200 ms"


def test_the_pause_control_sits_before_stop_and_toggles_to_resume(qtbot, tmp_path, monkeypatch) -> None:
    """设计稿帧④ 控制段的第一个命令是 ⏸ 暂停，排在停止前面。

    暂停不是停止的弱化版：它让 worker 停在下一个阶段边界，已跑完的阶段和当前最优原样
    留着，读者可以先去看一眼参数再决定继续还是停。所以按下之后按钮必须自己改口说
    「▶ 继续」——一枚按下去没有回执的暂停键，读者只能靠曲线动不动来猜它有没有生效。
    """
    panel = _panel(qtbot, tmp_path)
    calls: list[str] = []
    job = _FakeJob(())
    job.pause = lambda: calls.append("pause")  # type: ignore[method-assign]
    job.resume = lambda: calls.append("resume")  # type: ignore[method-assign]
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)

    bar = panel.findChild(QPushButton, "pauseFitButton")
    assert bar is not None
    # 没在跑的时候暂停是灰的：没有可暂停的东西时，一枚可点的暂停键是空承诺。
    assert bar.isEnabled() is False
    names = [button.objectName() for button in panel.command_bar.findChildren(QPushButton)]
    assert names.index("pauseFitButton") < names.index("cancelFitButton")

    panel.start_fit()
    assert bar.isEnabled() is True

    bar.click()
    assert calls == ["pause"]
    assert bar.text() == "▶ 继续"

    bar.click()
    assert calls == ["pause", "resume"]
    assert bar.text() == "⏸ 暂停"


def test_stopping_while_paused_leaves_the_pause_control_reset(qtbot, tmp_path, monkeypatch) -> None:
    """停止收工后暂停键回到未按下的样子，否则下一次开跑它还写着「▶ 继续」。"""
    panel = _panel(qtbot, tmp_path)
    job = _FakeJob(())
    job.pause = lambda: None  # type: ignore[method-assign]
    job.resume = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)

    panel.start_fit()
    bar = panel.findChild(QPushButton, "pauseFitButton")
    bar.click()
    assert bar.text() == "▶ 继续"

    panel._project_running_state(False)

    assert bar.text() == "⏸ 暂停"
    assert bar.isEnabled() is False


def test_the_skip_control_asks_the_worker_to_drop_the_current_stage(qtbot, tmp_path, monkeypatch) -> None:
    """设计稿帧④ 控制段的第二个命令：⏭ 跳过本阶段。

    粗搜已经出结果、读者不想再等这一层加工时，跳过让搜索直接进入下一层——作废的只有
    当前这一个阶段，已跑完的阶段和候选都留着。所以它和「停止」是两件事：停止之后没有
    的跑，跳过之后还有。
    """
    panel = _panel(qtbot, tmp_path)
    calls: list[str] = []
    job = _FakeJob(())
    job.skip_stage = lambda: calls.append("skip")  # type: ignore[method-assign]
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)

    button = panel.findChild(QPushButton, "skipStageButton")
    assert button is not None
    assert button.isEnabled() is False

    panel.start_fit()
    panel.controller.progress_changed.emit(api.FitProgress("curve", "C", 0, 1, 1.0, "search"))
    assert button.isEnabled() is True
    button.click()

    assert calls == ["skip"]
    names = [item.objectName() for item in panel.command_bar.findChildren(QPushButton)]
    assert names.index("pauseFitButton") < names.index("skipStageButton") < names.index("cancelFitButton")


def test_running_control_row_shows_only_the_three_commands_frame_four_draws(qtbot, tmp_path, monkeypatch) -> None:
    """设计稿帧④ 的控制段只有 ⏸ 暂停 / ⏭ 跳过本阶段 / ⏹ 停止并保留最优 三枚。

    跑起来之后「自动拟合」和「开始拟合」都按不动，留在原位只是两块灰，还把真正要用的
    三个命令挤到一边。「强制停止」是「停止」的升级而不是它的并列项：先按停止让 worker
    自己收尾，收不住时那一枚才该出现，否则读者第一眼看到的就是两个停止键，分不清该按哪个。
    """
    panel = _panel(qtbot, tmp_path)
    job = _FakeJob(())
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)

    assert panel.force_button.isVisibleTo(panel) is False

    panel.start_fit()
    visible = [
        button.objectName() for button in panel.command_bar.findChildren(QPushButton) if button.isVisibleTo(panel)
    ]
    assert visible == ["pauseFitButton", "skipStageButton", "cancelFitButton"]

    panel.cancel_button.click()
    assert panel.force_button.isVisibleTo(panel) is True

    panel._project_running_state(False)
    assert panel.force_button.isVisibleTo(panel) is False
    assert panel.automatic_button.isVisibleTo(panel) is True
