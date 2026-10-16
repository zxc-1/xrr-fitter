"""The progress card must track a real run's clock, not a guess about where time goes.

审计把 1201 点基准的进度事件回放进真实的 ``ProgressView``：60 s 的手动拟合第 6 s 就到 72%，
之后剩余时间在「00:01」与「仍在计算」之间来回跳；132 s 的拟合最后 60 s 停在 95% 且只显示
「--:--」；自动路线 profile 还没跑完就到 100%。三处病根各自独立：阶段权重是猜的（B 0.34，
实测 E 占 46–77%、手动 profile 最多 45%），估计器在进度没动的帧上也挪锚点，运行中的帧把
正在跑的那一颗种子算成已完成。

下面两条时间线是那次基准里单层 Gaussian 曲线的真实事件（只保留计数变化的帧，秒数相对首帧），
``r`` 是求解器还在跑这一颗时发的遥测帧（带 ``iteration``），``f`` 是这一颗收尾后的计数帧。
"""

from __future__ import annotations

import pytest
from tests.gui.test_fit_progress import _FakeJob, _project
from tests.gui.test_fit_route_is_named import _automatic

import xrr_fitter.api as api
from xrr_fitter.gui.fitting.progress import ProgressView, stage_text

MANUAL_RUN = """
0.0 A 1 513 f|0.2 A 128 513 f|0.4 A 256 513 f|0.6 A 384 513 f|0.8 A 512 513 f|1.6 A 513 513 f
1.9 B 1 2 r|2.5 B 1 2 f|2.8 B 2 2 r|3.3 B 2 2 f|3.4 C 1 3 f|3.5 C 2 3 f|3.5 C 3 3 f
3.6 D 1 3 r|4.0 D 1 3 f|4.1 D 2 3 r|4.5 D 2 3 f|4.6 D 3 3 r|4.6 D 3 3 f
5.1 E 1 4 r|9.8 E 1 4 f|10.3 E 2 4 r|18.7 E 2 4 f|19.2 E 3 4 r|25.2 E 3 4 f|25.7 E 4 4 r|32.7 E 4 4 f
32.7 basin-recovery 0 1 f|36.1 basin-recovery 1 1 f|36.2 bootstrap 0 8 f|37.0 bootstrap 8 8 f
37.0 profile 0 8 f|59.2 profile 8 8 f|59.2 finalizing 0 1 f|59.2 finalizing 1 1 f
"""

AUTOMATIC_RUN = """
0.0 A 1 513 f|0.2 A 128 513 f|0.4 A 256 513 f|0.6 A 384 513 f|0.8 A 512 513 f|1.6 A 513 513 f
1.9 B 1 2 r|2.5 B 1 2 f|2.8 B 2 2 r|3.3 B 2 2 f|3.4 C 1 3 f|3.5 C 2 3 f|3.5 C 3 3 f
3.6 D 1 3 r|4.1 D 1 3 f|4.2 D 2 3 r|4.5 D 2 3 f|4.6 D 3 3 r|4.6 D 3 3 f
5.1 E 1 4 r|9.7 E 1 4 f|10.2 E 2 4 r|17.5 E 2 4 f|18.0 E 3 4 r|23.2 E 3 4 f|23.6 E 4 4 r|29.2 E 4 4 f
29.2 finalizing 0 1 f|29.2 finalizing 1 1 f|29.3 profile 0 1 f|30.1 profile 1 1 f
30.1 finalizing 0 1 f|30.1 finalizing 1 1 f
"""


class _Clock:
    """A clock the replay sets by hand, so every frame lands at its recorded second."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _frames(timeline: str) -> list[tuple[float, api.FitProgress]]:
    frames = []
    for row in timeline.replace("\n", "|").split("|"):
        if not row.strip():
            continue
        seconds, stage, completed, total, kind = row.split()
        iteration = 1 if kind == "r" else None
        progress = api.FitProgress("curve", stage, int(completed), int(total), 1.0, "", iteration=iteration)
        frames.append((float(seconds), progress))
    return frames


def _largest_lead(view: ProgressView, clock: _Clock, timeline: str) -> float:
    """Replay a run and report how far the bar ever ran ahead of the elapsed share."""
    frames = _frames(timeline)
    end = frames[-1][0]
    lead = 0.0
    for seconds, progress in frames:
        clock.now = seconds
        view.set_progress(progress)
        lead = max(lead, view.bar.value() / view.bar.maximum() - seconds / end)
    return lead


@pytest.mark.parametrize(
    ("route", "timeline"), ((None, MANUAL_RUN), ("automatic", AUTOMATIC_RUN)), ids=("manual", "automatic")
)
def test_the_bar_never_runs_far_ahead_of_the_clock(qtbot, route, timeline) -> None:
    """进度条不应跑到时钟前面去：第 6 s 写着 72% 的那种进度条，读者据此安排的等待全是错的。

    落后一点是诚实的——profile 那 22 s 里没有任何帧，条只能停着；跑在前面则是在许诺一个
    不会来的完工时刻。两条路线的阶段开销差得远（自动路线没有 bootstrap，profile 也只占
    几个百分点），所以各用各的实测权重；不指定路线时按手动路线。
    """
    clock = _Clock()
    view = ProgressView(clock=clock)
    qtbot.addWidget(view)
    if route is not None:
        view.set_weighting(route)

    assert _largest_lead(view, clock, timeline) < 0.10
    view.freeze()


def test_the_automatic_route_uses_its_own_weights(qtbot, tmp_path, monkeypatch) -> None:
    """「自动拟合」启动的那一次，卡片按自动路线的权重走，而不是手动路线的。"""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    panel = FitPanel(ProjectDocument(_automatic(_project(tmp_path))))
    qtbot.addWidget(panel)
    monkeypatch.setattr(api, "preflight_automatic_fit", lambda *_a: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_automatic_fit_job", lambda *_a, **_k: _FakeJob())
    reference = ProgressView()
    qtbot.addWidget(reference)
    reference.set_weighting("automatic")
    frame = api.FitProgress("curve-0", "E", 2, 4, 1.0, "", iteration=3)
    try:
        assert panel.start_automatic_fit()
        panel.controller.progress_changed.emit(frame)
        reference.set_progress(frame)

        assert panel.progress_view.bar.value() == reference.bar.value()
    finally:
        panel.controller._finish()
        reference.freeze()


@pytest.mark.parametrize("stage", ("diagnostic-calibration", "diagnostics"))
def test_an_extra_step_is_named_in_chinese_and_keeps_the_ladder_position(qtbot, stage) -> None:
    """Poisson 诊断校准不在九阶梯上：它要有中文名，也不能把阶段序数报成 9/9。

    未登记的键原样上屏是一句英文；序数退到末位则说"快完了"，而校准之后还有 bootstrap、
    profile 与汇总。校准发生在基态复核之后，所以序数停在基态复核那一格。
    """
    view = ProgressView()
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "basin-recovery", 1, 1, 1.0, "basin recovery completed"))

    view.set_progress(api.FitProgress("curve", stage, 3, 999, 1.0, "Poisson diagnostic calibration 3/999"))

    label = view.stage_label.text()
    assert stage not in label
    assert "9/9" not in label
    assert view.status_summary().stage == "6 / 9"
    view.freeze()


@pytest.mark.parametrize("stage", ("uncertainty", "some-future-stage"))
def test_stage_names_never_fall_back_to_the_raw_key(stage) -> None:
    """导出预览按 ``stage_summaries`` 列阶段，里面有 ``uncertainty``；新加的键同样不能露英文。"""
    text = stage_text(stage)

    assert text != stage
    assert any("一" <= character <= "鿿" for character in text)


def test_the_bar_reaches_the_end_only_when_the_run_publishes(qtbot) -> None:
    """汇总帧 1/1 不是终点：自动路线在它之后还会再跑一轮 profile 与汇总。

    进度条与左栏只在结果真正发布时到 100%；在那之前最多停在 999 / 1000、99%。
    """
    view = ProgressView()
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "finalizing", 1, 1, 1.0, "completed"))

    assert view.bar.value() < view.bar.maximum()
    assert view.overall_percent() < 100

    view.complete()

    assert view.bar.value() == view.bar.maximum()
    assert view.meta_remaining.text() == "预计剩余 00:00"
    view.freeze()


def test_a_stalled_frame_does_not_move_the_rate_anchor() -> None:
    """进度没动的帧不是新读数：拿它当锚点，下一次前进就被算成「一瞬间走完了一大截」。"""
    from xrr_fitter.gui.fitting.progress import _RemainingEstimator

    estimator = _RemainingEstimator()
    estimator.update(0.0, 0.40)
    estimator.update(5.0, 0.45)
    estimator.update(9.0, 0.45)
    estimator.update(20.0, 0.60)

    remaining = estimator.remaining(20.0)
    assert remaining is not None
    assert abs(remaining - 40.0) < 1e-6
