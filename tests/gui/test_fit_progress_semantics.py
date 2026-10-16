"""The progress card says only what the worker has actually reported.

审计：暂停后卡片一直停在「已请求」，那一行连已用时间一起被换掉，读起来像计时停了，其实 worker
从不回报「已到安全点」——它没有这样的事件；E 阶段第四颗种子一开始事件里就是 4/4，卡片与指标
都照字面写成跑完了；卡片写 999 / 1000 时左栏把它四舍五入成 100%，两处对"跑完没有"答得不一样。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel
from tests.gui.test_fit_progress import _StepClock

import xrr_fitter.api as api
from xrr_fitter.gui.fitting.progress import ProgressView


def test_a_pause_request_keeps_the_clock_and_never_claims_a_pause(qtbot) -> None:
    """没有回执就只说「已请求」，而且计时照走：worker 可能还在跑完手上那一颗。"""
    # 取不空的假时钟：断言失败时心跳计时器还活着，再读一次也只是停在最后那个读数上。
    view = ProgressView(clock=_StepClock(100.0, 101.0, 105.0))
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "B", 1, 2, 1.0, "search"))

    view.set_pause_requested(True)
    view._tick()

    assert view.meta_elapsed.text() == "已用 00:05"
    assert "暂停已请求" in view.meta_remaining.text()
    assert "已暂停" not in view.meta_remaining.text() + view.meta_position.text()
    view.freeze()


def test_a_running_seed_is_not_counted_as_finished(qtbot) -> None:
    """E 阶段的遥测帧带的是「正在跑第几颗」，不是「跑完了几颗」。

    第四颗种子一开始，事件里就是 4/4；照字面读，阶梯写 4/4 (100%)、进度条跳到 E 的末端，
    而那一颗还要再跑 7 s。所以运行中的帧说「第 4/4 个」，条只按跑完的三颗走。
    """
    view = ProgressView()
    qtbot.addWidget(view)
    view.set_progress(api.FitProgress("curve", "E", 3, 4, 1.0, "completed final seed 3"))
    finished_three = view.bar.value()

    view.set_progress(api.FitProgress("curve", "E", 4, 4, 1.0, "DE generation (seed 4)", iteration=1))

    assert view.bar.value() == finished_three
    assert view.findChild(QLabel, "fitStageMetric_E").text() == "第 4/4 个"
    view.freeze()


def test_the_live_metrics_say_which_unit_is_running(qtbot) -> None:
    """右栏「阶段进度」同理：运行中的那一颗写「第 4 / 4 个」，收尾后才写 4 / 4。"""
    from xrr_fitter.gui.fitting.metrics import METRIC_STEP, LiveMetricsView

    view = LiveMetricsView()
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "E", 4, 4, 1.0, "", iteration=1))

    assert view.metric_text(METRIC_STEP) == "第 4 / 4 个"


def test_the_rail_never_rounds_a_running_bar_up_to_done(qtbot) -> None:
    """卡片写 999 / 1000 时左栏不能写 100%：两处对"跑完没有"必须给同一个答案。"""
    view = ProgressView()
    qtbot.addWidget(view)

    view.bar.setValue(999)

    assert view.overall_percent() == 99
