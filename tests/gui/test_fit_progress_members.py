"""A parallel independent batch keeps each dataset's own story on the progress card.

审计回放 91 s 的双数据集独立批量：卡片一换数据集就清空轨迹与预览，被清了 194 次、最后只剩
10 个点；阶段标签每秒在两条曲线之间跳；「全局目标 J」其实只是最后到达那一帧的数据集的值；
单条曲线的轨迹也会在每次换阶段时把横轴卷回 0。

两条曲线并行时卡片一次只讲一条：读者在左栏选中的那条，没选或它已收尾时讲先报到、还没收尾的那条。
进度条仍是整批的平均位置。
"""

from __future__ import annotations

from tests.gui.test_fit_progress import _FakeJob, _project, _StepClock

import xrr_fitter.api as api
from xrr_fitter.gui.fitting.progress import ProgressView


def _frame(owner, stage, completed, total, objective):
    return api.FitProgress(owner, stage, completed, total, objective, "")


def _view(qtbot, clock=None):
    view = ProgressView() if clock is None else ProgressView(clock=clock)
    qtbot.addWidget(view)
    return view


def test_interleaved_members_do_not_wipe_each_others_trace(qtbot) -> None:
    """两条曲线的帧交错到达，卡片上那条轨迹仍是一条曲线自己攒出来的完整序列。"""
    view = _view(qtbot)
    for objective in (10.0, 9.0, 8.0, 7.0):
        view.set_progress(_frame("a", "B", 1, 2, objective))
        view.set_progress(_frame("b", "B", 1, 2, objective * 100))

    assert list(view.trace_item.yData) == [10.0, 9.0, 8.0, 7.0]
    view.freeze()


def test_the_trace_axis_never_rolls_back_at_a_stage_change(qtbot) -> None:
    """横轴是这条曲线自己的总完成比例，换阶段时不会卷回 0。"""
    view = _view(qtbot)
    for stage, completed, total in (("B", 1, 2), ("B", 2, 2), ("C", 1, 3), ("D", 3, 3), ("E", 1, 4)):
        view.set_progress(_frame("a", stage, completed, total, 1.0))

    xs = list(view.trace_item.xData)
    assert xs == sorted(xs)
    view.freeze()


def test_the_card_stays_with_one_member_instead_of_hopping(qtbot) -> None:
    """没人选中时，卡片讲先报到的那条，不在两条并行的曲线之间每帧来回跳。"""
    view = _view(qtbot)
    view.set_progress(_frame("a", "D", 1, 3, 1.0))
    view.set_progress(_frame("b", "B", 1, 2, 1.0))

    assert view.stage_label.text().startswith("a · ")
    view.freeze()


def test_the_card_follows_the_dataset_the_reader_selected(qtbot) -> None:
    """读者在左栏点了哪条，卡片就讲哪条：阶段、阶梯、轨迹都换成它的。"""
    view = _view(qtbot)
    view.set_progress(_frame("a", "D", 1, 3, 1.0))
    view.set_progress(_frame("b", "B", 1, 2, 5.0))

    view.follow_dataset("b")

    assert view.stage_label.text().startswith("b · ")
    assert list(view.trace_item.yData) == [5.0]
    view.freeze()


def test_interleaved_stages_do_not_blank_the_remaining_time(qtbot) -> None:
    """两条曲线处在不同阶段时，每一帧都「换了阶段」；估计不能因此每帧清零。"""
    # 取不空的假时钟：断言失败时心跳计时器还活着，再读一次也只是停在最后那个读数上。
    view = _view(qtbot, clock=_StepClock(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    for completed in (1, 2, 3):
        view.set_progress(_frame("a", "E", completed, 4, 1.0))
        view.set_progress(_frame("b", "B", 1, 2, 1.0))

    assert "--:--" not in view.meta_remaining.text()
    view.freeze()


def test_the_objective_reading_belongs_to_the_dataset_on_the_card(qtbot, tmp_path, monkeypatch) -> None:
    """独立批量没有「全局」目标值：那一行报卡片上那条曲线的 J，并说出是谁的。"""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.metrics import METRIC_OBJECTIVE
    from xrr_fitter.gui.fitting.panel import FitPanel

    panel = FitPanel(ProjectDocument(_project(tmp_path, count=2)))
    qtbot.addWidget(panel)
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: _FakeJob())
    try:
        assert panel.start_fit()
        panel.controller.progress_changed.emit(_frame("curve-0", "B", 1, 2, 1.25))
        panel.controller.progress_changed.emit(_frame("curve-1", "B", 1, 2, 4.5))

        assert panel.live_metrics.metric_text(METRIC_OBJECTIVE) == "1.2500"
        assert "curve-0" in panel.live_metrics.metric_label(METRIC_OBJECTIVE)
    finally:
        panel.controller._finish()
