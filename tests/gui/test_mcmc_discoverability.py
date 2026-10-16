"""MCMC 的规矩要在用得着它的地方说出来。

审计：walkers 下限只写在左栏页脚，对话框里那个数值框旁边没有；关掉对话框后主窗口只剩
「N walkers · 采样中」，看不到跑到第几步、还要多久；取消一条跑了很久的链不问一句，整条直接
丢掉；非推荐候选的「运行 MCMC」灰着却不说为什么。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMessageBox
from tests.gui.test_results import _panel, _project_with_result, _two_candidate_result

import xrr_fitter.api as api


def _expert_panel(qtbot):
    return _panel(qtbot, _project_with_result(_two_candidate_result(), selected="candidate-a", expert=True))


def test_the_walkers_bound_sits_beside_its_spin_box(qtbot) -> None:
    """候选 A 有 17 个自由参数，所以 walkers 至少 36；数值框旁边就写着，改数时跟着改口。"""
    panel = _expert_panel(qtbot)
    hint = panel.mcmc_group.findChild(QLabel, "mcmcWalkersHint")
    assert hint is not None

    panel.walkers.setValue(10)

    assert "2·n_free + 2" in hint.text()
    assert "17→10" in hint.text() and "未满足" in hint.text()


def test_a_greyed_run_button_says_why(qtbot) -> None:
    """只有带着自己那份不确定度证据的候选能起采样；别的候选上那枚灰按钮要说清原因。"""
    panel = _expert_panel(qtbot)

    panel._candidate_requested("candidate-b")

    assert panel.selected_candidate_id() == "candidate-b"
    assert not panel.mcmc_button.isEnabled()
    assert "不确定度" in panel.mcmc_button.toolTip()


class _Job:
    def __init__(self) -> None:
        self.cancels = 0

    def cancel(self) -> None:
        self.cancels += 1


def test_cancelling_a_long_chain_asks_first(qtbot, monkeypatch) -> None:
    """跑了十分钟的链，取消就是十分钟白跑：先问一句，答「否」时链照跑。"""
    panel = _expert_panel(qtbot)
    job = _Job()
    panel.controller._job = job
    monkeypatch.setattr(panel.mcmc_progress, "elapsed_seconds", lambda: 600.0)
    answers = iter((QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: next(answers))
    try:
        panel._refresh_mcmc_buttons(True)
        panel.request_cancel()
        assert job.cancels == 0

        panel.request_cancel()
        assert job.cancels == 1
    finally:
        panel.controller._job = None


def test_the_sampling_progress_publishes_a_status_line(qtbot) -> None:
    """对话框关掉以后状态栏是唯一能看见采样的地方，所以进度视图把同一句摘要发出去。"""
    from xrr_fitter.gui.results.mcmc_progress import McmcProgressView

    now = [0.0]
    view = McmcProgressView(clock=lambda: now[0])
    qtbot.addWidget(view)
    published: list[str] = []
    view.summary_changed.connect(published.append)
    view.begin("curve", "candidate-a", api.McmcConfig(walkers=36, burn_in=100, production_steps=500))
    now[0] = 30.0

    view.set_progress(api.FitProgress("curve", "MCMC", 120, 600, 1.0, "", iteration=120, acceptance_rate=0.3))

    assert "120 / 600" in published[-1]
    assert "预计剩余" in published[-1]
    view.freeze()
