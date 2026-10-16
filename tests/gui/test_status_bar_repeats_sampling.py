"""采样跑着时，状态栏复述进度视图发出的那一句（F43）。

对话框关掉以后，状态栏是唯一还看得见采样的地方。进度视图已经把「采样 n / N 步 · 接受率 · 预计剩余」
发出来（``McmcProgressView.summary_changed``），停了发空串；窗口得把它接到底栏，否则那一句没人念。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel
from tests.gui.test_shell_status import _span_text


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def test_the_status_bar_repeats_the_sampling_summary_and_clears_it_when_sampling_stops(qtbot) -> None:
    window = _window(qtbot)
    summary = "采样 120 / 600 步 · 接受率 0.30 · 预计剩余 2:00"

    window.result_panel.mcmc_progress.summary_changed.emit(summary)

    assert window.findChild(QLabel, "fitPositionStatus").text() == summary
    assert summary in _span_text(window, "statusPositionSpan")

    window.result_panel.mcmc_progress.summary_changed.emit("")

    assert window.findChild(QLabel, "fitPositionStatus").text() == ""
