"""Pause/resume affordances must describe the same action to every user."""

import pytest
from tests.gui.test_fit_progress import _FakeJob, _project

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.main_window import MainWindow


def _running_window(qtbot, tmp_path, monkeypatch):
    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    job = _FakeJob()
    job.pause = lambda: None
    job.resume = lambda: None
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    assert window.fit_panel.start_fit()
    return window, job


def _assert_actions(buttons, text, name, help_text):
    for button in buttons:
        assert (button.text(), button.accessibleName(), button.toolTip()) == (text, name, help_text)


@pytest.mark.parametrize("toolbar", (False, True))
@pytest.mark.parametrize("finish_paused", (False, True))
def test_pause_actions_keep_text_accessibility_and_help_in_sync(qtbot, tmp_path, monkeypatch, toolbar, finish_paused):
    window, job = _running_window(qtbot, tmp_path, monkeypatch)
    panel = window.fit_panel
    buttons = (panel.pause_button, window.toolbar_pause_button)
    trigger = buttons[int(toolbar)]
    original_help = panel.pause_button.toolTip()
    try:
        trigger.click()

        _assert_actions(
            buttons, "▶ 继续", "继续拟合", "解除暂停请求，让已到达安全点的计算继续；保留已完成阶段与当前最优"
        )

        if finish_paused:
            job.events = (api.OperationEvent(0, "stopped"),)
            panel.controller.poll_now()
        else:
            trigger.click()

        _assert_actions(buttons, "⏸ 暂停", "暂停拟合", original_help)
    finally:
        job.events = (api.OperationEvent(1, "stopped"),)
        panel.controller.poll_now()
