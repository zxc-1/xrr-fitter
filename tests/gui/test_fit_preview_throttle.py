"""实时预览限流时，最后那一帧不能丢。

``FitPanel._project_preview`` 每 50 ms 最多转发一帧，窗口里的帧原来直接丢掉。一段局部搜索收尾时的
那几帧正好挤在同一个窗口里：画布于是停在倒数第二条曲线上，求解器手里那条更好的直到下一阶段出帧
才画出来——profile 那一段能沉默一分钟。所以窗口里的帧要留住最新的一个，窗口一过就画；运行收工
或换了数据集时留住的那一帧作废，不能盖在已发布的结果或另一条曲线上。
"""

from __future__ import annotations

from tests.gui.test_fit_progress import _FakeJob, _preview_progress, _project, _StepClock

import xrr_fitter.api as api


def _panel(qtbot, tmp_path, count=1):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    # 每一帧都比上一帧晚 0.02 s，始终落在 0.05 s 的窗口里。
    panel = FitPanel(ProjectDocument(_project(tmp_path, count)), clock=_StepClock(0.0, 0.02, 0.04, 0.06))
    qtbot.addWidget(panel)
    emitted: list[float] = []
    panel.preview_available.connect(lambda _qz, model: emitted.append(float(model[0])))
    return panel, emitted


def test_the_last_frame_of_a_burst_is_still_drawn(qtbot, tmp_path) -> None:
    panel, emitted = _panel(qtbot, tmp_path)

    panel._project_preview(_preview_progress(1.0))
    panel._project_preview(_preview_progress(2.0))

    qtbot.waitUntil(lambda: emitted == [1.0, 2.0], timeout=1000)


def test_a_held_frame_is_dropped_when_the_run_ends(qtbot, tmp_path, monkeypatch) -> None:
    """收工后画布换成已发布的结果；留到窗口过后才画的预览会把它盖住。"""
    panel, emitted = _panel(qtbot, tmp_path)
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: _FakeJob())
    assert panel.start_fit()
    panel._project_preview(_preview_progress(1.0))
    panel._project_preview(_preview_progress(2.0))

    panel.controller._finish()
    qtbot.wait(4 * int(1000 * panel.PREVIEW_MIN_INTERVAL_S))

    assert emitted == [1.0]


def test_a_held_frame_is_dropped_when_the_reader_switches_dataset(qtbot, tmp_path) -> None:
    """留住的是 curve-0 的模型；读者已经换到 curve-1 时，画出来就是把它叠在别人的观测上。"""
    panel, emitted = _panel(qtbot, tmp_path, count=2)
    panel._project_preview(_preview_progress(1.0))
    panel._project_preview(_preview_progress(2.0))

    panel.document.select_active_dataset("curve-1")
    qtbot.wait(4 * int(1000 * panel.PREVIEW_MIN_INTERVAL_S))

    assert emitted == [1.0]
