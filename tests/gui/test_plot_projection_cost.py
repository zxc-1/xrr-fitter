"""一次投影只做屏幕上要的那部分工作，而且切过去时内容是最新的。

审计实测一次重投影阻塞主线程 270–390 ms：预检把 10 张 matplotlib 视图在 scratch 画布上整套画一遍；
按步骤藏起来的页照画不误；每次投影把全部源文件重读一遍；字体那一遍对整张图 ``findobj``，把
``Axes.clear()`` 刚恢复成惰性的刻度对象一个个造出来。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from matplotlib.figure import Figure
from tests.gui.plot_cases_5 import _sld_project
from tests.gui.plot_support import _project_with_curves

import xrr_fitter.api as api
import xrr_fitter.gui.plots.diagnostics as diagnostics


def _panel(qtbot):
    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    qtbot.addWidget(panel)
    return panel


def _spy_figures(monkeypatch) -> list[object]:
    drawn: list[object] = []
    original = diagnostics.apply_figure_font

    def spy(figure):
        drawn.append(figure)
        return original(figure)

    monkeypatch.setattr(diagnostics, "apply_figure_font", spy)
    return drawn


def _spy_imports(monkeypatch) -> list[object]:
    calls: list[object] = []
    original = api.import_data

    def spy(source, *args, **kwargs):
        calls.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(api, "import_data", spy)
    return calls


def test_the_result_step_draws_no_matplotlib_pane_that_is_off_screen(qtbot, tmp_path, monkeypatch) -> None:
    project, _result = _sld_project(tmp_path, "curve")
    panel = _panel(qtbot)
    panel.set_step_scope(4)
    panel.project_project(project)
    drawn = _spy_figures(monkeypatch)

    panel.project_project(project)

    # 结果步的画布是反射率与残差两张 pyqtgraph 卡；剖面、候选比较、热图、不确定度都不在屏上。
    assert drawn == []


def test_an_off_screen_pane_is_brought_up_to_date_when_it_is_shown(qtbot, tmp_path, monkeypatch) -> None:
    project, _result = _sld_project(tmp_path, "curve")
    panel = _panel(qtbot)
    panel.set_step_scope(4)
    panel.project_project(project)
    drawn = _spy_figures(monkeypatch)

    panel.toolbar.sld_profile_action.setChecked(True)

    assert panel.view("sld").figure in drawn
    labels = {str(line.get_label()) for line in panel.view("sld").axes.lines}
    assert "SLD 虚部" in labels


def test_switching_to_a_hidden_analysis_page_shows_the_latest_selection(qtbot, tmp_path) -> None:
    project, _result = _sld_project(tmp_path, "curve")
    panel = _panel(qtbot)
    panel.set_step_scope(4)
    panel.project_project(project)
    dataset_id = project.datasets[0].dataset_id
    selected = replace(project.ui_state, selected_candidate_ids=((dataset_id, "candidate-b"),))
    panel.project_project(replace(project, ui_state=selected))

    panel.select_view("uncertainty")

    texts = " ".join(text.get_text() for axes in panel.view("uncertainty").figure.axes for text in axes.texts)
    assert "当前查看 candidate-b" in texts


def test_reprojecting_unchanged_sources_rereads_no_file(qtbot, tmp_path, monkeypatch) -> None:
    project = _project_with_curves(tmp_path, count=2)
    panel = _panel(qtbot)
    panel.project_project(project)
    calls = _spy_imports(monkeypatch)

    panel.project_project(project)
    second = project.datasets[1].dataset_id
    panel.project_project(replace(project, ui_state=replace(project.ui_state, active_dataset_id=second)))

    assert calls == []


def test_changed_import_parameters_or_source_bytes_are_reread(qtbot, tmp_path, monkeypatch) -> None:
    project = _project_with_curves(tmp_path, count=1)
    panel = _panel(qtbot)
    panel.project_project(project)
    calls = _spy_imports(monkeypatch)

    shifted = replace(project, datasets=(replace(project.datasets[0], import_angle_offset_deg=0.01),))
    panel.project_project(shifted)
    assert len(calls) == 1

    source = tmp_path / "curve-0.xy"
    rows = (f"{0.05 + index * 0.02:.8f} {900.0 / (index + 1):.12g}" for index in range(32))
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    panel.project_project(shifted)
    assert len(calls) == 2


def test_the_font_pass_does_not_build_the_lazy_tick_objects() -> None:
    figure = Figure()
    axes = figure.subplots()
    axes.plot(np.linspace(0.0, 10.0, 50), np.linspace(0.0, 1.0, 50))
    axes.set_title("标题")

    diagnostics.apply_figure_font(figure)

    assert len(axes.xaxis.__dict__.get("majorTicks", ())) <= 1
    assert axes.title.get_fontfamily() == diagnostics._cjk_font().get_family()


def test_every_text_on_the_uncertainty_page_wears_the_cjk_family(qtbot, tmp_path) -> None:
    """不走 ``findobj`` 之后，文字一处都不能漏——相关矩阵的读数栏挂在 ``inset_axes`` 里，不在 ``figure.axes``。"""
    from matplotlib.text import Text

    project, _result = _sld_project(tmp_path, "curve")
    panel = _panel(qtbot)
    panel.project_project(project)
    view = panel.view("uncertainty")

    view.canvas.draw()

    expected = tuple(diagnostics._cjk_font().get_family())
    families = {tuple(text.get_fontfamily()) for text in view.figure.findobj(match=Text) if text.get_text()}
    assert families == {expected}


def test_tick_labels_created_while_rendering_still_wear_the_cjk_family(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_batch_trends(("样品一", "样品二", "样品三"), (100.0, 120.0, 140.0), (50.0, 55.0, 60.0))
    view = panel.view("trend")

    view.canvas.draw()

    expected = tuple(diagnostics._cjk_font().get_family())
    labels = (*view.axes.get_xticklabels(), *view.axes.get_yticklabels())
    assert labels
    assert {tuple(label.get_fontfamily()) for label in labels} == {expected}
