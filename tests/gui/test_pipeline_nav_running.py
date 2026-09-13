"""拟合运行中，左栏那六步要停在「拟合」上。

设计稿帧④ 是运行中的那一屏：三步 done ✓、拟合 current、圆点写「4」、那行小字是
「进行中 62%」。此前左栏到不了这一步——``_determine_step`` 只看项目里存了什么，而「正在跑」
不是项目字段，所以运行中左栏仍然停在「参数」，「拟合」那行读的还是它的用途说明「运行自动
拟合」。同一次运行里右栏和画布都已经跟着 ``running_changed`` 收窄了（``apply_step_scope``
的 ``RUNNING_STEP_INDEX``），左栏是唯一没跟上的那一列：一块屏幕上两种说法。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path):
    """有数据、有结构、还没有结果：运行开始前项目就停在这个样子。"""
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "curve.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(AIR, (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),), SI)
    project = api.set_structure(project, "curve", structure)
    return api.select_active_dataset(project, "curve")


def _window(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(1)
    return window


def _description(nav, title: str) -> str:
    return nav.findChild(QLabel, f"pipelineDescription_{title}").text()


def _dot(nav, title: str) -> str:
    return nav.findChild(QLabel, f"pipelineDot_{title}").text()


def test_a_running_fit_moves_the_rail_to_the_fitting_step(qtbot, tmp_path) -> None:
    """运行中左栏停在「拟合」。

    项目字段一个都没变（还是那份有结构、没有结果的项目），所以这一步只可能来自运行状态。
    """
    window = _window(qtbot, _project(tmp_path))
    nav = window.pipeline_nav
    assert nav.current_step_index() == 2

    window.fit_panel.running_changed.emit(True)

    assert nav.current_step_index() == 3


def test_the_running_step_keeps_the_ordinal_and_marks_the_three_before_it_done(qtbot, tmp_path) -> None:
    """帧④ 的圆点：数据/结构/参数 是 ✓，拟合 是「4」。

    ``_apply_index`` 本来就照这个规矩画，这里要钉的是它收到的索引确实是 3——不然 ✓ 只会画到
    「结构」，而「参数」和「拟合」都还挂着序号。
    """
    window = _window(qtbot, _project(tmp_path))
    nav = window.pipeline_nav

    window.fit_panel.running_changed.emit(True)

    assert [_dot(nav, title) for title in ("数据", "结构", "参数")] == ["✓", "✓", "✓"]
    assert _dot(nav, "拟合") == "4"


def test_the_fitting_sub_line_reads_the_live_percentage(qtbot, tmp_path) -> None:
    """那行小字换成「进行中 NN%」，数字跟进度条同一个来源。

    ``_fit_state`` 读的是项目字段，运行中它只会说「未开始」——正在跑的时候说未开始。百分数
    取自 ``ProgressView`` 的进度条本身（``overall_percent``），所以左栏
    不可能报出一个和条形图不一样的位置。
    """
    window = _window(qtbot, _project(tmp_path))
    nav = window.pipeline_nav
    progress = window.fit_panel.progress_view

    window.fit_panel.running_changed.emit(True)
    progress.bar.setValue(620)

    assert _description(nav, "拟合") == "进行中 62%"


def test_leaving_the_run_hands_the_rail_back_to_the_project(qtbot, tmp_path) -> None:
    """运行结束，左栏交回项目状态算出来的那一步。

    运行中的高亮是覆盖而不是赋值：底下那个「项目走到哪了」的判断没有被改写，所以撤掉覆盖就该
    原样回到 2，「拟合」那行也回到项目字段的说法。
    """
    window = _window(qtbot, _project(tmp_path))
    nav = window.pipeline_nav
    window.fit_panel.running_changed.emit(True)
    assert nav.current_step_index() == 3

    window.fit_panel.running_changed.emit(False)

    assert nav.current_step_index() == 2
    assert _description(nav, "拟合") == "未开始"


def test_a_project_change_during_a_run_does_not_pull_the_rail_off_the_fitting_step(qtbot, tmp_path) -> None:
    """运行中项目变了（预览、检查点回写），左栏仍然停在「拟合」。

    ``_sync_state`` 挂在 ``project_changed`` 上，每次都重算索引；覆盖必须在那条路上也生效，
    否则运行途中任意一次项目回写都会把左栏弹回「参数」。
    """
    from xrr_fitter.gui.document import ProjectDocument

    project = _project(tmp_path)
    document = ProjectDocument(project)
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(document)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(1)
    window.fit_panel.running_changed.emit(True)

    thicker = api.StructureSpec(AIR, (api.LayerSpec("film", SIO2, 41.5, roughness_a=3.0),), SI)
    document.replace_project(api.set_structure(project, "curve", thicker))
    qtbot.wait(1)

    assert window.pipeline_nav.current_step_index() == 3


def test_the_progress_card_lives_in_the_canvas_not_the_inspector(qtbot, tmp_path) -> None:
    """设计稿帧④ 把「总进度」连同九阶段放在画布列，右栏只留控制与实时指标。

    实测反了过来：整块进度挤在 340px 的右栏里，九个阶段名被压成两行一条，而画布列
    仍在画一张与此刻无关的反射率图。进度是这一帧的主角，主角该占中间那一列。
    """
    window = _window(qtbot, _project(tmp_path))
    progress = window.fit_panel.progress_view

    assert window.canvas_column.isAncestorOf(progress)
    assert not window.inspector_column.isAncestorOf(progress)


def test_the_running_inspector_carries_the_live_metrics(qtbot, tmp_path) -> None:
    """设计稿帧④ 的右栏是三段：实时指标 / 各数据集目标值 / 控制。

    实测只有控制那一段，所以整屏里「此刻收敛到多少」只能从画布那条进度条上阶段行的
    附注里读——那是阶段的注脚，不是读数。指标那一段跟着运行那一步入场，也只在这一步
    入场：不在跑的时候它报的是上一轮的残值。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window(qtbot, _project(tmp_path))
    card = window.inspector_cards["inspectorLiveMetrics"]

    apply_step_scope(window, 3)
    assert card.isVisibleTo(window)
    assert window.inspector_column.isAncestorOf(window.live_metrics)

    apply_step_scope(window, 1)
    assert not card.isVisibleTo(window)
