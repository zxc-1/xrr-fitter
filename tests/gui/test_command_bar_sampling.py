"""帧⑤ 命令栏右端那枚执行命令：``▶ 运行 MCMC``。

设计稿七帧里，命令栏右端那枚逐帧换人：帧①③（反射率 / SLD 剖面）是实心的 ``⚡ 一键拟合``，
帧④（跑着）换成 ``⏸ 暂停`` 与 ``⏹ 停止``，帧⑤（不确定度）是 ``▶ 运行 MCMC``。实现此前只
分了前两档——切到不确定度那一页，栏上推荐的仍是「一键拟合」，而那一页每一段证据都出自采样：
读者要的下一步是把链接着跑下去，重新拟合只会把手上这份结果连着证据一起换掉。

这个文件钉四件事：换的是位置上的同一枚（不是多摆一枚）、字面照设计稿、点下去走的是检视器
里那条同一条采样命令、以及可用状态跟着那条命令而不是自己另判一次。
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import QPushButton, QToolButton
from tests.support.model_cases import dataset_project, final_fit_result

import xrr_fitter.api as api


def _project() -> api.XrrProject:
    """一份已经拟合完的项目——不确定度那一页得有结果可读才进得去。"""
    value = api.XrrProject.new((dataset_project(result=final_fit_result()),), master_seed=1201)
    value = replace(value, base_directory="/private/tmp")
    # ``XrrProject.new`` 不选数据集，而右栏与画布都是从「当前数据集」找结果的。
    return api.select_active_dataset(value, "curve")


def _window(qtbot):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project()))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _commands(window) -> tuple[QToolButton, QToolButton]:
    """命令栏右端那两枚执行命令：一键拟合、运行 MCMC。"""
    start = window.findChild(QToolButton, "startFitToolButton")
    mcmc = window.findChild(QToolButton, "runMcmcToolButton")
    assert start is not None, "startFitToolButton"
    assert mcmc is not None, "runMcmcToolButton"
    return start, mcmc


def _left_edge(widget, window) -> int:
    return widget.mapTo(window, widget.rect().topLeft()).x()


def test_the_uncertainty_view_puts_run_mcmc_where_the_fit_command_stands(qtbot) -> None:
    """切到不确定度那一页，右端那枚从 ``⚡ 一键拟合`` 换成 ``▶ 运行 MCMC``。

    两枚是同一个位置上的替换而不是并排：设计稿帧①③⑤ 的右端都恰好两枚（执行 + 导出…），
    三枚会把 ``导出…`` 挤向 ``☾``。
    """
    window = _window(qtbot)
    start, mcmc = _commands(window)

    assert (start.isVisibleTo(window), mcmc.isVisibleTo(window)) == (True, False)
    assert start.text() == "⚡ 一键拟合"

    window.plot_panel.select_view("uncertainty")

    assert (start.isVisibleTo(window), mcmc.isVisibleTo(window)) == (False, True)
    assert mcmc.text() == "▶ 运行 MCMC"


def test_leaving_the_uncertainty_view_gives_the_fit_command_back(qtbot) -> None:
    """换回别的分析页，右端那枚回到 ``⚡ 一键拟合``。

    与上一条是同一枚开关的两面。只钉「切进去会换」的话，实现可以换完就不换回来——读者在
    候选解那一页看着「运行 MCMC」，而那一页推荐的下一步本来是重拟合。
    """
    window = _window(qtbot)
    window.plot_panel.select_view("uncertainty")
    window.plot_panel.select_view("candidates")
    start, mcmc = _commands(window)

    assert (start.isVisibleTo(window), mcmc.isVisibleTo(window)) == (True, False)


def test_run_mcmc_is_not_the_screens_primary_call_to_action(qtbot) -> None:
    """``▶ 运行 MCMC`` 不是实心那枚——设计稿帧⑤ 给它的是普通 ``.btn``。

    帧①③ 的 ``⚡ 一键拟合`` 带 ``primary``：那一屏还没有结果或结果要重算，实心那枚就是
    唯一推荐的下一步。帧⑤ 手上已经有一份可读的结果，采样是往上补一层证据；两枚都实心，
    「整屏只推荐一个下一步」这条就不成立了。
    """
    window = _window(qtbot)
    window.plot_panel.select_view("uncertainty")
    start, mcmc = _commands(window)

    assert start.property("primary") is True
    assert bool(mcmc.property("primary")) is False


def test_run_mcmc_forwards_to_the_sampler_the_inspector_owns(qtbot) -> None:
    """点它走的是检视器里那条采样命令，不是另接一次控制器。

    照命令栏 ``⏸ 暂停`` / ``⏹ 停止`` 的同一条规矩：转发点击，于是 walkers 校验、候选解
    归属与「链跑着时不许再起一条」都只有一份实现。
    """
    from xrr_fitter.gui.chrome import refresh_command_bar

    window = _window(qtbot)
    window.plot_panel.select_view("uncertainty")
    _start, mcmc = _commands(window)

    window.result_panel.mcmc_button.setEnabled(True)
    refresh_command_bar(window)
    with qtbot.waitSignal(window.result_panel.mcmc_button.clicked, timeout=500):
        mcmc.click()


def test_run_mcmc_follows_the_samplers_own_enabled_state(qtbot) -> None:
    """能不能按，跟着检视器那枚采样按钮走。

    「这条候选解够不够跑 MCMC」由 ``candidate_is_mcmc_ready`` 判，链跑着时那枚也会自己灰
    掉。命令栏另判一次的话，两处迟早会各说各话——而按下去没反应的按钮比灰着的更难懂。
    """
    from xrr_fitter.gui.chrome import refresh_command_bar

    window = _window(qtbot)
    window.plot_panel.select_view("uncertainty")
    _start, mcmc = _commands(window)

    window.result_panel.mcmc_button.setEnabled(False)
    refresh_command_bar(window)
    assert mcmc.isEnabled() is False

    window.result_panel.mcmc_button.setEnabled(True)
    refresh_command_bar(window)
    assert mcmc.isEnabled() is True


def test_a_live_fit_takes_run_mcmc_off_the_bar_with_the_rest_of_the_group(qtbot) -> None:
    """拟合跑起来，整组让位给 ``⏸ 暂停`` / ``⏹ 停止``——新来的这枚不例外。

    帧④ 的右端只有那两枚。采样命令留在栏上会读成「这时候还能再起一条链」，而拟合与采样
    在实现里是互锁的。
    """
    from xrr_fitter.gui.chrome import set_command_bar_running

    window = _window(qtbot)
    window.plot_panel.select_view("uncertainty")
    start, mcmc = _commands(window)

    set_command_bar_running(window, True)

    assert (start.isVisibleTo(window), mcmc.isVisibleTo(window)) == (False, False)

    set_command_bar_running(window, False)

    assert (start.isVisibleTo(window), mcmc.isVisibleTo(window)) == (False, True)


def test_the_bar_keeps_the_designs_order_execute_then_export(qtbot) -> None:
    """``▶ 运行 MCMC`` 站在 ``导出…`` 左边，和它顶替的那枚同一个位置。

    设计稿右端自左向右是「执行 · 导出… · ☾」。换人时把新的一枚追加到组末尾，顺序就成了
    「导出… · 运行 MCMC」，同一枚命令于是在两页上左右横跳。
    """
    window = _window(qtbot)
    window.plot_panel.select_view("uncertainty")
    _start, mcmc = _commands(window)
    export = window.findChild(QPushButton, "exportResultsButton")
    assert export is not None

    assert _left_edge(mcmc, window) < _left_edge(export, window)
