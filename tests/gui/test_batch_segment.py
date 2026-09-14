"""帧①④ 命令栏的批量段。

两帧的 ``.cmdbar`` 都在 引导·专家 段右边挂一组 ``批量 独立|联合``，帧① 亮「独立」,
帧④ 亮「联合」。实测里这个选择是拟合卡上的一枚下拉框：读者要先把检视器滚到拟合卡
才知道这次拟合是三条曲线各自拟合还是三条共享一套层结构——而这件事决定了整屏参数
表的含义（共享 vs 独立），属于全局状态，不是某张卡的局部设置。

引导模式不给这个选择。帧② 的 cmdbar 只有 新建/打开/保存 加模式段，其导语写的是
「隐藏停靠面板与高级批量选项」。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QToolBar, QToolButton, QWidget
from tests.gui.plot_support import _project_with_curves


def _window(qtbot, project=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(None if project is None else ProjectDocument(project))
    qtbot.addWidget(window)
    return window


def _two_dataset_window(qtbot, tmp_path: Path):
    """联合拟合至少要两个数据集，所以能改成联合的项目必须先装两条曲线。"""
    return _window(qtbot, _project_with_curves(tmp_path, count=2))


def test_the_command_bar_carries_the_designs_batch_segment(qtbot) -> None:
    """``批量 独立|联合`` 与模式段并排，不藏在拟合卡里。"""
    window = _window(qtbot)

    group = window.findChild(QWidget, "batchModeGroup")
    toolbar = window.findChild(QToolBar, "mainToolbar")
    assert group is not None
    assert toolbar.isAncestorOf(group)

    label = window.findChild(QLabel, "batchModeLabel")
    assert label.text() == "批量"
    independent = window.findChild(QToolButton, "batchModeIndependent")
    joint = window.findChild(QToolButton, "batchModeJoint")
    assert [button.text() for button in (independent, joint)] == ["独立", "联合"]
    # 新项目是独立拟合，所以那一半已经亮着。
    assert (independent.isChecked(), joint.isChecked()) == (True, False)


def test_choosing_a_half_writes_the_mode_onto_the_project(qtbot, tmp_path: Path) -> None:
    """段是投影，状态归项目：点亮哪一半由 ``project.batch_mode`` 决定。

    手势要落在设计稿真的画着这一段的那一步上。批量段跟着画布走（见
    ``test_batch_mode_rides_with_the_reflectivity_canvas``），而刚导入两条曲线、还没建
    结构的项目停在 结构 那一步，画布上是 SLD 剖面——那一步栏上没有批量段。帧① 才是它
    亮着的那一帧，所以这里把窗口摆到第 4 步再点。

    摆位不做的话，这个测试测的不是写入路径而是 Qt 的一个副作用：命令栏收起一组是
    ``QAction::setVisible(False)``，而它顺手清掉 action 的 enabled 位；``QToolBar.addWidget``
    给的是 ``QWidgetAction``，它又把这个位同步到 ``defaultWidget``。于是被收起来的那枚
    ``QToolButton`` 是 disabled 的，``click()`` 连 ``clicked`` 都不发，项目自然没被改。
    点前先断言它可点，就是为了这个坑再回来时能一眼看出是控件没上台，而不是写入丢了。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _two_dataset_window(qtbot, tmp_path)
    apply_step_scope(window, 4)
    joint = window.findChild(QToolButton, "batchModeJoint")
    assert joint.isEnabled() is True

    joint.click()

    assert window.document.project.batch_mode == "joint"


def test_the_segment_follows_a_mode_the_project_changed_elsewhere(qtbot, tmp_path: Path) -> None:
    """一处状态两处控件：项目改了模式，段要跟着亮，不能两边各说一套。"""
    import xrr_fitter.api as api

    window = _two_dataset_window(qtbot, tmp_path)
    independent = window.findChild(QToolButton, "batchModeIndependent")
    joint = window.findChild(QToolButton, "batchModeJoint")

    window.document.replace_project(api.set_batch_mode(window.document.project, "joint"))

    assert (independent.isChecked(), joint.isChecked()) == (False, True)


def test_a_single_dataset_project_cannot_be_switched_to_joint(qtbot) -> None:
    """联合拟合要至少两个数据集；拒绝时段回到项目当前的那一半。

    服务层拒绝的方式是抛 ``ValueError``，此时段已经把「联合」点亮了——它是个
    ``QToolButton``，点击即改变自己的勾选态。段必须回读项目，否则屏幕上写着联合而
    项目里还是独立。
    """
    window = _window(qtbot)
    independent = window.findChild(QToolButton, "batchModeIndependent")
    joint = window.findChild(QToolButton, "batchModeJoint")

    joint.click()

    assert window.document.project.batch_mode == "independent"
    assert (independent.isChecked(), joint.isChecked()) == (True, False)
    assert "联合" in window.statusBar().currentMessage() or window.statusBar().currentMessage() != ""


def test_guided_mode_puts_the_batch_segment_away(qtbot) -> None:
    """帧② 的 cmdbar 没有批量段：「隐藏停靠面板与高级批量选项」。"""
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    group = window.findChild(QWidget, "batchModeGroup")

    window.set_guidance_visible(True)
    assert group.isVisibleTo(window) is False

    window.set_guidance_visible(False)
    assert group.isVisibleTo(window) is True


def test_a_running_fit_locks_the_batch_segment(qtbot) -> None:
    """拟合跑起来之后改批量模式没有意义：这次拟合已经按旧模式排好阶段了。"""
    window = _window(qtbot)
    group = window.findChild(QWidget, "batchModeGroup")

    window.fit_panel.running_changed.emit(True)
    assert group.isEnabled() is False

    window.fit_panel.running_changed.emit(False)
    assert group.isEnabled() is True


def test_the_fit_card_no_longer_repeats_the_choice(qtbot) -> None:
    """同一个选择不放两处：拟合卡上的下拉框让位给命令栏的段。

    两个控件写同一个字段时，读者要先判断哪个是真的。设计稿只画了命令栏这一处。
    """
    from PySide6.QtWidgets import QComboBox

    window = _window(qtbot)

    assert window.findChild(QComboBox, "batchModeSelector") is None
