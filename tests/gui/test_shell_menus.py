"""Menu and toolbar layout: command bar order, segments, guided mode reduction."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QPushButton, QToolBar, QToolButton, QWidget

GUIDED_COMMAND_BAR = ("newProjectButton", "openProjectButton", "saveProjectButton")
EXPERT_ONLY_COMMANDS = ("exportResultsButton",)


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _menu_action(window, object_name):
    """The menu's action for a command, wherever chrome parked it."""
    for action in window.actions():
        if action.objectName() == object_name:
            return action
    action = window.chrome_actions.get(object_name)
    assert action is not None, f"missing menu action: {object_name}"
    return action


def _command_bar_order(window) -> list[str]:
    """The command bar left-to-right, by object name, skipping unnamed fillers."""
    toolbar = window.findChild(QToolBar, "mainToolbar")
    names: list[str] = []
    for action in toolbar.actions():
        widget = toolbar.widgetForAction(action)
        if widget is not None and widget.objectName():
            names.append(widget.objectName())
    return names


def _button_visibility(window: QWidget, names: tuple[str, ...]) -> dict[str, bool]:
    """Which of ``names`` the command bar is currently showing."""
    return {name: window.findChild(QPushButton, name).isVisibleTo(window) for name in names}


def test_chrome_uses_a_toolbar_to_seat_the_command_bar(qtbot) -> None:
    """QToolBar 承载命令栏，不可移动，所有按钮都是它的后代。

    QToolBar 本身继承 QWidget，可以往里摆任何控件；设计稿里 新建/打开/保存 和 ⚡ 一键拟合
    都是普通的 ``.btn sm``，没有边框、没有图标，跟外面的任何其他按钮没有分别。把这一排画
    成工具栏（带边框的凸起按钮）是违反设计的，Qt 默认会把工具栏画成那样，所以 ``toolbar_
    style.qss`` 必须把框擦掉，同时保留 QToolBar 的语义（可拖出来的一整条）。

    不可移动是设计必需：整条贴着窗口上沿，下面就是步骤切换，而 Qt 工具栏默认可拖——不明确
    禁用的话，鼠标在命令栏上稍稍一抖就能把整条拽下来。
    """
    window = _window(qtbot)

    toolbar = window.findChild(QToolBar, "mainToolbar")
    assert toolbar is not None
    assert not toolbar.isMovable()
    new_button = window.findChild(QPushButton, "newProjectButton")
    export_button = window.findChild(QPushButton, "exportResultsButton")
    assert toolbar.isAncestorOf(new_button)
    assert toolbar.isAncestorOf(export_button)


def test_the_command_bar_seats_the_fit_and_export_commands_against_the_right_edge(qtbot) -> None:
    """设计稿四张专家帧的命令栏都是「左边一串项目/模式段 — 弹簧 — 右边执行与导出」。

    弹簧排在 一键拟合 之后时，执行命令贴着 批量 段坐在栏的中段，右边留出一大片空白：
    读起来像「模式选择的一部分」，而设计稿把它单独顶到右端正是为了让它读作这一屏的出口。
    """
    window = _window(qtbot)

    order = _command_bar_order(window)
    spring = order.index("commandBarSpring")
    assert order.index("projectActions") < spring
    assert order.index("workspaceModeSegment") < spring
    assert order.index("batchModeGroup") < spring
    for trailing in ("expertCommandGroup", "runningCommandGroup", "appearanceToggleButton"):
        assert spring < order.index(trailing), trailing
    assert order.index("expertCommandGroup") < order.index("appearanceToggleButton")


def test_the_commands_right_of_the_spring_carry_no_leading_divider(qtbot) -> None:
    """弹簧本身就是分隔：右端那几枚命令与左边隔着整片空白，再画一条竖线是重复的。"""
    window = _window(qtbot)

    for group_name in ("expertCommandGroup", "runningCommandGroup"):
        group = window.findChild(QWidget, group_name)
        assert group is not None, group_name
        rails = [child for child in group.findChildren(QFrame) if child.property("pipelineRail")]
        assert rails == [], f"{group_name} 仍带着 {len(rails)} 条竖线"


def test_the_primary_fit_pill_reads_the_designs_lightning_label(qtbot) -> None:
    """设计稿的实心那枚写作 ``⚡ 一键拟合``，字形在文字里而不是另挂一枚图标。

    字形挂在 ``iconText`` 上：工具栏那枚 QToolButton 读 ``iconText``，菜单读 ``text``，
    所以命令栏得到设计稿的字形，拟合 菜单仍是干净的「一键拟合」，而两处仍是同一条命令。
    """
    window = _window(qtbot)

    button = window.findChild(QToolButton, "startFitToolButton")
    assert button is not None
    assert button.text() == "⚡ 一键拟合"
    assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextOnly
    assert button.property("primary") is True


def test_command_bar_offers_the_designs_projection_segment(qtbot) -> None:
    """The two surfaces read as one exclusive control, the way the design draws them.

    A checkable menu entry says nothing while the menu is shut: the user cannot
    see which of the two projections is on screen without opening 视图. The
    design's segment has a lit half, so the answer is always on the command bar.
    """
    window = _window(qtbot)

    segment = window.findChild(QWidget, "workspaceModeSegment")
    toolbar = window.findChild(QToolBar, "mainToolbar")
    assert segment is not None
    assert toolbar.isAncestorOf(segment)

    guided = window.findChild(QToolButton, "workspaceModeGuided")
    expert = window.findChild(QToolButton, "workspaceModeExpert")
    assert [button.text() for button in (guided, expert)] == ["引导", "专家"]
    assert all(button.isCheckable() for button in (guided, expert))
    # A new window opens on the full workspace, so that is the half already lit.
    assert (guided.isChecked(), expert.isChecked()) == (False, True)


def test_command_bar_segment_switches_the_visible_projection(qtbot) -> None:
    window = _window(qtbot)
    guided = window.findChild(QToolButton, "workspaceModeGuided")
    expert = window.findChild(QToolButton, "workspaceModeExpert")

    guided.click()
    assert window.guidance_is_visible() is True
    assert window.inspector_column.isVisibleTo(window) is False

    expert.click()
    assert window.guidance_is_visible() is False
    assert window.inspector_column.isVisibleTo(window) is True


def test_command_bar_segment_and_the_view_menu_cannot_disagree(qtbot) -> None:
    """One state, two controls: the menu entry drives which half is lit."""
    window = _window(qtbot)
    guided = window.findChild(QToolButton, "workspaceModeGuided")
    expert = window.findChild(QToolButton, "workspaceModeExpert")
    action = window.chrome_actions["guidanceModeAction"]

    action.setChecked(True)
    assert (guided.isChecked(), expert.isChecked()) == (True, False)

    action.setChecked(False)
    assert (guided.isChecked(), expert.isChecked()) == (False, True)


def test_command_bar_segment_leaves_the_persisted_expert_depth_alone(qtbot) -> None:
    """Two controls wear the word 专家; only the inspector's one is the document's.

    The segment picks which surface is on screen, which is view state. 专家模式 in
    the parameters panel unlocks the deeper columns and is persisted on the
    project, so driving it from a surface switch would dirty a project the user
    only looked at and put an undo entry behind a glance.
    """
    window = _window(qtbot)
    guided = window.findChild(QToolButton, "workspaceModeGuided")
    expert = window.findChild(QToolButton, "workspaceModeExpert")
    before = window.document.project.ui_state.expert_mode

    guided.click()
    expert.click()

    assert window.document.project.ui_state.expert_mode is before
    assert window.document.is_dirty is False


def test_the_command_bar_carries_only_the_three_project_commands_the_design_draws(qtbot) -> None:
    """六张设计稿的 cmdbar 左端一律只有 新建/打开/保存。

    另存为 / 重载源 / 重链接源 在任何一帧里都没出现过，连专家模式那几帧也没有。设计稿
    把「开工要按的」和「一天按不到一次的」分在了两个层级：命令栏是常用动作的货架，
    文件 菜单才是全部动作的目录。把六颗都摆到货架上，等于让每次开工都要从六个里挑三个。

    三条命令并没有消失：``saveAsProjectAction`` / ``reloadSourceAction`` /
    ``relinkSourceAction`` 各自带快捷键留在 文件 菜单，所以货架变短而程序没有变小。
    """
    window = _window(qtbot)

    on_the_bar = [
        button.objectName() for button in window.project_actions.findChildren(QPushButton) if button.objectName()
    ]

    assert on_the_bar == list(GUIDED_COMMAND_BAR)
    for object_name in ("saveAsProjectAction", "reloadSourceAction", "relinkSourceAction"):
        assert _menu_action(window, object_name).isEnabled() in (True, False), object_name


def test_guided_mode_reduces_the_command_bar_to_what_frame_two_draws(qtbot) -> None:
    """Frame ②'s cmdbar is 新建/打开/保存 + the 引导·专家 segment, and nothing else.

    专家模式那几帧在同样三颗之后还有 ⚡一键拟合 / 导出…；帧② 一个都没有。把它们留在
    引导模式里，等于在还没有结果可导的第二步就把 导出结果 摆到面前，和这一帧自己的
    引子 「隐藏停靠面板与高级批量选项」 相反。

    这里是藏而不是删：每条都有 文件 或 拟合 菜单里的孪生项照常可用，所以货架变短而
    程序没有变小。
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    group = window.findChild(QWidget, "expertCommandGroup")
    assert group is not None

    window.set_guidance_visible(True)

    assert window.findChild(QWidget, "workspaceModeSegment").isVisibleTo(window) is True
    # A dict rather than a loop of asserts: it names every button that disagreed,
    # not merely the first, and keeps this check inside the complexity budget.
    assert _button_visibility(window, GUIDED_COMMAND_BAR) == dict.fromkeys(GUIDED_COMMAND_BAR, True)
    assert _button_visibility(window, EXPERT_ONLY_COMMANDS) == dict.fromkeys(EXPERT_ONLY_COMMANDS, False)
    assert group.isVisibleTo(window) is False

    window.set_guidance_visible(False)

    everything = (*GUIDED_COMMAND_BAR, *EXPERT_ONLY_COMMANDS)
    assert _button_visibility(window, everything) == dict.fromkeys(everything, True)
    assert group.isVisibleTo(window) is True


def test_the_idle_expert_bar_is_one_primary_fit_pill_beside_an_export_button(qtbot) -> None:
    """帧①③⑤ 的右端只有两枚：``⚡ 一键拟合``（primary 实心）和 ``导出…``。

    取消拟合 不在其中。空闲时它没有可取消的东西，而真的跑起来之后设计稿换的是 ⏸ 暂停 /
    ⏹ 停止 那一组——所以命令栏从来不需要一枚常驻的 取消。它连着 Esc 留在 拟合 菜单里。

    一键拟合 是这一栏唯一的 ``.btn primary``：整个界面每次只推荐一个下一步动作，实心那枚
    就是它。导出 写成 ``导出…``，省略号是「按下会开个对话框问你导到哪」的既有约定，和
    命令栏上其他直接生效的按钮区分开。
    """
    from PySide6.QtWidgets import QAbstractButton

    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    group = window.findChild(QWidget, "expertCommandGroup")

    # 认按钮用 ``objectName``，不用文字：那颗 pill 的文字是从 QAction 抄来的，
    # ``operation_state`` 每改一次可用状态就会重抄一遍（见 ``chrome._install_expert_commands``），
    # 拿它当身份会把「这一栏摆了哪两枚」和「这一枚此刻写着什么」两件事绑在一起。
    on_the_bar = [child.objectName() for child in group.findChildren(QAbstractButton) if child.isVisibleTo(group)]

    assert on_the_bar == ["startFitToolButton", "exportResultsButton"]
    fit_button = group.findChild(QAbstractButton, "startFitToolButton")
    assert fit_button.property("primary") is True
    assert window.export_button.text() == "导出…"
    # 取消 仍然可达，只是不在货架上。
    assert _menu_action(window, "cancelFitAction").isVisible() is True


def test_the_reduced_command_bar_keeps_every_command_reachable_from_a_menu(qtbot) -> None:
    """Hiding a button must not hide the command: the menu twin stays enabled.

    This is what makes the reduction safe.  A guided user who does need 另存为 or
    导出结果 finds it in 文件 / 拟合 exactly as before, so the guided bar is a
    smaller offer rather than a smaller program.
    """
    window = _window(qtbot)

    window.set_guidance_visible(True)

    for object_name in (
        "saveAsProjectAction",
        "reloadSourceAction",
        "relinkSourceAction",
        "exportResultsAction",
        "startFitAction",
        "cancelFitAction",
    ):
        assert _menu_action(window, object_name).isVisible() is True, object_name


def test_the_edit_menu_still_undoes_and_redoes_a_project_change(qtbot) -> None:
    """重设计后的外壳仍然保住 撤销/重做 这条路（计划 §11 判据 4 的第二项）。

    ``ProjectDocument`` 的两个栈一直都在，但连着它们的只有 编辑 菜单里这两个 QAction：
    重设计把 ``chrome.py`` 整条菜单栏重装了一遍，只要 ``undo_state_changed`` 那一根
    连线没接回来，撤销 就会永远是灰的——工程照样能改，只是改完退不回去，而这种坏法
    在别处一个断言都碰不到（本文件之前，全仓没有任何测试提到 ``undoAction``）。

    所以这里量三件事，而不是只量「菜单项存在」：栈空时两项都是灰的、改一次工程之后
    撤销 亮而 重做 仍灰、触发之后工程真的回到上一版且 重做 亮起来。快捷键一并核对，
    因为 Ctrl+Z 是这条路在设计稿里唯一露出的入口。
    """
    import xrr_fitter.api as api

    window = _window(qtbot)
    undo = _menu_action(window, "undoAction")
    redo = _menu_action(window, "redoAction")

    assert (undo.shortcut().toString(), redo.shortcut().toString()) == ("Ctrl+Z", "Ctrl+Shift+Z")
    assert (undo.isEnabled(), redo.isEnabled()) == (False, False)

    before = window.document.project.ui_state.expert_mode
    window.document.replace_project(api.set_expert_mode(window.document.project, not before))

    assert window.document.project.ui_state.expert_mode is (not before)
    assert (undo.isEnabled(), redo.isEnabled()) == (True, False)

    undo.trigger()

    assert window.document.project.ui_state.expert_mode is before
    assert (undo.isEnabled(), redo.isEnabled()) == (False, True)

    redo.trigger()

    assert window.document.project.ui_state.expert_mode is (not before)
    assert (undo.isEnabled(), redo.isEnabled()) == (True, False)
