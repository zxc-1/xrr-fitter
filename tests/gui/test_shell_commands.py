"""Command icons and menu integration: consistency across surfaces."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

COMMAND_SURFACES = (
    ("新建项目", "newProjectButton", "newProjectAction"),
    ("打开项目", "openProjectButton", "openProjectAction"),
    ("保存项目", "saveProjectButton", "saveProjectAction"),
    ("导出结果", "exportResultsButton", "exportResultsAction"),
)


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


def _icon_bytes(icon) -> bytes:
    """The rendered glyph, so two icons compare by what the user sees."""
    return bytes(icon.pixmap(16, 16).toImage().constBits())


def test_each_command_wears_one_icon_in_the_toolbar_and_the_menu(qtbot) -> None:
    """字形属于命令，命令栏则按设计稿只排字。

    从前是反过来的：命令栏画带图标的按钮，菜单里是光秃秃的文字，同一条命令要学两遍。
    修法不是给菜单也补一枚图标，而是把字形挂到 ``QAction`` 上——菜单从命令本身取字形，
    命令栏那几枚 ``.btn sm`` 保持设计稿的纯文字，两边说的仍是同一件事。
    """
    window = _window(qtbot)

    faults: list[str] = []
    for label, button_name, action_name in COMMAND_SURFACES:
        button = window.findChild(QPushButton, button_name)
        assert button is not None, f"{label}: 缺少工具栏按钮 {button_name}"
        action = _menu_action(window, action_name)
        if action.icon().isNull():
            faults.append(f"{label}: 菜单无图标")
        if not button.icon().isNull():
            faults.append(f"{label}: 命令栏按钮多了一枚图标")
        if not button.text():
            faults.append(f"{label}: 命令栏按钮没有文字")

    assert faults == []


def test_fit_commands_carry_their_icon_into_the_menu(qtbot) -> None:
    """The fit commands are one QAction per command; keep it that way."""
    window = _window(qtbot)

    for object_name in ("startFitAction", "cancelFitAction"):
        assert not _menu_action(window, object_name).icon().isNull(), object_name


def test_the_result_operations_the_inspector_dropped_live_in_the_fit_menu(qtbot) -> None:
    """设计稿帧① 右栏只有判定 / 结果值 / 候选解三段，清除与不确定度都不在栏里。

    把控件从检视器里撤掉不等于把命令撤掉。这两条是对「已经拿到的这份结果」做的操作，所以
    落在拟合菜单里，跟开始 / 取消 / 强制停止排在一起——检视器变窄是少给一个入口，不是
    少一个功能。
    """
    from PySide6.QtWidgets import QMenu

    window = _window(qtbot)

    menu = window.findChild(QMenu, "fitMenu")
    assert menu is not None
    texts = [action.text() for action in menu.actions()]
    assert "清除结果" in texts
    assert "不确定度分析…" in texts


def test_the_uncertainty_menu_entry_stays_expert_only(qtbot) -> None:
    """按钮此前只在专家模式露面；换成菜单项后这道门不能悄悄消失。"""
    import xrr_fitter.api as api

    window = _window(qtbot)
    action = _menu_action(window, "openUncertaintyAction")

    assert action.isVisible() is False

    window.document.replace_project(api.set_expert_mode(window.document.project, True))
    assert action.isVisible() is True


def test_no_two_commands_wear_the_same_icon(qtbot) -> None:
    """一个字形一条命令：看见圆圈就知道是「运行」，不该有第二条别的也画圆圈。

    相同字形让读者以为是同一条命令的两个入口，点了却是别的；只有字形各不同，才能凭形
    状记住这一条是什么。菜单藏在幕后，所以命令栏上任意两枚按钮不得撞字形。
    """
    window = _window(qtbot)

    seen: dict[bytes, str] = {}
    for label, _button_name, action_name in (
        *COMMAND_SURFACES,
        ("一键拟合", "", "startFitAction"),
        ("取消拟合", "", "cancelFitAction"),
    ):
        icon = _menu_action(window, action_name).icon()
        if icon.isNull():
            continue
        glyph = _icon_bytes(icon)
        assert glyph not in seen, f"{label} 与 {seen[glyph]} 用了同一个图标"
        seen[glyph] = label


def test_every_registered_command_wears_a_distinct_icon(qtbot) -> None:
    """QAction 是命令，菜单是这些命令的一张索引——每一行都必须有不同的字形。

    菜单逐行列命令，相同字形让读者误以为排了两次；只有字形各不同，才能凭形状记住是哪
    一条。测试范围只涵盖已经登记的 action（设了 ``objectName``），因为只有它们才会被
    明面上的入口取用。
    """
    from xrr_fitter.gui.command_icons import COMMAND_PIXMAPS, command_icon

    _window(qtbot)  # a styled QApplication, as every other icon test relies on
    seen: dict[bytes, str] = {}
    for command in COMMAND_PIXMAPS:
        icon = command_icon(command)
        assert not icon.isNull(), f"{command}: 命令无图标"
        glyph = _icon_bytes(icon)
        assert glyph not in seen, f"{command} 与 {seen[glyph]} 用了同一个图标"
        seen[glyph] = command
