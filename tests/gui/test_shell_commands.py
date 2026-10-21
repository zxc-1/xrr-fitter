"""Command icons and menu integration: consistency across surfaces."""

from __future__ import annotations

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QPushButton, QToolButton

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
    """The rendered glyph, so two icons compare by what the user sees.

    The image is held in a local: ``constBits()`` is a view into the image's
    buffer, and a temporary image is freed before ``bytes`` copies from it.  That
    only ever worked while the icon cached its pixmap; glyphs repainted on demand
    (``GlyphIconEngine``) hand back a fresh image each time.
    """
    image = icon.pixmap(16, 16).toImage()
    return bytes(image.constBits())


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


def test_the_rail_folder_shortcut_wears_the_folder_import_glyph(qtbot) -> None:
    """F45：机架抬头那枚文件夹键与 文件 ▸ 导入数据文件夹… 是同一条命令，就该是同一枚字形。

    它原先借的是平台的「打开文件夹」图标：macOS 上一只蓝文件夹、Windows 上一只黄的、离屏是
    一圈线框——三处长得都不像别处那枚「文件夹 + 向下箭头」，也不跟着深色外观换墨色，读者在
    菜单里学会的那枚到这里认不出来。
    """
    window = _window(qtbot)
    quick = window.findChild(QToolButton, "importFolderQuickButton")
    menu = _menu_action(window, "importFolderMenuAction")
    assert quick is not None and not menu.icon().isNull()

    assert _icon_bytes(quick.icon()) == _icon_bytes(menu.icon())


def _ink_box(host, widget) -> tuple[int, int]:
    """``widget`` 在 ``host`` 截图里落墨的宽与高（逻辑像素）；底色取控件左上角那一点。"""
    image = host.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    scale = image.width() / max(host.width(), 1)
    origin = widget.mapTo(host, QPoint(0, 0))

    def at(x: int, y: int) -> QColor:
        return QColor.fromRgba(image.pixel(round((origin.x() + x) * scale), round((origin.y() + y) * scale)))

    ground = at(0, 0)
    inked = [
        (x, y)
        for x in range(widget.width())
        for y in range(widget.height())
        if sum(abs(a - b) for a, b in zip(at(x, y).getRgb()[:3], ground.getRgb()[:3], strict=True)) > 60
    ]
    assert inked, "按钮上一个像素的墨都没有"
    xs, ys = zip(*inked, strict=True)
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def test_the_rail_folder_shortcut_draws_its_glyph_at_full_size(qtbot) -> None:
    """F45：通用的 ``QToolButton`` 规则给每枚按钮 4×8px 内边距、30px 最小高；这枚键钉死 24px 宽，
    16px 的字形于是只剩 6px 可画，换哪枚图标都是一团认不出的墨点。旁边那枚 ＋ 早就按设计稿去掉了
    这两样，文件夹键也得去掉。"""
    from xrr_fitter.gui.theme import build_stylesheet, light_palette

    window = _window(qtbot)
    palette = light_palette()
    window.setPalette(palette)
    window.setStyleSheet(build_stylesheet(palette))
    window.set_guidance_visible(False)
    window.resize(1280, 760)
    window.show()
    qtbot.waitExposed(window)
    quick = window.findChild(QToolButton, "importFolderQuickButton")

    width, height = _ink_box(window, quick)
    assert width >= 10 and height >= 10, (width, height, quick.size())
    # 键本身仍是 24px 见方：指针与触控都靠这一格（WCAG 2.5.8 的最小目标）。
    assert quick.width() >= 24 and quick.height() >= 24, quick.size()
