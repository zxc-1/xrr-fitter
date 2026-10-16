"""发货外观下，几处版面契约仍然成立；测试之间也不互相留下外观。

``create_application`` 总会装上 ``theme.apply_theme`` 的样式表，而多数 GUI 测试不走它，量到的是
一张没有样式表的离屏界面。命令栏明暗开关那条用例点两下之后，样式表就留在了整个会话唯一的
``QApplication`` 上：排在它后面的用例量到发货外观，单跑却量到裸界面——同一个用例两种结果。
这里显式装上同一份样式表再量，量完卸下；``tests/conftest.py`` 替忘了卸的用例兜底。
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QFont, QImage, QPalette
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QStyle, QStyleOptionButton
from tests.gui.paint_support import painted
from tests.gui.test_import_dialog import _write_curve


@pytest.fixture
def shipped_theme(qapp):
    from xrr_fitter.gui import theme

    previous = qapp.styleSheet()
    theme.apply_theme(qapp)
    yield theme
    qapp.setStyleSheet(previous)


def _assert_eyebrow(label: QLabel, theme) -> None:
    font = label.font()
    assert font.pixelSize() == theme.STEP_EYEBROW_FONT_PX, font
    assert font.weight() == QFont.Weight.Bold, font
    assert font.letterSpacing() == pytest.approx(theme.STEP_EYEBROW_TRACKING_PX, abs=0.05), font


def test_the_guided_eyebrow_keeps_its_font_under_the_shipped_theme(qtbot, shipped_theme) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.guidance.panel import GuidancePanel

    panel = GuidancePanel(ProjectDocument())
    qtbot.addWidget(panel)
    eyebrow = panel.findChild(QLabel, "structureStepEyebrow")

    _assert_eyebrow(eyebrow, shipped_theme)
    # 切明暗、改 stepState 都会重新 polish，而 polish 把字体退回它存下的那一份。
    shipped_theme.repolish(eyebrow)
    _assert_eyebrow(eyebrow, shipped_theme)


def test_the_import_dialog_fits_a_laptop_under_the_shipped_theme(qtbot, tmp_path, shipped_theme) -> None:
    """同一笔「笔记本放得下」的账，按发货外观再算一遍：样式表给输入框、按钮、分组框都加了内边距。"""
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    layout = dialog.layout()

    assert layout.totalMinimumSize().height() <= 640, layout.totalMinimumSize()
    assert layout.totalSizeHint().height() <= 800, layout.totalSizeHint()
    assert dialog.button_box.isVisibleTo(dialog)
    assert dialog.mapping_button.isVisibleTo(dialog)


def _over(colour: QColor, ground: QColor) -> QColor:
    """截图里透明的地方是窗口底色透上来的：把像素叠回窗口色上再比。"""
    alpha = colour.alphaF()
    return QColor.fromRgbF(
        *(
            alpha * mine + (1 - alpha) * under
            for mine, under in zip(
                (colour.redF(), colour.greenF(), colour.blueF()),
                (ground.redF(), ground.greenF(), ground.blueF()),
                strict=True,
            )
        )
    )


def _luminance(colour: QColor) -> float:
    linear = [part / 12.92 if part <= 0.03928 else ((part + 0.055) / 1.055) ** 2.4 for part in colour.getRgbF()[:3]]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: QColor, second: QColor) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _caret_rows(combo: QComboBox) -> tuple[list[int], float]:
    """下拉框右端那一段里每一行着墨的像素数，以及着墨处对框底的最大对比度（框是空的，没有文字）。"""
    # ``grab`` 交回预乘 alpha 的图，``pixel`` 原样给出预乘值；先换成直通 alpha，``_over`` 才叠得对。
    shot = painted(combo)
    shot = shot._replace(image=shot.image.convertToFormat(QImage.Format.Format_ARGB32))
    window = combo.palette().color(QPalette.ColorRole.Window)
    ground = _over(shot.at(4, combo.height() // 2), window)
    rows, peak = [], 1.0
    for y in range(3, combo.height() - 3):
        ratios = [
            _contrast(_over(shot.at(x, y), window), ground) for x in range(combo.width() * 3 // 4, combo.width() - 3)
        ]
        inked = [ratio for ratio in ratios if ratio >= 1.5]
        if inked:
            rows.append(len(inked))
            peak = max(peak, *inked)
    return rows, peak


@pytest.mark.parametrize("palette_name", ("light_palette", "dark_palette"))
def test_every_combo_box_draws_a_down_pointing_caret(qtbot, palette_name) -> None:
    """F46：设计稿 ``.inp.sel::after{content:"▾"}``——下拉框靠右有一枚 ▾，读者靠它认出这是选单。

    样式表给 ``::drop-down`` 定了样式，Qt 就不再画平台自带的箭头；不再补一条 ``::down-arrow``，
    那一格就空着，下拉框看上去和一只文本框没有分别。补上的那一枚得真是朝下的尖（一条横杠读起来
    是减号），而且它是认出控件的唯一记号，对框底至少要有 3:1。
    """
    from xrr_fitter.gui import theme

    palette = getattr(theme, palette_name)()
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.setPalette(palette)
    combo.setStyleSheet(theme.build_stylesheet(palette))
    combo.setFixedWidth(160)
    combo.show()
    qtbot.waitExposed(combo)

    rows, peak = _caret_rows(combo)
    assert len(rows) >= 3, rows
    assert rows == sorted(rows, reverse=True) and rows[0] > rows[-1], f"不是朝下收尖的三角：{rows}"
    assert peak >= 3.0, peak


DISABLED_TEXT_ROLES = (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText)


@pytest.mark.parametrize(
    ("palette_name", "tokens_name"), (("light_palette", "LIGHT_TOKENS"), ("dark_palette", "DARK_TOKENS"))
)
def test_disabled_text_reads_as_muted_in_both_appearances(palette_name, tokens_name) -> None:
    """禁用的字和停用的按钮用同一个词元：``QPushButton:disabled`` 写的是 muted，调色板的禁用组也得是。

    复选框、单选框、标签、菜单项、输入框都没有 ``:disabled`` 规则，它们的禁用字直接取调色板的
    Disabled 组。``setColor(role, colour)`` 一次写满三组，禁用组于是和正常的一模一样——深色下
    导出对话框里那几行锁定的勾选框看上去还能点。
    """
    from xrr_fitter.gui import theme

    palette = getattr(theme, palette_name)()
    muted = theme.token_colour(getattr(theme, tokens_name).muted_text)
    disabled = {
        role.name: palette.color(QPalette.ColorGroup.Disabled, role).name(QColor.NameFormat.HexArgb)
        for role in DISABLED_TEXT_ROLES
    }

    assert disabled == dict.fromkeys(disabled, muted.name(QColor.NameFormat.HexArgb))


def _label_contrast(box: QCheckBox, window: str) -> float:
    """勾选框文字那一段里，着墨处对窗口底色的最大对比度（方框本身不量：它两种状态画得一样）。"""
    from xrr_fitter.gui import theme

    option = QStyleOptionButton()
    option.initFrom(box)
    option.text = box.text()
    area = box.style().subElementRect(QStyle.SubElement.SE_CheckBoxContents, option, box)
    image = box.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    scale = image.width() / max(box.width(), 1)
    ground = theme.token_colour(window)
    return max(
        _contrast(_over(QColor.fromRgba(image.pixel(round(x * scale), round(y * scale))), ground), ground)
        for x in range(area.left(), area.right())
        for y in range(area.top(), area.bottom())
    )


def test_a_locked_check_box_looks_locked_in_the_dark_appearance(qtbot, qapp) -> None:
    """深色外观下一枚锁住的勾选框，字要比能点的那枚暗一档——否则读者看不出它点不动。"""
    from xrr_fitter.gui import theme

    theme.set_dark_appearance(qapp, True)
    boxes = [QCheckBox("锁定的一行"), QCheckBox("锁定的一行")]
    boxes[1].setEnabled(False)
    for box in boxes:
        qtbot.addWidget(box)
        box.setFixedWidth(160)
        box.show()
        qtbot.waitExposed(box)

    enabled, locked = (_label_contrast(box, theme.DARK_WINDOW) for box in boxes)
    assert locked < enabled, (locked, enabled)


def test_a_test_that_installs_the_theme_and_forgets_it(qapp) -> None:
    from xrr_fitter.gui import theme

    # 故意不卸：下一条用例只有在 conftest 替它收拾之后才能通过。
    theme.apply_theme(qapp)
    assert qapp.styleSheet()


def test_the_next_test_starts_without_the_shipped_stylesheet(qapp) -> None:
    assert qapp.styleSheet() == ""
