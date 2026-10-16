"""画布空状态那句提示不写死按键：按键由窗口按平台给出（macOS 上是 ⌘I，不是 Ctrl+I）。

审计实测：提示写死「快捷键 Ctrl+I」，macOS 上照着按什么也不会发生——菜单里那条命令显示的是 ⌘I。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel


def _hint(panel) -> str:
    label = panel.findChild(QLabel, "emptyStateHint")
    assert label is not None, "the empty state has no named hint"
    return label.text()


def test_the_empty_state_hint_names_no_key_until_the_window_supplies_one(qtbot) -> None:
    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    qtbot.addWidget(panel)
    assert "Ctrl" not in _hint(panel)
    assert "快捷键" not in _hint(panel)

    panel.set_import_shortcut_text("⌘I")
    assert _hint(panel).endswith("或快捷键 ⌘I")

    panel.set_import_shortcut_text("")
    assert "快捷键" not in _hint(panel)
