"""拟合卡上写出「本次」走哪条路线。

命令栏的一键拟合会按项目换路线（自动 / 手动 / 联合），提示只在悬停时出现；卡片是读者按下之前
正看着的那一处，所以那里常驻一行「本次：<路线> — <做法>」。路线由主窗口算（``fit_route``），
卡片只负责把它说出来。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel
from tests.gui.test_fit_progress import _panel


def test_the_card_names_the_route_it_will_take(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    line = panel.findChild(QLabel, "fitRouteLine")
    assert line is not None
    assert line.isHidden(), "no route has been named yet"

    panel.set_route("manual")

    assert line.text() == "本次：手动拟合 — 按参数表里的固定、边界与共享设置逐个数据集拟合"
    assert not line.isHidden()
