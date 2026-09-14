"""设计稿的 ``.badge``：一句只读的状态，画成带边框的胶囊。

配色规则本身在 :mod:`xrr_fitter.gui.theme` 里（``QLabel[badge="true"]`` 那几条）。这里
收的是每处都要重抄一遍的三件事：加上那个选择器属性、换类别时重刷样式、没话说时整枚
让开位置。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from xrr_fitter.gui import theme

#: 设计稿 ``.badge`` 的行高：2px 内边距 + 1px 边框，上下各一份，中间留给 11px 的字。
#: 定死高度是胶囊成形的条件——Qt 没有 ``999px`` 的等价写法，半径大到超过半高之后，
#: 两端是不是半圆就只由高度说了算。
HEIGHT_PX = 22

#: 设计稿给中性徽章起的名字。主题里它落在不带 ``statusKind`` 的那条基础规则上，所以
#: 对 Qt 而言就是「没有状态色」。
NEUTRAL_KIND = "mut"

KINDS = (NEUTRAL_KIND, "ok", "info", "warn", "error")


class Badge(QLabel):
    """一枚只读的状态胶囊。"""

    HEIGHT_PX = HEIGHT_PX

    def __init__(self, object_name: str, kind: str = NEUTRAL_KIND, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setProperty("badge", True)
        self.setFixedHeight(HEIGHT_PX)
        # 横向只占自己那句话的宽度：徽章是并排铺开的，被拉伸的那一枚会把同一行里
        # 其余几枚挤到看不出是同一组。
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._kind = ""
        self.set_kind(kind)
        # 没有初始文字，所以先让开位置；``setText`` 会在有话说时把它放回来。
        self.hide()

    def kind(self) -> str:
        """当前类别，用设计稿的叫法（中性是 ``mut``，不是空串）。"""
        return self._kind or NEUTRAL_KIND

    def set_kind(self, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"unsupported badge kind: {kind}")
        self._kind = "" if kind == NEUTRAL_KIND else kind
        theme.set_status_kind(self, self._kind)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt 的命名
        """写入这句话；写空等于「此刻没有可报的状态」，整枚让开位置。

        留一个空胶囊会被读成「有个状态，只是没加载出来」，而那正是它此刻不能表示的
        意思。
        """
        super().setText(text)
        self.setVisible(bool(text))
