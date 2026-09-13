"""按逻辑坐标读一个控件实际画出来的颜色。

``QWidget.grab().toImage()`` 交回来的是**设备像素**图：Retina 上它的边长是控件逻辑宽度的
两倍。而 ``geometry()``、``mapTo()``、``visualItemRect()``、``childAt()`` 说的全是逻辑坐标。
拿逻辑坐标直接去 ``image.pixel()`` 取样，在 2× 屏上会落到半程之外——量的是隔壁那块的底色，
而这类断言恰好都在「机架色 / 画布色 / 缝线色」之间分胜负，取错位置读到的正是相邻那一种，
于是测试既不报错也不说真话。

所以取样一律先按 ``图宽 / 控件逻辑宽`` 折回去。控件必须已经 ``show()`` 并 ``waitExposed``，
否则 ``grab()`` 用的是还没落定的尺寸。
"""

from __future__ import annotations

from typing import NamedTuple

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QWidget


class Painted(NamedTuple):
    """一张截图，加上把逻辑坐标折成设备像素要乘的那个比例。"""

    image: QImage
    scale: float

    def at(self, x: int, y: int) -> QColor:
        """控件逻辑坐标 (x, y) 上实际画出来的颜色。"""
        return QColor.fromRgba(self.image.pixel(round(x * self.scale), round(y * self.scale)))

    def at_point(self, point: QPoint) -> QColor:
        return self.at(point.x(), point.y())


def painted(host: QWidget) -> Painted:
    """截一次 ``host``，之后可以按逻辑坐标反复取样。

    一次截图多点取样，而不是每点截一次：同一帧里比较两块颜色时，重截会把中间发生的
    重绘（悬停、焦点框）混进来。
    """
    image = host.grab().toImage()
    return Painted(image, image.width() / max(host.width(), 1))


def painted_at(host: QWidget, point: QPoint) -> QColor:
    """只取一点时的便捷写法。"""
    return painted(host).at_point(point)
