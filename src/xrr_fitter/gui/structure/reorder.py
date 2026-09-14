"""拖动排序：把树里的一次拖放翻译成一次层序变更。

设计稿的层堆叠副标题写着「拖动排序」，帧③ 的说明也写「可增删、拖动排序」，每行左端
都画着 ``⋮⋮`` 拖柄。上移/下移两个按钮做的是同一件事，但一次只挪一格，把第 5 层搬到
最上面要点四下。

这里不让 Qt 自己搬 item。结构是唯一的事实来源，树是照结构重绘出来的：让 Qt 先移动一
行、提交之后再重绘一次，同一次拖放会被算作两次移动。所以 :meth:`dropEvent` 只负责算
出「哪一个组件、落到哪个位置」，然后把这两个数发出去，一行都不动。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QTreeWidgetItem, QWidget

from xrr_fitter.gui.sizing import ContentSizedTree


class ReorderableTree(ContentSizedTree):
    """A stack tree whose rows can be dragged, reporting moves by component index."""

    # 载荷是组件索引，不是行号：树的首末两行是空气与基底两个半无限介质。
    component_moved = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        # 拖的是整行，不是某一格；只允许拖顶层行，周期块里的子层没有独立的层序。
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

    def _component_index(self, item: QTreeWidgetItem | None) -> int | None:
        if item is None or item.parent() is not None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return value if isinstance(value, int) else None

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Report where the dragged component landed, without moving any row.

        不调用基类：基类会就地搬 item，而树随后会照提交后的结构整体重绘，同一次拖放
        就成了两次移动。落点算不出组件（拖到空气或基底那两行外侧）时忽略这次拖放，层
        排不到半无限介质外面去。
        """
        source = self._component_index(self.currentItem())
        target = self.itemAt(event.position().toPoint())
        destination = self._component_index(target)
        if source is None or destination is None:
            event.ignore()
            return
        event.setDropAction(Qt.DropAction.MoveAction)
        event.accept()
        if destination != source:
            self.component_moved.emit(source, destination)

    def startDrag(self, supported_actions) -> None:  # noqa: N802 - Qt override
        """Only offer a drag for rows that are components.

        空气和基底没有层序可言，能把它们拖起来只会给出一个必然被拒的落点。
        """
        if self._component_index(self.currentItem()) is None:
            return
        super().startDrag(supported_actions)
