"""Exercise the owned drag lifecycle while replacing only the OS drag loop."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QDrag, QDropEvent
from tests.gui.test_structure_reorder_selection import _assert_selection, _layer, _mouse_select, _panel


def _start_drag(tree):
    # Bound the legacy C++ drag loop on a headless platform. The application-
    # owned QDrag calls the replacement below and returns synchronously.
    cancel = QTimer(tree)
    cancel.setSingleShot(True)
    cancel.timeout.connect(QDrag.cancel)
    cancel.start(100)
    try:
        tree.startDrag(Qt.DropAction.MoveAction)
    finally:
        cancel.stop()


@pytest.mark.parametrize("source,destination", ((1, 0), (0, 1), (1, 1)))
def test_completed_drag_preserves_all_rows_and_component_selection(qtbot, tmp_path, monkeypatch, source, destination):
    components = (_layer("film0"), _layer("film1"))
    panel = _panel(qtbot, tmp_path, components)
    tree = _mouse_select(qtbot, panel, source)
    executions = []
    moves = []
    tree.component_moved.connect(lambda *args: moves.append(args))

    def execute(drag, actions, default_action=Qt.DropAction.IgnoreAction):
        executions.append(drag)
        assert drag.source() is tree
        assert actions == Qt.DropAction.MoveAction
        assert default_action == Qt.DropAction.MoveAction
        assert drag.mimeData().hasFormat("application/x-qabstractitemmodeldatalist")
        target = tree.visualItemRect(tree.topLevelItem(destination + 1))
        event = QDropEvent(
            QPointF(target.center()),
            actions,
            drag.mimeData(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        tree.dropEvent(event)
        assert event.isAccepted()
        return event.dropAction()

    monkeypatch.setattr(QDrag, "exec", execute)
    _start_drag(tree)
    qtbot.wait(1)

    expected = list(components)
    expected.insert(destination, expected.pop(source))
    assert len(executions) == 1
    assert panel.structure.components == tuple(expected)
    assert tree.topLevelItemCount() == len(components) + 2
    assert all(tree.itemWidget(tree.topLevelItem(index), 0) is not None for index in range(tree.topLevelItemCount()))
    assert moves == ([] if source == destination else [(source, destination)])
    _assert_selection(panel, destination, components[source])


def test_cancelled_drag_preserves_model_rows_and_selection(qtbot, tmp_path, monkeypatch):
    components = (_layer("film0"), _layer("film1"))
    panel = _panel(qtbot, tmp_path, components)
    tree = _mouse_select(qtbot, panel, 1)
    before = panel.document.project
    executions = []

    def cancel(drag, *_args):
        executions.append(drag)
        return Qt.DropAction.IgnoreAction

    monkeypatch.setattr(QDrag, "exec", cancel)
    _start_drag(tree)

    assert len(executions) == 1
    assert panel.document.project is before
    assert tree.topLevelItemCount() == 4
    _assert_selection(panel, 1, components[1])
