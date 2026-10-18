"""Selection follows a component through real structure commands and reloads."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.structure.panel import StructurePanel

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _layer(name: str, thickness: float = 40.0) -> api.LayerSpec:
    return api.LayerSpec(name, SIO2, thickness, roughness_a=3.0)


def _panel(qtbot, tmp_path, components):
    curve = tmp_path / "selection.xy"
    curve.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    project = api.add_dataset(api.new_project(), curve, api.InstrumentSpec())
    structure = api.StructureSpec(AIR, tuple(components), SI)
    panel = StructurePanel(ProjectDocument(api.set_structure(project, "selection", structure)))
    qtbot.addWidget(panel)
    panel.resize(680, 600)
    panel.show()
    panel.activateWindow()
    qtbot.waitExposed(panel)
    qtbot.waitUntil(panel.isActiveWindow)
    return panel


def _mouse_select(qtbot, panel, index):
    tree = panel.editor.tree
    item = tree.topLevelItem(index + 1)
    tree.scrollToItem(item)
    qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(item).center())
    assert tree.currentItem() is item
    assert tree.hasFocus()
    return tree


def _assert_selection(panel, index, component):
    editor = panel.editor
    item = editor.tree.currentItem()
    assert item is not None
    assert item.data(0, Qt.ItemDataRole.UserRole) == index
    assert panel.structure.components[index] is component
    assert editor.stack.selected_index() == index
    assert editor.remove_action.isEnabled()
    assert editor.up_action.isEnabled() == (index > 0)
    assert editor.down_action.isEnabled() == (index + 1 < len(panel.structure.components))


def _assert_no_selection(editor):
    assert editor.tree.currentItem() is None
    assert editor.stack.selected_index() is None
    assert not editor.remove_action.isEnabled()
    assert not editor.up_action.isEnabled()
    assert not editor.down_action.isEnabled()


@pytest.mark.parametrize(
    "selected",
    (
        _layer("film1"),
        api.PeriodicBlock("periodic", (_layer("cell"),), repeats=2),
        _layer("film0", thickness=60.0),
    ),
    ids=("layer", "periodic-block", "duplicate-name"),
)
def test_shortcuts_keep_the_moved_component_selected(qtbot, tmp_path, selected):
    first = _layer("film0")
    panel = _panel(qtbot, tmp_path, (first, selected))
    tree = _mouse_select(qtbot, panel, 1)
    events = []
    panel.editor.component_selected.connect(lambda index, component: events.append((index, component)))

    QTest.keySequence(tree, QKeySequence("Ctrl+Up"))

    assert panel.structure.components == (selected, first)
    _assert_selection(panel, 0, selected)
    assert events[-1] == (0, selected)

    # No second click: the next shortcut must still act on the component just moved.
    QTest.keySequence(tree, QKeySequence("Ctrl+Down"))

    assert panel.structure.components == (first, selected)
    _assert_selection(panel, 1, selected)
    assert events[-1] == (1, selected)


def _click_menu_action(qtbot, editor, action):
    menu = editor.row_menu
    point = editor.tree.visualItemRect(editor.tree.currentItem()).bottomLeft()
    menu.popup(editor.tree.viewport().mapToGlobal(point))
    qtbot.waitUntil(menu.isVisible)
    assert action.isEnabled()
    qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
    qtbot.waitUntil(lambda: not menu.isVisible())


def test_menu_moves_keep_the_same_component_selected(qtbot, tmp_path):
    first, selected = _layer("film0"), _layer("film1")
    panel = _panel(qtbot, tmp_path, (first, selected))
    _mouse_select(qtbot, panel, 1)

    _click_menu_action(qtbot, panel.editor, panel.editor.up_action)

    assert panel.structure.components == (selected, first)
    _assert_selection(panel, 0, selected)

    _click_menu_action(qtbot, panel.editor, panel.editor.down_action)

    assert panel.structure.components == (first, selected)
    _assert_selection(panel, 1, selected)


def test_drop_commit_keeps_selection_for_the_next_keyboard_move(qtbot, tmp_path):
    first, selected = _layer("film0"), _layer("film1")
    panel = _panel(qtbot, tmp_path, (first, selected))
    tree = _mouse_select(qtbot, panel, 1)
    target = tree.visualItemRect(tree.topLevelItem(1))
    position = QPointF(target.center().x(), target.top() + 2)
    mime = QMimeData()
    event = QDropEvent(
        position, Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )

    # Exercise Qt's drop handler and its real project commit, not a native OS drag loop.
    tree.dropEvent(event)
    QApplication.processEvents()

    assert event.isAccepted()
    assert panel.structure.components == (selected, first)
    _assert_selection(panel, 0, selected)

    QTest.keySequence(tree, QKeySequence("Ctrl+Down"))

    assert panel.structure.components == (first, selected)
    _assert_selection(panel, 1, selected)


@pytest.mark.parametrize(
    ("count", "removed", "next_index"),
    ((3, 1, 1), (2, 1, 0), (1, 0, None)),
    ids=("next-neighbour", "previous-neighbour", "empty-stack"),
)
def test_delete_selects_the_nearest_surviving_component(qtbot, tmp_path, count, removed, next_index):
    components = tuple(_layer(f"film{index}") for index in range(count))
    panel = _panel(qtbot, tmp_path, components)
    tree = _mouse_select(qtbot, panel, removed)
    remaining = components[:removed] + components[removed + 1 :]

    QTest.keySequence(tree, panel.editor.remove_action.shortcut())

    assert panel.structure.components == remaining
    if next_index is None:
        _assert_no_selection(panel.editor)
    else:
        _assert_selection(panel, next_index, remaining[next_index])


def test_deleting_an_earlier_component_tracks_identity_not_equal_values(qtbot, tmp_path):
    equal_layer, selected = _layer("duplicate"), _layer("duplicate")
    panel = _panel(qtbot, tmp_path, (_layer("first"), equal_layer, selected))
    _mouse_select(qtbot, panel, 2)
    assert equal_layer == selected
    assert equal_layer is not selected

    panel.remove_component(0)

    assert panel.structure.components == (equal_layer, selected)
    _assert_selection(panel, 1, selected)


def test_noop_move_does_not_reset_selection(qtbot, tmp_path):
    selected = _layer("film1")
    panel = _panel(qtbot, tmp_path, (_layer("film0"), selected))
    _mouse_select(qtbot, panel, 1)
    before = panel.document.project

    changed = panel.move_component(1, 1)

    assert changed is False
    assert panel.document.project is before
    _assert_selection(panel, 1, selected)
