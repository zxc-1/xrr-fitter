"""在层堆叠里点一层，参数表停到那一层的参数上。

审计实测：选中 a-Si 之后要调它的厚度，得回到参数表的十几行里自己找。这里钉住：读者在层堆叠
（示意图或树）里点了哪一层，参数表就滚到那一组；而编辑引起的重建（同一份结构刷新、选中被恢复）
不会把读者正在改的那一格拽走。
"""

from __future__ import annotations

from tests.gui.control_operation_driver import setup_project

import xrr_fitter.api as api


def _window(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.set_expert_mode(setup_project(tmp_path, layers=2), True)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def _tree_item(editor, index):
    tree = editor.tree
    return next(
        tree.topLevelItem(row)
        for row in range(tree.topLevelItemCount())
        if tree.topLevelItem(row).data(0, 256) == index
    )


def test_clicking_a_layer_row_focuses_its_parameters(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path)
    editor = window.structure_panel.editor
    table = window.parameters_panel.parameter_table

    item = _tree_item(editor, 1)
    editor.tree.setCurrentItem(item)
    editor.tree.itemClicked.emit(item, 0)

    assert table.currentRow() in table.group_rows("component.1")


def test_clicking_a_band_in_the_diagram_focuses_its_parameters(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path)
    table = window.parameters_panel.parameter_table

    window.structure_panel.editor.stack.component_selected.emit(0)

    assert table.currentRow() in table.group_rows("component.0")


def test_a_rebuild_does_not_move_the_cell_being_edited(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path)
    table = window.parameters_panel.parameter_table
    window.structure_panel.editor.stack.component_selected.emit(0)
    elsewhere = table.group_rows("component.1")[-1]
    table.setCurrentCell(elsewhere, 0)

    dataset = window.document.project.datasets[0]
    window.structure_panel.set_structure(dataset.structure)

    assert table.currentRow() == elsewhere
