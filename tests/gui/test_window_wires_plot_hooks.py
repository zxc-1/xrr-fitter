"""窗口把读者的操作接到画布上：选中的层在 SLD 剖面上高亮，空状态提示念出导入的平台按键。

画布自己提供了这两样（``highlight_component``、``set_import_shortcut_text``），单测都过；但没有窗口
去调用它们，读者在堆叠里点了层，剖面照旧不认，空状态提示也照旧不提按键。
"""

from __future__ import annotations

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QLabel
from tests.gui.test_layer_pick_reaches_the_parameter_table import _tree_item, _window
from tests.gui.test_sld_component_highlight import _highlight_artists, _highlight_line

from xrr_fitter.gui.plots.sld import component_span_nm


def _highlighted_span(window) -> tuple[float, float]:
    xdata = _highlight_line(window.plot_panel).get_xdata()
    return float(min(xdata)), float(max(xdata))


def _expected_span(window, index: int) -> tuple[float, float]:
    span = component_span_nm(window.document.project.datasets[0].structure, index)
    assert span is not None
    return span


def _inside(span: tuple[float, float], expected: tuple[float, float]) -> bool:
    return expected[0] - 1e-9 <= span[0] < span[1] <= expected[1] + 1e-9


def test_clicking_a_layer_row_highlights_its_segment_of_the_profile(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path)
    editor = window.structure_panel.editor
    assert _highlight_artists(window.plot_panel) == []

    item = _tree_item(editor, 1)
    editor.tree.setCurrentItem(item)
    editor.tree.itemClicked.emit(item, 0)

    assert _inside(_highlighted_span(window), _expected_span(window, 1))


def test_clicking_a_band_in_the_diagram_highlights_its_segment_of_the_profile(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path)

    window.structure_panel.editor.stack.component_selected.emit(0)

    assert _inside(_highlighted_span(window), _expected_span(window, 0))


def test_the_empty_canvas_names_the_import_key_the_menu_shows(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    shortcut = window.data_panel.import_files_action.shortcut()
    native = shortcut.toString(QKeySequence.SequenceFormat.NativeText)
    assert native

    hint = window.plot_panel.findChild(QLabel, "emptyStateHint").text()

    assert hint.endswith(f"或快捷键 {native}")
