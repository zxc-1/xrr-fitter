"""M7 import/export dialog structural contracts."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QTableWidget,
)


def _write_curve(path: Path, count: int = 64) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + i * 0.02:.6f} {1000.0 / (i + 1):.12g}" for i in range(count)) + "\n",
        encoding="utf-8",
    )
    return path


def _window(qtbot, project=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    document = ProjectDocument() if project is None else ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(1)
    return window


def _import_dialog(qtbot, tmp_path):
    """An import dialog over one readable curve and one file with no data columns.

    Mockup frame ⑥ previews the whole batch at once, so a test about that table
    needs both outcomes present: a file the reader can import, and a file that
    has to stay on screen carrying its own failure.
    """
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    bad = tmp_path / "blank_run.dat"
    bad.write_text("# 无数值列\n", encoding="utf-8")
    paths = (_write_curve(tmp_path / "aSi_ML_25C.xy"), bad)
    dialog = ImportDialog(paths)
    qtbot.addWidget(dialog)
    return dialog


def _preview_table(qtbot, tmp_path, count: int):
    """An import dialog over ``count`` readable curves, and its preview table.

    The dialog is returned alongside the table because it owns it: drop the
    reference and Qt takes the C++ table with it mid-assertion.
    """
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    paths = tuple(_write_curve(tmp_path / f"run_{index:02d}.xy") for index in range(count))
    dialog = ImportDialog(paths)
    qtbot.addWidget(dialog)
    table = dialog.findChild(QTableWidget, "importPreviewTable")
    assert table is not None, "no preview table to measure"
    return dialog, table


def _table_content_height(table) -> int:
    """The height the rows and header actually occupy."""
    rows = sum(table.rowHeight(row) for row in range(table.rowCount()))
    return rows + table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth()


def test_import_dialog_previews_every_file_with_its_row_count(qtbot, tmp_path) -> None:
    """Mockup frame ⑥ heads a 文件/角度/列映射/行数/状态 grid over the whole selected batch.

    The dialog shipped a newline-joined path label plus a plot of ``paths[0]``,
    so a ten-file import showed one curve and nine unexamined names: nothing on
    screen said how many rows each file carried before it was imported.
    """
    from xrr_fitter.gui.data.import_dialog import PREVIEW_HEADERS, PREVIEW_READY

    dialog = _import_dialog(qtbot, tmp_path)

    table = dialog.findChild(QTableWidget, "importPreviewTable")
    assert table is not None, "no per-file preview table"
    headers = tuple(table.horizontalHeaderItem(column).text() for column in range(table.columnCount()))
    assert headers == PREVIEW_HEADERS
    assert table.rowCount() == 2, "every selected file gets its own row"
    assert table.item(0, 0).text() == "aSi_ML_25C.xy"
    assert table.item(0, 3).text() == "64"
    assert table.item(0, 4).text() == PREVIEW_READY


def test_import_dialog_right_aligns_the_row_counts(qtbot, tmp_path) -> None:
    """Mockup ``td.num`` right-aligns the count column so the digits line up.

    Left-aligned counts make 64 and 640 start at the same pixel across a batch
    of mixed-length files, which defeats the one comparison this column exists
    for.
    """
    from PySide6.QtCore import Qt

    dialog = _import_dialog(qtbot, tmp_path)

    table = dialog.findChild(QTableWidget, "importPreviewTable")
    alignment = table.item(0, 3).textAlignment()
    assert alignment & Qt.AlignmentFlag.AlignRight, "row counts read as a text column"


def test_import_dialog_keeps_a_failed_file_visible_and_marked(qtbot, tmp_path) -> None:
    """Mockup frame ⑥: 坏文件不静默吞掉而是显式标红并可跳过。

    ``_refresh_preview`` folded the parse error of ``paths[0]`` into one hint
    line and never looked at the remaining files, so a source with no numeric
    columns left no trace: the import went ahead and the dataset was missing.
    """
    from xrr_fitter.gui.data.import_dialog import PREVIEW_FAILED

    dialog = _import_dialog(qtbot, tmp_path)

    table = dialog.findChild(QTableWidget, "importPreviewTable")
    assert table.item(1, 0).text() == "blank_run.dat"
    assert table.item(1, 3).text() == "0"
    status = table.item(1, 4)
    assert status.text() == PREVIEW_FAILED


def test_export_dialog_groups_its_option_under_a_headed_section(qtbot) -> None:
    """Mockup frame ⑥ heads the export choice with a ``.groupbox`` 数据格式.

    The dialog shipped one bare checkbox stacked straight onto the button box,
    so the only pre-export decision read as an afterthought floating above 确定
    rather than as the data-format choice it is.
    """
    from xrr_fitter.gui.export.dialog import OrtOptionDialog

    dialog = OrtOptionDialog()
    qtbot.addWidget(dialog)

    group = dialog.findChild(QGroupBox, "exportFormatsGroup")
    assert group is not None, "the export option sits outside any headed section"
    assert group.title() == "数据格式"
    assert group.findChild(QCheckBox, "ortOptionCheckbox") is not None


def test_export_dialog_previews_the_manifest_seed(qtbot) -> None:
    """Mockup frame ⑥ prints ``master_seed`` in a 清单预览 before exporting.

    The seed is what makes an export bit-for-bit reproducible, and it appeared
    nowhere in the dialog: the reader confirmed the export without seeing the
    one number a reviewer needs in order to reproduce it.
    """
    from xrr_fitter.gui.export.dialog import OrtOptionDialog

    dialog = OrtOptionDialog(master_seed=20240822)
    qtbot.addWidget(dialog)

    preview = dialog.findChild(QLabel, "exportManifestPreview")
    assert preview is not None, "no manifest preview in the export dialog"
    assert "master_seed" in preview.text()
    assert "20240822" in preview.text()


def test_import_button_counts_what_it_will_and_will_not_import(qtbot, tmp_path) -> None:
    """Mockup frame ⑥ labels the confirm button with the batch it will act on.

    A bare 导入 over a mixed batch hides the arithmetic the reader most needs:
    that one of the chosen files carries no data and will be skipped.  Naming
    both numbers on the button puts that consequence where the click happens.
    """
    dialog = _import_dialog(qtbot, tmp_path)
    dialog.select_beam_kind("monochromatic")

    text = dialog.import_button().text()

    assert "导入" in text, "the confirm button stopped naming its own action"
    assert "1 个" in text, f"no importable count on the button: {text!r}"
    assert "跳过 1 个" in text, f"the skipped file is unnamed on the button: {text!r}"


def test_import_button_stays_quiet_when_every_file_is_readable(qtbot, tmp_path) -> None:
    """A clean batch must not advertise a 跳过 clause it has no failures for."""
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    paths = (_write_curve(tmp_path / "a.xy"), _write_curve(tmp_path / "b.xy"))
    dialog = ImportDialog(paths)
    qtbot.addWidget(dialog)
    dialog.select_beam_kind("monochromatic")

    text = dialog.import_button().text()

    assert "2 个" in text, f"no importable count on the button: {text!r}"
    assert "跳过" not in text, f"a clean batch invented a skip clause: {text!r}"


def test_import_preview_table_is_exactly_as_tall_as_its_rows(qtbot, tmp_path) -> None:
    """Mockup frame ⑥ ends the preview grid at its last file.

    ``QTableWidget`` reports Qt's fixed default height whatever it holds, so a
    four-file batch was drawn with a blank strip hanging below the last row —
    dead space that reads as "more files, failed to load".
    """
    dialog, table = _preview_table(qtbot, tmp_path, 4)

    assert table.sizeHint().height() == _table_content_height(table), "the preview grid outruns its own rows"


def test_import_preview_table_caps_a_long_batch_at_the_shared_ceiling(qtbot, tmp_path) -> None:
    """A folder import must yield to its scrollbar at the house row ceiling.

    Sizing to content without a cap would let a 30-file folder import grow the
    dialog past the screen, so the table stops at the same ceiling the dock
    trees use rather than at whatever height Qt happens to default to.
    """
    from xrr_fitter.gui.sizing import VISIBLE_ROW_CEILING

    dialog, table = _preview_table(qtbot, tmp_path, 30)

    capped = VISIBLE_ROW_CEILING * table.rowHeight(0)
    capped += table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth()
    assert table.sizeHint().height() == capped, "a long batch ignored the shared row ceiling"


def test_import_dialog_labels_its_cancel_button_in_chinese(qtbot, tmp_path) -> None:
    """Mockup frame ⑥ ends the import footer with 取消, not Qt's English default.

    ``accessibility._standard_button_name`` already knows this copy, but it only
    ever reached ``accessibleName``: no dialog runs ``configure_accessibility``
    on itself, so a reader of a fully Chinese dialog was handed an English
    ``Cancel`` -- and a screen-reader user heard 取消 for a button that did not
    say it.
    """
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "curve.xy"),))
    qtbot.addWidget(dialog)

    cancel = dialog.button_box.button(QDialogButtonBox.StandardButton.Cancel)
    assert cancel.text() == "取消", f"English standard button in a Chinese dialog: {cancel.text()!r}"


def test_export_dialog_labels_both_footer_buttons_in_chinese(qtbot) -> None:
    """Mockup frame ⑥ pairs 取消 with a confirm named after the action it performs.

    The mockup footer reads ``取消`` / ``导出到文件夹…``: the confirm says where
    the files land, so the next dialog -- a directory chooser -- is expected
    rather than surprising.
    """
    from xrr_fitter.gui.export.dialog import OrtOptionDialog

    dialog = OrtOptionDialog()
    qtbot.addWidget(dialog)

    box = dialog.findChild(QDialogButtonBox, "ortOptionButtons")
    assert box is not None, "no footer button box in the export dialog"
    assert box.button(QDialogButtonBox.StandardButton.Cancel).text() == "取消"
    assert box.button(QDialogButtonBox.StandardButton.Ok).text() == "导出到文件夹…"


def test_column_mapping_dialog_labels_its_footer_in_chinese(qtbot) -> None:
    """The import flow's second level is Chinese too, footer included.

    ``ColumnMappingDialog`` opens from 高级列映射… inside the import dialog, so a
    reader who has just seen a Chinese footer meets an English one a click later.
    It builds its own ``QDialogButtonBox`` rather than growing on the shared
    template, so the template's localization never reaches it.
    """
    from xrr_fitter.gui.data.import_dialog import ColumnMappingDialog

    dialog = ColumnMappingDialog()
    qtbot.addWidget(dialog)

    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).text() == "取消"
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).text() == "应用映射"


def test_shared_dialog_template_labels_its_footer_in_chinese(qtbot) -> None:
    """Every dialog built on the shared template inherits a Chinese footer.

    ``StyledDialog`` is the template new dialogs are meant to grow into, so the
    localization belongs in the template rather than in each subclass -- a
    dialog written tomorrow should not have to remember this call.
    """
    from xrr_fitter.gui.dialog_template import StyledDialog

    dialog = StyledDialog("示例对话框", "说明文字")
    qtbot.addWidget(dialog)

    box = dialog.button_box
    assert box.button(QDialogButtonBox.StandardButton.Cancel).text() == "取消"
    assert box.button(QDialogButtonBox.StandardButton.Ok).text() == "确定"
