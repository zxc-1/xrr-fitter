"""Import feedback belongs to one document, not the lifetime of a window."""

from dataclasses import replace

import pytest
from tests.gui.data_import_support import _panel, _saved_preset, _write_curve

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument


def _imported_panel(qtbot, tmp_path):
    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))
    bad = tmp_path / "P2 Nb.xy"
    bad.write_text("no numeric data\n", encoding="utf-8")
    panel.import_paths((_write_curve(tmp_path / "P1 Zr.xy"), bad))
    assert panel.batch_preview_table.rowCount() == 2
    assert panel.failure_table.rowCount() == 1
    return panel


def _table_rows(table):
    return tuple(
        tuple(table.item(row, column).text() for column in range(table.columnCount()))
        for row in range(table.rowCount())
    )


def _feedback(panel):
    return tuple((table.isHidden(), _table_rows(table)) for table in (panel.batch_preview_table, panel.failure_table))


def _replace_document(document, replacement, path):
    if replacement == "new":
        document.new()
    else:
        document.open(path)


@pytest.mark.parametrize("replacement", ("new", "open"))
def test_document_replacement_clears_prior_import_feedback(qtbot, tmp_path, replacement):
    panel = _imported_panel(qtbot, tmp_path)
    path = tmp_path / "other.xrrproj.json"
    api.save_project(api.new_project(), path)

    _replace_document(panel.document, replacement, path)

    assert _feedback(panel) == ((True, ()), (True, ()))


def test_open_failure_keeps_existing_import_feedback(qtbot, tmp_path):
    panel = _imported_panel(qtbot, tmp_path)
    before = _feedback(panel)

    with pytest.raises(FileNotFoundError):
        panel.document.open(tmp_path / "missing.xrrproj.json")

    assert _feedback(panel) == before


@pytest.mark.parametrize("replacement", ("new", "open"))
def test_failed_replacement_projection_keeps_import_feedback(qtbot, tmp_path, replacement):
    panel = _imported_panel(qtbot, tmp_path)
    document = panel.document
    original = document.project
    before = _feedback(panel)
    path = tmp_path / "other.xrrproj.json"
    api.save_project(api.new_project(), path)

    def reject_incoming(project):
        if project is not original:
            raise ValueError("projection rejected")

    document.register_project_projection(reject_incoming)
    with pytest.raises(ValueError, match="projection rejected"):
        _replace_document(document, replacement, path)

    assert document.project is original
    assert _feedback(panel) == before


def test_save_and_dataset_edit_keep_current_import_feedback(qtbot, tmp_path):
    panel = _imported_panel(qtbot, tmp_path)
    before = _feedback(panel)

    panel.document.save(tmp_path / "current.xrrproj.json")
    panel.document.select_active_dataset("P1")

    assert _feedback(panel) == before
