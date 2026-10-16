"""Rendered output affordances cannot silently mutate input or evidence."""

from dataclasses import replace

import pytest
from PySide6.QtWidgets import QDialogButtonBox
from tests.gui.control_operation_driver import source_curve
from tests.gui.control_readonly_helpers import attempt_table_edit, scroll_roundtrip
from tests.gui.test_control_operations_dialogs import preset_project
from tests.gui.test_control_operations_results import result_fixture

import xrr_fitter.api as api


def test_import_feedback_tables_reject_inline_edits(controls, tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    source_curve(folder / "valid.xy")
    (folder / "bad.xy").write_text("not valid numeric data\n", encoding="utf-8")
    window = controls.window(preset_project())
    controls.modals(
        lambda: controls.named_click(window, "importFolderQuickButton"),
        [
            lambda dialog: controls.file(dialog, folder),
            lambda dialog: controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok),
        ],
    )
    assert len(window.document.project.datasets) == 1
    before = window.document.project
    for table in (window.data_panel.failure_table, window.data_panel.batch_preview_table):
        attempt_table_edit(controls, table)
    assert window.document.project is before


def test_result_value_cells_reject_inline_edits(controls, tmp_path):
    window = controls.window(result_fixture(tmp_path))
    table = controls.widget(window, "resultValueTable")
    before = window.document.project
    attempt_table_edit(controls, table)
    assert window.document.project is before


def automatic_summary():
    layer = api.AutomaticLayerResult("point-0", 0, "Zr", 120.0, 4.0, 3.1e-5, 2e-7, 0.92, 6.52, 1.03, 6.72, None)
    datasets = tuple(
        api.AutomaticDatasetSummary(
            f"point-{i}", api.AutomaticStatus.PASSED, True, None, (replace(layer, dataset_id=f"point-{i}"),)
        )
        for i in range(40)
    )
    uniformity = tuple(
        api.LayerUniformitySummary(f"group-{i}", 0, "Zr", 2, 121.0, 120.0, 122.0, 1.0, 0.83, 1.65) for i in range(30)
    )
    return api.AutomaticResultSummary("readonly-fixture", datasets, uniformity)


@pytest.mark.parametrize("kind", ("points", "uniformity"))
def test_automatic_summary_tables_are_readonly_and_scrollable(controls, kind):
    from xrr_fitter.gui.results.automatic import AutomaticPointLayerTable, AutomaticUniformityTable

    table = AutomaticPointLayerTable() if kind == "points" else AutomaticUniformityTable()
    controls.qtbot.addWidget(table)
    table.project_summary(automatic_summary())
    table.resize(650, 220)
    table.show()
    controls.qtbot.waitExposed(table)
    attempt_table_edit(controls, table)
    scroll_roundtrip(controls, table, "tableScrollbars")
    controls.records[-1]["setup"] = (
        "standalone production table projected from a synthetic result-summary fixture; not a numerical fit"
    )
