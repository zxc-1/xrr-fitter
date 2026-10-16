"""Real navigation/export inputs on an explicitly constructed result fixture.

No optimizer is faked or timed here.  The fixture is the same fully serializable
result used by export integration tests; these tests cover GUI/export I/O only.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox
from tests.gui.control_readonly_helpers import scroll_roundtrip
from tests.gui.test_export_dialog import _fitted_project


def result_fixture(tmp_path, *, multiple=False):
    value = _fitted_project(tmp_path)
    if not multiple:
        return value
    dataset = value.datasets[0]
    result = dataset.last_valid_result
    first = result.candidates[0]
    second = replace(first, candidate_id="second", objective=first.objective + 0.1)
    archived = replace(first, candidate_id="archived", stop_reason="early_eliminated", objective=first.objective + 0.2)
    result = replace(result, candidates=(first, second, archived))
    return replace(value, datasets=(replace(dataset, last_valid_result=result),))


def _candidate_click(controls, window, candidate_id):
    view = window.result_panel.candidates
    row = next(
        index for index in range(view.count()) if view.item(index).data(Qt.ItemDataRole.UserRole) == candidate_id
    )
    item = view.item(row)
    controls.reach(view.viewport(), view.visualItemRect(item).center())
    controls.qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualItemRect(item).center())


def _adopt(controls, window):
    controls.named_click(window, "adoptCandidateButton")


def _adopted(window):
    return dict(window.document.project.ui_state.selected_candidate_ids).get("curve")


def test_result_candidate_rows_and_archived_confirmation(controls, tmp_path):
    window = controls.window(result_fixture(tmp_path, multiple=True))
    original = window.document.project
    _candidate_click(controls, window, "second")
    assert window.plot_panel.selected_candidate_id() == "second"
    assert window.document.project is original and not window.document.is_dirty
    controls.record(
        "candidateList:rowSelection",
        "click second row",
        "plot follows the viewed row; project and dirty flag unchanged",
    )
    _adopt(controls, window)
    assert _adopted(window) == "second"
    controls.record("adoptCandidateButton", "click Adopt on the viewed row", "only adoption persists the candidate")
    _candidate_click(controls, window, "archived")
    no = [lambda dialog: controls.click(dialog.button(QMessageBox.StandardButton.No))]
    controls.modals(lambda: _adopt(controls, window), no)
    assert _adopted(window) == "second"
    assert window.plot_panel.selected_candidate_id() == "archived"
    controls.record(
        "archivedCandidateConfirm:No",
        "view archived row; Adopt; confirm No",
        "inspection is allowed but persistent candidate remains unchanged",
    )
    yes = [lambda dialog: controls.click(dialog.button(QMessageBox.StandardButton.Yes))]
    controls.modals(lambda: _adopt(controls, window), yes)
    assert _adopted(window) == "archived"
    controls.record(
        "archivedCandidateConfirm:Yes",
        "Adopt archived row; confirm Yes",
        "explicit confirmation persists archived candidate",
    )


def test_result_clear_menu_removes_result_and_export_readiness(controls, tmp_path):
    window = controls.window(result_fixture(tmp_path))
    yes = [lambda dialog: controls.click(dialog.button(QMessageBox.StandardButton.Yes))]
    controls.modals(lambda: controls.menu(window, "clearResultsAction"), yes)
    assert window.document.project.datasets[0].last_valid_result is None
    assert window.result_panel.candidates.count() == 0
    assert not window.findChild(QAction, "exportResultsAction").isEnabled()
    controls.record(
        "clearResultsAction", "Fit menu Clear results", "result/rows cleared and export becomes unavailable"
    )


def test_guidance_all_header_cells_and_ctas_are_read_only(controls, tmp_path):
    window = controls.window(result_fixture(tmp_path))
    original = window.document.project
    controls.named_click(window, "workspaceModeGuided")
    for key in ("importStep", "structureStep", "fitStep", "resultStep"):
        controls.named_click(window, f"guidanceStep_{key}Cell")
        assert window.guidance.current_step() == key
        controls.record(f"guidanceStep_{key}Cell", "click guidance header cell", "requested page is active")
        controls.qtbot.wait(220)
    controls.named_click(window, "guidanceStep_importStepCell")
    controls.qtbot.wait(220)
    for key, target in (("importStep", "structureStep"), ("structureStep", "fitStep"), ("fitStep", "resultStep")):
        controls.named_click(window, f"{key}Action")
        assert window.guidance.current_step() == target
        controls.record(f"{key}Action", "click Next CTA", "onboarding advances without launching an operation")
        controls.qtbot.wait(220)
    controls.named_click(window, "resultStepAction")
    assert not window.guidance_is_visible()
    controls.named_click(window, "workspaceModeGuided")
    controls.named_click(window, "guidanceStep_structureStepCell")
    controls.qtbot.wait(220)
    controls.named_click(window, "structureStepManual")
    assert not window.guidance_is_visible()
    assert window.document.project is original and not window._operation_is_running()
    controls.record(
        ("resultStepAction", "structureStepManual"),
        "complete guidance and manual exit CTA",
        "returns to homepage; immutable project is unchanged",
    )


def test_pipeline_each_reached_step_by_mouse_and_keyboard(controls, tmp_path):
    window = controls.window(result_fixture(tmp_path))
    original = window.document.project
    for index, name in enumerate(("数据", "结构", "参数", "拟合", "结果")):
        controls.named_click(window, f"pipelineStep_{name}")
        assert window.pipeline_nav.current_step_index() == index
        row = controls.widget(window, f"pipelineStep_{name}")
        controls.qtbot.keyClick(row, Qt.Key.Key_Return)
        controls.qtbot.keyClick(row, Qt.Key.Key_Space)
        assert window.pipeline_nav.current_step_index() == index
        controls.record(f"pipelineStep_{name}", "click row and press Return/Space", "reached step becomes selected")
    assert window.document.project is original


def _assert_export_manifest(manifest):
    assert manifest is not None
    suffixes = set()
    for record in manifest.files:
        path = manifest.run_directory / record.path
        assert path.is_file()
        assert len(path.read_bytes()) == record.size
        assert sha256(path.read_bytes()).hexdigest() == record.sha256
        suffixes.add(path.suffix)
    assert {".xlsx", ".json", ".png", ".svg", ".ort", ".csv"} <= suffixes


def fixed_export_options(driver, dialog):
    names = (
        "xlsxOptionCheckbox",
        "overviewPngCheckbox",
        "residualsPngCheckbox",
        "parameterTableCheckbox",
        "correlationCheckbox",
        "manifestCheckbox",
    )
    for name in names:
        widget = driver.widget(dialog, name)
        assert not widget.isEnabled() and widget.isChecked()
        driver.qtbot.mouseClick(widget, Qt.MouseButton.LeftButton)
        assert widget.isChecked()
        driver.record(
            name,
            "attempt click on fixed export output checkbox",
            "required output remains enabled in the manifest; checkbox is intentionally not an option",
        )


@pytest.mark.parametrize("surface", ("toolbar", "menu", "shortcut"))
def test_all_export_formats_via_real_dialogs(controls, tmp_path, surface):
    window = controls.window(result_fixture(tmp_path / "fixture"))
    destination = tmp_path / "export"
    destination.mkdir()

    def options(dialog):
        fixed_export_options(controls, dialog)
        for name in ("ortOptionCheckbox", "csvOptionCheckbox", "sldSvgCheckbox"):
            widget = controls.widget(dialog, name)
            before = widget.isChecked()
            controls.click(widget)
            assert widget.isChecked() is not before
            if not widget.isChecked():
                controls.click(widget)
            assert widget.isChecked()
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    def summary(dialog):
        editor = controls.widget(dialog, "exportSummaryText")
        text = editor.toPlainText()
        assert "导出完成" in text and "sha256" in text
        controls.qtbot.keyClicks(editor, "do not overwrite the manifest")
        assert editor.isReadOnly() and editor.toPlainText() == text
        controls.record(
            "readOnly:exportSummaryText",
            "attempt to type into completed export summary",
            "the published manifest text stays unchanged",
        )
        scroll_roundtrip(controls, editor, "evidenceScrollbars")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Close)

    activate = {
        "toolbar": lambda: controls.named_click(window, "exportResultsButton"),
        "menu": lambda: controls.menu(window, "exportResultsAction"),
        "shortcut": lambda: controls.shortcut(window, "exportResultsAction"),
    }[surface]
    controls.modals(activate, [options, lambda dialog: controls.file(dialog, destination), summary])
    manifest = window.export_workflow.manifest
    _assert_export_manifest(manifest)
    identity = {
        "toolbar": "exportResultsButton",
        "menu": "exportResultsAction",
        "shortcut": "exportShortcut:Ctrl+Shift+E",
    }[surface]
    controls.record(
        (
            identity,
            "ortOptionCheckbox",
            "csvOptionCheckbox",
            "sldSvgCheckbox",
            "ortOptionButtons:Ok",
            "fileDialog:ExportDirectory",
            "exportSummaryButtons:Close",
        ),
        f"{surface} export; toggle optional formats; type directory; close summary",
        "all six output formats are real files, lengths/SHA-256 equal manifest",
        setup="serializable export integration result fixture; no optimizer run or fake worker",
    )


@pytest.mark.parametrize("where", ("options", "directory"))
def test_export_cancel_at_each_dialog_writes_nothing(controls, tmp_path, where):
    window = controls.window(result_fixture(tmp_path))
    original = window.document.project
    if where == "options":
        handlers = [lambda dialog: controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)]
    else:
        handlers = [
            lambda dialog: controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok),
            lambda dialog: controls.file(dialog, None),
        ]
    controls.modals(lambda: controls.named_click(window, "exportResultsButton"), handlers)
    assert window.document.project is original and window.export_workflow.manifest is None
    assert not list(tmp_path.glob("run-*"))
    controls.record(
        "ortOptionButtons:Cancel" if where == "options" else "fileDialog:ExportDirectory",
        f"Cancel at {where}",
        "no manifest/export is published and project unchanged",
    )


def test_inference_evidence_dialog_tabs_and_escape(controls, tmp_path):
    window = controls.window(result_fixture(tmp_path))
    controls.menu(window, "viewInferenceEvidenceAction")
    dialog = window.result_panel._uncertainty_dialog
    assert dialog.isVisible() and not window.result_panel.mcmc_group.isVisibleTo(dialog)
    pages = window.result_panel.uncertainty.pages
    for index, key in enumerate(("correlation", "profile", "sld_bands", "posterior")):
        controls.tab(pages, index)
        assert pages.currentWidget().isVisible()
        controls.record(
            f"uncertaintyPages:{key}",
            "click evidence tab",
            "corresponding evidence page is active, even when result says unavailable",
        )
    controls.qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    assert not dialog.isVisible()
    controls.record(
        "viewInferenceEvidenceAction",
        "Fit menu evidence; four tabs; Escape",
        "read-only dialog opens and closes without starting worker",
    )
