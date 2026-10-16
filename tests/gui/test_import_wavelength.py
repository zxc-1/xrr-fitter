"""Replay the clean-home real-data wavelength mistake at the GUI boundary."""

import pytest
from PySide6.QtWidgets import QDialog
from tests.gui.data_import_support import _panel, _saved_preset, _write_headed_curve

from xrr_fitter.gui.data.import_dialog import PREVIEW_MEASUREMENT, PREVIEW_READY, ImportDialog


def _dialog(qtbot, paths, **kwargs):
    dialog = ImportDialog(tuple(paths), **kwargs)
    qtbot.addWidget(dialog)
    return dialog


def test_declared_wavelength_preview_is_not_a_false_conflict_before_beam_choice(qtbot, tmp_path):
    path = _write_headed_curve(tmp_path / "scan.xy", "# wavelength=0.6889 A")
    dialog = _dialog(qtbot, (path,))

    assert dialog.beam_kind() is None
    assert dialog.preview_table.item(0, 3).text() == "32"
    assert dialog.preview_table.item(0, 4).text() == PREVIEW_READY
    assert not dialog.import_button().isEnabled()


def test_import_prefills_declared_wavelength_without_guessing_beam(qtbot, tmp_path):
    path = _write_headed_curve(tmp_path / "PETRA1046 DNTTPDIF.xy", "# 2theta_deg intensity; wavelength=0.6889 A")
    dialog = _dialog(qtbot, (path,))
    assert dialog.mono_wavelength.value() == pytest.approx(0.6889)
    assert dialog.beam_kind() is None
    dialog.select_beam_kind("monochromatic")
    assert dialog.import_button().isEnabled()
    assert "0.6889" in dialog.wavelength_status.text()


def test_typed_wavelength_conflict_is_visible_and_cannot_be_accepted(qtbot, tmp_path):
    path = _write_headed_curve(tmp_path / "scan.xy", "# wavelength=0.6889 A")
    dialog = _dialog(qtbot, (path,))
    dialog.select_beam_kind("monochromatic")
    dialog.mono_wavelength.setValue(1.5406)
    assert not dialog.import_button().isEnabled()
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.mono_wavelength.setValue(0.6889)
    assert dialog.import_button().isEnabled()
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_saved_preset_is_not_silently_overwritten_by_header(qtbot, tmp_path):
    path = _write_headed_curve(tmp_path / "scan.xy", "# wavelength=0.6889 A")
    preset = _saved_preset()
    dialog = _dialog(qtbot, (path,), preset=preset, folder_mode=True)
    assert dialog.beam_spec() == preset.beam
    assert not dialog.import_button().isEnabled()


def test_clean_home_import_persists_header_wavelength_in_dataset_and_preset(qtbot, tmp_path, monkeypatch):
    path = _write_headed_curve(tmp_path / "PETRA1046 DNTTPDIF.xy", "# 2theta_deg intensity; wavelength=0.6889 A")
    panel = _panel(qtbot)

    def accept(dialog):
        dialog.select_beam_kind("monochromatic")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ImportDialog, "exec", accept)
    panel._confirm_import((path,), folder=False)
    project = panel.document.project
    assert project.measurement_preset.beam.wavelength_a == pytest.approx(0.6889)
    assert project.datasets[0].beam.wavelength_a == pytest.approx(0.6889)


def test_plain_header_wavelength_prefills_the_actual_import(qtbot, tmp_path):
    path = _write_headed_curve(tmp_path / "scan.xy", "wavelength = 0.6889 A\n2theta_deg intensity")
    dialog = _dialog(qtbot, (path,))

    assert dialog.mono_wavelength.value() == pytest.approx(0.6889)
    assert dialog.beam_kind() is None
    assert not dialog.import_button().isEnabled()
    dialog.select_beam_kind("monochromatic")
    assert dialog.preview_table.item(0, 4).text() == PREVIEW_READY
    assert dialog.import_button().isEnabled()


def test_recursive_folder_rescans_wavelength_before_validating_preview(qtbot, tmp_path):
    root = tmp_path / "batch"
    source = _write_headed_curve(root / "nested" / "P1 Zr.xy", "# wavelength=0.6889 A")
    dialog = _dialog(qtbot, (root,), folder_mode=True)
    assert not dialog.source_paths()

    dialog.recursive_check.setChecked(True)

    assert dialog.source_paths() == (source,)
    assert dialog.mono_wavelength.value() == pytest.approx(0.6889)
    assert "0.6889" in dialog.wavelength_status.text()
    assert dialog.preview_table.item(0, 4).text() == PREVIEW_READY
    assert not dialog.import_button().isEnabled()
    dialog.select_beam_kind("monochromatic")
    assert dialog.import_button().isEnabled()


def test_disabling_recursion_removes_rows_and_refreshes_wavelength_message(qtbot, tmp_path):
    root = tmp_path / "batch"
    _write_headed_curve(root / "nested" / "P1 Zr.xy", "# wavelength=0.6889 A")
    dialog = _dialog(qtbot, (root,), folder_mode=True)
    dialog.recursive_check.setChecked(True)
    dialog.select_beam_kind("monochromatic")

    dialog.recursive_check.setChecked(False)

    assert not dialog.source_paths()
    assert dialog.preview_table.rowCount() == 0
    assert not dialog.import_button().isEnabled()
    assert dialog.import_button().text() == "导入 0 个"
    assert "未声明" in dialog.wavelength_status.text()


def test_recursive_wavelength_conflict_is_refreshed_in_both_directions(qtbot, tmp_path):
    root = tmp_path / "batch"
    _write_headed_curve(root / "P1 Zr.xy", "# wavelength=0.6889 A")
    _write_headed_curve(root / "nested" / "P2 Zr.xy", "# wavelength=0.7093 A")
    dialog = _dialog(qtbot, (root,), folder_mode=True)
    dialog.select_beam_kind("monochromatic")

    dialog.recursive_check.setChecked(True)

    assert "不一致" in dialog.wavelength_status.text()
    assert dialog.mono_wavelength.value() == pytest.approx(0.6889)
    assert dialog.import_button().text() == "导入 1 个（跳过 1 个失败）"
    dialog.recursive_check.setChecked(False)
    assert "不一致" not in dialog.wavelength_status.text()
    assert "0.6889" in dialog.wavelength_status.text()
    assert dialog.import_button().text() == "导入 1 个"


def test_recursive_wavelength_scan_never_overwrites_saved_preset(qtbot, tmp_path):
    root = tmp_path / "batch"
    _write_headed_curve(root / "nested" / "P1 Zr.xy", "# wavelength=0.6889 A")
    preset = _saved_preset()
    dialog = _dialog(qtbot, (root,), folder_mode=True, preset=preset)

    dialog.recursive_check.setChecked(True)

    assert dialog.beam_spec() == preset.beam
    assert dialog.mono_wavelength.value() == preset.beam.wavelength_a
    assert dialog.preview_table.item(0, 4).text() == PREVIEW_MEASUREMENT
    assert not dialog.import_button().isEnabled()


@pytest.mark.parametrize("bad_first", (True, False))
@pytest.mark.parametrize(
    "bad_content",
    (
        b"# wavelength=invalid\n0.1 10\n0.2 9\n",
        None,
        b"\xff\xfe invalid UTF-8",
        b"# wavelength=0.7093 A\nno numeric data\n",
    ),
    ids=("invalid_wavelength", "missing_file", "invalid_encoding", "declared_but_no_data"),
)
def test_unreadable_wavelength_does_not_block_valid_batch_member(qtbot, tmp_path, bad_first, bad_content):
    bad = tmp_path / "P0 Zr.xy"
    if bad_content is not None:
        bad.write_bytes(bad_content)
    good = _write_headed_curve(tmp_path / "P1 Zr.xy", "# wavelength=0.6889 A")
    paths = (bad, good) if bad_first else (good, bad)
    dialog = _dialog(qtbot, paths)
    dialog.select_beam_kind("monochromatic")

    assert dialog.mono_wavelength.value() == pytest.approx(0.6889)
    assert dialog.import_button().isEnabled()
    assert dialog.import_button().text() == "导入 1 个（跳过 1 个失败）"
    assert dialog.preview_table.item(paths.index(good), 4).text() == PREVIEW_READY
    assert dialog.preview_table.item(paths.index(bad), 4).toolTip()
    assert bad.name in dialog.wavelength_status.toolTip()


def test_all_invalid_wavelengths_keep_reason_visible_and_cannot_import(qtbot, tmp_path):
    bad = _write_headed_curve(tmp_path / "P0 Zr.xy", "# wavelength=invalid")
    dialog = _dialog(qtbot, (bad,))
    dialog.select_beam_kind("monochromatic")

    assert "wavelength" in dialog.wavelength_status.text()
    assert bad.name in dialog.wavelength_status.toolTip()
    assert not dialog.import_button().isEnabled()
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_partial_batch_reaches_project_with_correct_wavelength_and_failure_reason(qtbot, tmp_path, monkeypatch):
    bad = _write_headed_curve(tmp_path / "P0 Zr.xy", "# wavelength=invalid")
    good = _write_headed_curve(tmp_path / "P1 Zr.xy", "# wavelength=0.6889 A")
    panel = _panel(qtbot)

    def accept(dialog):
        dialog.select_beam_kind("monochromatic")
        assert dialog.import_button().isEnabled()
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(ImportDialog, "exec", accept)
    panel._confirm_import((bad, good), folder=False)

    project = panel.document.project
    assert len(project.datasets) == 1
    assert project.datasets[0].beam.wavelength_a == pytest.approx(0.6889)
    assert project.measurement_preset.beam.wavelength_a == pytest.approx(0.6889)
    assert panel.failure_table.rowCount() == 1
    assert panel.failure_table.item(0, 0).text() == bad.name
    assert "wavelength" in panel.failure_table.item(0, 1).text()
