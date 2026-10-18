"""Wavelength prefill follows reparsing after an explicit column correction."""

import pytest
from tests.gui.data_import_support import _saved_preset, _write_headed_curve

from xrr_fitter.gui.data.import_dialog import PREVIEW_MEASUREMENT, PREVIEW_READY, ColumnMappingDialog, ImportDialog


def _dialog(qtbot, paths, **kwargs):
    dialog = ImportDialog(tuple(paths), **kwargs)
    qtbot.addWidget(dialog)
    return dialog


@pytest.mark.parametrize("noise_model", ("robust_log", "gaussian", "poisson"))
@pytest.mark.parametrize("use_dialog", (False, True))
def test_corrected_mapping_rescans_declared_wavelength(qtbot, tmp_path, monkeypatch, noise_model, use_dialog):
    path = tmp_path / "P1 Zr.xy"
    path.write_text(
        "# wavelength=0.6889 A\n"
        + "\n".join(f"nan {100 - index} {0.1 + index * 0.01:.6f} 0.1" for index in range(40))
        + "\n",
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, (path,), noise_model=noise_model)
    dialog.select_beam_kind("monochromatic")
    assert not dialog.import_button().isEnabled()

    def correct(mapping_dialog):
        mapping_dialog.two_theta.setValue(3)
        mapping_dialog.intensity.setValue(2)
        mapping_dialog.sigma_enabled.setChecked(True)
        mapping_dialog.intensity_sigma.setValue(4)
        mapping_dialog.accept()
        return mapping_dialog.result()

    if use_dialog:
        monkeypatch.setattr(ColumnMappingDialog, "exec", correct)
        dialog._edit_column_mapping()
    else:
        dialog.set_column_mapping(two_theta=2, intensity=1, intensity_sigma=3)

    assert dialog.mono_wavelength.value() == pytest.approx(0.6889)
    assert "0.6889" in dialog.wavelength_status.text()
    assert not dialog.wavelength_status.toolTip()
    assert dialog.preview_table.item(0, 4).text() == PREVIEW_READY
    assert dialog.import_button().isEnabled()


def test_mapping_refresh_does_not_replace_saved_wavelength(qtbot, tmp_path):
    path = _write_headed_curve(tmp_path / "scan.xy", "# wavelength=0.6889 A")
    preset = _saved_preset()
    dialog = _dialog(qtbot, (path,), preset=preset)

    dialog.set_column_mapping()

    assert dialog.mono_wavelength.value() == preset.beam.wavelength_a
    assert dialog.preview_table.item(0, 4).text() == PREVIEW_MEASUREMENT
    assert not dialog.import_button().isEnabled()


def test_reapplying_same_mapping_does_not_undo_explicit_wavelength_edit(qtbot, tmp_path):
    path = _write_headed_curve(tmp_path / "scan.xy", "# wavelength=0.6889 A")
    dialog = _dialog(qtbot, (path,))
    dialog.set_column_mapping()
    dialog.select_beam_kind("monochromatic")
    dialog.mono_wavelength.setValue(1.5406)

    dialog.set_column_mapping()

    assert dialog.mono_wavelength.value() == pytest.approx(1.5406)
    assert dialog.preview_table.item(0, 4).text() == PREVIEW_MEASUREMENT
    assert not dialog.import_button().isEnabled()
