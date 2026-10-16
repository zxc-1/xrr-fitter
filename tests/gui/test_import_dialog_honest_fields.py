"""F50h：导入对话框的字段只说真话——没有按不动的开关，位数不铺满，用不上的波长不让填。"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QCheckBox
from tests.gui.test_import_dialog import _write_curve


def _dialog(qtbot, tmp_path):
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    return dialog


def test_numbers_drop_trailing_zeros_but_keep_every_digit(qtbot, tmp_path) -> None:
    dialog = _dialog(qtbot, tmp_path)

    assert dialog.mono_wavelength.text() == "1.5406 Å"
    assert dialog.wavelength_1.text() == "1.54056 Å"
    assert dialog.sample_length.text() == "10 mm"
    # 只是不写尾零：八位小数照旧存得下，Cu Kα₁ 的 1.540598 不会被四舍五入成别的光。
    dialog.mono_wavelength.setValue(1.54059817)
    assert dialog.mono_wavelength.value() == pytest.approx(1.54059817, abs=1e-12)
    assert dialog.mono_wavelength.text() == "1.54059817 Å"


def test_no_checkbox_sits_permanently_disabled(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.data.import_dialog import LOG_INTENSITY_TEXT

    dialog = _dialog(qtbot, tmp_path)

    dead = [
        box.objectName() for box in dialog.findChildren(QCheckBox) if box.isVisibleTo(dialog) and not box.isEnabled()
    ]
    assert dead == []
    # 取对数这件事仍然说得出来，只是写成一句陈述，和「跳过表头行：自动」同一种格。
    statement = dialog.log_intensity_field
    assert statement.isReadOnly() and statement.isVisibleTo(dialog)
    assert statement.text() == LOG_INTENSITY_TEXT and statement.toolTip()


@pytest.mark.parametrize("mixed", (False, True))
def test_only_the_chosen_beams_wavelengths_are_editable(qtbot, tmp_path, mixed) -> None:
    dialog = _dialog(qtbot, tmp_path)
    kalpha = (dialog.wavelength_1, dialog.wavelength_2, dialog.intensity_ratio)
    # 还没选光路时两组都开着：校验行正在请读者去选，先灰掉哪一组都是替他做了选择。
    assert dialog.mono_wavelength.isEnabled() and all(editor.isEnabled() for editor in kalpha)

    (dialog.mixed_button if mixed else dialog.mono_button).setChecked(True)

    assert dialog.mono_wavelength.isEnabled() is not mixed
    assert all(editor.isEnabled() is mixed for editor in kalpha)
