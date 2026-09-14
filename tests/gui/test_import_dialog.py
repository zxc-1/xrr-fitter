"""What ``ImportDialog`` asks before a manual import, and what it refuses to guess.

The dialog is the one place a beam, an instrument, a column mapping, and an angle
convention get declared by hand, so every field here is answerable only by the
person doing the import. It therefore blocks on the ones with no safe default
(beam kind, column indices) and states its evidence for the ones it detects
(角度约定) instead of back-filling them silently.

Layout belongs to the same contract: a control the user cannot reach on a 1440×900
screen is a control that was never asked.
"""

from __future__ import annotations

from math import asin, degrees

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QSpinBox,
)
from tests.gui.data_import_support import _panel, _write_curve, _write_headed_curve, api


def test_import_dialog_requires_explicit_beam_choice(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)

    assert dialog.beam_kind() is None
    assert dialog.import_button().isEnabled() is False
    assert dialog.validation_text() == "请选择光路类型：单色 / 混合 Kα"

    dialog.select_beam_kind("mixed_kalpha")

    assert dialog.import_button().isEnabled() is True
    assert dialog.beam_spec() == api.BeamSpec("mixed_kalpha")


def test_import_dialog_displays_and_uses_monochromatic_wavelength(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    dialog.select_beam_kind("monochromatic")
    editor = dialog.findChild(QDoubleSpinBox, "monochromaticWavelengthEditor")
    assert editor is not None
    editor.setValue(1.2345)

    assert dialog.beam_spec().wavelength_a == pytest.approx(1.2345)


def test_import_dialog_column_mapping_cancel_and_validation(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    assert dialog.column_mapping() is None

    dialog.set_column_mapping(
        two_theta=2,
        intensity=3,
        intensity_sigma=4,
        resolution=5,
        resolution_kind="sigma_q_a_inv",
    )
    assert dialog.column_mapping() == api.DataColumnMapping(2, 3, 4, 5, "sigma_q_a_inv")

    dialog.cancel_column_mapping()
    assert dialog.column_mapping() is None
    with pytest.raises(ValueError, match="distinct nonnegative"):
        dialog.set_column_mapping(two_theta=0, intensity=0)


def test_column_mapping_dialog_blocks_invalid_mapping_with_inline_error(qtbot) -> None:
    from xrr_fitter.gui.data.import_dialog import ColumnMappingDialog

    dialog = ColumnMappingDialog()
    qtbot.addWidget(dialog)
    dialog.findChild(QSpinBox, "twoThetaColumnEditor").setValue(0)
    dialog.findChild(QSpinBox, "intensityColumnEditor").setValue(0)
    buttons = dialog.findChild(QDialogButtonBox)

    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    error = dialog.findChild(QLabel, "columnMappingError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert error is not None and error.isVisible()
    assert "distinct nonnegative" in error.text()


def test_import_dialog_exposes_real_instrument_choices_and_geometry_fields(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    footprint = dialog.findChild(QComboBox, "footprintModeEditor")
    background = dialog.findChild(QComboBox, "backgroundModelEditor")
    resolution = dialog.findChild(QComboBox, "resolutionDomainEditor")
    assert [footprint.itemData(index) for index in range(footprint.count())] == [
        "geometry",
        "fit",
        "none",
    ]
    footprint.setCurrentIndex(footprint.findData("geometry"))
    dialog.findChild(QDoubleSpinBox, "sampleLengthEditor").setValue(10.0)
    dialog.findChild(QDoubleSpinBox, "beamWidthEditor").setValue(0.1)
    background.setCurrentIndex(background.findData("powerlaw"))
    resolution.setCurrentIndex(resolution.findData("theta"))

    instrument = dialog.instrument_spec()

    assert instrument.footprint_mode == "geometry"
    assert instrument.footprint_spill_angle_deg == pytest.approx(degrees(asin(0.01)))
    assert instrument.background_kind == "powerlaw"
    assert instrument.resolution_domain == "theta"


def test_import_action_keeps_parented_dialog_as_modal_window(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    source = _write_curve(tmp_path / "sample.xy")
    panel = _panel(qtbot)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: ([str(source)], ""),
    )

    def reject(dialog):
        observed.update(
            parent=dialog.parent(),
            is_window=dialog.isWindow(),
            is_modal=dialog.isModal(),
        )
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ImportDialog, "exec", reject)

    panel.findChild(QPushButton, "importFilesButton").click()

    assert observed == {"parent": panel, "is_window": True, "is_modal": True}


def test_import_preview_table_keeps_three_rows_when_the_dialog_is_squeezed(qtbot, tmp_path) -> None:
    """The batch preview stops being a preview once the layout can crush it.

    ``ContentSizedTable`` promises a three-row floor, but it promised it in
    ``sizeHint`` only, and a hint is what a layout gives away first.  Under the two
    rigid 仪器/光路 forms the table was handed its ``minimumSizeHint`` -- header plus
    two rows -- so the design's five-file batch (帧⑥: aSi_ML_25C … blank_run.dat,
    where the last row is the one that matters, ``✕ 无数据列``) opened showing the
    two files nobody needed to check.
    """
    from xrr_fitter.gui.data.import_dialog import ImportDialog
    from xrr_fitter.gui.sizing import VISIBLE_ROW_FLOOR

    paths = tuple(_write_curve(tmp_path / f"sample{index}.xy") for index in range(5))
    dialog = ImportDialog(paths)
    qtbot.addWidget(dialog)
    table = dialog.preview_table
    header = table.horizontalHeader().sizeHint().height()
    floor = VISIBLE_ROW_FLOOR * table.rowHeight(0) + header

    assert table.minimumSizeHint().height() >= floor, (
        f"{table.minimumSizeHint().height()}px leaves "
        f"{(table.minimumSizeHint().height() - header) // table.rowHeight(0)} rows of {VISIBLE_ROW_FLOOR}"
    )


def test_import_dialog_fits_a_laptop_without_hiding_its_controls(qtbot, tmp_path) -> None:
    """Every control has to be reachable on the shortest screen the app supports.

    The dialog stacked 光路 (5 rows) and 仪器 (6 rows) as two full-width forms, which
    is 408px of rigid minimum before the preview table or the curve, and asked for
    933px in total -- taller than the work area of a 1440×900 display, where a
    window manager crops the footer and the 导入 button goes with it.  The design
    lays these out two-per-row (帧⑥'s ``.wrap2``), which is what makes them fit.
    """
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    layout = dialog.layout()

    assert layout.totalMinimumSize().height() <= 640, layout.totalMinimumSize()
    assert layout.totalSizeHint().height() <= 800, layout.totalSizeHint()
    assert dialog.button_box.isVisibleTo(dialog)

    # 宽度是同一笔账的另一半。两栏并排之后，每个数值框都按「1000000.00000000」——它永远
    # 不会显示的那个最宽值——要走 141px，八个加起来把对话框的下限顶到 1035px。设计稿那张
    # 导入卡不到 1035 的三分之二宽，而这 1035 是下限：用户想把它收窄也收不动。
    assert layout.totalMinimumSize().width() <= 720, layout.totalMinimumSize()


def test_choosing_the_incident_angle_convention_relabels_the_preview(qtbot, tmp_path) -> None:
    """θ 那一档要能按下去，按下去之后预览表得说它现在按 θ 读。

    这一格是这次改动唯一看得见的地方：约定选错了，行数与「✓ 就绪」照样成立，导进来的
    角度却整段差一倍。所以角度列必须跟着选择走，而不是钉死在 ``2θ``。
    """
    from xrr_fitter.gui.data.import_dialog import (
        PREVIEW_ANGLE_THETA,
        PREVIEW_ANGLE_TWO_THETA,
        ImportDialog,
    )

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)

    assert dialog.theta_convention.isEnabled()
    assert dialog.angle_convention() == "two_theta"
    assert dialog.preview_table.item(0, 1).text() == PREVIEW_ANGLE_TWO_THETA == "2θ"

    dialog.theta_convention.setChecked(True)

    assert dialog.angle_convention() == "theta"
    assert dialog.preview_table.item(0, 1).text() == PREVIEW_ANGLE_THETA == "θ"
    # 预览按 θ 读的是同一份文件，行数不该因为换了轴而变；变的只有轴。
    assert dialog.preview_table.item(0, 3).text() == "32"


def test_detecting_the_convention_moves_the_choice_to_the_axis_the_header_names(qtbot, tmp_path) -> None:
    """表头自报入射角，检测就把选择挪过去，并把依据原文摆出来。

    这一栏此前只能手选，而选错不报错——导进来的角度整段差一倍。文件自己写了轴名的时候，
    让用户去核对表头再手动对上一栏，是把机器读得出来的事推给人。
    """
    from xrr_fitter.gui.data.import_dialog import ANGLE_THETA_TEXT, ImportDialog

    dialog = ImportDialog((_write_headed_curve(tmp_path / "incident.xy", "# Omega Intensity"),))
    qtbot.addWidget(dialog)

    assert dialog.two_theta_convention.isChecked()
    assert dialog.angle_convention() == "two_theta"
    assert dialog.convention_status.text() == ""

    dialog.detect_convention.click()

    assert dialog.theta_convention.isChecked()
    assert dialog.angle_convention() == "theta"
    assert ANGLE_THETA_TEXT in dialog.convention_status.text()
    assert "# Omega Intensity" in dialog.convention_status.text()


def test_detecting_the_convention_leaves_the_choice_alone_without_a_declared_axis(qtbot, tmp_path) -> None:
    """表头没写轴名就不动选择，只说明为什么没动。

    静默回填才是这里真正的危险：把没有依据的猜测写进单选，用户会以为那是文件说的。
    ``_write_curve`` 写的正是这一类——纯数值，一行表头都没有。
    """
    from xrr_fitter.gui.data.import_dialog import DETECT_UNCHANGED, ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "bare.xy"),))
    qtbot.addWidget(dialog)

    dialog.detect_convention.click()

    assert dialog.two_theta_convention.isChecked()
    assert dialog.angle_convention() == "two_theta"
    assert dialog.convention_status.text().endswith(DETECT_UNCHANGED)
    # 光说「没动」不够：没动的理由得写出来，否则用户不知道是文件没写还是检测坏了。
    assert dialog.convention_status.text() != DETECT_UNCHANGED


def test_detecting_the_convention_reports_a_batch_that_disagrees_with_itself(qtbot, tmp_path) -> None:
    """一批里两种轴都被声明过，就谁也不回填。

    角度约定这一栏是整批共用的（``MeasurementPreset`` 一份），所以两份文件各写一个轴
    时没有「多数票」可取——挑一个就等于让另一份整段翻倍。
    """
    from xrr_fitter.gui.data.import_dialog import DETECT_CONFLICT, ImportDialog

    dialog = ImportDialog(
        (
            _write_headed_curve(tmp_path / "incident.xy", "# Omega Intensity"),
            _write_headed_curve(tmp_path / "scattering.xy", "# 2Theta Intensity"),
        )
    )
    qtbot.addWidget(dialog)

    dialog.detect_convention.click()

    assert dialog.two_theta_convention.isChecked()
    assert dialog.convention_status.text() == DETECT_CONFLICT
