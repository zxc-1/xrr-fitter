"""Keep compact GUI contracts under the stylesheet used by the real entry point."""

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint
from PySide6.QtGui import QColor, QFont, QFontInfo
from PySide6.QtWidgets import QLabel
from tests.gui.data_import_support import _write_curve
from tests.gui.test_guidance import STEP_NAMES, _guided

from xrr_fitter.gui import theme
from xrr_fitter.gui.application import create_application
from xrr_fitter.gui.data.import_dialog import ImportDialog


@pytest.fixture(autouse=True)
def _deliver_deferred_deletes(qapp):
    # 换应用级主题前先完成前一用例的 Qt 销毁，避免重刷待销毁的绘图对象。
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    yield
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(params=("light", "dark"))
def startup_theme(qapp, request):
    palette = theme._resolved_copy(qapp.palette())
    stylesheet = qapp.styleSheet()
    light_palette = theme.LIGHT_PALETTE
    qapp.setPalette(theme.light_palette() if request.param == "light" else theme.dark_palette())
    create_application([])
    try:
        yield qapp
    finally:
        theme.LIGHT_PALETTE = light_palette
        qapp.setPalette(palette)
        qapp.setStyleSheet(stylesheet)


def _assert_eyebrows(window, application):
    accent = QColor(theme.palette_tokens(application.palette()).accent)
    for name in STEP_NAMES:
        label = window.guidance.findChild(QLabel, f"{name}Eyebrow")
        assert QFontInfo(label.font()).pixelSize() == theme.STEP_EYEBROW_FONT_PX
        assert label.font().weight() == QFont.Weight.Bold
        assert label.font().letterSpacingType() == QFont.SpacingType.AbsoluteSpacing
        assert label.font().letterSpacing() == pytest.approx(theme.STEP_EYEBROW_TRACKING_PX, abs=1 / 64)
        assert label.palette().color(label.foregroundRole()) == accent


def test_step_eyebrows_keep_their_font_through_startup_and_restyle(qtbot, startup_theme):
    window = _guided(qtbot)
    window.autosave.stop()
    _assert_eyebrows(window, startup_theme)

    for palette in (theme.dark_palette(), theme.light_palette()):
        startup_theme.setPalette(palette)
        theme.apply_theme(startup_theme)
        qtbot.wait(1)
        _assert_eyebrows(window, startup_theme)


@pytest.mark.parametrize("folder_mode", (False, True))
def test_styled_import_keeps_its_controls_inside_a_laptop_dialog(qtbot, tmp_path, startup_theme, folder_mode):
    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),), folder_mode=folder_mode)
    qtbot.addWidget(dialog)
    assert dialog.layout().totalMinimumSize().width() <= 720, dialog.layout().totalMinimumSize()
    assert dialog.layout().totalMinimumSize().height() <= 640, dialog.layout().totalMinimumSize()
    assert dialog.layout().totalSizeHint().height() <= 800, dialog.layout().totalSizeHint()

    dialog.resize(720, 640)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.width() <= 720
    assert dialog.height() <= 640
    controls = (
        dialog.preview_table,
        dialog.angle_row,
        dialog.advanced_box,
        dialog.button_box,
        dialog.mono_button,
        dialog.mixed_button,
        dialog.mono_wavelength,
        dialog.wavelength_1,
        dialog.wavelength_2,
        dialog.intensity_ratio,
        dialog.instrument_id,
        dialog.footprint,
        dialog.sample_length,
        dialog.beam_width,
        dialog.background,
        dialog.resolution_domain,
        dialog.mapping_button,
    )
    for control in controls:
        assert control.isVisibleTo(dialog)
        bounds = control.rect().translated(control.mapTo(dialog, QPoint()))
        assert dialog.rect().contains(bounds), (control.objectName(), bounds)
