"""Saved SLD sampling display and the actual pyqtgraph context menu."""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from tests.gui.control_operation_driver import controls as controls
from tests.gui.plot_cases_5 import _sld_project
from tests.gui.test_control_operations_plots import data_window, live_point

import xrr_fitter.api as api


def _angle_ranges(panel):
    return tuple(tuple(panel.view(key).plot_item.vb.viewRange()[0]) for key in ("raw", "log"))


def _stored_range(window):
    active = window.document.active_dataset_id
    dataset = next(item for item in window.document.project.datasets if item.dataset_id == active)
    return tuple(dataset.fit_range_two_theta_deg)


def _assert_full_extent(plot):
    angles, _values = plot.observed_item.getData()
    low, high = plot.plot_item.vb.viewRange()[0]
    assert low <= float(np.min(angles))
    assert high >= float(np.max(angles))


def _assert_readonly_range_mode(window, project):
    assert window.plot_panel.interaction_mode() == "range"
    assert window.document.project is project


def test_saved_sld_bands_toggle_and_alignment_use_real_inputs(controls, tmp_path):
    project, _result = _sld_project(tmp_path, "curve")
    saved = tmp_path / "saved-sld.xrrproj.json"
    api.save_project(project, saved)
    window = controls.window(api.load_project(saved))
    # 后验可信带画在结果步的拟合剖面上；结构步画的是标称剖面，不叠别人的带（F16）。
    controls.named_click(window, "pipelineStep_结果")
    panel = window.plot_panel
    controls.context_menu(panel.toolbar, panel.toolbar.rect().center(), "plotToolAction:sld_profile")
    before = window.document.project
    report = before.datasets[0].last_valid_result.uncertainty.mcmc
    samples = report.samples_physical.copy()
    assert panel.sld_bands_toggle.isEnabled() and panel.sld_bands_toggle.isChecked()
    assert panel.view("sld").axes.collections
    controls.click(panel.sld_bands_toggle)
    assert not panel.view("sld").axes.collections
    controls.click(panel.sld_bands_toggle)
    assert panel.view("sld").axes.collections
    for index, label in ((1, "表面界面"), (0, "基底界面")):
        controls.select(panel.sld_align_selector, index)
        assert f"对齐 {label}" in panel.view("sld").axes.get_title()
        assert panel.view("sld").axes.collections
    assert window.document.project is before
    np.testing.assert_array_equal(report.samples_physical, samples)
    controls.record(
        ("sldBandsToggle", "sldAlignSelector"),
        "click bands off/on; select surface then substrate alignment",
        "real sample replay updates visible bands without changing saved samples or project",
        setup="serialized four-sample display fixture; real band computation, no optimizer or new MCMC",
    )


def test_pyqtgraph_right_click_view_all_restores_data_extent(controls, tmp_path):
    window = data_window(controls, tmp_path)
    plot = window.plot_panel.view("raw")
    window.activateWindow()
    controls.qtbot.waitUntil(window.isActiveWindow)
    x, y = plot.observed_item.getData()
    viewport, point = live_point(plot, x[len(x) // 2], y[len(y) // 2])
    wheel = QWheelEvent(
        QPointF(point),
        QPointF(viewport.mapToGlobal(point)),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(viewport, wheel)
    zoomed = plot.plot_item.vb.viewRange()[0]

    def view_all(menu):
        action = next(item for item in menu.actions() if item.text() == "View All")
        assert action.isVisible() and action.isEnabled()
        controls.qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())

    controls.popup(lambda: controls.qtbot.mouseClick(viewport, Qt.MouseButton.RightButton, pos=point), view_all)
    restored = plot.plot_item.vb.viewRange()[0]
    _assert_full_extent(plot)
    assert restored[1] - restored[0] > zoomed[1] - zoomed[0]
    controls.record(
        "pyqtgraph:contextMenusAndAutoRange",
        "QTest right-click live canvas; click View All",
        "actual ViewBox menu restores full data bounds after Qt wheel zoom",
        setup="wheel event prepares a zoomed viewport; right-click and menu selection use QTest",
    )


@pytest.mark.parametrize("visible_range", (None, (0.28, 1.08)), ids=("no-range", "selected-range"))
def test_range_navigation_actions_preserve_mode_and_project(controls, tmp_path, visible_range):
    window = data_window(controls, tmp_path)
    panel = window.plot_panel
    # View-only fixture setup: this does not count as a range-drag operation.
    if visible_range is not None:
        panel.show_range(*visible_range)
    # 每个数据集都存着拟合范围（导入时默认是全部数据）；不另画时，画着的就是存下的那个（F17）。
    focused = visible_range or _stored_range(window)
    assert panel.visible_range() == pytest.approx(focused)
    controls.named_click(window, "plotModeRange")
    controls.qtbot.wait(20)
    assert panel.interaction_mode() == "range"
    before_project = window.document.project
    controls.context_menu(panel.toolbar, panel.toolbar.rect().center(), "plotToolAction:zoom_to_range")
    np.testing.assert_allclose(_angle_ranges(panel), (focused, focused))
    _assert_readonly_range_mode(window, before_project)
    controls.context_menu(panel.toolbar, panel.toolbar.rect().center(), "plotToolAction:reset_zoom")
    for key in ("raw", "log"):
        _assert_full_extent(panel.view(key))
    _assert_readonly_range_mode(window, before_project)
    controls.record(
        ("plotModeRange", "plotToolAction:zoom_to_range", "plotToolAction:reset_zoom"),
        "click range mode; use actual zoom-to-range and reset context actions",
        "both angle views focus exactly, or stay unchanged without a range; reset spans all data and preserves mode/project",
        setup=f"view-only visible_range={visible_range!r} fixture; no range-drag success asserted",
    )
