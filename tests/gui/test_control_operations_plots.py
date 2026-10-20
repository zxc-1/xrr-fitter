"""Plot input sweeps at Qt boundaries, not Matplotlib callback invocations."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from tests.gui.control_operation_driver import controls as controls
from tests.gui.control_operation_driver import setup_project

import xrr_fitter.api as api


def data_window(driver, tmp_path):
    window = driver.window(setup_project(tmp_path, datasets=2))
    driver.menu(window, "expertModeAction")
    driver.named_click(window, "pipelineStep_数据")
    driver.menu(window, "plotViewAction:raw")
    return window


def live_point(plot, x, y):
    view = plot.scene().views()[0]
    return view.viewport(), view.mapFromScene(plot.plot_item.vb.mapViewToScene(QPointF(float(x), float(y))))


def drag(driver, target, first, last):
    # Offscreen menus do not always return activation to the owning window.
    target.window().activateWindow()
    driver.qtbot.waitUntil(target.window().isActiveWindow)
    driver.qtbot.mouseMove(target, first)
    # Flush each input separately: pyqtgraph drops moves arriving within 10 ms.
    driver.qtbot.wait(20)
    driver.qtbot.mousePress(target, Qt.MouseButton.LeftButton, pos=first)
    driver.qtbot.wait(20)
    driver.qtbot.mouseMove(target, last, delay=20)
    driver.qtbot.wait(20)
    driver.qtbot.mouseRelease(target, Qt.MouseButton.LeftButton, pos=last)
    QApplication.processEvents()


def test_plot_context_modes_overlay_and_navigation(controls, tmp_path):
    window = data_window(controls, tmp_path)
    panel = window.plot_panel
    toolbar = panel.toolbar
    for mode in ("mask", "range", "view"):
        identity = f"plotToolAction:{mode}"
        controls.context_menu(toolbar, toolbar.rect().center(), identity)
        assert panel.interaction_mode() == mode
        controls.record(identity, f"Qt context request; click {mode}", "exclusive plot mode follows menu action")
    for name, mode in (("plotNavZoom", "zoom"), ("plotNavPan", "pan")):
        controls.named_click(window, name)
        assert panel.navigation_mode() == mode
        controls.record(name, "click navigation button", "live plot navigation mode changes")
    controls.named_click(window, "plotNavHome")
    controls.context_menu(toolbar, toolbar.rect().center(), "plotToolAction:overlay")
    assert toolbar.overlay_action.isChecked()
    controls.record(
        ("plotNavHome", "plotToolAction:overlay"),
        "Home and overlay context action",
        "view resets and overlay is enabled",
    )


def test_range_handle_drag_commits_fit_window_and_navigation(controls, tmp_path):
    window = data_window(controls, tmp_path)
    panel = window.plot_panel
    controls.named_click(window, "plotModeRange")
    plot = panel.view("raw")
    # Let the newly installed GraphicsScene region enter the item index.
    controls.qtbot.wait(1)
    low, high = plot.range_item.getRegion()
    y = sum(plot.plot_item.vb.viewRange()[1]) / 2
    target = low + (high - low) * 0.2
    viewport, first = live_point(plot, low, y)
    _, last = live_point(plot, target, y)
    drag(controls, viewport, first, last)
    selected = window.document.project.datasets[0].fit_range_two_theta_deg
    assert selected is not None and selected[0] == pytest.approx(target, abs=(high - low) * 0.03)
    controls.record(
        ("plotModeRange", "reflectivity:rangeDrag"),
        "enable range and drag the left boundary",
        "stored fitting range follows the real Qt drag",
    )
    for key in ("zoom_to_range", "reset_zoom"):
        controls.context_menu(panel.toolbar, panel.toolbar.rect().center(), f"plotToolAction:{key}")
        assert all(np.isfinite(axis).all() for axis in plot.plot_item.vb.viewRange())
        controls.record(
            f"plotToolAction:{key}",
            "click range navigation context item",
            "view limits remain finite and operation is dispatched",
        )


def test_mask_click_cursor_and_wheel_use_live_canvas(controls, tmp_path):
    window = data_window(controls, tmp_path)
    panel = window.plot_panel
    plot = panel.view("raw")
    x, y = plot.observed_item.getData()
    viewport, point = live_point(plot, x[10], y[10])
    controls.context_menu(panel.toolbar, panel.toolbar.rect().center(), "plotToolAction:mask")
    before = tuple(window.document.project.datasets[0].fit_mask)
    controls.qtbot.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=point)
    after = tuple(window.document.project.datasets[0].fit_mask)
    assert before[10] and not after[10] and sum(a != b for a, b in zip(before, after, strict=False)) == 1
    controls.record(
        "reflectivity:pointMaskClick",
        "click observed point in mask mode",
        "only the corresponding prepared point is excluded",
    )
    controls.context_menu(panel.toolbar, panel.toolbar.rect().center(), "plotToolAction:view")
    controls.qtbot.mouseMove(viewport, point)
    controls.qtbot.waitUntil(lambda: panel.cursor_readout.text() != panel.CURSOR_IDLE_HINT, timeout=1500)
    controls.record(
        "plot:cursorReadout", "move mouse over live curve", "rendered coordinate readout replaces idle hint"
    )
    old = plot.plot_item.vb.viewRange()
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
    new = plot.plot_item.vb.viewRange()
    assert new[0][1] - new[0][0] < old[0][1] - old[0][0]
    controls.record(
        "plot:wheelZoom", "deliver Qt wheel event to live viewport", "positive wheel delta decreases visible x span"
    )
    controls.records[-1]["modality"] = "qt_wheel_event"


def mpl_point(canvas, axes, x, y):
    px, py = axes.transData.transform((x, y))
    ratio = canvas.device_pixel_ratio
    return QPoint(round(px / ratio), round(canvas.height() - py / ratio))


def sld_endpoints(view, kind):
    axes = view.axes
    line = next(line for line in axes.lines if line.get_label() == f"_{kind}_0")
    xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    if kind == "interface":
        y = sum(axes.get_ylim()) / 2
        return (float(xs[0]), y), (float(xs[0]) + (axes.get_xlim()[1] - float(xs[0])) / 2, y)
    if kind == "level":
        x = float(np.mean(xs))
        return (x, float(ys[0])), (x, float(ys[0]) * 0.95)
    return (float(xs[-1]), float(ys[0])), (float(xs[-1]) + 0.25, float(ys[0]))


@pytest.mark.parametrize("kind", ("interface", "level", "roughness"))
def test_sld_handle_drag_commits_structure(controls, tmp_path, kind):
    project = setup_project(tmp_path)
    # Separate the roughness tip from the interface; zero sigma overlaps them.
    structure = replace(project.datasets[0].structure, backing_roughness_a=3.0)
    window = controls.window(api.set_structure(project, "curve", structure))
    controls.named_click(window, "pipelineStep_结构")
    view = window.plot_panel.view("sld")
    view.canvas.draw()
    first, last = sld_endpoints(view, kind)
    before = window.document.project.datasets[0].structure
    drag(controls, view.canvas, mpl_point(view.canvas, view.axes, *first), mpl_point(view.canvas, view.axes, *last))
    after = window.document.project.datasets[0].structure
    assert after != before
    expected = {"interface": "thickness_a", "level": "density_scale"}
    if kind == "roughness":
        assert after.backing_roughness_a > before.backing_roughness_a
    else:
        assert getattr(after.components[0], expected[kind]) != getattr(before.components[0], expected[kind])
    controls.record(
        f"sld:{kind}Drag",
        f"drag displayed {kind} handle through Qt canvas",
        "the immutable structure is updated once on release",
    )
