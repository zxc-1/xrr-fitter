"""Movable inner splits, fixed rails, and overflow controls at Qt input boundaries."""

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QScrollArea, QSplitter
from tests.gui.control_operation_driver import setup_project
from tests.gui.control_readonly_helpers import scroll_roundtrip
from tests.gui.test_control_operations_plots import drag
from tests.gui.test_control_operations_results import result_fixture
from tests.gui.test_fit_progress_layout import _joint_view


def sized_window(driver, tmp_path, *, layers=1):
    window = driver.window(setup_project(tmp_path, layers=layers))
    window.resize(1272, 693)
    driver.qtbot.waitExposed(window)
    return window


def test_fixed_rails_reject_drag_and_layout_menu_resets_saved_widths(controls, tmp_path):
    window = sized_window(controls, tmp_path)
    splitter = window.workspace_splitter
    before = splitter.sizes()
    handle = splitter.handle(1)
    assert not handle.isEnabled()
    drag(controls, handle, handle.rect().center(), handle.rect().center() + QPoint(100, 0))
    assert splitter.sizes() == before
    left = controls.widget(window, "leftSplitter", QSplitter)
    assert left.handleWidth() == 0
    controls.record(
        ("workspaceSplitter", "leftSplitter"),
        "attempt outer-rail handle drag; inspect content-sized zero-width inner handle",
        "outer widths stay fixed; left rail has no user drag affordance",
    )
    # Persisted/custom geometry is fixture setup, not a claimed drag operation.
    splitter.setSizes((300, 500, 380))
    controls.menu(window, "resetLayoutAction")
    assert splitter.sizes() == before
    controls.record(
        "resetLayoutAction",
        "View menu Reset Layout after custom geometry setup",
        "documented fixed column widths return",
    )


@pytest.mark.parametrize("name,step", (("canvasSplitter", "结构"), ("plotSplitter", "结果")))
def test_inner_splitter_handle_moves_rendered_sections(controls, tmp_path, name, step):
    project = result_fixture(tmp_path) if name == "plotSplitter" else setup_project(tmp_path)
    window = controls.window(project)
    controls.named_click(window, f"pipelineStep_{step}")
    splitter = controls.widget(window, name, QSplitter)
    indexes = [i for i in range(1, splitter.count()) if splitter.handle(i).isVisible()]
    assert indexes
    handle = splitter.handle(indexes[-1])
    # The layer stack's floor follows its content, so the handle may already sit on its
    # lower bound -- it does wherever the fonts make the stack exactly as tall as that
    # floor.  Drag toward whichever side the splitter itself says has room.
    low, high = splitter.getRange(indexes[-1])
    room_up, room_down = handle.y() - low, high - handle.y()
    assert max(room_up, room_down) > 0, f"{name} handle has no travel: {(low, high)} at {handle.y()}"
    offset = -min(35, room_up) if room_up >= room_down else min(35, room_down)
    before = splitter.sizes()
    drag(controls, handle, handle.rect().center(), handle.rect().center() + QPoint(0, offset))
    assert splitter.sizes() != before
    assert sum(splitter.sizes()) == sum(before)
    controls.record(
        name,
        f"drag visible {name} handle {'upward' if offset < 0 else 'downward'} toward its free side",
        "section heights change without losing total available height",
    )


@pytest.mark.parametrize("font_pixels", (12, 18))
def test_structural_and_inspector_overflow_is_reachable(controls, tmp_path, font_pixels):
    window = sized_window(controls, tmp_path, layers=20)
    font = window.font()
    font.setPixelSize(font_pixels)
    window.setFont(font)
    controls.named_click(window, "pipelineStep_结构")
    controls.qtbot.wait(10)
    for name, identity in (("structurePanelScroll", "structurePaneContents"), ("inspectorScroll", "inspectorScroll")):
        area = controls.widget(window, name, QScrollArea)
        if area.verticalScrollBar().maximum() > 0:
            scroll_roundtrip(controls, area, identity)
        else:
            assert area.widget().height() <= area.viewport().height()
            assert area.widget().height() >= area.widget().minimumSizeHint().height()
            controls.record(
                identity,
                "inspect fully visible content at current font size",
                "no scrolling needed: content fits without compression",
                setup=f"window 1272x693, font {font_pixels}px; geometry evidence, not a scroll operation",
            )


def test_progress_card_scrolls_to_its_last_stage(controls):
    view = _joint_view(controls.qtbot, 640, 220, font_pixels=16)
    area = controls.widget(view, "fitProgressScroll", QScrollArea)
    card = area.widget()
    chrome_height = view.height() - area.viewport().height()
    view.resize(view.width(), min(view.height(), chrome_height + card.minimumSizeHint().height() - 1))
    controls.qtbot.waitUntil(lambda: area.verticalScrollBar().isVisible() and area.verticalScrollBar().maximum() > 0)
    assert card.height() >= card.minimumSizeHint().height()
    scroll_roundtrip(controls, area, "fitProgressScroll")
