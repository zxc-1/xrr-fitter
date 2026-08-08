"""Dockable panel workspace.

The fixed three-column split could not adapt to "this step is only about the
structure" or "I just want a big plot". Each side panel is now a QDockWidget the
user can move, float, stack, or hide, with the plot as the central widget
because it is the workspace's subject rather than one panel among peers.

A restored layout is untrusted: saveState() bytes belong to a particular Qt
build, so anything unreadable falls back to the default arrangement instead of
refusing to open the project.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDockWidget

import xrr_fitter.api as api

DOCK_NAMES = (
    "dataDock",
    "structureDock",
    "parametersDock",
    "fitDock",
    "resultsDock",
)


def _window(qtbot):
    """Build a window on the expert surface, where the docks are the subject.

    Guidance is the opening surface and deliberately hides the docks, so these
    contracts switch to the expert surface first.
    """
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def _docks(window) -> dict[str, QDockWidget]:
    return {
        name: window.findChild(QDockWidget, name) for name in DOCK_NAMES
    }


def test_every_side_panel_is_a_dock(qtbot) -> None:
    window = _window(qtbot)

    for name, dock in _docks(window).items():
        assert dock is not None, name


def test_the_plot_is_the_central_widget_not_a_dock(qtbot) -> None:
    """The plot is the subject of the workspace, so it is never dockable."""
    window = _window(qtbot)

    assert window.central_stack.currentWidget() is window.plot_panel
    assert window.findChild(QDockWidget, "plotDock") is None


def test_docks_are_movable_floatable_and_closable(qtbot) -> None:
    window = _window(qtbot)

    for name, dock in _docks(window).items():
        features = dock.features()
        assert features & QDockWidget.DockWidgetFeature.DockWidgetMovable, name
        assert features & QDockWidget.DockWidgetFeature.DockWidgetFloatable, name
        assert features & QDockWidget.DockWidgetFeature.DockWidgetClosable, name


def test_view_menu_offers_one_toggle_per_dock(qtbot) -> None:
    window = _window(qtbot)

    for name, dock in _docks(window).items():
        action = window.chrome_actions[f"dockToggle:{name}"]
        assert action is dock.toggleViewAction(), name
        assert action.isCheckable() is True


def test_hiding_a_dock_leaves_the_others_alone(qtbot) -> None:
    window = _window(qtbot)
    window.show()
    docks = _docks(window)

    docks["resultsDock"].hide()

    assert docks["resultsDock"].isVisibleTo(window) is False
    assert docks["dataDock"].isVisibleTo(window) is True


def test_reset_layout_restores_every_dock(qtbot) -> None:
    window = _window(qtbot)
    window.show()
    docks = _docks(window)
    docks["resultsDock"].hide()
    docks["fitDock"].hide()

    window.reset_dock_layout()

    for name, dock in docks.items():
        assert dock.isVisibleTo(window) is True, name


def test_layout_round_trips_through_a_saved_project(qtbot, tmp_path: Path) -> None:
    window = _window(qtbot)
    window.show()
    _docks(window)["resultsDock"].hide()
    window.capture_dock_layout()
    state = window.document.project.ui_state.dock_state
    assert state

    target = tmp_path / "layout.xrrproj.json"
    api.save_project(window.document.project, target)
    reopened = _window(qtbot)
    reopened.document.open(target)

    assert reopened.document.project.ui_state.dock_state == state
    assert _docks(reopened)["resultsDock"].isVisibleTo(reopened) is False


def test_unreadable_dock_state_falls_back_to_the_default_layout(qtbot) -> None:
    """A byte string from another Qt build must not make a project unopenable."""
    window = _window(qtbot)
    window.show()

    corrupt = api.set_dock_state(window.document.project, "!!!not-base64!!!")
    window.document.replace_project(corrupt)

    for name, dock in _docks(window).items():
        assert dock.isVisibleTo(window) is True, name


def test_empty_dock_state_leaves_the_default_layout_untouched(qtbot) -> None:
    window = _window(qtbot)
    window.show()

    assert window.document.project.ui_state.dock_state == ""
    for name, dock in _docks(window).items():
        assert dock.isVisibleTo(window) is True, name
