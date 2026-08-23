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
    return {name: window.findChild(QDockWidget, name) for name in DOCK_NAMES}


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


def test_raising_a_tabbed_dock_does_not_restore_mid_notification(qtbot) -> None:
    """Selecting a tabbed dock must not rebuild the layout being changed.

    The analysis docks are tabbed, so raising one is what a user does by
    clicking its tab. That emits ``visibilityChanged`` on both the raised and
    the displaced dock, and capturing there replaces the project, whose
    ``project_changed`` handler calls ``restoreState``. Rebuilding every dock
    while Qt is still delivering its own dock notification tears down the
    widgets Qt is iterating, which crashes the process rather than raising.
    """
    window = _window(qtbot)
    window.show()
    restores: list[object] = []
    real_restore = window.restore_dock_layout
    window.restore_dock_layout = lambda project: (
        restores.append(project),
        real_restore(project),
    )[1]

    window.docks["resultsDock"].raise_()

    assert restores == []


def test_raising_a_tabbed_dock_still_persists_the_new_arrangement(qtbot) -> None:
    """Deferring the capture must not drop it: the tab change still persists."""
    window = _window(qtbot)
    window.show()
    assert window.docks["parametersDock"].isVisible() is True

    window.docks["resultsDock"].raise_()
    qtbot.waitUntil(lambda: window.document.project.ui_state.dock_state != "")

    assert window.docks["resultsDock"].isVisible() is True


AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _left_column(qtbot, tmp_path: Path, *, datasets: int, layers: int):
    """A window whose left column holds the requested dataset and layer counts.

    The project is handed to the document the window is built on, rather than
    swapped in afterwards, so the source-validation records the panels consult
    are populated the same way opening a project populates them.
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.new_project()
    for index in range(datasets):
        project = api.add_dataset(
            project,
            _write_curve(tmp_path / f"curve{index}.xy"),
            api.InstrumentSpec(instrument_id="left-column"),
        )
    if datasets and layers:
        structure = api.StructureSpec(
            AIR,
            tuple(api.LayerSpec(f"layer{index}", SIO2, 40.0, roughness_a=3.0) for index in range(layers)),
            SI,
        )
        project = api.set_structure(project, "curve0", structure)
    if datasets:
        project = api.select_active_dataset(project, "curve0")
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def test_dataset_list_height_follows_the_datasets_it_holds(
    qtbot,
    tmp_path: Path,
) -> None:
    """One dataset must not ask for the height that ten datasets need.

    A default scrolling view reports a fixed 192px whatever it holds, so the
    two stacked left-column panels handed Qt identical requests and were given
    identical shares -- see VISIBLE_ROW_FLOOR in results/candidates.py for the
    same defect fixed in the result dock.
    """
    one = _left_column(qtbot, tmp_path / "one", datasets=1, layers=0)
    many = _left_column(qtbot, tmp_path / "many", datasets=10, layers=0)

    assert one.data_panel.tree.sizeHint().height() < many.data_panel.tree.sizeHint().height()


def test_structure_list_height_follows_the_layers_it_holds(
    qtbot,
    tmp_path: Path,
) -> None:
    two = _left_column(qtbot, tmp_path / "two", datasets=1, layers=2)
    twelve = _left_column(qtbot, tmp_path / "twelve", datasets=1, layers=12)

    assert two.structure_panel.editor.tree.sizeHint().height() < twelve.structure_panel.editor.tree.sizeHint().height()


def test_one_dataset_does_not_claim_as_much_height_as_a_twelve_layer_stack(
    qtbot,
    tmp_path: Path,
) -> None:
    """The reported complaint: a two-row dataset table crowding out the layers."""
    window = _left_column(qtbot, tmp_path, datasets=1, layers=12)

    datasets = window.data_panel.tree.sizeHint().height()
    layers = window.structure_panel.editor.tree.sizeHint().height()
    assert datasets < layers, f"dataset list asks {datasets}px, layer list {layers}px"
