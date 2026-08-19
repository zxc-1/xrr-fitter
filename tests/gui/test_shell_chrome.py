"""Shell chrome contract: menus, toolbar, status bar, theme, and empty states."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMenuBar,
    QPushButton,
    QStatusBar,
    QToolBar,
    QWidget,
)

import xrr_fitter.api as api


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_main_window_has_menu_bar_with_workflow_menus(qtbot) -> None:
    window = _window(qtbot)

    bar = window.findChild(QMenuBar, "mainMenuBar")
    assert bar is not None
    titles = [action.text() for action in bar.actions()]
    assert titles == ["文件", "视图", "拟合", "帮助"]

    file_menu = bar.actions()[0].menu()
    names = [action.objectName() for action in file_menu.actions() if not action.isSeparator()]
    for required in (
        "newProjectAction",
        "openProjectAction",
        "saveProjectAction",
        "saveAsProjectAction",
        "importFilesMenuAction",
        "importFolderMenuAction",
        "reloadSourceAction",
        "relinkSourceAction",
        "exportResultsAction",
    ):
        assert required in names


def test_toolbar_hosts_project_commands_and_export(qtbot) -> None:
    window = _window(qtbot)

    toolbar = window.findChild(QToolBar, "mainToolbar")
    assert toolbar is not None
    assert not toolbar.isMovable()
    new_button = window.findChild(QPushButton, "newProjectButton")
    export_button = window.findChild(QPushButton, "exportResultsButton")
    assert toolbar.isAncestorOf(new_button)
    assert toolbar.isAncestorOf(export_button)


def test_status_bar_reports_readiness_and_active_dataset(qtbot) -> None:
    window = _window(qtbot)

    bar = window.findChild(QStatusBar, "mainStatusBar")
    assert bar is not None
    readiness = window.findChild(QLabel, "fitReadinessStatus")
    dataset = window.findChild(QLabel, "activeDatasetStatus")
    assert readiness is not None and dataset is not None
    assert "导入" in readiness.text()
    assert dataset.text() == "无活动数据集"


def test_view_menu_switches_plot_views_and_syncs_expert_mode(qtbot) -> None:
    window = _window(qtbot)

    bar = window.findChild(QMenuBar, "mainMenuBar")
    view_menu = bar.actions()[1].menu()
    by_name = {action.objectName(): action for action in view_menu.actions()}

    log_action = by_name["plotViewAction:log"]
    log_action.trigger()
    assert window.plot_panel.current_view_key() == "log"

    expert_action = by_name["expertModeAction"]
    assert expert_action.isCheckable()
    expert_action.setChecked(True)
    assert window.parameters_panel.expert_toggle.isChecked()
    window.parameters_panel.expert_toggle.setChecked(False)
    assert not expert_action.isChecked()


def test_view_menu_groups_cover_every_diagnostic_tab() -> None:
    """The menu's grouping is hand-written, so a new tab must be added to it.

    ``_sync_view_actions`` looks up an action for every ``TAB_SPECS`` entry, so a
    tab missing from the grouping raises ``KeyError`` on the next view change
    rather than merely going unlisted in the menu.
    """
    from xrr_fitter.gui.chrome import VIEW_GROUPS
    from xrr_fitter.gui.plots.diagnostics import TAB_SPECS

    grouped = tuple(key for _title, keys in VIEW_GROUPS for key in keys)

    assert sorted(grouped) == sorted(key for key, _title, _description in TAB_SPECS)
    assert len(grouped) == len(set(grouped))


def test_theme_applies_idempotent_application_stylesheet(qtbot) -> None:
    from xrr_fitter.gui.theme import apply_theme

    application = QApplication.instance()
    previous = application.styleSheet()
    try:
        first = apply_theme(application)
        assert first
        assert application.styleSheet() == first
        assert apply_theme(application) == first
    finally:
        application.setStyleSheet(previous)


def test_plot_panel_shows_import_guidance_until_data_arrives(qtbot) -> None:
    from tests.support.model_cases import prepared_data

    window = _window(qtbot)
    panel = window.plot_panel

    empty_state = panel.findChild(QWidget, "plotEmptyState")
    assert empty_state is not None
    assert empty_state.isVisibleTo(panel)
    assert not panel.tabs.isVisibleTo(panel)

    panel.set_dataset("sample", prepared_data())

    assert not empty_state.isVisibleTo(panel)
    assert panel.tabs.isVisibleTo(panel)


def test_empty_state_import_button_opens_import_dialog(qtbot, monkeypatch) -> None:
    window = _window(qtbot)
    calls: list[str] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: (calls.append("opened"), ([], ""))[1],
    )

    button = window.findChild(QPushButton, "emptyStateImportButton")
    assert button is not None
    button.click()

    assert calls == ["opened"]


def test_open_project_activates_first_dataset_when_none_selected(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "a.xy"), api.InstrumentSpec())
    project = api.add_dataset(project, _write_curve(tmp_path / "b.xy"), api.InstrumentSpec())
    project = api.select_active_dataset(project, None)
    target = tmp_path / "project.xrrproj.json"
    api.save_project(project, target)

    document = ProjectDocument()
    document.open(target)

    assert document.active_dataset_id == "a"
    assert not document.is_dirty


def test_fit_progress_view_is_hidden_while_idle(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "c.xy"), api.InstrumentSpec())
    panel = FitPanel(ProjectDocument(project))
    qtbot.addWidget(panel)

    assert not panel.progress_view.isVisibleTo(panel)

    panel._project_running_state(True)
    assert panel.progress_view.isVisibleTo(panel)

    panel._project_running_state(False)
    assert not panel.progress_view.isVisibleTo(panel)


def test_dataset_tree_presents_compact_columns_with_details_label(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "d.xy"), api.InstrumentSpec())
    panel = DataPanel(ProjectDocument(project))
    qtbot.addWidget(panel)

    tree = panel.tree
    assert not tree.isColumnHidden(0)
    assert not tree.isColumnHidden(4)
    for column in (1, 2, 3, 5):
        assert tree.isColumnHidden(column)

    details = panel.findChild(QLabel, "datasetDetails")
    assert details is not None
    assert "d.xy" in details.text()
    assert panel.sha256_text("d")[:12] in details.text()


def test_readiness_is_reported_once_across_status_bar_and_fit_panel(
    qtbot,
    tmp_path,
) -> None:
    """The same readiness verdict must not occupy two labels at once.

    The status bar owns the persistent readiness report. The fit panel's label
    is for operation outcomes (cancelled, failed, finished), so while idle and
    ready it stays quiet instead of restating what the status bar already says.
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.new_project()
    project = api.add_dataset(
        project,
        _write_curve(tmp_path / "e.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.20), 40.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    project = api.set_structure(project, "e", structure)
    project = api.set_expert_mode(project, True)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.show()

    bar_text = window.findChild(QLabel, "fitReadinessStatus").text()
    assert "就绪" in bar_text
    assert window.fit_panel.status_text() != bar_text
