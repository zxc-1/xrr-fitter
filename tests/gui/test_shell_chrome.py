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


def test_dataset_details_fields_each_stay_on_one_line(qtbot, tmp_path) -> None:
    from math import asin, degrees

    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    # A named instrument with geometry footprint needs 434 px on one line, so in the
    # ~320 px dock the field wrapped.  Chinese permits a break between any two
    # characters, so the break landed mid-word and "分辨率 q" rendered as "分辨" over
    # "率 q" - the unit split away from the quantity it belongs to.
    instrument = api.InstrumentSpec(
        instrument_id="Rigaku SmartLab",
        footprint_mode="geometry",
        sample_length_mm=10.0,
        beam_width_mm=0.1,
        footprint_spill_angle_deg=degrees(asin(0.1 / 10.0)),
    )
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "d.xy"), instrument)
    panel = DataPanel(ProjectDocument(project))
    qtbot.addWidget(panel)

    details = panel.findChild(QLabel, "datasetDetails")

    # Narrowing the label to dock width must not buy it another line: one field
    # that no longer fits has to be shortened, never broken across two lines.
    assert details.heightForWidth(320) == details.heightForWidth(1200)
    assert not details.wordWrap()


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


def _fitted_window(qtbot, tmp_path, confidence, objective):
    """A window whose one dataset already carries a persisted fit result."""
    from dataclasses import replace

    import numpy as np
    from tests.support.model_cases import fit_candidate, fit_result

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "q.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.20), 40.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    project = api.set_structure(project, "q", structure)
    # The per-point arrays must match the curve written above, which the plot
    # panel validates before it will project the result at all.
    size = 64
    candidate = replace(
        fit_candidate("candidate-a", objective),
        qz_a_inv=np.linspace(0.015, 0.25, size),
        model_normalized=np.geomspace(0.9, 2e-5, size),
        log_residuals_decades=np.full(size, 0.1),
        weighted_residuals=np.zeros(size),
    )
    result = api.FitResult.from_search(
        fit_result(candidate),
        confidence=confidence,
        uncertainty=None,
        classification_evidence=(),
    )
    project = replace(project, datasets=(replace(project.datasets[0], last_valid_result=result),))
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.show()
    return window


def test_status_bar_reports_the_quality_of_the_fit_that_already_ran(qtbot, tmp_path) -> None:
    """Readiness answers "can I start", never "was the last answer any good".

    A finished fit whose confidence is 不可信 left the bar still reading 已就绪，
    which is true about starting again and misleading about what is on screen.
    The verdict and its objective therefore get their own label.
    """
    window = _fitted_window(qtbot, tmp_path, api.ConfidenceClass.UNTRUSTED, 12.5)

    quality = window.findChild(QLabel, "fitQualityStatus")
    assert quality is not None
    assert "不可信" in quality.text()
    assert "12.5" in quality.text()
    # The readiness label keeps answering only its own question.
    assert "就绪" in window.findChild(QLabel, "fitReadinessStatus").text()


def test_quality_status_carries_the_semantic_kind_matching_its_verdict(qtbot, tmp_path) -> None:
    """An untrusted result must not be painted in the same colour as a trusted one."""
    untrusted = _fitted_window(qtbot, tmp_path / "a", api.ConfidenceClass.UNTRUSTED, 12.5)
    trusted = _fitted_window(qtbot, tmp_path / "b", api.ConfidenceClass.TRUSTED, 0.02)

    assert untrusted.findChild(QLabel, "fitQualityStatus").property("statusKind") == "error"
    assert trusted.findChild(QLabel, "fitQualityStatus").property("statusKind") == "ok"


def test_quality_status_stays_empty_until_a_fit_has_run(qtbot) -> None:
    """With no result there is no verdict, so the label claims nothing."""
    window = _window(qtbot)

    assert window.findChild(QLabel, "fitQualityStatus").text() == ""


# Each command that chrome surfaces twice, with the toolbar widget and the menu
# entry that run the same callback.  The two objects are separate instances by
# construction, so nothing but a shared icon registry keeps them looking alike.
COMMAND_SURFACES = (
    ("新建项目", "newProjectButton", "newProjectAction"),
    ("打开项目", "openProjectButton", "openProjectAction"),
    ("保存项目", "saveProjectButton", "saveProjectAction"),
    ("项目另存为", "saveAsProjectButton", "saveAsProjectAction"),
    ("重新加载数据源", "reloadSourceButton", "reloadSourceAction"),
    ("重新链接数据源", "relinkSourceButton", "relinkSourceAction"),
    ("导出结果", "exportResultsButton", "exportResultsAction"),
)


def _menu_action(window, object_name):
    """The menu's action for a command, wherever chrome parked it."""
    for action in window.actions():
        if action.objectName() == object_name:
            return action
    action = window.chrome_actions.get(object_name)
    assert action is not None, f"missing menu action: {object_name}"
    return action


def _icon_bytes(icon) -> bytes:
    """The rendered glyph, so two icons compare by what the user sees."""
    return bytes(icon.pixmap(16, 16).toImage().constBits())


def test_each_command_wears_one_icon_in_the_toolbar_and_the_menu(qtbot) -> None:
    """A command must look the same everywhere it is offered.

    The toolbar drives the project commands through QPushButton while the menu
    drives them through QAction, so "新建" was a labelled icon on the toolbar and
    bare text in the menu - the user has to learn the command twice.  Hanging the
    icon off the widget is what allows the two to drift; it belongs to the command.
    """
    window = _window(qtbot)

    faults: list[str] = []
    for label, button_name, action_name in COMMAND_SURFACES:
        button = window.findChild(QPushButton, button_name)
        assert button is not None, f"{label}: 缺少工具栏按钮 {button_name}"
        action = _menu_action(window, action_name)
        toolbar_bare = button.icon().isNull()
        menu_bare = action.icon().isNull()
        if toolbar_bare:
            faults.append(f"{label}: 工具栏无图标")
        if menu_bare:
            faults.append(f"{label}: 菜单无图标")
        if not toolbar_bare and not menu_bare and _icon_bytes(button.icon()) != _icon_bytes(action.icon()):
            faults.append(f"{label}: 工具栏与菜单图标不一致")

    assert faults == []


def test_fit_commands_carry_their_icon_into_the_menu(qtbot) -> None:
    """The fit commands are one QAction per command; keep it that way."""
    window = _window(qtbot)

    for object_name in ("startFitAction", "cancelFitAction"):
        assert not _menu_action(window, object_name).icon().isNull(), object_name


def test_no_two_commands_wear_the_same_icon(qtbot) -> None:
    """A shared glyph makes two commands one, which is the opposite of the point."""
    window = _window(qtbot)

    seen: dict[bytes, str] = {}
    for label, _button_name, action_name in (
        *COMMAND_SURFACES,
        ("一键拟合", "", "startFitAction"),
        ("取消拟合", "", "cancelFitAction"),
    ):
        icon = _menu_action(window, action_name).icon()
        if icon.isNull():
            continue
        glyph = _icon_bytes(icon)
        assert glyph not in seen, f"{label} 与 {seen[glyph]} 用了同一个图标"
        seen[glyph] = label


def test_every_registered_command_wears_a_distinct_icon(qtbot) -> None:
    """The registry is the source of truth, so no two commands may share a glyph.

    The menu-level test above only sees the commands chrome surfaces twice; a
    command with no toolbar twin (导入文件夹, 强制停止) could still collide and go
    unnoticed.  Worse, two QStyle pixmaps the native style lacks fall back to the
    same generic glyph, so "SP_DirLinkIcon != SP_DirOpenIcon" is true of the enums
    yet false of what renders.  Distinctness therefore has to be asserted against
    the rendered bytes of every registered command, not against the enum values.
    """
    from xrr_fitter.gui.command_icons import COMMAND_PIXMAPS, command_icon

    _window(qtbot)  # a styled QApplication, as every other icon test relies on
    seen: dict[bytes, str] = {}
    for command in COMMAND_PIXMAPS:
        icon = command_icon(command)
        assert not icon.isNull(), f"{command}: 命令无图标"
        glyph = _icon_bytes(icon)
        assert glyph not in seen, f"{command} 与 {seen[glyph]} 用了同一个图标"
        seen[glyph] = command
