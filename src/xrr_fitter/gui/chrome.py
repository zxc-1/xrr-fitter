"""Menu bar, toolbar, and status bar assembly for the main window.

Chrome reuses the window's existing QAction and QPushButton objects so every
command keeps one identity across menus, the toolbar, keyboard shortcuts, and
accessibility metadata.  Only presentation lives here; workflows stay on the
window and its panels.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QLabel,
    QMenu,
    QMenuBar,
    QMessageBox,
    QStatusBar,
    QStyle,
    QToolBar,
    QWidget,
)

from xrr_fitter.gui import messages, theme
from xrr_fitter.gui.plots.diagnostics import TAB_SPECS

# The SLD profile is no longer a selectable view; it is a permanent companion
# pane, so the menu lists only the switchable diagnostic tabs.
# The view menu sections, which group the diagnostic tabs by the question they
# answer rather than by tab order. Every tab needs an entry here: the menu is
# the only way to reach a view once the tab bar scrolls, and the checked-state
# sync walks TAB_SPECS expecting to find an action for each key.
VIEW_GROUPS = (
    ("反射率", ("log", "raw", "qz4", "residual")),
    ("诊断", ("candidates", "residual_map", "parameter_map", "uncertainty", "trend")),
)

BUTTON_ICONS = (
    ("newProjectButton", QStyle.StandardPixmap.SP_FileIcon),
    ("openProjectButton", QStyle.StandardPixmap.SP_DirOpenIcon),
    ("saveProjectButton", QStyle.StandardPixmap.SP_DialogSaveButton),
    ("reloadSourceButton", QStyle.StandardPixmap.SP_BrowserReload),
    ("relinkSourceButton", QStyle.StandardPixmap.SP_FileLinkIcon),
)

ABOUT_TEXT = (
    "XRR Fitter\n\n"
    "X 射线反射率全自动拟合桌面应用。\n"
    "导入 .xy / .dat / .txt 反射率数据，初始化样品结构，"
    "一键拟合并导出结果。\n\n"
    "支持的 Python 边界：xrr_fitter.api"
)


def _window_action(window: QWidget, object_name: str) -> QAction:
    for action in window.actions():
        if action.objectName() == object_name:
            return action
    raise LookupError(f"missing window action: {object_name}")


def _action(
    window: QWidget,
    object_name: str,
    text: str,
    callback,
    *,
    checkable: bool = False,
) -> QAction:
    action = QAction(text, window)
    action.setObjectName(object_name)
    action.setToolTip(text)
    action.setStatusTip(text)
    action.setCheckable(checkable)
    if checkable:
        action.toggled.connect(callback)
    else:
        action.triggered.connect(lambda _checked=False: callback())
    return action


def install_chrome(window: QWidget) -> None:
    """Assemble toolbar, menus, and the status bar around existing commands."""
    window.chrome_actions = {}
    _install_toolbar(window)
    _install_menu_bar(window)
    _install_status_bar(window)
    _connect_view_sync(window)


def _install_toolbar(window: QWidget) -> None:
    toolbar = QToolBar("主工具栏", window)
    toolbar.setObjectName("mainToolbar")
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    style = window.style()
    for name, pixmap in BUTTON_ICONS:
        button = window.project_actions.button(name)
        button.setIcon(style.standardIcon(pixmap))
    toolbar.addWidget(window.project_actions)
    toolbar.addSeparator()
    start = _window_action(window, "startFitAction")
    cancel = _window_action(window, "cancelFitAction")
    start.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    cancel.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
    toolbar.addAction(start)
    toolbar.addAction(cancel)
    # Export sits directly after the fit commands.  An expanding spacer used to
    # push it to the far right edge, which left the middle of the toolbar empty
    # and put the button furthest from the actions that produce what it exports.
    toolbar.addSeparator()
    toolbar.addWidget(window.export_button)
    window.addToolBar(toolbar)


def _install_menu_bar(window: QWidget) -> None:
    bar = QMenuBar(window)
    bar.setObjectName("mainMenuBar")
    _install_file_menu(window, bar)
    _install_view_menu(window, bar)
    _install_fit_menu(window, bar)
    _install_help_menu(window, bar)
    window.setMenuBar(bar)


def _install_file_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("文件", bar)
    menu.setObjectName("fileMenu")
    for name in ("newProjectAction", "openProjectAction"):
        menu.addAction(_window_action(window, name))
    menu.addSeparator()
    for name in ("saveProjectAction", "saveAsProjectAction"):
        menu.addAction(_window_action(window, name))
    menu.addSeparator()
    specs = (
        (
            "importFilesMenuAction",
            "导入数据文件…",
            window.data_panel.import_files_button.click,
        ),
        (
            "importFolderMenuAction",
            "导入数据文件夹…",
            window.data_panel.import_folder_button.click,
        ),
        ("reloadSourceAction", "重新加载数据源", window.reload_source_dialog),
        ("relinkSourceAction", "重新链接数据源…", window.relink_source_dialog),
    )
    for object_name, text, callback in specs:
        action = _action(window, object_name, text, callback)
        window.chrome_actions[object_name] = action
        menu.addAction(action)
        if object_name in ("importFolderMenuAction", "relinkSourceAction"):
            menu.addSeparator()
    menu.addAction(_window_action(window, "exportResultsAction"))
    bar.addMenu(menu)


def _install_view_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("视图", bar)
    menu.setObjectName("viewMenu")
    group = QActionGroup(menu)
    group.setExclusive(True)
    for title, keys in VIEW_GROUPS:
        menu.addSection(title)
        for key in keys:
            menu.addAction(_view_action(window, group, key))
    _install_dock_actions(window, menu)
    menu.addSeparator()
    guidance = _action(
        window,
        "guidanceModeAction",
        "引导模式",
        window.set_guidance_visible,
        checkable=True,
    )
    guidance.setChecked(window.guidance_is_visible())
    window.chrome_actions["guidanceModeAction"] = guidance
    menu.addAction(guidance)
    menu.addSeparator()
    expert = _action(
        window,
        "expertModeAction",
        "专家模式",
        window.parameters_panel.expert_toggle.setChecked,
        checkable=True,
    )
    window.chrome_actions["expertModeAction"] = expert
    window.parameters_panel.expert_toggle.toggled.connect(expert.setChecked)
    expert.setChecked(window.parameters_panel.expert_toggle.isChecked())
    menu.addAction(expert)
    bar.addMenu(menu)


def _install_dock_actions(window: QWidget, menu: QMenu) -> None:
    """Expose each dock's own toggle plus a way back to the default layout.

    ``toggleViewAction`` is reused rather than wrapped so the menu check state
    and the dock's visibility can never disagree.
    """
    menu.addSection("面板")
    for name, dock in window.docks.items():
        action = dock.toggleViewAction()
        action.setObjectName(f"dockToggle:{name}")
        window.chrome_actions[f"dockToggle:{name}"] = action
        menu.addAction(action)
    reset = _action(
        window,
        "resetDockLayoutAction",
        "重置面板布局",
        window.reset_dock_layout,
    )
    window.chrome_actions["resetDockLayoutAction"] = reset
    menu.addAction(reset)


def _view_action(window: QWidget, group: QActionGroup, key: str) -> QAction:
    title = next(title for name, title, _description in TAB_SPECS if name == key)
    action = _action(
        window,
        f"plotViewAction:{key}",
        title,
        lambda view_key=key: _select_view(window, view_key),
    )
    action.setCheckable(True)
    group.addAction(action)
    window.chrome_actions[action.objectName()] = action
    return action


def _select_view(window: QWidget, key: str) -> None:
    try:
        window.plot_panel.select_view(key)
    except ValueError:
        window.statusBar().showMessage("该诊断视图仅在专家模式下可用", 4000)
        _sync_view_actions(window)


def _install_fit_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("拟合", bar)
    menu.setObjectName("fitMenu")
    cancel = _window_action(window, "cancelFitAction")
    menu.addAction(_window_action(window, "startFitAction"))
    menu.addAction(cancel)
    force = _action(
        window,
        "forceStopFitAction",
        "强制停止拟合",
        window.fit_panel.controller.force_stop,
    )
    force.setEnabled(cancel.isEnabled())
    cancel.changed.connect(lambda: force.setEnabled(cancel.isEnabled()))
    window.chrome_actions["forceStopFitAction"] = force
    menu.addAction(force)
    bar.addMenu(menu)


def _install_help_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("帮助", bar)
    menu.setObjectName("helpMenu")
    about = _action(
        window,
        "aboutAction",
        "关于 XRR Fitter",
        lambda: QMessageBox.about(window, "关于 XRR Fitter", ABOUT_TEXT),
    )
    window.chrome_actions["aboutAction"] = about
    menu.addAction(about)
    bar.addMenu(menu)


def _install_status_bar(window: QWidget) -> None:
    bar = QStatusBar(window)
    bar.setObjectName("mainStatusBar")
    bar.setSizeGripEnabled(False)
    readiness = QLabel(bar)
    readiness.setObjectName("fitReadinessStatus")
    dataset = QLabel(bar)
    dataset.setObjectName("activeDatasetStatus")
    source = window.project_actions.source_status_label
    bar.addWidget(readiness, 1)
    bar.addPermanentWidget(dataset)
    bar.addPermanentWidget(source)
    window.setStatusBar(bar)


def _connect_view_sync(window: QWidget) -> None:
    window.plot_panel.view_changed.connect(lambda _index: _sync_view_actions(window))
    window.parameters_panel.expert_toggle.toggled.connect(lambda _checked: _sync_view_actions(window))
    _sync_view_actions(window)


def _sync_view_actions(window: QWidget) -> None:
    current = window.plot_panel.current_view_key()
    for key, _title, _description in TAB_SPECS:
        action = window.chrome_actions[f"plotViewAction:{key}"]
        action.setChecked(key == current)


def _active_dataset_text(window: QWidget) -> str:
    dataset_id = window.document.active_dataset_id
    if dataset_id is None:
        return "无活动数据集"
    dataset = next(
        (value for value in window.document.project.datasets if value.dataset_id == dataset_id),
        None,
    )
    if dataset is None:
        return "无活动数据集"
    return f"数据集：{dataset.display_name or dataset.dataset_id}"


def refresh_status(window: QWidget, readiness: object) -> None:
    """Project readiness, active dataset, and source-action enablement."""
    label = window.findChild(QLabel, "fitReadinessStatus")
    if label is None:
        return
    label.setText(messages.readiness_text(readiness.message))
    theme.set_status_kind(label, "ok" if readiness.ready else "warn")
    dataset_label = window.findChild(QLabel, "activeDatasetStatus")
    dataset_label.setText(_active_dataset_text(window))
    has_active = window.document.active_dataset_id is not None
    for name in ("reloadSourceAction", "relinkSourceAction"):
        window.chrome_actions[name].setEnabled(has_active)
