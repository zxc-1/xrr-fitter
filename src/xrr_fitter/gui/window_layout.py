"""Stable three-column MainWindow widget and action assembly."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from xrr_fitter.gui.command_icons import command_icon
from xrr_fitter.gui.data.panel import DataPanel
from xrr_fitter.gui.fitting.panel import FitPanel
from xrr_fitter.gui.guidance.panel import GuidancePanel
from xrr_fitter.gui.parameters.panel import ParametersPanel
from xrr_fitter.gui.plots.panel import PlotPanel
from xrr_fitter.gui.project.actions import ProjectActions
from xrr_fitter.gui.results.panel import ResultsPanel
from xrr_fitter.gui.structure.panel import StructurePanel

WORKFLOW_ACTION_SPECS = (
    ("startFitAction", "一键拟合", "Ctrl+Return", "start_fit"),
    ("cancelFitAction", "取消拟合", "Esc", "cancel_fit"),
    ("exportResultsAction", "导出结果", "Ctrl+Shift+E", "export_results_dialog"),
)


DOCK_SPECS = (
    ("dataDock", "数据集", "数据集面板", Qt.DockWidgetArea.LeftDockWidgetArea),
    ("structureDock", "样品结构", "样品结构面板", Qt.DockWidgetArea.LeftDockWidgetArea),
    ("parametersDock", "参数", "拟合参数面板", Qt.DockWidgetArea.RightDockWidgetArea),
    ("fitDock", "拟合", "拟合控制面板", Qt.DockWidgetArea.RightDockWidgetArea),
    ("resultsDock", "结果", "拟合结果面板", Qt.DockWidgetArea.RightDockWidgetArea),
)

DOCK_FEATURES = (
    QDockWidget.DockWidgetFeature.DockWidgetMovable
    | QDockWidget.DockWidgetFeature.DockWidgetFloatable
    | QDockWidget.DockWidgetFeature.DockWidgetClosable
)


def _dock(name: str, title: str, accessible: str, inner: QWidget) -> QDockWidget:
    dock = QDockWidget(title)
    dock.setObjectName(name)
    dock.setAccessibleName(accessible)
    dock.setFeatures(DOCK_FEATURES)
    dock.setAllowedAreas(
        Qt.DockWidgetArea.LeftDockWidgetArea
        | Qt.DockWidgetArea.RightDockWidgetArea
        | Qt.DockWidgetArea.BottomDockWidgetArea
    )
    dock.setWidget(inner)
    return dock


def _scrolled(name: str, inner: QWidget) -> QScrollArea:
    """Wrap a panel so a narrow dock scrolls instead of clipping."""
    scroll = QScrollArea()
    scroll.setObjectName(name)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(inner)
    return scroll


def _guidance_actions(window: object) -> dict:
    """Bind each guided step's action to the panel that already performs it.

    Guidance never reimplements a workflow; it routes to the same buttons and
    methods the expert surface uses, so both paths share one code path.
    """
    return {
        "import_files": window.data_panel.import_files_button.click,
        "initialize_structure": window.structure_panel.initialize_button.click,
        "start_fit": window.start_fit,
    }


def install_workspace(window: object, document: object) -> None:
    """Build the plot-centred workspace with one dock per side panel."""
    window.project_actions = ProjectActions(window, document)
    window.data_panel = DataPanel(document)
    window.structure_panel = StructurePanel(document)
    window.parameters_panel = ParametersPanel(document)
    window.fit_panel = FitPanel(document)
    window.result_panel = ResultsPanel(document)
    window.plot_panel = PlotPanel()
    window.export_button = QPushButton("导出结果")
    window.export_button.setObjectName("exportResultsButton")
    window.export_button.setAccessibleName("导出拟合结果")
    window.export_button.setToolTip("将当前项目的拟合结果导出到所选目录")
    window.export_button.setIcon(command_icon("export_results_dialog"))
    window.export_button.clicked.connect(window.export_results_dialog)
    widgets = {
        "dataDock": window.data_panel,
        "structureDock": window.structure_panel,
        "parametersDock": window.parameters_panel,
        "fitDock": window.fit_panel,
        "resultsDock": window.result_panel,
    }
    window.docks = {}
    for name, title, accessible, area in DOCK_SPECS:
        inner = widgets[name]
        dock = _dock(name, title, accessible, _scrolled(f"{name}Scroll", inner))
        window.addDockWidget(area, dock)
        window.docks[name] = dock
    # Guidance and the plot share the central slot: they are two surfaces onto
    # one project, so a stack swaps them without either being rebuilt.
    window.guidance = GuidancePanel(document, _guidance_actions(window))
    window.central_stack = QStackedWidget()
    window.central_stack.setObjectName("centralStack")
    window.central_stack.addWidget(window.guidance)
    window.central_stack.addWidget(window.plot_panel)
    window.setCentralWidget(window.central_stack)
    window.setDockOptions(
        QMainWindow.DockOption.AnimatedDocks
        | QMainWindow.DockOption.AllowNestedDocks
        | QMainWindow.DockOption.AllowTabbedDocks
    )
    # Stacking parameters, fit, and results vertically gives each a third of the
    # height, and all three then overflow at the documented minimum size. Tabbing
    # them gives whichever is in front the full height; a user who wants two at
    # once can now drag one out, which is the point of a dock layout.
    window.tabifyDockWidget(window.docks["parametersDock"], window.docks["fitDock"])
    window.tabifyDockWidget(window.docks["fitDock"], window.docks["resultsDock"])
    window.docks["parametersDock"].raise_()
    window.resizeDocks(
        [window.docks["dataDock"], window.docks["parametersDock"]],
        [320, 380],
        Qt.Orientation.Horizontal,
    )
    window._default_dock_state = window.saveState()


def install_workflow_actions(window: object) -> None:
    window._workflow_actions = {}
    for object_name, text, shortcut, callback_name in WORKFLOW_ACTION_SPECS:
        action = QAction(text, window)
        action.setObjectName(object_name)
        action.setShortcut(QKeySequence(shortcut))
        action.setToolTip(text)
        action.setStatusTip(text)
        action.setIcon(command_icon(callback_name))
        callback = getattr(window, callback_name)
        action.triggered.connect(lambda _checked=False, operation=callback: operation())
        window.addAction(action)
        window._workflow_actions[object_name] = action
