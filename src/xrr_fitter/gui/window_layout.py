"""Stable three-column MainWindow widget and action assembly."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from xrr_fitter.gui.data.panel import DataPanel
from xrr_fitter.gui.fitting.panel import FitPanel
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


def _column(name: str, label: str) -> QWidget:
    widget = QWidget()
    widget.setObjectName(name)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    heading = QLabel(label)
    heading.setObjectName(f"{name}Heading")
    heading.hide()
    layout.addWidget(heading)
    layout.addStretch(0)
    return widget


def _install_project_column(window: object, document: object) -> QWidget:
    project_column = _column("projectColumn", "项目与数据")
    project_layout = project_column.layout()
    window.project_actions = ProjectActions(window, document)
    window.data_panel = DataPanel(document)
    window.data_panel.layout().insertWidget(0, window.project_actions)
    window.structure_panel = StructurePanel(document)
    window.left_splitter = QSplitter(Qt.Orientation.Vertical)
    window.left_splitter.setObjectName("leftSplitter")
    window.left_splitter.addWidget(window.data_panel)
    window.left_splitter.addWidget(window.structure_panel)
    window.left_splitter.setSizes(list(document.project.ui_state.left_splitter_sizes))
    project_layout.insertWidget(project_layout.count() - 1, window.left_splitter, 1)
    project_column.setSizePolicy(
        QSizePolicy.Policy.Ignored,
        QSizePolicy.Policy.Expanding,
    )
    project_column.setMinimumWidth(320)
    return project_column


def _install_plot_column(window: object) -> QWidget:
    plot_column = _column("plotColumn", "反射率与 SLD")
    window.plot_panel = PlotPanel()
    plot_layout = plot_column.layout()
    plot_layout.insertWidget(plot_layout.count() - 1, window.plot_panel)
    return plot_column


def _install_analysis_column(window: object, document: object) -> QWidget:
    analysis_column = _column("analysisColumn", "参数与结果")
    window.parameters_panel = ParametersPanel(document)
    window.fit_panel = FitPanel(document)
    window.result_panel = ResultsPanel(document)
    window.export_button = QPushButton("导出结果")
    window.export_button.setObjectName("exportResultsButton")
    window.export_button.setAccessibleName("导出拟合结果")
    window.export_button.setToolTip("将当前项目的拟合结果导出到所选目录")
    window.export_button.clicked.connect(window.export_results_dialog)
    analysis_layout = analysis_column.layout()
    for widget in (
        window.parameters_panel,
        window.fit_panel,
        window.result_panel,
        window.export_button,
    ):
        analysis_layout.insertWidget(analysis_layout.count() - 1, widget)
    analysis_column.setSizePolicy(
        QSizePolicy.Policy.Ignored,
        QSizePolicy.Policy.Expanding,
    )
    analysis_column.setMinimumWidth(380)
    return analysis_column


def install_workspace(window: object, document: object) -> None:
    splitter = QSplitter(Qt.Orientation.Horizontal)
    window.workspace_splitter = splitter
    splitter.setObjectName("workspaceSplitter")
    splitter.setChildrenCollapsible(False)
    columns = (
        _install_project_column(window, document),
        _install_plot_column(window),
        _install_analysis_column(window, document),
    )
    for widget in columns:
        splitter.addWidget(widget)
    for index in range(splitter.count()):
        splitter.setCollapsible(index, False)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setStretchFactor(2, 0)
    splitter.setSizes(list(document.project.ui_state.workspace_splitter_sizes))
    window.setCentralWidget(splitter)


def install_workflow_actions(window: object) -> None:
    window._workflow_actions = {}
    for object_name, text, shortcut, callback_name in WORKFLOW_ACTION_SPECS:
        action = QAction(text, window)
        action.setObjectName(object_name)
        action.setShortcut(QKeySequence(shortcut))
        action.setToolTip(text)
        action.setStatusTip(text)
        callback = getattr(window, callback_name)
        action.triggered.connect(
            lambda _checked=False, operation=callback: operation()
        )
        window.addAction(action)
        window._workflow_actions[object_name] = action
