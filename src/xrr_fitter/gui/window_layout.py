"""Stable three-column MainWindow widget and action assembly."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
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


def _column(name: str) -> QWidget:
    widget = QWidget()
    widget.setObjectName(name)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    return widget


class AnalysisSection(QFrame):
    """A titled analysis card whose header toggles its body.

    Three permanently expanded cards overflowed the analysis column at the
    documented minimum window size, leaving the lower ones reachable only by
    scrolling. Collapsing is per section rather than mutually exclusive, because
    reading fit progress against the parameter table needs two open at once.
    """

    def __init__(self, title: str, name: str, inner: QWidget) -> None:
        super().__init__()
        self.setObjectName(name)
        self.setProperty("sectionCard", True)
        self.body = inner
        self.toggle = QToolButton()
        self.toggle.setObjectName(f"{name}Header")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setAccessibleName(f"{title}分区")
        self.toggle.setToolTip(f"展开或折叠{title}分区")
        self.toggle.setProperty("sectionHeader", True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.toggle.toggled.connect(self.set_expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.toggle)
        layout.addWidget(inner)

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the body, keeping the header reachable either way."""
        if self.toggle.isChecked() != expanded:
            self.toggle.setChecked(expanded)
            return
        self.body.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


def _install_project_column(window: object, document: object) -> QWidget:
    project_column = _column("projectColumn")
    window.project_actions = ProjectActions(window, document)
    window.data_panel = DataPanel(document)
    window.structure_panel = StructurePanel(document)
    window.left_splitter = QSplitter(Qt.Orientation.Vertical)
    window.left_splitter.setObjectName("leftSplitter")
    window.left_splitter.addWidget(window.data_panel)
    window.left_splitter.addWidget(window.structure_panel)
    window.left_splitter.setSizes(list(document.project.ui_state.left_splitter_sizes))
    project_column.layout().addWidget(window.left_splitter, 1)
    project_column.setSizePolicy(
        QSizePolicy.Policy.Ignored,
        QSizePolicy.Policy.Expanding,
    )
    project_column.setMinimumWidth(320)
    return project_column


def _install_plot_column(window: object) -> QWidget:
    plot_column = _column("plotColumn")
    window.plot_panel = PlotPanel()
    plot_column.layout().addWidget(window.plot_panel, 1)
    return plot_column


def _install_analysis_column(window: object, document: object) -> QWidget:
    analysis_column = _column("analysisColumn")
    window.parameters_panel = ParametersPanel(document)
    window.fit_panel = FitPanel(document)
    window.result_panel = ResultsPanel(document)
    window.export_button = QPushButton("导出结果")
    window.export_button.setObjectName("exportResultsButton")
    window.export_button.setAccessibleName("导出拟合结果")
    window.export_button.setToolTip("将当前项目的拟合结果导出到所选目录")
    window.export_button.clicked.connect(window.export_results_dialog)
    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 4, 0)
    content_layout.setSpacing(8)
    specs = (
        ("参数", "parametersSection", window.parameters_panel),
        ("拟合", "fitSection", window.fit_panel),
        ("结果", "resultsSection", window.result_panel),
    )
    window.analysis_sections = {}
    for title, name, panel in specs:
        section = AnalysisSection(title, name, panel)
        window.analysis_sections[name] = section
        content_layout.addWidget(section)
    # The results section starts collapsed: a fresh project has no result to
    # read, and keeping all three open is what pushed the column past the
    # viewport at 1280x760.
    window.analysis_sections["resultsSection"].set_expanded(False)
    content_layout.addStretch(1)
    scroll = QScrollArea()
    scroll.setObjectName("analysisScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    analysis_column.layout().addWidget(scroll, 1)
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
