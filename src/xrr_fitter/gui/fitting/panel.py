"""Fit controls, readiness, progress, and project publication."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import messages, theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.controller import FitController
from xrr_fitter.gui.fitting.progress import ProgressView


class FitPanel(QWidget):
    """Coordinate preflight and operation events without owning a worker."""

    running_changed = Signal(bool)
    result_published = Signal(object)
    checkpoint_published = Signal(object)
    operation_failed = Signal(object)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("fitPanel")
        self.controller = FitController(self)
        self.progress_view = ProgressView(self)
        self._readiness = api.FitReadiness(False, "尚未检查拟合条件")
        self._build_controls()
        self._connect_controller()
        document.project_changed.connect(self._refresh_readiness)
        document.source_validation_changed.connect(self._refresh_readiness)
        self._refresh_readiness()

    def _build_controls(self) -> None:
        self.batch_selector = QComboBox()
        self.batch_selector.setObjectName("batchModeSelector")
        self.batch_selector.addItem("独立拟合", "independent")
        self.batch_selector.addItem("联合拟合", "joint")
        self.batch_selector.currentIndexChanged.connect(self._batch_mode_selected)
        batch_label = QLabel("批量模式")
        batch_label.setObjectName("batchModeLabel")
        batch_label.setBuddy(self.batch_selector)
        self.start_button = QPushButton("开始拟合")
        self.start_button.setObjectName("startFitButton")
        self.start_button.setProperty("primary", True)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("cancelFitButton")
        self.force_button = QPushButton("强制停止")
        self.force_button.setObjectName("forceStopFitButton")
        self.start_button.clicked.connect(self.start_fit)
        self.cancel_button.clicked.connect(self.controller.cancel)
        self.force_button.clicked.connect(self.controller.force_stop)
        self.cancel_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.cancel_shortcut.setObjectName("cancelFitShortcut")
        self.cancel_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.cancel_shortcut.activated.connect(self.controller.cancel)
        self.status_label = QLabel()
        self.status_label.setObjectName("fitStatusLabel")
        self.status_label.setWordWrap(True)
        batch_row = QHBoxLayout()
        batch_row.addWidget(batch_label)
        batch_row.addWidget(self.batch_selector, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button, 1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.force_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(batch_row)
        layout.addLayout(buttons)
        layout.addWidget(self.progress_view)
        layout.addWidget(self.status_label)
        self._project_running_state(False)

    def _connect_controller(self) -> None:
        self.controller.running_changed.connect(self._project_running_state)
        self.controller.progress_changed.connect(self.progress_view.set_progress)
        self.controller.checkpoint_ready.connect(self._publish_checkpoint)
        self.controller.fit_finished.connect(self._publish_fit_result)
        self.controller.cancelled.connect(self._show_cancelled)
        self.controller.failed.connect(self._show_failure)

    @property
    def is_running(self) -> bool:
        return self.controller.is_running

    def status_text(self) -> str:
        return self.status_label.text()

    def start_fit(self, _checked: bool = False, checkpoint_path=None) -> bool:
        readiness = api.preflight_fit(self.document.project)
        self._readiness = readiness
        self._show_readiness()
        if not readiness.ready:
            self._refresh_controls()
            return False
        self.progress_view.reset()
        return self.controller.start_fit(self.document.project, checkpoint_path)

    def set_batch_mode(self, mode: str) -> bool:
        current = self.document.project
        updated = api.set_batch_mode(current, mode)
        if updated is current:
            return False
        self.document.replace_project(updated)
        return True

    def _batch_mode_selected(self, _index: int) -> None:
        mode = str(self.batch_selector.currentData())
        try:
            self.set_batch_mode(mode)
        except ValueError as error:
            self.status_label.setText(str(error))
            self._sync_batch_selector()

    def _publish_checkpoint(self, project: api.XrrProject) -> None:
        self.document.replace_project(project)
        self.checkpoint_published.emit(project)

    def _publish_fit_result(self, result: api.ProjectFitResult) -> None:
        self.document.replace_project(result.updated_project)
        self._show_status("拟合完成", kind="ok")
        self.result_published.emit(result)

    def _show_cancelled(self, reason: str) -> None:
        self._show_status(f"已取消：{reason}", kind="warn")

    def _show_failure(self, error: api.OperationError) -> None:
        self._show_status(messages.operation_error_text(error), kind="error")
        self.operation_failed.emit(error)

    def _show_status(self, text: str, *, kind: str) -> None:
        self.status_label.setText(text)
        theme.set_status_kind(self.status_label, kind)

    def _show_readiness(self) -> None:
        self._show_status(
            messages.readiness_text(self._readiness.message),
            kind="ok" if self._readiness.ready else "warn",
        )

    def _project_running_state(self, running: bool) -> None:
        self.progress_view.setVisible(running)
        self._refresh_controls(running)
        self.running_changed.emit(running)

    def _refresh_readiness(self, *_args) -> None:
        try:
            self._readiness = api.preflight_fit(self.document.project)
        except (OSError, ValueError) as error:
            self._readiness = api.FitReadiness(False, str(error))
        if not self.is_running:
            self._show_readiness()
        self._sync_batch_selector()
        self._refresh_controls()

    def _sync_batch_selector(self) -> None:
        blocker = QSignalBlocker(self.batch_selector)
        index = self.batch_selector.findData(self.document.project.batch_mode)
        self.batch_selector.setCurrentIndex(index)
        del blocker

    def _refresh_controls(self, running: bool | None = None) -> None:
        active = self.is_running if running is None else running
        self.start_button.setEnabled(self._readiness.ready and not active)
        self.cancel_button.setEnabled(active)
        self.force_button.setEnabled(active)
        self.batch_selector.setEnabled(not active)
