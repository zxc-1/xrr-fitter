"""Fit controls, readiness, progress, and project publication."""

from __future__ import annotations

import time

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
    preview_available = Signal(object, object)

    # Minimum wall-clock gap between forwarded preview curves. A local search
    # can emit previews many times per second; repainting every one makes the
    # canvas flicker without adding information, so intermediate frames are
    # dropped and only the newest survivor of each window is drawn.
    PREVIEW_MIN_INTERVAL_S = 0.2

    def __init__(
        self,
        document: ProjectDocument,
        *,
        clock: object = time.monotonic,
    ) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("fitPanel")
        self._clock = clock
        self._preview_last_emit: float | None = None
        self._checkpoint_saved = False
        self.controller = FitController(self)
        self.progress_view = ProgressView(self)
        self._readiness = api.FitReadiness(False, "尚未检查拟合条件")
        self._automatic_readiness = api.FitReadiness(
            False,
            "尚未检查自动拟合条件",
        )
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
        self.batch_label = QLabel("批量模式")
        self.batch_label.setObjectName("batchModeLabel")
        self.batch_label.setBuddy(self.batch_selector)
        self.automatic_button = QPushButton("自动拟合")
        self.automatic_button.setObjectName("startAutomaticFitButton")
        self.automatic_button.setProperty("primary", True)
        self.automatic_button.setAccessibleName("启动自动拟合")
        self.automatic_button.setToolTip("运行项目中所有待拟合的自动数据集")
        self.start_button = QPushButton("开始拟合")
        self.start_button.setObjectName("startFitButton")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("cancelFitButton")
        self.force_button = QPushButton("强制停止")
        self.force_button.setObjectName("forceStopFitButton")
        self.automatic_button.clicked.connect(self.start_automatic_fit)
        self.start_button.clicked.connect(self.start_fit)
        self.cancel_button.clicked.connect(self._request_cancel)
        self.force_button.clicked.connect(self.controller.force_stop)
        self.cancel_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.cancel_shortcut.setObjectName("cancelFitShortcut")
        self.cancel_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.cancel_shortcut.activated.connect(self._request_cancel)
        self.status_label = QLabel()
        self.status_label.setObjectName("fitStatusLabel")
        self.status_label.setWordWrap(True)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_label)
        batch_row.addWidget(self.batch_selector, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.automatic_button, 1)
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
        self.controller.progress_changed.connect(self._project_preview)
        self.controller.checkpoint_ready.connect(self._publish_checkpoint)
        self.controller.fit_finished.connect(self._publish_fit_result)
        self.controller.cancelled.connect(self._show_cancelled)
        self.controller.failed.connect(self._show_failure)

    def _project_preview(self, progress: api.FitProgress) -> None:
        """Forward preview curves at a bounded rate to avoid canvas flicker."""
        qz = progress.preview_qz_a_inv
        model = progress.preview_model_normalized
        if qz is None or model is None:
            return
        now = self._clock()
        last = self._preview_last_emit
        if last is not None and now - last < self.PREVIEW_MIN_INTERVAL_S:
            return
        self._preview_last_emit = now
        self.preview_available.emit(qz, model)

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
        self._preview_last_emit = None
        self._checkpoint_saved = False
        self.progress_view.reset()
        self.progress_view.set_joint_layout(self._joint_layout())
        return self.controller.start_fit(self.document.project, checkpoint_path)

    def _request_cancel(self) -> None:
        """Give immediate feedback, then ask the worker to stop gracefully.

        The worker cannot abandon the seed it is mid-evaluation on, so a cancel
        request has visible latency. Announcing it on the progress view keeps the
        button press from feeling ignored while the current seed winds down.
        """
        if not self.is_running:
            return
        self.progress_view.mark_cancelling()
        self.controller.cancel()

    def start_automatic_fit(
        self,
        import_batch_id: str | None = None,
        checkpoint_path=None,
    ) -> bool:
        if isinstance(import_batch_id, bool):
            import_batch_id = None
        readiness = api.preflight_automatic_fit(
            self.document.project,
            import_batch_id,
        )
        self._automatic_readiness = readiness
        self._show_readiness(readiness)
        if not readiness.ready:
            self._refresh_controls()
            return False
        self.progress_view.reset()
        return self.controller.start_automatic_fit(
            self.document.project,
            import_batch_id,
            checkpoint_path,
        )

    def _joint_layout(self) -> object | None:
        """Describe the joint layout so progress frames name their members.

        A joint run's progress carries no single owning dataset, so without this
        the view can only say "联合拟合". The layout is a cheap read of persisted
        fields; an independent project has none, so the banner stays hidden.
        """
        if self.document.project.batch_mode != "joint":
            return None
        return api.describe_joint_layout(self.document.project)

    def set_batch_mode(self, mode: str) -> bool:
        current = self.document.project
        updated = api.set_batch_mode(current, mode)
        if updated is current:
            return False
        self.document.replace_project(updated)
        return True

    def has_resumable_checkpoint(self) -> bool:
        """Report whether any dataset holds a checkpoint the fit will resume.

        Resume is implicit: a fit reads each dataset's stored checkpoint and
        continues from it. That is invisible in the UI, so the start button
        would silently pick up hours-old partial work. Detecting the checkpoint
        lets the panel relabel the action and say so before the user commits.
        """
        return any(
            dataset.checkpoint is not None
            for dataset in self.document.project.datasets
        )

    def _batch_mode_selected(self, _index: int) -> None:
        mode = str(self.batch_selector.currentData())
        try:
            self.set_batch_mode(mode)
        except ValueError as error:
            self.status_label.setText(str(error))
            self._sync_batch_selector()

    def _publish_checkpoint(self, project: api.XrrProject) -> None:
        self._checkpoint_saved = True
        self.document.replace_project(project)
        self.checkpoint_published.emit(project)

    def _publish_fit_result(self, result: api.ProjectFitResult) -> None:
        self.document.replace_project(result.updated_project)
        self._show_status("拟合完成", kind="ok")
        self.result_published.emit(result)

    def _show_cancelled(self, reason: str) -> None:
        # A bare "已取消" leaves the user unsure whether the interrupted work
        # survived; naming the checkpoint state tells them if a resume is
        # possible instead of making them guess.
        if self._checkpoint_saved:
            hint = "，已保存检查点，可从中恢复"
        else:
            hint = "，本次未产生检查点"
        self._show_status(f"已取消：{reason}{hint}", kind="warn")

    def _show_failure(self, error: api.OperationError) -> None:
        self._show_status(messages.operation_error_text(error), kind="error")
        self.operation_failed.emit(error)

    def _show_status(self, text: str, *, kind: str) -> None:
        self.status_label.setText(text)
        theme.set_status_kind(self.status_label, kind)

    def _show_readiness(self, readiness: api.FitReadiness | None = None) -> None:
        """Report only what the status bar does not already say.

        The status bar carries the persistent readiness verdict. Repeating a
        successful one here put the identical sentence on screen twice, so a
        ready project leaves this label empty and it is reserved for blocking
        reasons and operation outcomes.
        """
        value = self._readiness if readiness is None else readiness
        if value.ready:
            self._show_status("", kind="")
            return
        self._show_status(messages.readiness_text(value.message), kind="warn")

    def _project_running_state(self, running: bool) -> None:
        if not running:
            # The worker is done, so stop the live clock; the last rendered
            # elapsed/remaining values stay put instead of ticking on forever.
            self.progress_view.freeze()
        self.progress_view.setVisible(running)
        self._refresh_controls(running)
        self.running_changed.emit(running)

    def _refresh_readiness(self, *_args) -> None:
        try:
            self._readiness = api.preflight_fit(self.document.project)
            self._automatic_readiness = api.preflight_automatic_fit(
                self.document.project
            )
        except (OSError, ValueError) as error:
            self._readiness = api.FitReadiness(False, str(error))
            self._automatic_readiness = api.FitReadiness(False, str(error))
        if not self.is_running:
            readiness = (
                self._readiness
                if self.document.project.ui_state.expert_mode
                else self._automatic_readiness
            )
            self._show_readiness(readiness)
        self._sync_batch_selector()
        self._sync_mode_visibility()
        self._refresh_controls()

    def _sync_mode_visibility(self) -> None:
        expert = self.document.project.ui_state.expert_mode
        self.batch_label.setVisible(expert)
        self.batch_selector.setVisible(expert)
        self.start_button.setVisible(expert)

    def _sync_batch_selector(self) -> None:
        blocker = QSignalBlocker(self.batch_selector)
        index = self.batch_selector.findData(self.document.project.batch_mode)
        self.batch_selector.setCurrentIndex(index)
        del blocker

    def _refresh_controls(self, running: bool | None = None) -> None:
        active = self.is_running if running is None else running
        self.start_button.setEnabled(self._readiness.ready and not active)
        self.automatic_button.setEnabled(
            self._automatic_readiness.ready and not active
        )
        self.cancel_button.setEnabled(active)
        self.force_button.setEnabled(active)
        self.batch_selector.setEnabled(not active)
        self._refresh_start_label()

    def _refresh_start_label(self) -> None:
        """Relabel the start action when a resume from checkpoint is pending."""
        if self.has_resumable_checkpoint():
            self.start_button.setText("继续拟合")
            self.start_button.setToolTip("检测到检查点，将从上次进度继续拟合")
        else:
            self.start_button.setText("开始拟合")
            self.start_button.setToolTip("")
