"""Fit controls, readiness, progress, and project publication."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import messages, theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.controller import FitController
from xrr_fitter.gui.fitting.metrics import LiveMetricsView
from xrr_fitter.gui.fitting.progress import ProgressView
from xrr_fitter.gui.wrapping import command_bar

# 暂停键的两种面孔。按下之后必须改口，否则读者只能靠曲线动不动来猜它是否生效。
PAUSE_TEXT = "⏸ 暂停"
RESUME_TEXT = "▶ 继续"


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
    PREVIEW_MIN_INTERVAL_S = 0.05

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
        self._paused = False
        self._cancel_requested = False
        self._skip_available = False
        self.controller = FitController(self)
        self.progress_view = ProgressView(self)
        # 检视器把它摆在自己那一段里（``inspectorLiveMetrics``），但喂它的是本面板的
        # 那条进度信号，所以由本面板构造与持有：接线在构造期一次连好，视图搬到哪一列
        # 都不影响。
        self.live_metrics = LiveMetricsView()
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
        # 批量模式（独立/联合）不在这张卡上：它决定整屏参数表读作共享还是独立，
        # 是项目级状态，设计稿把它画在命令栏 引导·专家 段的右边。这里只保留
        # ``set_batch_mode``，命令栏的段调它。
        self.automatic_button = QPushButton("自动拟合")
        self.automatic_button.setObjectName("startAutomaticFitButton")
        self.automatic_button.setProperty("primary", True)
        self.automatic_button.setAccessibleName("启动自动拟合")
        self.automatic_button.setToolTip("运行项目中所有待拟合的自动数据集")
        self.start_button = QPushButton("开始拟合")
        self.start_button.setObjectName("startFitButton")
        # 设计稿帧④ 控制段的第一个命令。暂停停在下一个阶段边界，不丢已跑完的阶段，
        # 所以它与「停止并保留最优」是两件事而不是强弱两档。
        self.pause_button = QPushButton(PAUSE_TEXT)
        self.pause_button.setObjectName("pauseFitButton")
        self.pause_button.setAccessibleName("暂停拟合")
        self.pause_button.setToolTip("在下一个阶段边界停住，保留已完成阶段与当前最优")
        # 设计稿帧④ 控制段的第二个命令。跳过作废的只有当前这一个阶段：停止之后没有
        # 的跑，跳过之后还有。
        self.skip_button = QPushButton("⏭ 跳过本阶段")
        self.skip_button.setObjectName("skipStageButton")
        self.skip_button.setAccessibleName("跳过当前拟合阶段")
        self.skip_button.setToolTip("作废当前阶段，直接进入下一阶段；已完成阶段与候选都保留")
        # 设计稿帧④ 控制段的措辞：这个命令保留当前最优候选与已跑完的阶段，「取消」把它
        # 说反了——字面意思与行为相反的按钮，跑到一半的人多半不敢按。
        self.cancel_button = QPushButton("⏹ 停止并保留最优")
        self.cancel_button.setObjectName("cancelFitButton")
        self.cancel_button.setAccessibleName("停止拟合并保留最优候选")
        self.force_button = QPushButton("强制停止")
        self.force_button.setObjectName("forceStopFitButton")
        self.automatic_button.clicked.connect(self.start_automatic_fit)
        self.start_button.clicked.connect(self.start_fit)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.skip_button.clicked.connect(self.controller.skip_stage)
        self.cancel_button.clicked.connect(self._request_cancel)
        self.force_button.clicked.connect(self.controller.force_stop)
        self.cancel_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.cancel_shortcut.setObjectName("cancelFitShortcut")
        self.cancel_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.cancel_shortcut.activated.connect(self._request_cancel)
        self.stop_help = QLabel("停止后保留当前最优候选，可直接进入结果复核，不丢弃已完成阶段。")
        self.stop_help.setObjectName("fitStopHelp")
        self.stop_help.setProperty("mutedText", True)
        self.stop_help.setWordWrap(True)
        self.status_label = QLabel()
        self.status_label.setObjectName("fitStatusLabel")
        self.status_label.setWordWrap(True)
        # 四枚按钮换行，而不是把四枚之和（296px）当成地板：检视器视口只有 322px 且水平
        # 滚动条是关掉的，一行放不下时超出的按钮不是滚动而是无声裁掉——「强制停止」是
        # 拟合跑飞时唯一的出路，它被裁掉时屏幕上没有任何东西说明少了一个命令。
        self.command_bar, buttons = command_bar(self, name="fitCommandBar")
        buttons.addWidget(self.automatic_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.skip_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.force_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(self.command_bar)
        layout.addWidget(self.stop_help)
        layout.addWidget(self.progress_view)
        layout.addWidget(self.status_label)
        self._project_running_state(False)

    def _connect_controller(self) -> None:
        self.controller.running_changed.connect(self._project_running_state)
        self.controller.progress_changed.connect(self.progress_view.set_progress)
        self.controller.progress_changed.connect(self.live_metrics.set_progress)
        self.controller.progress_changed.connect(self._project_preview)
        self.controller.progress_changed.connect(self._project_skip_state)
        self.controller.poll_interval_changed.connect(self.progress_view.set_refresh_interval_ms)
        self.controller.checkpoint_ready.connect(self._publish_checkpoint)
        self.controller.fit_finished.connect(self._publish_fit_result)
        self.controller.cancelled.connect(self._show_cancelled)
        self.controller.failed.connect(self._show_failure)

    def _project_skip_state(self, progress: api.FitProgress) -> None:
        self._skip_available = progress.stage in {"A", "B", "C", "D", "E"}
        self._refresh_controls()

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
        layout = self._joint_layout()
        self.progress_view.set_joint_layout(layout)
        self._reset_metrics(layout)
        return self.controller.start_fit(self.document.project, checkpoint_path)

    def _toggle_pause(self) -> None:
        """在下一个阶段边界停住/放行，并让按钮自己报出当前处在哪一边。"""
        if not self.is_running:
            return
        if self._paused:
            self.controller.resume()
        else:
            self.controller.pause()
        self._paused = not self._paused
        self.pause_button.setText(RESUME_TEXT if self._paused else PAUSE_TEXT)

    def _request_cancel(self) -> None:
        """Give immediate feedback, then ask the worker to stop gracefully.

        The worker cannot abandon the seed it is mid-evaluation on, so a cancel
        request has visible latency. Announcing it on the progress view keeps the
        button press from feeling ignored while the current seed winds down.
        """
        if not self.is_running:
            return
        self.progress_view.mark_cancelling()
        # 「强制停止」是「停止」的升级而不是并列项：先按停止让 worker 自己收尾，收不住时
        # 那一枚才露面，读者不会一上来就面对两个停止键。
        self._cancel_requested = True
        self._refresh_controls()
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
        self._reset_metrics(self._joint_layout())
        return self.controller.start_automatic_fit(
            self.document.project,
            import_batch_id,
            checkpoint_path,
        )

    def _reset_metrics(self, layout: object | None) -> None:
        """Clear last run's readings and name the datasets a joint run shares.

        A joint progress event carries no dataset, so the objective table has
        nothing to list until the membership is handed over; an independent run
        builds its rows from the events themselves and passes none.
        """
        self.live_metrics.reset()
        members = () if layout is None else tuple(layout.dataset_ids)
        self.live_metrics.set_members(members)

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
        return any(dataset.checkpoint is not None for dataset in self.document.project.datasets)

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
        self._skip_available = False
        if not running:
            # The worker is done, so stop the live clock; the last rendered
            # elapsed/remaining values stay put instead of ticking on forever.
            self.progress_view.freeze()
            # 收工后暂停键回到未按下的样子，否则下一次开跑它还写着「▶ 继续」。
            self._paused = False
            self.pause_button.setText(PAUSE_TEXT)
        self.progress_view.setVisible(running)
        self.live_metrics.setVisible(running)
        self._refresh_controls(running)
        self.running_changed.emit(running)

    def _refresh_readiness(self, *_args) -> None:
        try:
            self._readiness = api.preflight_fit(self.document.project)
            self._automatic_readiness = api.preflight_automatic_fit(self.document.project)
        except (OSError, ValueError) as error:
            self._readiness = api.FitReadiness(False, str(error))
            self._automatic_readiness = api.FitReadiness(False, str(error))
        if not self.is_running:
            readiness = self._readiness if self.document.project.ui_state.expert_mode else self._automatic_readiness
            self._show_readiness(readiness)
        self._sync_mode_visibility()
        self._refresh_controls()

    def _sync_mode_visibility(self, running: bool | None = None) -> None:
        # 跑起来之后两枚启动键都按不动，留在原位只是两块灰，还把真正要用的三个命令挤到一边。
        active = self.is_running if running is None else running
        self.start_button.setVisible(self.document.project.ui_state.expert_mode and not active)
        self.automatic_button.setVisible(not active)

    def _refresh_controls(self, running: bool | None = None) -> None:
        active = self.is_running if running is None else running
        if not active:
            self._cancel_requested = False
        self._sync_mode_visibility(active)
        self.force_button.setVisible(active and self._cancel_requested)
        self.start_button.setEnabled(self._readiness.ready and not active)
        self.automatic_button.setEnabled(self._automatic_readiness.ready and not active)
        self.cancel_button.setEnabled(active)
        self.pause_button.setEnabled(active)
        self.skip_button.setEnabled(active and self._skip_available and not self._cancel_requested)
        self.force_button.setEnabled(active)
        self._refresh_start_label()

    def _refresh_start_label(self) -> None:
        """Relabel the start action when a resume from checkpoint is pending."""
        if self.has_resumable_checkpoint():
            self.start_button.setText("继续拟合")
            self.start_button.setToolTip("检测到检查点，将从上次进度继续拟合")
        else:
            self.start_button.setText("开始拟合")
            self.start_button.setToolTip("")
