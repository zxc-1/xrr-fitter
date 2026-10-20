"""Fit controls, readiness, progress, and project publication."""

from __future__ import annotations

import time
from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
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
from xrr_fitter.gui.document import ProjectDocument, dropped_fit_work
from xrr_fitter.gui.fitting.controller import FitController
from xrr_fitter.gui.fitting.metrics import LiveMetricsView
from xrr_fitter.gui.fitting.progress import ProgressView
from xrr_fitter.gui.noise import NOISE_MODE_LABELS, NOISE_MODE_REQUIREMENTS
from xrr_fitter.gui.project import dialogs
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
    # 停止与强制停止能不能按变了。菜单、命令栏那几枚是同一条命令的别的入口，跟着这一声改。
    stop_state_changed = Signal()

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
        # 限流窗口里到达的最新一帧，窗口一过由这只单发计时器画出来。
        self._preview_pending: api.FitProgress | None = None
        self._preview_flush = QTimer(self)
        self._preview_flush.setObjectName("fitPreviewFlush")
        self._preview_flush.setSingleShot(True)
        self._preview_flush.setInterval(round(1000 * self.PREVIEW_MIN_INTERVAL_S))
        self._preview_flush.timeout.connect(self._flush_preview)
        self._checkpoint_saved = False
        self._paused = False
        self._cancel_requested = False
        self._force_requested = False
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
        document.project_changed.connect(self._follow_active_dataset)
        document.source_validation_changed.connect(self._refresh_readiness)
        self._refresh_readiness()
        self._follow_active_dataset()

    def _build_controls(self) -> None:
        self._build_noise_controls()
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
        # 所以它与「停止并保留进度」是两件事而不是强弱两档。
        self.pause_button = QPushButton(PAUSE_TEXT)
        self.pause_button.setObjectName("pauseFitButton")
        self._refresh_pause_label()
        # 设计稿帧④ 控制段的第二个命令。跳过作废的只有当前这一个阶段：停止之后没有
        # 的跑，跳过之后还有。
        self.skip_button = QPushButton("⏭ 跳过本阶段")
        self.skip_button.setObjectName("skipStageButton")
        self.skip_button.setAccessibleName("跳过当前拟合阶段")
        self.skip_button.setToolTip("作废当前阶段，直接进入下一阶段；已完成阶段与候选都保留")
        # 「取消」把这个命令说反了（它不作废已跑完的部分），设计稿的「保留最优」又说过头：
        # 搜索在取消时只交出检查点，跑到一半的那条数据集不产生新结果。所以只承诺保留进度。
        self.cancel_button = QPushButton("⏹ 停止并保留进度")
        self.cancel_button.setObjectName("cancelFitButton")
        self.cancel_button.setAccessibleName("停止拟合并保留已完成的进度")
        self.force_button = QPushButton("强制停止")
        self.force_button.setObjectName("forceStopFitButton")
        self.automatic_button.clicked.connect(self.start_automatic_fit)
        self.start_button.clicked.connect(self.start_fit)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.skip_button.clicked.connect(self.controller.skip_stage)
        self.cancel_button.clicked.connect(self.request_cancel)
        self.force_button.clicked.connect(self.request_force_stop)
        # Esc 只在拟合卡里停拟合：在别处按 Esc 是在取消输入或关弹窗，不该连带停掉一次运行。
        self.cancel_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.cancel_shortcut.setObjectName("cancelFitShortcut")
        self.cancel_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.cancel_shortcut.activated.connect(self.request_cancel)
        self.stop_help = QLabel(
            "停止后，已跑完的数据集保留结果；没跑完的只保存检查点，不产生新结果，再次拟合时从检查点继续。"
        )
        self.stop_help.setObjectName("fitStopHelp")
        self.stop_help.setProperty("mutedText", True)
        self.stop_help.setWordWrap(True)
        self.status_label = QLabel()
        self.status_label.setObjectName("fitStatusLabel")
        self.status_label.setWordWrap(True)
        # 「本次：<路线> — <做法>」。路线由主窗口按项目算（``MainWindow.fit_route``）再交给这里；
        # 没交之前不画，免得空着一行「本次：」。
        self.route_label = QLabel()
        # 本次路线的键（``messages.FIT_ROUTES``）；窗口没交之前是 ``None``，两枚启动键照旧按高级开关摆。
        self._route: str | None = None
        self.route_label.setObjectName("fitRouteLine")
        self.route_label.setAccessibleName("本次拟合路线")
        self.route_label.setProperty("mutedText", True)
        self.route_label.setWordWrap(True)
        self.route_label.hide()
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
        noise_row = QHBoxLayout()
        noise_row.addWidget(self.noise_label)
        noise_row.addWidget(self.noise_selector, 1)
        layout.addLayout(noise_row)
        layout.addWidget(self.noise_requirements)
        layout.addWidget(self.route_label)
        layout.addWidget(self.command_bar)
        layout.addWidget(self.stop_help)
        layout.addWidget(self.progress_view)
        layout.addWidget(self.status_label)
        self._project_running_state(False)

    def _build_noise_controls(self) -> None:
        self.noise_selector = QComboBox()
        self.noise_selector.setObjectName("noiseModelSelector")
        self.noise_selector.setAccessibleName("噪声模式")
        for mode, label in NOISE_MODE_LABELS.items():
            self.noise_selector.addItem(label, mode)
        self.noise_label = QLabel("噪声模式")
        self.noise_label.setBuddy(self.noise_selector)
        self.noise_requirements = QLabel()
        self.noise_requirements.setObjectName("noiseModelRequirements")
        self.noise_requirements.setWordWrap(True)
        self.noise_selector.currentIndexChanged.connect(self._noise_mode_selected)

    def _noise_mode_selected(self, _index: int) -> None:
        current = self.document.project
        config = replace(current.fit_config, noise_model=str(self.noise_selector.currentData()))
        updated = api.set_fit_config(current, config)
        if updated is current:
            return
        if not self._may_drop_fit_work("切换噪声模式", current, updated):
            # 下拉框自己已经跳到了新选项；读回项目，让它停在没换的那一档。
            self._sync_noise_selector()
            return
        self.document.replace_project(updated)

    def _may_drop_fit_work(self, change: str, current: api.XrrProject, updated: api.XrrProject) -> bool:
        lost = dropped_fit_work(current, updated)
        return not lost or dialogs.confirm_dropping_results(self, change, len(lost))

    def _sync_noise_selector(self) -> None:
        mode = self.document.project.fit_config.noise_model
        blocker = QSignalBlocker(self.noise_selector)
        self.noise_selector.setCurrentIndex(self.noise_selector.findData(mode))
        del blocker
        self.noise_requirements.setText(NOISE_MODE_REQUIREMENTS[mode])

    def _connect_controller(self) -> None:
        self.controller.running_changed.connect(self._project_running_state)
        # 进度卡先收，指标后收：卡片决定了这一帧是不是它正在讲的那条曲线的，指标照它的口径走。
        self.controller.progress_changed.connect(self.progress_view.set_progress)
        self.controller.progress_changed.connect(self._project_metrics)
        self.controller.progress_changed.connect(self._project_preview)
        self.controller.progress_changed.connect(self._project_skip_state)
        self.controller.poll_interval_changed.connect(self.progress_view.set_refresh_interval_ms)
        self.controller.checkpoint_ready.connect(self._publish_checkpoint)
        self.controller.fit_finished.connect(self._publish_fit_result)
        self.controller.cancelled.connect(self._show_cancelled)
        self.controller.failed.connect(self._show_failure)

    def _project_metrics(self, progress: api.FitProgress) -> None:
        """Every frame feeds the per-dataset table; only the card's own frame the readings."""
        if self.progress_view.draws(progress):
            self.live_metrics.set_progress(progress)
        else:
            self.live_metrics.record(progress)

    def _follow_active_dataset(self, *_args) -> None:
        """The card tells the run of the dataset selected in the left rail, if it is in the run."""
        if self.progress_view.follow_dataset(self.document.active_dataset_id):
            self.live_metrics.show_frame(self.progress_view.focused_frame())

    def _project_skip_state(self, progress: api.FitProgress) -> None:
        self._skip_available = progress.stage in {"A", "B", "C", "D", "E"}
        self._refresh_controls()

    def _project_preview(self, progress: api.FitProgress) -> None:
        """Forward preview curves at a bounded rate to avoid canvas flicker.

        A frame inside the window is held rather than dropped -- the newest one wins
        -- and drawn once the window has passed: the frames a local search ends on
        arrive together, and dropping them left the canvas on the second-to-last
        curve until the next stage spoke, which during profile can be a minute.
        """
        if not self._shows_preview(progress):
            return
        now = self._clock()
        last = self._preview_last_emit
        if last is not None and now - last < self.PREVIEW_MIN_INTERVAL_S:
            self._preview_pending = progress
            if not self._preview_flush.isActive():
                self._preview_flush.start()
            return
        self._emit_preview(progress, now)

    def _shows_preview(self, progress: api.FitProgress) -> bool:
        """A frame carrying a curve of the dataset the canvas shows."""
        return (
            progress.dataset_id is not None
            and progress.dataset_id == self.document.active_dataset_id
            and progress.preview_qz_a_inv is not None
            and progress.preview_model_normalized is not None
        )

    def _emit_preview(self, progress: api.FitProgress, now: float) -> None:
        self._drop_pending_preview()
        self._preview_last_emit = now
        self.preview_available.emit(progress.preview_qz_a_inv, progress.preview_model_normalized)

    def _flush_preview(self) -> None:
        """Draw the frame held through the window, if it still belongs on the canvas."""
        pending = self._preview_pending
        if pending is not None and self._shows_preview(pending):
            self._emit_preview(pending, self._clock())
        self._drop_pending_preview()

    def _drop_pending_preview(self) -> None:
        self._preview_pending = None
        self._preview_flush.stop()

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
        layout = self._joint_layout()
        members = tuple(dataset.dataset_id for dataset in self.document.project.datasets)
        self._prepare_run(members, layout, "manual" if layout is None else "joint")
        return self.controller.start_fit(self.document.project, checkpoint_path)

    def _toggle_pause(self) -> None:
        """Report the request, not an unobserved acknowledgement from all workers."""
        if not self.is_running or self._cancel_requested:
            return
        if self._paused:
            self.controller.resume()
        else:
            self.controller.pause()
        self._paused = not self._paused
        self.progress_view.set_pause_requested(self._paused)
        self._refresh_pause_label()
        self._refresh_controls()

    def _refresh_pause_label(self) -> None:
        """Keep the visible action, screen-reader name and help in agreement."""
        text, name, help_text = (
            (RESUME_TEXT, "继续拟合", "解除暂停请求，让已到达安全点的计算继续；保留已完成阶段与当前最优")
            if self._paused
            else (PAUSE_TEXT, "暂停拟合", "请求在下一个安全检查点停住，保留已完成阶段与当前最优")
        )
        self.pause_button.setText(text)
        self.pause_button.setAccessibleName(name)
        self.pause_button.setToolTip(help_text)

    def request_cancel(self) -> None:
        """Give immediate feedback, then ask the worker to stop gracefully.

        The worker cannot abandon the seed it is mid-evaluation on, so a cancel
        request has visible latency. Announcing it on the progress view keeps the
        button press from feeling ignored while the current seed winds down.
        Every stop entrance (button, Esc, menu, command bar, window close) comes
        through here so the card always says what was asked.
        """
        if not self.can_request_stop():
            return
        self.progress_view.mark_cancelling()
        # 「强制停止」是「停止」的升级而不是并列项：先按停止让 worker 自己收尾，收不住时
        # 那一枚才露面，读者不会一上来就面对两个停止键。
        self._cancel_requested = True
        self._refresh_controls()
        self.controller.cancel()

    def request_force_stop(self) -> None:
        """Escalate a pending stop the worker has not wound down from."""
        if not self.can_force_stop():
            return
        self._force_requested = True
        self._refresh_controls()
        self.controller.force_stop()

    def can_request_stop(self) -> bool:
        return self.is_running and not self._cancel_requested

    def can_force_stop(self) -> bool:
        return self.is_running and self._cancel_requested and not self._force_requested

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
        self._prepare_run(self._automatic_members(import_batch_id), None, "automatic")
        return self.controller.start_automatic_fit(
            self.document.project,
            import_batch_id,
            checkpoint_path,
        )

    def _prepare_run(self, members: tuple[str, ...], layout: object | None, route: str) -> None:
        self._preview_last_emit = None
        self._drop_pending_preview()
        self._checkpoint_saved = False
        self.progress_view.reset()
        self.progress_view.set_weighting(route)
        self.progress_view.set_members(members)
        self.progress_view.set_joint_layout(layout)
        self._reset_metrics(layout)

    def _automatic_members(self, import_batch_id: str | None) -> tuple[str, ...]:
        runnable = {api.AutomaticStatus.PENDING, api.AutomaticStatus.REFINING, api.AutomaticStatus.REVIEW}
        return tuple(
            dataset.dataset_id
            for dataset in self.document.project.datasets
            if dataset.automation.role is not api.AutomaticRole.MANUAL
            and dataset.automation.status in runnable
            and (import_batch_id is None or dataset.automation.import_batch_id == import_batch_id)
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

    def set_route(self, route: str) -> None:
        """Name the route the next fit takes and offer only its start; the window works it out from the project.

        未开高级选项时，卡片上露出的那一枚就是命令栏一键拟合走的那条路（F56）：同一件事不该有两个名字，
        也不该在手动路线下只摆一枚灰着的「自动拟合」。
        """
        title, detail = messages.FIT_ROUTES[route]
        self._route = route
        self.route_label.setText(f"本次：{title} — {detail}")
        self.route_label.show()
        self._sync_mode_visibility()

    def set_batch_mode(self, mode: str) -> bool:
        """Switch the batch mode; False when nothing changed or the reader kept their results."""
        current = self.document.project
        updated = api.set_batch_mode(current, mode)
        if updated is current or not self._may_drop_fit_work("切换批量模式", current, updated):
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
        # 一次运行在撤销栈里只占一格：第一次写回开一格，之后的检查点与最终结果都并进这一格。
        self.document.adopt_operation_result(project, amend=self._checkpoint_saved)
        self._checkpoint_saved = True
        self.checkpoint_published.emit(project)

    def _publish_fit_result(self, result: api.ProjectFitResult) -> None:
        self.document.adopt_operation_result(result.updated_project, amend=self._checkpoint_saved)
        self.progress_view.complete()
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
        self._show_status(f"已取消：{messages.cancellation_text(reason)}{hint}", kind="warn")

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
            # The canvas now shows the committed result; a held preview drawn after
            # this would paint the searching model back over it.
            self._drop_pending_preview()
            # 收工后暂停键回到未按下的样子，否则下一次开跑它还写着「▶ 继续」。
            self._paused = False
            self._refresh_pause_label()
            self._refresh_readiness(show_status=False)
        self.progress_view.setVisible(running)
        self.live_metrics.setVisible(running)
        self._refresh_controls(running)
        self.running_changed.emit(running)

    def _refresh_readiness(self, *_args, show_status: bool = True) -> None:
        # A checkpoint changes evidence, not the running job's immutable inputs.
        # Re-read sources and compile only when another start can be requested.
        if not self.is_running:
            self._check_readiness()
        if not self.is_running and show_status:
            readiness = self._readiness if self.document.project.ui_state.expert_mode else self._automatic_readiness
            self._show_readiness(readiness)
        self._sync_noise_selector()
        self._sync_mode_visibility()
        self._refresh_controls()

    def _check_readiness(self) -> None:
        try:
            self._readiness = api.preflight_fit(self.document.project)
            self._automatic_readiness = api.preflight_automatic_fit(self.document.project)
        except (OSError, ValueError) as error:
            self._readiness = api.FitReadiness(False, str(error))
            self._automatic_readiness = api.FitReadiness(False, str(error))

    def _sync_mode_visibility(self, running: bool | None = None) -> None:
        # 跑起来之后两枚启动键都按不动，留在原位只是两块灰，还把真正要用的三个命令挤到一边。
        # 专家两条路都要；未开高级选项时只露本次路线那一枚（``set_route``）。
        active = self.is_running if running is None else running
        expert = self.document.project.ui_state.expert_mode
        routed_by_table = self._route in {"manual", "joint"}
        self.start_button.setVisible((expert or routed_by_table) and not active)
        self.automatic_button.setVisible((expert or not routed_by_table) and not active)

    def _refresh_controls(self, running: bool | None = None) -> None:
        active = self.is_running if running is None else running
        if not active:
            self._cancel_requested = False
            self._force_requested = False
        self._sync_mode_visibility(active)
        self.force_button.setVisible(active and self._cancel_requested)
        self.start_button.setEnabled(self._readiness.ready and not active)
        self.automatic_button.setEnabled(self._automatic_readiness.ready and not active)
        can_request = active and not self._cancel_requested
        self.cancel_button.setEnabled(can_request)
        self.pause_button.setEnabled(can_request)
        self.skip_button.setEnabled(can_request and self._skip_available and not self._paused)
        self.force_button.setEnabled(active and not self._force_requested)
        self.noise_selector.setEnabled(not active)
        self._refresh_start_label()
        self.stop_state_changed.emit()

    def _refresh_start_label(self) -> None:
        """Relabel the start action when a resume from checkpoint is pending."""
        if self.has_resumable_checkpoint():
            self.start_button.setText("继续拟合")
            self.start_button.setToolTip("检测到检查点，将从上次进度继续拟合")
        else:
            self.start_button.setText("开始拟合")
            self.start_button.setToolTip("")
