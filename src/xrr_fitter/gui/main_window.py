"""Main application shell and asynchronous close coordination.

The window composes visible panels while :class:`ProjectDocument` owns the
immutable project identity.  Plot projection is registered synchronously at
that document boundary; ordinary Qt signals run only after a successful
precommit.  Close handling remains asynchronous so active workers never block
the GUI event loop.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import messages, status_bar
from xrr_fitter.gui.accessibility import (
    configure_accessibility,
    configure_focus_navigation,
)
from xrr_fitter.gui.chrome import install_chrome, refresh_run_summary, set_command_bar_guided
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.export.dialog import ExportWorkflow
from xrr_fitter.gui.operation_state import (
    has_exportable_results,
    operation_controllers,
    operation_is_running,
    refresh_operation_state,
)
from xrr_fitter.gui.project.autosave import AutosaveController
from xrr_fitter.gui.window_layout import (
    MINIMUM_SHELL_HEIGHT,
    MINIMUM_SHELL_WIDTH,
    apply_default_columns,
    apply_step_scope,
    install_workflow_actions,
    install_workspace,
    refresh_sampling_footer,
)
from xrr_fitter.gui.workspace import (
    WorkspaceView,
    capture_project,
    configure_splitters,
    restore_project,
)


class MainWindow(QMainWindow):
    """Own a document and the stable three-column desktop workspace."""

    def __init__(
        self,
        document: ProjectDocument | None = None,
        *,
        operation_controller: object | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("mainWindow")
        self.document = ProjectDocument() if document is None else document
        self._operation_controller: object | None = None
        self._close_pending = False
        self._resume_user_close = False
        self._force_close_prompt: QMessageBox | None = None
        self._workspace_released = False
        self._columns_settled = False
        self._capture_scheduled = False
        # 帧③ 底栏第一段的判据：层堆叠被提交过一次编辑，而这次编辑还没存盘。只看
        # ``document.is_dirty`` 太松——切换活动数据集也会置脏；只看「站在结构这一步」是撒
        # 谎——读者可以什么都没改就走过来。所以要两件事的与，编辑那一半记在这里。
        self._structure_edit_unsaved = False
        # Owned by the window so a pending capture dies with it. A bare
        # QTimer.singleShot would still fire after the window is gone and touch a
        # deleted C++ object, which in a long test run disturbs whichever window
        # is active by then.
        self._capture_timer = QTimer(self)
        self._capture_timer.setObjectName("workspaceCaptureTimer")
        self._capture_timer.setSingleShot(True)
        self._capture_timer.setInterval(0)
        self._capture_timer.timeout.connect(self._capture_scheduled_layout)
        self._close_cancel_timer = QTimer(self)
        self._close_cancel_timer.setObjectName("closeCancelTimer")
        self._close_cancel_timer.setSingleShot(True)
        self._close_cancel_timer.setInterval(5000)
        self._close_cancel_timer.timeout.connect(self.close_cancel_deadline_reached)
        self.autosave = AutosaveController(self.document, parent=self)
        self.autosave.start()
        install_workspace(self, self.document)
        self.workspace_view = WorkspaceView.from_root(self)
        # Stretch and collapse rules live with the persistence view because both
        # concern the same two splitters, and ``install_workspace`` runs before
        # that view exists.  Without them the canvas is not the column that
        # absorbs spare width and a side column can be dragged shut.
        configure_splitters(self.workspace_view)
        self.export_workflow = ExportWorkflow(
            self.document,
            is_running=self._operation_is_running,
        )
        install_workflow_actions(self)
        install_chrome(self)
        self._register_project_projections()
        self._connect_workflows()
        configure_accessibility(self)
        configure_focus_navigation(self)
        # Default to the full workspace. Guided mode is available from the View
        # menu for users who prefer a step-by-step workflow.
        self.set_guidance_visible(False)
        self._refresh_operation_state()
        self._refresh_window_title()
        self.setMinimumSize(MINIMUM_SHELL_WIDTH, MINIMUM_SHELL_HEIGHT)
        if operation_controller is not None:
            self.set_operation_controller(operation_controller)

    def _connect_workflows(self) -> None:
        self.plot_panel.fit_range_requested.connect(self._plot_range_requested)
        self.plot_panel.point_mask_requested.connect(self._plot_point_requested)
        self.plot_panel.view_changed.connect(self._plot_tab_changed)
        self.plot_panel.import_requested.connect(self.data_panel.import_files_button.click)
        self.plot_panel.structure_edit_requested.connect(self._plot_structure_edited)
        # Geometry is only worth persisting when a handle moved, and
        # ``splitterMoved`` is emitted for that alone: ``setSizes``, which is how
        # the defaults, ``reset_layout`` and a restore are all applied, stays
        # silent.  That is what makes this signal enough on its own, with none of
        # the "is a restore in flight" bookkeeping the dock notifications needed.
        # The three columns are pinned (``window_layout._pin_columns``), so what
        # this can still report is the left column's own split.
        for splitter in (self.workspace_splitter, self.workspace_view.left_splitter):
            if splitter is not None:
                splitter.splitterMoved.connect(self._layout_changed)
        self.guidance.leave_requested.connect(lambda: self.set_guidance_visible(False))
        self.guidance.step_changed.connect(self._refresh_guidance_status)
        self.result_panel.candidate_selected.connect(self._project_candidate)
        self.result_panel.candidate_inspected.connect(self._project_candidate)
        self.fit_panel.running_changed.connect(self._operation_running_changed)
        self.fit_panel.preview_available.connect(self._project_preview)
        self.fit_panel.running_changed.connect(self._discard_preview_when_idle)
        # 帧④ 左栏的成员行小字报的是这一次运行（「拟合中 · J=2.14↓」），所以运行状态和
        # 进度都得走到数据面板——它自己订阅不到求解器。
        self.fit_panel.running_changed.connect(self.data_panel.set_running)
        self.fit_panel.controller.progress_changed.connect(self.data_panel.set_run_progress)
        # Guided mode hides the inspector holding the progress card, so the card
        # publishes its position for the status bar to repeat.
        self.fit_panel.progress_view.summary_changed.connect(self._project_run_summary)
        self.result_panel.controller.running_changed.connect(self._operation_running_changed)
        # 帧⑤ 左栏页脚那句 walkers 下界预告的是「下一次采样会不会被拦」，所以它跟着 spin box
        # 走而不是跟着报告走：读者把 walkers 从 32 调到 8，页脚要当即改口说未满足。
        self.result_panel.walkers.valueChanged.connect(self._refresh_sampling_footer)
        self.document.project_changed.connect(self._refresh_operation_state)
        self.document.dirty_changed.connect(self._refresh_window_title)
        self.document.path_changed.connect(self._refresh_window_title)
        # 结构面板提交一次编辑之后底栏要改口（帧③「结构已修改（未保存）」）。这一路必须自己
        # 再刷一遍状态：``replace_project`` 先发 ``project_changed``（底栏由此刷过一次），
        # ``structure_changed`` 才跟着出来，闩上的时候上一次刷新已经过去了。
        self.structure_panel.structure_changed.connect(self._structure_edited)
        self.document.dirty_changed.connect(self._dirty_changed)

    def _structure_edited(self) -> None:
        """记下「这一叠层被改过」，并把底栏第一段重刷一遍。"""
        if self._structure_edit_unsaved:
            return
        self._structure_edit_unsaved = True
        self._refresh_operation_state()

    def _dirty_changed(self, dirty: bool) -> None:
        """存过盘之后那句话必须消失，否则它报的是「改过」而不是「没存」。"""
        if dirty or not self._structure_edit_unsaved:
            return
        self._structure_edit_unsaved = False
        self._refresh_operation_state()

    @property
    def structure_edit_unsaved(self) -> bool:
        """层堆叠被改过、而这次改动还没存盘。底栏第一段读它。"""
        return self._structure_edit_unsaved and self.document.is_dirty

    @property
    def close_pending(self) -> bool:
        return self._close_pending

    @property
    def close_cancel_timer(self) -> QTimer:
        return self._close_cancel_timer

    @property
    def force_close_prompt(self) -> QMessageBox | None:
        return self._force_close_prompt

    def _register_project_projections(self) -> None:
        projections = (self._project_plots, self._project_workspace)
        registered = []
        try:
            for projection in projections:
                self.document.register_project_projection(projection)
                registered.append(projection)
                projection(self.document.project)
        except Exception:
            for projection in reversed(registered):
                self.document.unregister_project_projection(projection)
            raise

    def _project_plots(self, project: api.XrrProject) -> None:
        self.plot_panel.project_project(project)

    def _project_workspace(self, project: api.XrrProject) -> None:
        restore_project(self.workspace_view, project)

    def _project_candidate(self, candidate_id: str) -> None:
        dataset_id = self.document.active_dataset_id
        dataset = next(
            (value for value in self.document.project.datasets if value.dataset_id == dataset_id),
            None,
        )
        if dataset is None or dataset.last_valid_result is None:
            raise RuntimeError("candidate projection requires an active fit result")
        self.plot_panel.set_result(dataset.last_valid_result, candidate_id)

    def _project_preview(self, qz_a_inv: object, model_normalized: object) -> None:
        """Show the searching model without touching committed project state."""
        self.plot_panel.set_preview_curve(qz_a_inv, model_normalized)

    def _plot_structure_edited(self, structure: api.StructureSpec) -> None:
        """Commit a structure hand-edited by dragging on the SLD companion pane.

        The drag emits the whole edited :class:`StructureSpec`; routing it
        through the structure panel's ``set_structure`` means a dragged edit and
        a typed edit take exactly the same validated, refit-triggering path.
        """
        self.structure_panel.set_structure(structure)

    def _discard_preview_when_idle(self, running: bool) -> None:
        if not running:
            self.plot_panel.clear_preview_curve()

    def _plot_range_requested(self, lower: float, upper: float) -> None:
        dataset_id = self.document.active_dataset_id
        if dataset_id is None:
            raise RuntimeError("plot range requires an active dataset")
        self.data_panel.set_fit_range(dataset_id, lower, upper)

    def _plot_point_requested(self, index: int) -> None:
        dataset_id = self.document.active_dataset_id
        if dataset_id is None:
            raise RuntimeError("plot point mask requires an active dataset")
        self.data_panel.set_point_enabled(dataset_id, index, False)

    def _plot_tab_changed(self, index: int) -> None:
        """换了画布那一页，右栏跟着重算一次该摆哪三段。

        帧① 与帧⑤ 在实现里是同一个流程步，差别只在画布列停在哪一页——不确定度页问的是「这条
        链能不能当证据用」，右栏得换成收敛/后验/自助那三段。步骤没变，所以 ``step_changed``
        那条边一声不响，这一声必须由换页自己发。
        """
        self._capture_workspace()
        apply_step_scope(self)

    def _layout_changed(self, *_args) -> None:
        """Persist the geometry a moved handle produced.

        A drag emits ``splitterMoved`` on every mouse move and every capture
        replaces the project, so persisting inline would push one undo entry per
        pixel crossed.  The window-owned single-shot timer collapses a whole drag
        into the single width it ended on.

        While a fit is running, preview updates repaint the plot at high
        frequency; replacing the project in that window re-projects every panel
        mid-paint, so capture waits until the fit ends.
        """
        if self._operation_is_running():
            return
        if self._capture_scheduled:
            return
        self._capture_scheduled = True
        self._capture_timer.start()

    def _capture_scheduled_layout(self) -> None:
        """Capture the settled widths once the drag's own delivery is finished."""
        self._capture_scheduled = False
        if self._workspace_released:
            return
        self._capture_workspace()

    def showEvent(self, event: object) -> None:
        """Divide the shell's real width the first time there is one to divide.

        A splitter built before its window is shown has no width, so budgets
        handed out at construction are applied against zero and Qt spreads the
        three columns evenly instead.  The first show is the earliest moment the
        design's ``264px 1fr 340px`` can actually be met; the project's own widths
        are applied straight after, so a layout the user saved still wins.
        """
        super().showEvent(event)
        if not self._columns_settled:
            self._columns_settled = True
            apply_default_columns(self.workspace_splitter)
            restore_project(self.workspace_view, self.document.project)

    def _capture_workspace(self) -> None:
        current = self.document.project
        updated = capture_project(current, self.workspace_view)
        if updated is current:
            return
        try:
            self.document.replace_project(updated)
        except Exception:
            restore_project(self.workspace_view, current)

    def set_guidance_visible(self, visible: bool) -> None:
        """Swap between the guided flow and the three-column workspace.

        Both surfaces read the same document, so switching changes only what is
        on screen.  Frame ② draws no ``appbody`` at all -- only ``cmdbar`` →
        ``wizhead`` → ``wizbody`` → ``statusbar`` -- so the guided step owns the
        full shell and both side columns step aside.  ``.appbody.two``
        (``grid-template-columns:264px 1fr``) is defined in the mockup and never
        applied to anything, so it cannot be read as leaving the rail up; the
        frame's own lead states the requirement outright: 「隐藏停靠面板与高级批量
        选项」.  Keeping the rail also put two competing numberings on one screen,
        the六段管线 beside the header's 第 N 步 / 共 4 步.
        """
        self.central_stack.setCurrentWidget(self.guidance if visible else self.plot_panel)
        self.inspector_column.setVisible(not visible)
        self.nav_column.setVisible(not visible)
        # 层堆叠栏在画布列里是 ``central_stack`` 的兄弟，换页换不掉它：不显式收起，专家结构
        # 编辑器就压在引导抬头上方，而第 2 步自己已经用平实语言把同一个结构列了一遍。
        #
        # 只有收起这一侧写在这里。放回去归 ``apply_step_scope``：专家模式下层堆叠是否在画布
        # 里由走到哪一步说话（帧③ 有、帧① 没有），这里无条件放回就会在结果态把两张图各挤掉
        # 一半——而这正是从引导切回专家最常落到的那一步。
        stack_pane = self.canvas_column.findChild(QWidget, "structurePanelScroll")
        if stack_pane is not None and visible:
            stack_pane.setVisible(False)
        apply_step_scope(self)
        set_command_bar_guided(self, visible)
        action = self.chrome_actions.get("guidanceModeAction")
        if action is not None and action.isChecked() != visible:
            action.setChecked(visible)
        self._refresh_guidance_status()
        # 底栏第一段报的是模式（引导模式 / 项目就绪），换模式就得重投一次；那一段还带着
        # 语义色的圆点，不重投会留着上一模式的颜色。
        self._refresh_operation_state()

    def _refresh_guidance_status(self, *_args) -> None:
        """Say which guided step is open, in the one place a guided user can see.

        Guided mode takes the inspector away, so the status bar is the only surface
        left that can report position at all.  Expert mode has no step to report,
        so the segment hides rather than keeping a number that stopped moving.

        序号与它两边的说明文字分开写：设计稿帧② 这一段是「第 <b>2</b> 步 / 共 4 步」，
        只有序号加粗，所以「第 」与「 步 / 共 4 步」是段里另外两片（见 ``status_bar``）。
        """
        label = self.findChild(QLabel, "guidanceStepStatus")
        tail = self.findChild(QLabel, "guidanceStepTail")
        if label is None or tail is None:
            return
        visible = self.guidance_is_visible()
        status_bar.allow(self, ("statusStepSpan",), allowed=visible)
        if visible:
            names = self.guidance.step_names()
            label.setText(str(names.index(self.guidance.current_step()) + 1))
            tail.setText(f" 步 / 共 {len(names)} 步")
        status_bar.sync(self)

    def guidance_is_visible(self) -> bool:
        return self.central_stack.currentWidget() is self.guidance

    def reset_layout(self) -> None:
        """Return the columns to the widths the design gives them.

        The way back from a drag that went too far.  It runs the same arithmetic
        the first show performs, so "reset" and "as opened" are one arrangement
        rather than two that can drift apart, and the result is captured because
        a reset the next open discards is not a reset.
        """
        apply_default_columns(self.workspace_splitter)
        self._capture_workspace()

    def _refresh_window_title(self, *_args) -> None:
        """Reflect the project name and unsaved state in the window title.

        The title uses Qt's ``[*]`` modification placeholder so each platform
        renders the unsaved marker natively (the close-button dot on macOS, a
        leading asterisk elsewhere) instead of a hand-placed asterisk that the
        window manager would not understand.

        破折号后面写的是这个软件的正式名。``README`` 首行、``docs/user-guide.md`` 与设计稿五帧
        的标题栏（``aSi_multilayer_2024-03 ● — XRR Fitter``）用的都是它——标题栏是这个名字最
        显眼的一处，只有这里写成一句功能描述，读者在任务栏、窗口列表和文档里看到的就是两个
        不同的软件。
        """
        path = self.document.path
        name = path.stem if path is not None else "未命名项目"
        self.setWindowTitle(f"{name}[*] — XRR Fitter")
        self.setWindowModified(self.document.is_dirty)

    def _require_idle(self, operation: str) -> None:
        if self._operation_is_running():
            raise RuntimeError(f"cannot {operation} while an operation is running")

    def new_project(self, *, discard_unsaved: bool = False) -> None:
        self._require_idle("create a project")
        if self.document.is_dirty and not discard_unsaved:
            raise RuntimeError("unsaved project changes require explicit discard")
        with self.autosave.discard_after_replacement():
            self.document.new()

    def open_project(
        self,
        path: str | object,
        *,
        discard_unsaved: bool = False,
    ) -> None:
        self._require_idle("open a project")
        if self.document.is_dirty and not discard_unsaved:
            raise RuntimeError("unsaved project changes require explicit discard")
        with self.autosave.discard_after_replacement():
            self.document.open(path)
        self._offer_draft_recovery()

    def maybe_recover_draft(self, path: str | object) -> bool:
        """Open a project and offer to restore any leftover autosave draft."""
        self.document.open(path)
        return self._offer_draft_recovery()

    def _offer_draft_recovery(self) -> bool:
        """Prompt to adopt a leftover draft, or clear it when declined.

        A draft beside the just-opened project means the previous session ended
        without a clean save; adopting it restores that unsaved work as dirty so
        the user can persist it, while declining removes the stale draft.
        """
        if not self.autosave.has_recoverable_draft():
            return False
        if self._confirm_recover_draft():
            self.document.replace_project(self.autosave.recover(), dirty=True)
            return True
        self.autosave.discard_draft()
        return False

    def _confirm_recover_draft(self) -> bool:
        response = QMessageBox.question(
            self,
            "恢复自动保存草稿",
            "检测到上次会话遗留的自动保存草稿。是否恢复其中未保存的更改？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return response == QMessageBox.StandardButton.Yes

    def save_project(self, path: str | object | None = None):
        self._require_idle("save a project")
        self.document.save(path)
        # A successful save supersedes any draft, so clear it to avoid a false
        # recovery offer on the next open.
        self.autosave.discard_draft()
        return self.document.path

    def start_fit(self) -> bool:
        if self._operation_is_running():
            raise RuntimeError("cannot start fit while an operation is running")
        return self.fit_panel.start_fit()

    def cancel_fit(self) -> None:
        self.fit_panel.controller.cancel()

    def export_results(self, directory: str | object):
        manifest = self.export_workflow.export_results(directory)
        self._refresh_operation_state()
        return manifest

    def export_results_dialog(self) -> object | None:
        manifest = self.export_workflow.export_results_dialog(self)
        self._refresh_operation_state()
        return manifest

    def export_summary_text(self) -> str:
        return self.export_workflow.summary_text

    def fit_is_ready(self) -> bool:
        return self._fit_readiness().ready

    def fit_readiness_text(self) -> str:
        readiness = self._fit_readiness()
        return "已就绪" if readiness.ready else messages.readiness_text(readiness.message)

    def _fit_readiness(self) -> api.FitReadiness:
        try:
            return api.preflight_fit(self.document.project)
        except (OSError, ValueError) as error:
            return api.FitReadiness(False, str(error))

    def select_active_dataset(self, dataset_id: str | None) -> None:
        self.document.select_active_dataset(dataset_id)

    def new_project_dialog(self, _checked: bool = False) -> None:
        self.project_actions.new_project()

    def open_project_dialog(self, _checked: bool = False) -> None:
        self.project_actions.open_project()

    def save_project_dialog(self, _checked: bool = False) -> None:
        self.project_actions.save_project(save_as=False)

    def save_project_as_dialog(self, _checked: bool = False) -> None:
        self.project_actions.save_project(save_as=True)

    def reload_source_dialog(self, _checked: bool = False) -> None:
        self.project_actions.reload_source()

    def relink_source_dialog(self, _checked: bool = False) -> None:
        self.project_actions.relink_source()

    def source_hash_status(self, dataset_id: str) -> str:
        return self.document.source_status(dataset_id)

    def source_warning_text(self, dataset_id: str) -> str:
        return self.document.source_warning(dataset_id)

    def set_operation_controller(self, controller: object) -> None:
        if self._operation_controller is not None:
            raise RuntimeError("operation controller is already attached")
        controller.running_changed.connect(self._resume_pending_close)
        controller.running_changed.connect(self._refresh_operation_state)
        self._operation_controller = controller
        self._refresh_operation_state()

    def _operation_is_running(self) -> bool:
        return operation_is_running(self)

    def _operation_controllers(self) -> tuple[object, ...]:
        return operation_controllers(self)

    def _operation_running_changed(self, running: bool) -> None:
        self._refresh_operation_state()
        self._resume_pending_close(running)

    def _has_exportable_results(self) -> bool:
        return has_exportable_results(self)

    def _refresh_operation_state(self, *_args) -> None:
        refresh_operation_state(self)

    def _refresh_sampling_footer(self, *_args) -> None:
        refresh_sampling_footer(self)

    def _project_run_summary(self, summary: object) -> None:
        refresh_run_summary(self, summary)

    def _is_user_close(self, event: object) -> bool:
        if self._resume_user_close:
            return True
        if not self.isVisible():
            return False
        spontaneous = event.spontaneous
        return bool(spontaneous())

    def _confirm_discard_changes(self) -> bool:
        response = QMessageBox.question(
            self,
            "未保存的项目更改",
            "当前项目有未保存的更改。是否放弃这些更改并继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def _accept_idle_close(self, event: object) -> None:
        user_close = self._is_user_close(event)
        self._resume_user_close = False
        if user_close and self.document.is_dirty and not self._confirm_discard_changes():
            event.ignore()
            return
        event.accept()
        self._release_workspace()

    def _release_workspace(self) -> None:
        if self._workspace_released:
            return
        self._workspace_released = True
        # A clean shutdown must not leave a recovery draft behind; only an
        # uncleanly terminated session (which never reaches here) should.
        self.autosave.stop()
        self.autosave.discard_draft()
        self.document.unregister_project_projection(self._project_plots)
        self.document.unregister_project_projection(self._project_workspace)
        self.plot_panel.release_resources()

    def closeEvent(self, event: object) -> None:
        if not self._operation_is_running():
            self._accept_idle_close(event)
            return
        event.ignore()
        if self._close_pending:
            return
        self._close_pending = True
        self._close_cancel_timer.start()
        for controller in self._operation_controllers():
            if controller.is_running:
                controller.cancel()

    def _resume_pending_close(self, running: bool) -> None:
        if running or self._operation_is_running() or not self._close_pending:
            return
        self._close_cancel_timer.stop()
        prompt = self._force_close_prompt
        self._force_close_prompt = None
        if prompt is not None:
            prompt.deleteLater()
        self._close_pending = False
        self._resume_user_close = True
        QTimer.singleShot(0, self.close)

    def _force_prompt_finished(self, _result: int) -> None:
        prompt = self._force_close_prompt
        if prompt is None:
            return
        response = prompt.standardButton(prompt.clickedButton())
        self._force_close_prompt = None
        prompt.deleteLater()
        if not self._close_pending:
            return
        if response != QMessageBox.StandardButton.Yes:
            self._close_pending = False
            self._close_cancel_timer.stop()
            return
        for controller in self._operation_controllers():
            if controller.is_running:
                controller.force_stop()

    def close_cancel_deadline_reached(self) -> None:
        self._close_cancel_timer.stop()
        if not self._close_pending or self._force_close_prompt is not None:
            return
        if not self._operation_is_running():
            self._resume_pending_close(False)
            return
        prompt = QMessageBox(self)
        prompt.setObjectName("forceClosePrompt")
        prompt.setWindowTitle("拟合仍在运行")
        prompt.setAccessibleName("确认强制结束后台任务")
        prompt.setIcon(QMessageBox.Icon.Warning)
        prompt.setText(
            "后台任务未在 5 秒内停止。强制结束可能丢失尚未写入的检查点，并可能损坏未完成的工作状态。是否继续？"
        )
        prompt.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        prompt.setDefaultButton(QMessageBox.StandardButton.No)
        prompt.setModal(True)
        prompt.finished.connect(self._force_prompt_finished)
        self._force_close_prompt = prompt
        prompt.open()
