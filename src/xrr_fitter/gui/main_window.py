"""Main application shell and asynchronous close coordination.

The window composes visible panels while :class:`ProjectDocument` owns the
immutable project identity.  Plot projection is registered synchronously at
that document boundary; ordinary Qt signals run only after a successful
precommit.  Close handling remains asynchronous so active workers never block
the GUI event loop.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QTimer
from PySide6.QtWidgets import QMainWindow, QMessageBox

import xrr_fitter.api as api
from xrr_fitter.gui import messages
from xrr_fitter.gui.accessibility import (
    configure_accessibility,
    configure_focus_navigation,
)
from xrr_fitter.gui.chrome import install_chrome
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.export.dialog import ExportWorkflow
from xrr_fitter.gui.operation_state import (
    has_exportable_results,
    operation_controllers,
    operation_is_running,
    refresh_operation_state,
)
from xrr_fitter.gui.project.autosave import AutosaveController
from xrr_fitter.gui.window_layout import install_workflow_actions, install_workspace
from xrr_fitter.gui.workspace import (
    WorkspaceView,
    capture_project,
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
        self._restoring_docks = False
        self._docks_settled = False
        self._close_cancel_timer = QTimer(self)
        self._close_cancel_timer.setObjectName("closeCancelTimer")
        self._close_cancel_timer.setSingleShot(True)
        self._close_cancel_timer.setInterval(5000)
        self._close_cancel_timer.timeout.connect(self.close_cancel_deadline_reached)
        self.autosave = AutosaveController(self.document, parent=self)
        self.autosave.start()
        install_workspace(self, self.document)
        self.workspace_view = WorkspaceView.from_root(self)
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
        self._refresh_operation_state()
        self._refresh_window_title()
        self.setMinimumSize(1280, 760)
        if operation_controller is not None:
            self.set_operation_controller(operation_controller)

    def _connect_workflows(self) -> None:
        self.plot_panel.fit_range_requested.connect(self._plot_range_requested)
        self.plot_panel.point_mask_requested.connect(self._plot_point_requested)
        self.plot_panel.view_changed.connect(self._plot_tab_changed)
        self.plot_panel.import_requested.connect(
            self.data_panel.import_files_button.click
        )
        for dock in self.docks.values():
            dock.dockLocationChanged.connect(self._dock_layout_changed)
            dock.topLevelChanged.connect(self._dock_layout_changed)
            dock.visibilityChanged.connect(self._dock_layout_changed)
        self.result_panel.candidate_selected.connect(self._project_candidate)
        self.result_panel.candidate_inspected.connect(self._project_candidate)
        self.fit_panel.running_changed.connect(self._operation_running_changed)
        self.fit_panel.preview_available.connect(self._project_preview)
        self.fit_panel.running_changed.connect(self._discard_preview_when_idle)
        self.result_panel.controller.running_changed.connect(
            self._operation_running_changed
        )
        self.document.project_changed.connect(self._restore_workspace)
        self.document.project_changed.connect(self._refresh_operation_state)
        self.document.dirty_changed.connect(self._refresh_window_title)
        self.document.path_changed.connect(self._refresh_window_title)

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
            (
                value
                for value in self.document.project.datasets
                if value.dataset_id == dataset_id
            ),
            None,
        )
        if dataset is None or dataset.last_valid_result is None:
            raise RuntimeError("candidate projection requires an active fit result")
        self.plot_panel.set_result(dataset.last_valid_result, candidate_id)

    def _project_preview(self, qz_a_inv: object, model_normalized: object) -> None:
        """Show the searching model without touching committed project state."""
        self.plot_panel.set_preview_curve(qz_a_inv, model_normalized)

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
        if index != self.document.project.ui_state.plot_tab_index:
            self._capture_workspace()

    def _dock_layout_changed(self, *_args) -> None:
        """Persist a rearrangement the user actually made.

        Docks emit these signals while the default layout is assembled, while a
        restore is in flight, and as a side effect of the first show; capturing
        then would either dirty a freshly opened project or write back the state
        being applied.
        """
        if self._restoring_docks or not self._docks_settled:
            return
        self.capture_dock_layout()

    def showEvent(self, event: object) -> None:
        """Adopt the shown geometry as the default the user is compared against.

        ``saveState`` before the first show carries no real geometry, so the
        pre-show baseline would never equal a post-show capture and every window
        would look rearranged.
        """
        super().showEvent(event)
        if not self._docks_settled:
            self._default_dock_state = self.saveState()
            self._docks_settled = True
            self.restore_dock_layout(self.document.project)

    def _capture_workspace(self) -> None:
        current = self.document.project
        updated = capture_project(current, self.workspace_view)
        if updated is current:
            return
        try:
            self.document.replace_project(updated)
        except Exception:
            restore_project(self.workspace_view, current)

    def _restore_workspace(self, project: api.XrrProject) -> None:
        restore_project(self.workspace_view, project)
        self.restore_dock_layout(project)

    def capture_dock_layout(self) -> bool:
        """Persist the current dock arrangement onto the project.

        A layout matching the construction default persists as the empty string
        rather than its byte encoding, so merely showing a window (which Qt
        reports as a dock visibility change) cannot dirty an untouched project.
        """
        current_state = self.saveState()
        state = (
            ""
            if current_state == self._default_dock_state
            else bytes(current_state.toBase64().data()).decode("ascii")
        )
        current = self.document.project
        updated = api.set_dock_state(current, state)
        if updated is current:
            return False
        self.document.replace_project(updated)
        return True

    def restore_dock_layout(self, project: api.XrrProject) -> bool:
        """Apply a persisted arrangement, falling back to the default layout.

        ``saveState`` bytes belong to the Qt build that produced them, so a
        project written by another version can legitimately fail to restore.
        That must not make the project unopenable, so any rejection or decoding
        failure leaves the default arrangement in place.
        """
        state = project.ui_state.dock_state
        if not state:
            return False
        try:
            payload = QByteArray.fromBase64(state.encode("ascii"))
        except UnicodeEncodeError:
            payload = QByteArray()
        self._restoring_docks = True
        try:
            if payload.isEmpty() or not self.restoreState(payload):
                self._apply_default_dock_layout()
                return False
        finally:
            self._restoring_docks = False
        return True

    def reset_dock_layout(self) -> None:
        """Return every dock to the arrangement built at construction."""
        self._restoring_docks = True
        try:
            self._apply_default_dock_layout()
        finally:
            self._restoring_docks = False

    def _apply_default_dock_layout(self) -> None:
        self.restoreState(self._default_dock_state)
        for dock in self.docks.values():
            dock.show()

    def _refresh_window_title(self, *_args) -> None:
        """Reflect the project name and unsaved state in the window title.

        The title uses Qt's ``[*]`` modification placeholder so each platform
        renders the unsaved marker natively (the close-button dot on macOS, a
        leading asterisk elsewhere) instead of a hand-placed asterisk that the
        window manager would not understand.
        """
        path = self.document.path
        name = path.stem if path is not None else "未命名项目"
        self.setWindowTitle(f"{name}[*] — XRR 全自动拟合")
        self.setWindowModified(self.document.is_dirty)

    def _require_idle(self, operation: str) -> None:
        if self._operation_is_running():
            raise RuntimeError(f"cannot {operation} while an operation is running")

    def new_project(self, *, discard_unsaved: bool = False) -> None:
        self._require_idle("create a project")
        if self.document.is_dirty and not discard_unsaved:
            raise RuntimeError("unsaved project changes require explicit discard")
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
            "后台任务未在 5 秒内停止。强制结束可能丢失尚未写入的检查点，"
            "并可能损坏未完成的工作状态。是否继续？"
        )
        prompt.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        prompt.setDefaultButton(QMessageBox.StandardButton.No)
        prompt.setModal(True)
        prompt.finished.connect(self._force_prompt_finished)
        self._force_close_prompt = prompt
        prompt.open()
