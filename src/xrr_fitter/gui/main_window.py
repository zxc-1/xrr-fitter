"""Main application shell and asynchronous close coordination.

The window composes visible panels while :class:`ProjectDocument` owns the
immutable project identity. Plot projection is registered synchronously at
that document boundary; ordinary Qt signals run only after a successful
precommit. Close handling remains asynchronous so active workers never block
the GUI event loop.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QSplitter, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui.data.panel import DataPanel
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.parameters.panel import ParametersPanel
from xrr_fitter.gui.plots.panel import PlotPanel
from xrr_fitter.gui.project.actions import ProjectActions
from xrr_fitter.gui.structure.panel import StructurePanel


def _column(name: str, label: str) -> QWidget:
    widget = QWidget()
    widget.setObjectName(name)
    layout = QVBoxLayout(widget)
    heading = QLabel(label)
    heading.setObjectName(f"{name}Heading")
    layout.addWidget(heading)
    layout.addStretch(1)
    return widget


class MainWindow(QMainWindow):
    """Own the immutable document and stable three-column shell."""

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
        self._close_cancel_timer = QTimer(self)
        self._close_cancel_timer.setObjectName("closeCancelTimer")
        self._close_cancel_timer.setSingleShot(True)
        self._close_cancel_timer.setInterval(5000)
        self._close_cancel_timer.timeout.connect(self.close_cancel_deadline_reached)
        self._install_workspace()
        self._register_plot_projection()
        self._connect_plot_workflow()
        self.setWindowTitle("XRR 全自动拟合")
        self.setMinimumSize(1280, 760)
        if operation_controller is not None:
            self.set_operation_controller(operation_controller)

    def _install_workspace(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter = splitter
        splitter.setObjectName("workspaceSplitter")
        splitter.setChildrenCollapsible(False)
        for name, label in (
            ("projectColumn", "项目与数据"),
            ("plotColumn", "反射率与 SLD"),
            ("analysisColumn", "参数与结果"),
        ):
            column = _column(name, label)
            splitter.addWidget(column)
            if name == "projectColumn":
                self.project_actions = ProjectActions(self, self.document)
                self.data_panel = DataPanel(self.document)
                self.structure_panel = StructurePanel(self.document)
                for panel in (
                    self.project_actions,
                    self.data_panel,
                    self.structure_panel,
                ):
                    column.layout().insertWidget(column.layout().count() - 1, panel)
            elif name == "plotColumn":
                self.plot_panel = PlotPanel()
                column.layout().insertWidget(
                    column.layout().count() - 1,
                    self.plot_panel,
                )
            elif name == "analysisColumn":
                self.parameters_panel = ParametersPanel(self.document)
                column.layout().insertWidget(
                    column.layout().count() - 1,
                    self.parameters_panel,
                )
        for index in range(splitter.count()):
            splitter.setCollapsible(index, False)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(self.document.project.ui_state.workspace_splitter_sizes))
        self.setCentralWidget(splitter)

    def _register_plot_projection(self) -> None:
        self.document.register_project_projection(self._project_plots)
        try:
            self._project_plots(self.document.project)
        except Exception:
            self.document.unregister_project_projection(self._project_plots)
            raise

    def _connect_plot_workflow(self) -> None:
        self.plot_panel.fit_range_requested.connect(self._plot_range_requested)
        self.plot_panel.point_mask_requested.connect(self._plot_point_requested)
        self.plot_panel.view_changed.connect(self._plot_tab_changed)
        self.workspace_splitter.splitterMoved.connect(self._workspace_changed)

    def _project_plots(self, project: api.XrrProject) -> None:
        self.plot_panel.project_project(project)

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
        if index == self.document.project.ui_state.plot_tab_index:
            return
        self._commit_workspace(plot_tab_index=index)

    def _workspace_changed(self, _position: int, _index: int) -> None:
        self._commit_workspace()

    def _commit_workspace(self, *, plot_tab_index: int | None = None) -> None:
        current = self.document.project
        state = replace(
            current.ui_state,
            workspace_splitter_sizes=tuple(self.workspace_splitter.sizes()),
            plot_tab_index=(
                current.ui_state.plot_tab_index
                if plot_tab_index is None
                else plot_tab_index
            ),
        )
        updated = api.set_workspace_state(current, state)
        if updated == current:
            return
        try:
            self.document.replace_project(updated)
        except Exception:
            blocker = QSignalBlocker(self.workspace_splitter)
            self.workspace_splitter.setSizes(
                list(current.ui_state.workspace_splitter_sizes)
            )
            del blocker

    @property
    def close_pending(self) -> bool:
        return self._close_pending

    @property
    def close_cancel_timer(self) -> QTimer:
        return self._close_cancel_timer

    @property
    def force_close_prompt(self) -> QMessageBox | None:
        return self._force_close_prompt

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

    def save_project(self, path: str | object | None = None):
        self._require_idle("save a project")
        self.document.save(path)
        return self.document.path

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
        self._operation_controller = controller

    def _operation_controllers(self) -> tuple[object, ...]:
        return (() if self._operation_controller is None else (self._operation_controller,))

    def _operation_is_running(self) -> bool:
        return any(controller.is_running for controller in self._operation_controllers())

    def _is_user_close(self, event: object) -> bool:
        if self._resume_user_close:
            return True
        if not self.isVisible():
            return False
        return bool(event.spontaneous())

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
