"""Minimal desktop shell and asynchronous close coordination."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QSplitter, QVBoxLayout, QWidget

from xrr_fitter.gui.document import ProjectDocument


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
        self.setWindowTitle("XRR 全自动拟合")
        self.setMinimumSize(1280, 760)
        if operation_controller is not None:
            self.set_operation_controller(operation_controller)

    def _install_workspace(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("workspaceSplitter")
        splitter.setChildrenCollapsible(False)
        for name, label in (
            ("projectColumn", "项目与数据"),
            ("plotColumn", "反射率与 SLD"),
            ("analysisColumn", "参数与结果"),
        ):
            splitter.addWidget(_column(name, label))
        for index in range(splitter.count()):
            splitter.setCollapsible(index, False)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    @property
    def close_pending(self) -> bool:
        return self._close_pending

    @property
    def close_cancel_timer(self) -> QTimer:
        return self._close_cancel_timer

    @property
    def force_close_prompt(self) -> QMessageBox | None:
        return self._force_close_prompt

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
