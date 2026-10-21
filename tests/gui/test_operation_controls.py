"""Operation-state gates for homepage commands and their alternate entrances."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractButton

import xrr_fitter.api as api

PROJECT_BUTTON_NAMES = (
    "newProjectButton",
    "openProjectButton",
    "saveProjectButton",
    "datasetAddButton",
    "importFolderQuickButton",
    "importFilesButton",
    "importFolderButton",
    "changeMeasurementPresetButton",
    "emptyStateImportButton",
    "emptyStateImportFolderButton",
)
PROJECT_ACTION_NAMES = (
    "newProjectAction",
    "openProjectAction",
    "saveProjectAction",
    "saveAsProjectAction",
    "importFilesMenuAction",
    "importFolderMenuAction",
    "reloadSourceAction",
    "relinkSourceAction",
)
HISTORY_ACTION_NAMES = ("undoAction", "redoAction")
FIT_STOP_BUTTON_NAMES = (
    "pauseFitButton",
    "cancelFitButton",
    "toolbarPauseButton",
    "toolbarStopButton",
)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _window(qtbot, tmp_path: Path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(),
    )
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    return window


class _FakeRunningJob:
    def cancel(self) -> None:
        pass


class _FakeExternalOperation(QObject):
    running_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.is_running = False

    def cancel(self) -> None:
        self.is_running = False
        self.running_changed.emit(False)


def _action(window, object_name: str) -> QAction:
    for action in (*window.actions(), *window.chrome_actions.values()):
        if action.objectName() == object_name:
            return action
    raise AssertionError(f"missing action {object_name}")


def _button(window, object_name: str) -> QAbstractButton:
    button = window.findChild(QAbstractButton, object_name)
    assert button is not None, object_name
    return button


def _assert_objects(window, names, getter, enabled: bool) -> None:
    for name in names:
        assert getter(window, name).isEnabled() is enabled, name


def _assert_buttons(window, enabled: bool) -> None:
    _assert_objects(window, PROJECT_BUTTON_NAMES, _button, enabled)


def _assert_actions(window, names, enabled: bool) -> None:
    _assert_objects(window, names, _action, enabled)


# 导入快捷键就挂在 ``importFilesMenuAction`` / ``importFolderMenuAction`` 上，禁用这两条命令
# 就禁用了按键，所以 PROJECT_ACTION_NAMES 那一组已经覆盖它们。


def _assert_running_mutating_routes(window) -> None:
    _assert_buttons(window, enabled=False)
    _assert_actions(window, PROJECT_ACTION_NAMES + HISTORY_ACTION_NAMES, enabled=False)


def _assert_idle_mutating_routes(window) -> None:
    _assert_buttons(window, enabled=True)
    _assert_actions(window, PROJECT_ACTION_NAMES, enabled=True)


def _start_fit(window):
    controller = window.fit_panel.controller
    controller._job = _FakeRunningJob()
    controller.running_changed.emit(True)
    return controller


def _stop_fit(controller) -> None:
    controller._job = None
    controller.running_changed.emit(False)


def _start_mcmc(window):
    controller = window.result_panel.controller
    controller._job = _FakeRunningJob()
    controller.running_changed.emit(True)
    return lambda: _stop_fit(controller)


def _start_external(window):
    controller = _FakeExternalOperation()
    window.set_operation_controller(controller)
    controller.is_running = True
    controller.running_changed.emit(True)
    return controller.cancel


def test_running_fit_disables_mutating_home_routes(qtbot, tmp_path: Path) -> None:
    """Every homepage route that can mutate the project is gated during a fit."""
    window = _window(qtbot, tmp_path)
    controller = _start_fit(window)
    _assert_running_mutating_routes(window)
    _stop_fit(controller)


def test_running_fit_keeps_stop_controls_available(qtbot, tmp_path: Path) -> None:
    """A running fit keeps the controls needed to pause or stop it."""
    window = _window(qtbot, tmp_path)
    controller = _start_fit(window)
    assert _action(window, "cancelFitAction").isEnabled() is True
    _assert_objects(window, FIT_STOP_BUTTON_NAMES, _button, enabled=True)
    _stop_fit(controller)


@pytest.mark.parametrize(
    "start_operation",
    (_start_mcmc, _start_external),
    ids=("mcmc", "external"),
)
def test_non_fit_operations_disable_mutating_home_routes(
    qtbot,
    tmp_path: Path,
    start_operation,
) -> None:
    """MCMC and attached external work use the same project mutation gate."""
    window = _window(qtbot, tmp_path)
    stop_operation = start_operation(window)
    try:
        _assert_running_mutating_routes(window)
    finally:
        stop_operation()


def test_mutating_home_routes_restore_after_fit_stops(qtbot, tmp_path: Path) -> None:
    """The idle projection restores source/import commands and shortcuts."""
    window = _window(qtbot, tmp_path)
    controller = _start_fit(window)
    _stop_fit(controller)
    _assert_idle_mutating_routes(window)
    assert _action(window, "undoAction").isEnabled() is window.document.can_undo
    assert _action(window, "redoAction").isEnabled() is window.document.can_redo
