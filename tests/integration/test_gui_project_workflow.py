"""Complete desktop workflow contracts across the assembled Qt shell.

The focused GUI suites own individual controls. These integration cases prove
that the final MainWindow composes those controls around one ProjectDocument,
routes fit and candidate state into every projection, and preserves the same
fitted project through save, export, reopen, and deterministic teardown.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QPushButton
from shiboken6 import isValid

import xrr_fitter.api as api
from tests.support.model_cases import final_fit_result, fit_candidate, simple_structure


class _FakeJob:
    def __init__(self, events: tuple[api.OperationEvent, ...]) -> None:
        self._events = events
        self.is_running = True

    def poll(self) -> tuple[api.OperationEvent, ...]:
        events, self._events = self._events, ()
        if any(event.kind == "stopped" for event in events):
            self.is_running = False
        return events

    def cancel(self) -> None:
        pass

    def force_stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _candidate(data: api.PreparedData, candidate_id: str, objective: float):
    return replace(
        fit_candidate(candidate_id, objective),
        ranking_objective=objective,
        unit_vector=np.asarray([objective]),
        qz_a_inv=data.qz_a_inv,
        model_normalized=data.intensity_normalized,
        log_residuals_decades=np.zeros(data.qz_a_inv.size),
        weighted_residuals=np.zeros(data.qz_a_inv.size),
    )


def _project(tmp_path: Path, *, fitted: bool = True) -> api.XrrProject:
    source = _write_curve(tmp_path / "curve.xy")
    beam = api.BeamSpec("monochromatic")
    value = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id="gui-workflow", footprint_mode="none"),
        beam=beam,
    )
    value = api.set_structure(value, "curve", simple_structure())
    if not fitted:
        return value
    data = api.import_data(source, beam)
    result = final_fit_result(
        _candidate(data, "candidate-a", 0.2),
        _candidate(data, "candidate-b", 0.3),
    )
    dataset = replace(value.datasets[0], last_valid_result=result)
    value = replace(value, datasets=(dataset,))
    return api.select_candidate(value, "curve", "candidate-a")


def _actions(window) -> dict[str, QAction]:
    return {
        action.objectName(): action
        for action in window.findChildren(QAction)
        if action.objectName()
    }


EXPECTED_ACTIONS = {
    "openProjectAction": ("打开项目", "Ctrl+O"),
    "saveProjectAction": ("保存项目", "Ctrl+S"),
    "saveAsProjectAction": ("另存为", "Ctrl+Shift+S"),
    "startFitAction": ("一键拟合", "Ctrl+Return"),
    "cancelFitAction": ("取消拟合", "Esc"),
    "exportResultsAction": ("导出结果", "Ctrl+Shift+E"),
}


def _assert_workspace_layout(window) -> None:
    assert window.objectName() == "mainWindow"
    assert window.left_splitter.count() == 2
    assert window.fit_panel.parent() is not None
    assert window.result_panel.parent() is not None


def _assert_fitted_projection(window) -> None:
    assert window.result_panel.candidate_count() == 2
    assert window.result_panel.selected_candidate_id() == "candidate-a"
    assert window.plot_panel.selected_candidate_id() == "candidate-a"
    assert window.fit_is_ready() is True
    assert window.fit_readiness_text() == "已就绪"


def _assert_action_contract(window) -> None:
    actions = _actions(window)
    observed = {
        name: (actions[name].text(), actions[name].shortcut().toString())
        for name in EXPECTED_ACTIONS
    }
    assert observed == EXPECTED_ACTIONS
    assert actions["startFitAction"].isEnabled()
    assert not actions["cancelFitAction"].isEnabled()
    assert actions["exportResultsAction"].isEnabled()
    assert window.findChild(QPushButton, "exportResultsButton").isEnabled()


def test_main_window_composes_complete_fitted_workspace(qtbot, tmp_path: Path) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    target = tmp_path / "fitted.xrrproj.json"
    api.save_project(_project(tmp_path), target)
    window = MainWindow()
    qtbot.addWidget(window)

    window.open_project(target)
    window.resize(1280, 760)
    window.show()
    qtbot.wait(1)

    _assert_workspace_layout(window)
    _assert_fitted_projection(window)
    _assert_action_contract(window)


def test_candidate_selection_updates_project_plot_and_export_state(
    qtbot,
    tmp_path: Path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)

    assert window.result_panel.select_candidate("candidate-b") is True

    assert window.document.project.ui_state.selected_candidate_ids == (
        ("curve", "candidate-b"),
    )
    assert window.result_panel.selected_candidate_id() == "candidate-b"
    assert window.plot_panel.selected_candidate_id() == "candidate-b"
    assert _actions(window)["exportResultsAction"].isEnabled()


def test_expert_mcmc_controls_remain_readable_at_documented_window_size(
    qtbot,
    tmp_path: Path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.set_expert_mode(_project(tmp_path), True)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.resize(1600, 900)
    window.show()
    qtbot.wait(1)

    # The controls live in an on-demand dialog rather than the analysis column,
    # so readability is asserted where the user actually meets them.
    dialog = window.result_panel.open_uncertainty_dialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(1)

    controls = (
        window.result_panel.walkers,
        window.result_panel.burn_in,
        window.result_panel.production,
        window.result_panel.thin,
        window.result_panel.mcmc_button,
        window.result_panel.cancel_button,
        window.result_panel.force_button,
    )
    assert all(widget.height() >= widget.minimumSizeHint().height() for widget in controls)
    assert all(widget.geometry().intersected(other.geometry()).isEmpty()
               for index, widget in enumerate(controls)
               for other in controls[index + 1:])


def test_fit_completion_projects_result_and_updates_operation_actions(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    original = _project(tmp_path, fitted=False)
    fitted = _project(tmp_path)
    result = api.ProjectFitResult("independent", (), (), fitted)
    job = _FakeJob(
        (
            api.OperationEvent(0, "fit_result", fit_result=result),
            api.OperationEvent(1, "stopped"),
        )
    )
    monkeypatch.setattr(api, "preflight_fit", lambda _project: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    window = MainWindow(ProjectDocument(original))
    qtbot.addWidget(window)

    assert window.start_fit() is True
    assert _actions(window)["cancelFitAction"].isEnabled()
    window.fit_panel.controller.poll_now()

    assert window.document.project is fitted
    assert window.result_panel.candidate_count() == 2
    assert window.plot_panel.selected_candidate_id() == "candidate-a"
    assert not _actions(window)["cancelFitAction"].isEnabled()
    assert _actions(window)["exportResultsAction"].isEnabled()


def test_save_export_reopen_and_delete_complete_window(
    qtbot,
    tmp_path: Path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    target = tmp_path / "saved.xrrproj.json"

    assert window.save_project(target) == target
    manifest = window.export_results(tmp_path / "exports")
    reopened = MainWindow()
    qtbot.addWidget(reopened)
    reopened.open_project(target)

    assert manifest.run_directory.is_dir()
    assert str(manifest.run_directory) in window.export_summary_text()
    assert reopened.result_panel.selected_candidate_id() == "candidate-a"
    assert reopened.plot_panel.selected_candidate_id() == "candidate-a"
    assert window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(window)
