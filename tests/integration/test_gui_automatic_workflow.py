from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QTableWidget
from tests.support.model_cases import final_fit_result, fit_candidate

import xrr_fitter.api as api
from xrr_fitter.model.parameters import ParameterValue


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


class FakeJob:
    is_running = True

    def __init__(self, project: api.XrrProject) -> None:
        candidate = replace(
            fit_candidate(),
            parameters=tuple(
                ParameterValue(name, value, lower, upper)
                for name, value, lower, upper in (
                    ("component.0.thickness_a", 100.0, 1.0, 1000.0),
                    ("component.0.roughness_a", 3.0, 0.0, 20.0),
                    ("component.0.density_scale", 1.0, 0.5, 1.5),
                    ("component.1.thickness_a", 10.0, 2.0, 50.0),
                    ("component.1.roughness_a", 3.0, 0.0, 20.0),
                    ("component.1.density_scale", 1.0, 1.0, 1.0),
                )
            ),
            qz_a_inv=np.linspace(0.01, 0.2, 32),
            model_normalized=np.linspace(1.0, 0.1, 32),
            log_residuals_decades=np.zeros(32),
            weighted_residuals=np.zeros(32),
        )
        dataset = replace(
            project.datasets[0],
            last_valid_result=final_fit_result(candidate),
        )
        checkpoint = replace(project, datasets=(dataset, *project.datasets[1:]))
        self.events = (
            api.OperationEvent(0, "checkpoint", checkpoint=checkpoint),
            api.OperationEvent(1, "cancelled", cancellation="requested"),
            api.OperationEvent(2, "stopped"),
        )

    def poll(self):
        events, self.events = self.events, ()
        if events:
            self.is_running = False
        return events

    def cancel(self) -> None:
        self.is_running = False

    def force_stop(self) -> None:
        self.is_running = False

    def close(self) -> None:
        pass


def test_opening_a_shipped_example_enables_the_automatic_fit_button(
    qtbot,
    tmp_path,
) -> None:
    """The shipped examples exist to demonstrate the headline automatic action.

    They are built rather than imported, so they used to carry neither the
    project preset nor the automation markers the preflight requires, leaving
    「自动拟合」 greyed out on exactly the projects meant to showcase it.
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.io.examples import write_examples

    destination = tmp_path / "examples"
    write_examples(destination)
    window = MainWindow(ProjectDocument())
    qtbot.addWidget(window)

    for stem in ("single-layer", "mo-si-periodic"):
        window.document.open(destination / f"{stem}.xrrproj.json")

        assert window.fit_panel.automatic_button.isEnabled(), stem


def test_partial_import_waits_for_manual_fit_keeps_failure_recovery_and_publishes_curve(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    preset = api.MeasurementPreset(
        "integration-lab",
        api.BeamSpec("monochromatic", wavelength_a=1.5406),
        api.InstrumentSpec(instrument_id="integration-lab"),
    )
    project = replace(api.new_project(), measurement_preset=preset)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    starts: list[tuple[api.XrrProject, str | None]] = []

    def start(value, import_batch_id=None, checkpoint_path=None):
        del checkpoint_path
        starts.append((value, import_batch_id))
        return FakeJob(value)

    monkeypatch.setattr(api, "start_automatic_fit_job", start, raising=False)
    valid = _write_curve(tmp_path / "P1 Zr.xy")
    bad = _write_curve(tmp_path / "bad-name.xy")

    result = window.data_panel.import_paths((valid, bad))

    dataset = window.document.project.datasets[0]
    assert (
        starts,
        dataset.automation.status.value,
        dataset.last_valid_result,
        window.fit_panel.automatic_button.isEnabled(),
    ) == ([], "pending", None, True)

    window.fit_panel.automatic_button.click()

    assert (
        len(starts),
        starts[0][0] is result.updated_project,
        starts[0][1],
    ) == (1, True, None)

    window.fit_panel.controller.poll_now()

    failures = window.data_panel.findChild(QTableWidget, "importFailureTable")
    assert (
        window.document.project.datasets[0].last_valid_result is not None,
        failures.rowCount(),
        failures.item(0, 0).text(),
        bool(failures.item(0, 2).text()),
    ) == (True, 1, "bad-name.xy", True)
