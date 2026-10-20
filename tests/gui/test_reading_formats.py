"""One J, one unit, one word per quantity, wherever the same fit is read back.

The audit found the same objective read as ``J=0.01`` on the rail and
``J = 0.00759`` in the status bar (``.2f`` against ``:g``), the noise model and
residual unit leaking as ``robust_log [decade]``, a periodic block's sublayer density
labelled 相对密度 beside a value in g·cm⁻³, its top-interface roughness left in Å
while every other roughness reads nm, an integer repeat count printed as ``20.000``,
and cancelling the substrate question reported as a failed import.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QDialog, QMessageBox
from tests.gui.data_import_support import _panel, _saved_preset, _write_curve
from tests.gui.test_export_dialog import _fitted_project

import xrr_fitter.api as api
from xrr_fitter.model.parameters import ParameterValue

SMALL_J = 0.007593


def _with_objective(project: api.XrrProject, objective: float) -> api.XrrProject:
    dataset = project.datasets[0]
    result = dataset.last_valid_result
    candidates = tuple(replace(candidate, objective=objective) for candidate in result.candidates)
    return replace(project, datasets=(replace(dataset, last_valid_result=replace(result, candidates=candidates)),))


def _j_values(text: str) -> list[str]:
    return re.findall(r"J ?= ?([-+0-9.eE]+)", text)


def test_every_summary_reads_the_same_objective_the_same_way(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.chrome import _fit_metrics_text
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.navigation.panel import _result_state

    project = _with_objective(_fitted_project(tmp_path), SMALL_J)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)

    readings = {
        "rail": _j_values(_result_state(window.document.project)),
        "status bar": _j_values(_fit_metrics_text(window)),
        "dataset tooltip": _j_values(window.data_panel._fit_status_tooltip("curve")),
    }

    assert readings == {name: ["0.00759"] for name in readings}


def test_the_running_card_does_not_round_a_small_objective_to_zero(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    panel = _panel(qtbot, ProjectDocument(_fitted_project(tmp_path)))
    panel.set_running(True)
    panel.set_run_progress(api.FitProgress("curve", "B", 1, 10, SMALL_J, "search"))

    assert _j_values(panel.run_subline("curve")) == ["0.00759"]


def test_the_status_bar_names_the_noise_model_in_chinese(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.chrome import _fit_metrics_text
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_fitted_project(tmp_path)))
    qtbot.addWidget(window)
    text = _fit_metrics_text(window)

    assert "robust_log" not in text and "decade" not in text
    assert "稳健对数" in text


def _definition(name: str, display_name: str, unit: str, **changes) -> api.ParameterDefinition:
    values = {
        "name": name,
        "display_name": display_name,
        "unit": unit,
        "category": "structure",
        "initial": 1.0,
        "lower": 0.0,
        "upper": 100.0,
        "transform": "linear",
        "locked": False,
    }
    values.update(changes)
    return api.ParameterDefinition(**values)


def test_a_periodic_sublayer_density_shown_absolute_is_not_called_relative() -> None:
    from xrr_fitter.gui.parameters.grouping import display_unit, row_name

    definition = _definition("component.0.layer.0.density_scale", "Mo 相对密度", "")

    assert display_unit(definition, 10.2) == "g·cm⁻³"
    assert row_name(definition.display_name, "Mo/Si", 10.2) == "Mo 密度 ρ"
    # 拿不到体密度时行上仍是倍率，名字也仍说「相对」。
    assert row_name(definition.display_name, "Mo/Si", None) == "Mo 相对密度 ρ"


def test_a_periodic_block_top_roughness_reads_in_nm_like_every_other_roughness() -> None:
    from xrr_fitter.gui.parameters.grouping import display_scale, display_unit

    top = _definition("component.0.top_roughness_a", "Mo/Si 顶界面粗糙度", "Å")
    layer = _definition("component.1.roughness_a", "SiO2 入射侧粗糙度", "Å")

    assert (display_unit(top), display_scale(top)) == (display_unit(layer), display_scale(layer)) == ("nm", 0.1)


def test_an_integer_parameter_reads_without_decimals_in_the_result_table(qtbot) -> None:
    from tests.support.model_cases import final_fit_result, fit_candidate

    from xrr_fitter.gui.results.values import ResultValueTable

    definitions = (
        _definition("component.0.repeats", "Mo/Si 重复次数", "", locked=True, integer=True, initial=20.0),
        _definition("instrument.scale", "强度标度", "", category="instrument"),
    )
    values = (
        ParameterValue("component.0.repeats", 20.0, 20.0, 20.0),
        ParameterValue("instrument.scale", 0.982, 0.0, 100.0),
    )
    candidate = replace(fit_candidate("candidate-a", 0.2), parameters=values, unit_vector=np.zeros(len(values)))
    result = replace(final_fit_result(candidate), parameter_definitions=definitions)
    table = ResultValueTable()
    qtbot.addWidget(table)

    table.project_result(result, "candidate-a", {"component.0": "Mo/Si"})

    shown = [
        table.item(row, 1).text()
        for row in range(table.rowCount())
        if table.item(row, 0) is not None and table.item(row, 0).text().startswith("重复次数")
    ]
    assert shown == ["20"]


def test_cancelling_the_substrate_question_cancels_quietly(qtbot, tmp_path, monkeypatch) -> None:
    from xrr_fitter.gui.data.substrate_dialog import SubstrateDialog
    from xrr_fitter.gui.document import ProjectDocument

    project = replace(api.new_project(), measurement_preset=_saved_preset())
    panel = _panel(qtbot, ProjectDocument(project))
    source = _write_curve(tmp_path / "P1 Si+Zr.xy")
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, title, text, *_a: errors.append((title, text)))
    monkeypatch.setattr(SubstrateDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)

    panel._confirm_import((source,), folder=False)

    assert errors == []
    assert panel.document.project.datasets == ()
