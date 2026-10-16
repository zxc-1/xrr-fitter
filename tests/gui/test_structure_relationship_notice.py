"""A structure edit that retires parameter work has to say what it retired."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QLabel

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)
FILM = api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path: Path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.structure.panel import StructurePanel

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(instrument_id="relationship-notice"),
    )
    dataset_id = project.datasets[0].dataset_id
    project = api.select_active_dataset(project, dataset_id)
    project = api.set_structure(project, dataset_id, api.StructureSpec(AIR, (FILM,), SI))
    thickness = next(
        item for item in api.describe_parameters(project, dataset_id) if item.name == "component.0.thickness_a"
    )
    fixed = api.ParameterSetting(
        thickness.name, thickness.initial, thickness.lower, thickness.upper, api.ParameterFreedom.FIXED
    )
    project = api.set_parameter_settings(project, dataset_id, (fixed,))
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel, thickness.display_name


def test_a_value_edit_keeps_settings_and_says_nothing(qtbot, tmp_path: Path) -> None:
    panel, _caption = _panel(qtbot, tmp_path)
    structure = panel.structure

    assert panel.set_structure(replace(structure, components=(replace(FILM, roughness_a=4.0),)))

    notice = panel.findChild(QLabel, "structureRelationshipNotice")
    assert notice is not None and notice.isHidden()
    assert panel.document.project.datasets[0].parameter_settings != ()


def test_an_edit_that_retires_a_setting_names_it(qtbot, tmp_path: Path) -> None:
    panel, caption = _panel(qtbot, tmp_path)
    structure = panel.structure
    cap = api.LayerSpec("cap", SIO2, 20.0)

    assert panel.set_structure(replace(structure, components=(cap, FILM)))

    notice = panel.findChild(QLabel, "structureRelationshipNotice")
    assert notice is not None and not notice.isHidden()
    assert "已移除" in notice.text()
    assert caption in notice.text()
    assert panel.document.project.datasets[0].parameter_settings == ()
