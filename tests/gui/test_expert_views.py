from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QCheckBox

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path):
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
        SI,
    )
    return api.set_structure(project, "sample", structure)


def test_expert_mode_toggle_updates_only_project_ui_state(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    document = ProjectDocument(_project(tmp_path))
    panel = ParametersPanel(document)
    qtbot.addWidget(panel)
    original = document.project
    updated = api.set_expert_mode(original, True)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "set_expert_mode",
        lambda project, enabled: (
            calls.append((project, enabled)),
            updated,
        )[1],
    )

    panel.set_expert_mode(True)

    assert calls == [(original, True)]
    assert document.project is updated
    assert document.project.datasets == original.datasets


def test_expert_mode_projects_expert_only_parameter_rows(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    standard = panel.row_names

    panel.set_expert_mode(True)

    assert len(panel.row_names) > len(standard)
    assert "instrument.absolute_sigma_a_inv" in panel.row_names
    assert panel.expert_mode is True


def test_main_window_composes_data_structure_and_parameter_panels(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)

    assert window.data_panel.document is window.document
    assert window.structure_panel.document is window.document
    assert window.parameters_panel.document is window.document
    assert window.structure_panel.structure == window.document.project.datasets[0].structure
    assert "component.0.thickness_a" in window.parameters_panel.row_names


def test_expert_toggle_control_has_visible_name_and_reflects_document(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    toggle = window.findChild(QCheckBox, "expertModeToggle")

    assert toggle is not None
    assert toggle.text() == "专家模式"
    assert toggle.accessibleName() == "切换专家参数"

    toggle.setChecked(True)

    assert window.document.project.ui_state.expert_mode is True
