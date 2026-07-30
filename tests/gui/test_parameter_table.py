from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget

import xrr_fitter.api as api


AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path):
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(instrument_id="parameter-gui"),
    )
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
        SI,
    )
    return api.set_structure(project, "sample", structure)


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    return panel


def test_parameter_table_shows_required_fields(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")

    assert table is not None
    assert [table.horizontalHeaderItem(index).text() for index in range(table.columnCount())] == [
        "参数",
        "初值",
        "下限",
        "上限",
        "单位",
        "锁定",
    ]
    assert "component.0.thickness_a" in panel.row_names
    assert "instrument.scale" in panel.row_names
    assert all(not definition.expert_only for definition in panel.visible_definitions)


def test_length_parameters_display_nm_but_emit_angstrom_settings(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    name = "component.0.thickness_a"
    displayed = panel.display_values(name)

    panel.set_display_parameter(
        name,
        initial=4.5,
        lower=2.0,
        upper=10.0,
        locked=False,
    )

    setting = panel.document.project.datasets[0].parameter_settings[0]
    assert displayed[0] == pytest.approx(4.0)
    assert panel.display_unit(name) == "nm"
    assert setting == api.ParameterSetting(name, 45.0, 20.0, 100.0)


def test_user_edit_in_parameter_table_commits_display_value_and_lock(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    name = "component.0.thickness_a"
    row = panel.row_names.index(name)

    table.item(row, 1).setText("4.5")
    table.item(row, 5).setCheckState(Qt.CheckState.Checked)

    setting = next(
        value
        for value in panel.document.project.datasets[0].parameter_settings
        if value.name == name
    )
    assert setting.initial == pytest.approx(45.0)
    assert setting.locked is True


def test_parameter_commit_routes_only_through_set_parameter_settings(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    definition = next(item for item in panel.definitions if item.name == "instrument.scale")
    setting = api.ParameterSetting(
        definition.name,
        1.5,
        definition.lower,
        definition.upper,
    )
    updated = api.set_parameter_settings(original, "sample", (setting,))
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "set_parameter_settings",
        lambda project, dataset_id, settings: (
            calls.append((project, dataset_id, settings)),
            updated,
        )[1],
    )

    panel.set_parameter(
        definition.name,
        initial=setting.initial,
        lower=setting.lower,
        upper=setting.upper,
        locked=False,
    )

    assert calls == [(original, "sample", (setting,))]
    assert panel.document.project is updated


def test_parameter_failure_preserves_project_table_and_signal(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    name = "instrument.scale"
    before = panel.document.project
    before_values = panel.display_values(name)
    events: list[object] = []
    panel.settings_changed.connect(events.append)
    monkeypatch.setattr(
        api,
        "set_parameter_settings",
        lambda *_args: (_ for _ in ()).throw(ValueError("settings rejected")),
    )

    with pytest.raises(ValueError, match="settings rejected"):
        panel.set_parameter(name, initial=2.0, lower=0.1, upper=10.0, locked=False)

    assert panel.document.project is before
    assert panel.display_values(name) == before_values
    assert events == []


def test_parameter_refresh_and_noop_commit_preserve_exact_setting_bytes(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    definition = next(item for item in panel.definitions if item.name == "instrument.scale")
    panel.set_parameter(
        definition.name,
        initial=definition.initial,
        lower=definition.lower,
        upper=definition.upper,
        locked=definition.locked,
    )
    persisted = panel.document.project
    setting = persisted.datasets[0].parameter_settings[0]
    events: list[object] = []
    panel.settings_changed.connect(events.append)

    changed = panel.set_parameter(
        setting.name,
        initial=setting.initial,
        lower=setting.lower,
        upper=setting.upper,
        locked=setting.locked,
    )

    assert changed is False
    assert panel.document.project is persisted
    assert panel.document.project.datasets[0].parameter_settings == (setting,)
    assert events == []


def test_unstructured_active_dataset_clears_parameter_projection(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "empty.xy"),
        api.InstrumentSpec(),
    )
    panel = ParametersPanel(ProjectDocument(project))
    qtbot.addWidget(panel)

    assert panel.definitions == ()
    assert panel.row_names == ()
