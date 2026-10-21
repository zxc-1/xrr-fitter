"""参数表跟着选中的层走，并且一组能一次固定或放开。"""

from __future__ import annotations

from pathlib import Path

import xrr_fitter.api as api

SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path, layers: int = 6):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "sample.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        tuple(api.LayerSpec(f"film{index}", SIO2, 400.0, roughness_a=3.0) for index in range(layers)),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    panel = ParametersPanel(ProjectDocument(api.set_structure(project, "sample", structure)))
    qtbot.addWidget(panel)
    return panel


def test_focusing_a_component_scrolls_to_its_group_and_selects_its_first_row(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.resize(420, 300)
    panel.show()
    qtbot.waitExposed(panel)
    table = panel.parameter_table

    panel.focus_component(5)

    assert table.current_name() is not None and table.current_name().startswith("component.5.")
    caption_row = table.currentRow() - 1
    assert table.item(caption_row, 0).text().startswith("film5")
    assert table.verticalScrollBar().value() > 0
    assert table.viewport().rect().intersects(table.visualItemRect(table.item(caption_row, 0)))


def test_a_group_can_be_fixed_and_freed_in_one_action(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, layers=2)
    free = {item.name for item in panel.definitions if item.name.startswith("component.1.") and not item.locked}
    assert free

    panel.set_group_freedom("component.1", api.ParameterFreedom.FIXED)

    settings = {item.name: item for item in panel.document.project.datasets[0].parameter_settings}
    assert {name for name, item in settings.items() if item.freedom is api.ParameterFreedom.FIXED} == free
    assert not any(name.startswith("component.0.") for name in settings)
    assert "已固定" in panel.status_label.text()

    panel.set_group_freedom("component.1", api.ParameterFreedom.FREE)

    settings = panel.document.project.datasets[0].parameter_settings
    assert all(item.freedom is api.ParameterFreedom.FREE for item in settings if item.name in free)


def test_the_row_menu_offers_the_group_actions_and_a_caption_row_has_a_menu_too(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, layers=2)

    names = {action.objectName() for action in panel._row_context_menu("component.1.thickness_a").actions()}
    assert {"fixGroupAction", "freeGroupAction"} <= names

    caption = {action.objectName() for action in panel._group_context_menu("component.1").actions()}
    assert caption == {"fixGroupAction", "freeGroupAction"}
