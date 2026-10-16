"""同一层的密度在参数表、选中层卡、结果表里是同一个数：有效密度 = 体密度 × density_scale。

此前参数表写「相对密度 0.5」，卡片写体密度 2.2 g·cm⁻³，结果表写 1.1 g·cm⁻³；卡片的界限条
还按 scale=1 定位。读者在三处读到三个数，分不清哪一个是拟合器真正用的。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import xrr_fitter.api as api

DENSITY = "component.0.density_scale"
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.2)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path, scale: float = 0.5):
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "sample.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", SIO2, 400.0, roughness_a=3.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    project = api.set_structure(project, "sample", structure)
    declared = next(item for item in api.describe_parameters(project, "sample") if item.name == DENSITY)
    setting = api.ParameterSetting(DENSITY, scale, declared.lower, declared.upper)
    return api.set_parameter_settings(project, "sample", (setting,)), declared


def _panel(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def test_the_parameter_table_shows_effective_density_and_stores_the_scale(qtbot, tmp_path) -> None:
    project, declared = _project(tmp_path)
    panel = _panel(qtbot, project)
    table = panel.parameter_table
    row = table.row_names.index(DENSITY)

    assert table.item(row, 0).text() == "密度 ρ（g·cm⁻³）"
    assert panel.display_unit(DENSITY) == "g·cm⁻³"
    assert panel.display_values(DENSITY) == pytest.approx((1.1, declared.lower * 2.2, declared.upper * 2.2))

    table.item(row, 1).setText("1.32")

    setting = next(item for item in panel.document.project.datasets[0].parameter_settings if item.name == DENSITY)
    assert setting.initial == pytest.approx(0.6)
    assert (setting.lower, setting.upper) == pytest.approx((declared.lower, declared.upper))


def test_the_layer_card_shows_the_same_effective_density_and_its_bar_uses_the_real_scale(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.structure.selection import SelectedLayerCard, layer_bounds

    project, declared = _project(tmp_path)
    definitions = api.describe_parameters(project, "sample")
    layer = project.datasets[0].structure.components[0]
    commits: list[api.LayerSpec] = []
    card = SelectedLayerCard(
        lambda _index, value: commits.append(value),
        bounds_for=lambda index: layer_bounds(definitions, index, layer.material),
    )
    qtbot.addWidget(card)

    card.show_component(0, layer)

    assert card.density_editor.value() == pytest.approx(1.1)
    expected = (0.5 - declared.lower) / (declared.upper - declared.lower)
    assert card.density_rail.fraction() == pytest.approx(expected)

    card.density_editor.setValue(1.32)
    card.density_editor.editingFinished.emit()

    # 卡片改体密度、留着 scale：有效密度正好是读者键入的那个数。
    assert commits[-1].material.bulk_density_g_cm3 == pytest.approx(2.64)
    assert commits[-1].density_scale == layer.density_scale
