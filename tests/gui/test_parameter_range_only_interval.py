"""「仅范围」那一档的区间属于 setting，切档不能拿声明的硬边界把它冲掉。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

import xrr_fitter.api as api

NAME = "component.0.thickness_a"


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "sample.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.2), 40.0, roughness_a=3.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    panel = ParametersPanel(ProjectDocument(api.set_structure(project, "sample", structure)))
    qtbot.addWidget(panel)
    return panel


def _setting(panel) -> api.ParameterSetting:
    return next(value for value in panel.document.project.datasets[0].parameter_settings if value.name == NAME)


def _shown_bounds(panel) -> tuple[float, float]:
    table = panel.parameter_table
    row = table.row_names.index(NAME)
    return table.entered_value(table.item(row, 2)), table.entered_value(table.item(row, 3))


def test_range_only_row_shows_and_keeps_its_own_interval_through_every_gear(qtbot, tmp_path) -> None:
    """35–45 Å 是读者自己写的区间；切到「仅范围」再切回去，它都得原样留着。

    「仅范围」编译出来的声明保留声明的硬边界（区间改走 soft_range 先验），表要是照声明画，
    那一行显示的就是硬边界；下一次切档整行读回，硬边界被当成新区间写进 setting，自定义区间
    从此找不回来。
    """
    panel = _panel(qtbot, tmp_path)
    panel.set_parameter(NAME, initial=40.0, lower=35.0, upper=45.0, freedom=api.ParameterFreedom.FREE)
    table = panel.parameter_table
    row = table.row_names.index(NAME)

    table.item(row, 0).setCheckState(Qt.CheckState.PartiallyChecked)

    assert _setting(panel) == api.ParameterSetting(NAME, 40.0, 35.0, 45.0, api.ParameterFreedom.RANGE_ONLY)
    assert _shown_bounds(panel) == pytest.approx((3.5, 4.5))

    table.setCurrentCell(table.row_names.index(NAME), 0)
    panel.disposition.fixed_button.click()
    assert _setting(panel) == api.ParameterSetting(NAME, 40.0, 35.0, 45.0, api.ParameterFreedom.FIXED)

    panel.disposition.range_button.click()
    assert _setting(panel).freedom is api.ParameterFreedom.RANGE_ONLY
    assert _shown_bounds(panel) == pytest.approx((3.5, 4.5))

    panel.disposition.free_button.click()
    assert _setting(panel) == api.ParameterSetting(NAME, 40.0, 35.0, 45.0, api.ParameterFreedom.FREE)


def test_range_only_interval_edit_commits_the_typed_interval(qtbot, tmp_path) -> None:
    """「仅范围」那一行的下限格里改的是区间本身，提交的也必须是区间本身。"""
    panel = _panel(qtbot, tmp_path)
    panel.set_parameter(NAME, initial=40.0, lower=35.0, upper=45.0, freedom=api.ParameterFreedom.RANGE_ONLY)
    table = panel.parameter_table
    row = table.row_names.index(NAME)

    table.item(row, 2).setText("3")

    assert _setting(panel) == api.ParameterSetting(NAME, 40.0, 30.0, 45.0, api.ParameterFreedom.RANGE_ONLY)
    assert panel.display_values(NAME) == pytest.approx((4.0, 3.0, 4.5))
