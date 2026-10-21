"""源文件缺失时画布要说清楚，并指出去哪儿修；那时掩膜与范围手势一律不接。

审计里复现过：读源失败被吞掉，画布只写「暂无可用数据」，读者以为是数据本身空；此时在掩膜模式
下点一下还会抛 ``KeyError``——面板记着当前数据集的名字，却没有它的数据。
"""

from __future__ import annotations

import pytest
from tests.gui.plot_support import _project_with_curves


def _missing(qtbot, tmp_path):
    from xrr_fitter.gui.plots.panel import PlotPanel

    project = _project_with_curves(tmp_path, count=1)
    (tmp_path / "curve-0.xy").unlink()
    panel = PlotPanel()
    qtbot.addWidget(panel)
    panel.project_project(project)
    return panel, project


def test_a_missing_source_is_named_on_the_canvas_with_the_way_to_fix_it(qtbot, tmp_path) -> None:
    panel, _project = _missing(qtbot, tmp_path)

    for key in ("log", "raw"):
        text = panel.view(key).placeholder_text() or ""
        assert "源文件缺失或无法读取" in text
        assert "重新链接数据源" in text
        assert "curve-0.xy" in text
    sld_texts = " ".join(item.get_text() for item in panel.view("sld").axes.texts)
    assert "源文件缺失或无法读取" in sld_texts


def test_a_missing_source_disables_the_mask_and_range_gestures(qtbot, tmp_path) -> None:
    panel, _project = _missing(qtbot, tmp_path)

    actions = panel.mode_actions()
    assert actions["mask"].isEnabled() is False
    assert actions["range"].isEnabled() is False
    panel.set_interaction_mode("mask")
    assert panel.interaction_mode() == "view"
    # 原先这一下会穿过 _active_data 抛 KeyError：面板记着数据集名字，却没有它的数据。
    assert panel.request_point_mask(0) is False
    assert panel.select_fit_range(0.1, 0.2) is False


def test_restoring_the_source_brings_the_gestures_back(qtbot, tmp_path) -> None:
    from tests.gui.plot_support import _write_curve

    panel, project = _missing(qtbot, tmp_path)
    _write_curve(tmp_path / "curve-0.xy")

    panel.project_project(project)

    assert panel.view("raw").placeholder_text() is None
    assert panel.mode_actions()["mask"].isEnabled() is True
    panel.set_interaction_mode("mask")
    assert panel.interaction_mode() == "mask"
    assert panel.request_point_mask(0) is True


@pytest.mark.parametrize("damage", ("garbage", "empty"))
def test_an_unreadable_source_is_reported_the_same_way(qtbot, tmp_path, damage) -> None:
    from xrr_fitter.gui.plots.panel import PlotPanel

    project = _project_with_curves(tmp_path, count=1)
    (tmp_path / "curve-0.xy").write_text("not numbers\n" if damage == "garbage" else "", encoding="utf-8")
    panel = PlotPanel()
    qtbot.addWidget(panel)

    panel.project_project(project)

    assert "源文件缺失或无法读取" in (panel.view("log").placeholder_text() or "")
