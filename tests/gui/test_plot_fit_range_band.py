"""拟合范围色带画的是「当前数据集存下的那个范围」，每次投影都按它重画。

审计里复现过两种错：面板只记一份范围、而且从不读项目里的 ``fit_range_two_theta_deg``——加载一个
限定在中段的项目时画布上没有色带；切到另一个全范围的数据集后，画布上还留着上一个数据集的范围。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.gui.plot_support import _drag_range, _project_with_curves
from tests.support.model_cases import prepared_data

import xrr_fitter.api as api


def _restricted(project, dataset_id: str, keep: slice):
    dataset = next(item for item in project.datasets if item.dataset_id == dataset_id)
    mask = np.zeros(len(dataset.fit_mask), dtype=bool)
    mask[keep] = True
    return api.set_fit_mask(project, dataset_id, mask)


def _active(project, dataset_id: str):
    return replace(project, ui_state=replace(project.ui_state, active_dataset_id=dataset_id))


def _band(panel, key: str = "raw"):
    item = panel.view(key).fit_range_item
    return None if item is None else tuple(sorted(float(value) for value in item.getRegion()))


def _panel(qtbot):
    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    qtbot.addWidget(panel)
    return panel


def test_projecting_a_project_draws_the_saved_fit_range(qtbot, tmp_path) -> None:
    project = _project_with_curves(tmp_path, count=1)
    dataset_id = project.datasets[0].dataset_id
    project = _restricted(project, dataset_id, slice(5, 21))
    saved = project.datasets[0].fit_range_two_theta_deg
    panel = _panel(qtbot)

    panel.project_project(project)

    assert panel.visible_range() == pytest.approx(saved)
    for key in ("raw", "log"):
        assert _band(panel, key) == pytest.approx(saved)


def test_switching_datasets_draws_the_new_datasets_range_not_the_last_one(qtbot, tmp_path) -> None:
    project = _project_with_curves(tmp_path, count=2)
    first, second = (item.dataset_id for item in project.datasets)
    project = _active(_restricted(project, first, slice(5, 21)), first)
    panel = _panel(qtbot)
    panel.project_project(project)

    switched = _active(project, second)
    panel.project_project(switched)

    full = next(item for item in switched.datasets if item.dataset_id == second).fit_range_two_theta_deg
    assert panel.visible_range() == pytest.approx(full)
    assert _band(panel) == pytest.approx(full)


def test_a_reprojection_replaces_a_dragged_but_uncommitted_range(qtbot, tmp_path) -> None:
    """拖出来的范围只是请求；项目重新投影时画的是项目里真正存下的那一个（撤销之后也是）。"""
    project = _project_with_curves(tmp_path, count=1)
    saved = project.datasets[0].fit_range_two_theta_deg
    panel = _panel(qtbot)
    panel.project_project(project)
    panel.set_interaction_mode("range")
    _drag_range(panel, 0.2, 0.3)
    assert panel.visible_range() == pytest.approx((0.2, 0.3))

    panel.project_project(project)

    assert panel.visible_range() == pytest.approx(saved)
    assert _band(panel) == pytest.approx(saved)


def test_selecting_another_dataset_does_not_inherit_the_previous_band(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_dataset("first", prepared_data(size=4))
    panel.set_dataset("second", prepared_data(size=4))
    panel.select_dataset("first")
    panel.show_range(0.5, 2.5)

    panel.select_dataset("second")

    assert panel.visible_range() is None
    assert _band(panel) is None


def test_zoom_to_range_uses_the_saved_range_of_the_active_dataset(qtbot, tmp_path) -> None:
    project = _project_with_curves(tmp_path, count=1)
    dataset_id = project.datasets[0].dataset_id
    project = _restricted(project, dataset_id, slice(5, 21))
    saved = project.datasets[0].fit_range_two_theta_deg
    panel = _panel(qtbot)
    panel.project_project(project)

    assert panel.zoom_to_range() is True

    low, high = panel.view("raw").plot_item.vb.viewRange()[0]
    assert (low, high) == pytest.approx(saved)
