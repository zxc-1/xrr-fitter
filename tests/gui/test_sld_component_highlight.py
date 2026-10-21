"""选中结构里的一个组件，SLD 剖面上把它那一段加粗、垫一条色带；取消选择、换数据集、结构变了都撤掉。

层堆叠示意图与参数表已经跟着读者选的那一层走（F34），剖面却不认这次选择：读者在堆叠里点了氧化层，还得
自己去剖面上数台阶，才知道哪一段是它。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from tests.gui.plot_cases_5 import _projected_sld_panel, _simple_structure, _sld_project
from tests.gui.test_plot_dark_series import dark  # noqa: F401 - pytest fixture shared with the dark-theme cases
from tests.support.model_cases import prepared_data

import xrr_fitter.api as api
from xrr_fitter.gui import theme

HIGHLIGHT_GID = "sldComponentHighlight"


def _highlight_artists(panel) -> list[object]:
    axes = panel.view("sld").axes
    return [artist for artist in (*axes.lines, *axes.patches) if artist.get_gid() == HIGHLIGHT_GID]


def _highlight_line(panel):
    return next(artist for artist in _highlight_artists(panel) if hasattr(artist, "get_xdata"))


def _structure_panel(qtbot, tmp_path):
    panel, _result = _projected_sld_panel(qtbot, tmp_path)
    panel.set_step_scope(1)
    return panel


def _luminance(colour: object) -> float:
    channels = [value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4 for value in to_rgb(colour)]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(first: object, second: object) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_highlighting_a_component_emphasises_its_segment_of_the_profile(qtbot, tmp_path) -> None:
    panel = _structure_panel(qtbot, tmp_path)
    assert _highlight_artists(panel) == []
    lines = panel.view("sld").axes.lines
    profile_width = max(line.get_linewidth() for line in lines if line.get_label() == "当前 SLD 剖面")

    panel.highlight_component(0)

    line = _highlight_line(panel)
    thickness_nm = float(_simple_structure().components[0].thickness_a) / 10.0
    xdata = np.asarray(line.get_xdata(), dtype=float)
    assert len(_highlight_artists(panel)) == 2
    assert line.get_linewidth() > profile_width
    assert xdata.min() >= -1e-9 and xdata.max() <= thickness_nm + 1e-9


def test_the_highlight_reads_on_both_the_light_and_the_dark_canvas(qtbot, tmp_path, dark) -> None:  # noqa: F811
    panel = _structure_panel(qtbot, tmp_path)
    panel.highlight_component(0)
    figure = panel.view("sld").figure
    assert _contrast(_highlight_line(panel).get_color(), figure.get_facecolor()) >= 3.0

    dark()

    band = next(artist for artist in _highlight_artists(panel) if not hasattr(artist, "get_xdata"))
    assert _contrast(_highlight_line(panel).get_color(), figure.get_facecolor()) >= 3.0
    assert to_hex(band.get_facecolor(), keep_alpha=False) == to_hex(theme.current_accent())


def test_clearing_the_selection_or_changing_the_structure_drops_the_highlight(qtbot, tmp_path) -> None:
    panel = _structure_panel(qtbot, tmp_path)
    panel.highlight_component(0)
    panel.highlight_component(None)
    assert _highlight_artists(panel) == []

    project, _result = _sld_project(tmp_path, "curve")
    dataset = project.datasets[0]
    first = dataset.structure.components[0]
    thicker = replace(
        dataset.structure,
        components=(replace(first, thickness_a=first.thickness_a + 5.0), *dataset.structure.components[1:]),
    )
    panel.highlight_component(0)
    panel.project_project(api.set_structure(project, dataset.dataset_id, thicker))
    assert _highlight_artists(panel) == []

    panel.project_project(project)
    assert _highlight_artists(panel) == [], "撤销回原结构也不该让已经撤掉的高亮自己回来"


def test_switching_dataset_drops_the_highlight(qtbot, tmp_path) -> None:
    panel = _structure_panel(qtbot, tmp_path)
    panel.highlight_component(0)

    panel.set_dataset("other", prepared_data(size=4))
    project, _result = _sld_project(tmp_path, "curve")
    panel.project_project(project)

    assert _highlight_artists(panel) == []
