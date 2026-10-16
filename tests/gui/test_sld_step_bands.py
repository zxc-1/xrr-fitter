"""结构步画的是读者正在编辑的标称剖面，不配以拟合结果为中心的不确定度带；结果步要看得到拟合 SLD。

审计里复现过：结构步把以拟合结果为中心的 MCMC 带叠在标称剖面上，图例却写「当前 SLD 剖面」——
读者会把那条带读成「我刚改的这套结构有多确定」；而到了结果步，画布上根本没有拟合出来的 SLD。
"""

from __future__ import annotations

from tests.gui.plot_cases_5 import _projected_sld_panel

from xrr_fitter.gui.plots.sld import BAND_PAIRS


def _band_artists(panel) -> list[object]:
    band_labels = {label for *_, label in BAND_PAIRS}
    return [item for item in panel.view("sld").axes.collections if item.get_label() in band_labels]


def _named_curves(panel) -> tuple[str, ...]:
    labels = (str(line.get_label()) for line in panel.view("sld").axes.lines)
    return tuple(label for label in labels if not label.startswith("_"))


def test_the_structure_step_draws_no_fit_centred_band_over_the_nominal_profile(qtbot, tmp_path) -> None:
    panel, result = _projected_sld_panel(qtbot, tmp_path)
    assert _band_artists(panel), "前置：结果态那张图上有带"

    panel.set_step_scope(1)

    assert _band_artists(panel) == []
    assert result.uncertainty.sld_bands.caption() not in panel.view("sld").axes.get_title()
    # 开关此刻管不着这张图：灰掉并说明原因，而不是亮着却点了没反应。
    assert panel.sld_bands_toggle.isEnabled() is False
    assert "拟合" in panel.sld_bands_toggle.toolTip()


def test_returning_to_the_result_step_keeps_the_readers_band_choice(qtbot, tmp_path) -> None:
    panel, _result = _projected_sld_panel(qtbot, tmp_path)
    panel.set_step_scope(4)
    panel.sld_bands_toggle.setChecked(False)

    panel.set_step_scope(1)
    panel.set_step_scope(4)

    assert panel.sld_bands_toggle.isEnabled() is True
    assert panel.sld_bands_toggle.isChecked() is False
    assert _band_artists(panel) == []


def test_the_result_step_offers_the_fitted_sld_profile(qtbot, tmp_path) -> None:
    panel, _result = _projected_sld_panel(qtbot, tmp_path)
    panel.set_step_scope(4)
    action = panel.toolbar.sld_profile_action
    assert "sld" not in panel.canvas_pane_keys(), "默认仍是设计稿帧① 的两张卡"
    assert action.isEnabled() is True

    action.setChecked(True)

    assert panel.canvas_pane_keys() == ("reflectivity", "residual", "sld")
    assert panel.sld_pane.isVisibleTo(panel) is True
    # 结果步画的是拟合候选，不是标称：候选的实部与虚部都在。
    assert "SLD 虚部" in _named_curves(panel)
    assert _band_artists(panel), "结果步的带以拟合结果为中心，画在拟合曲线上是对的"


def test_the_fitted_profile_entry_is_inert_outside_the_result_steps(qtbot, tmp_path) -> None:
    panel, _result = _projected_sld_panel(qtbot, tmp_path)
    action = panel.toolbar.sld_profile_action

    panel.set_step_scope(1)
    assert action.isEnabled() is False
    panel.set_step_scope(3)
    assert action.isEnabled() is False
    assert "sld" not in panel.canvas_pane_keys()
