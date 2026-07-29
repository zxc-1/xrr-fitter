"""Plot contract cases, partition 1; collected via test_plots.py."""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403

def test_plot_panel_has_all_diagnostic_tabs(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.tab_titles() == TAB_TITLES
    assert panel.view_keys() == (
        "raw",
        "log",
        "qz4",
        "residual",
        "sld",
        "candidates",
        "uncertainty",
        "trend",
    )

@pytest.mark.parametrize(
    ("code", "label"),
    (
        ("suspected_unmodeled_footprint", "疑似未建模的足迹效应"),
        ("suspected_diffuse_background", "疑似漫散射背景"),
        ("nevot_croce_applicability_exceeded", "Nevot-Croce 适用范围超限"),
        ("ideal_reflectivity_above_one", "理想反射率超过 1"),
        ("gauss_hermite_unconverged", "Gauss-Hermite 积分未收敛"),
    ),
)
def test_known_diagnostic_codes_have_chinese_labels_and_keep_technical_details(
    code,
    label,
) -> None:
    from xrr_fitter.gui.plots.diagnostics import diagnostic_text

    text = diagnostic_text(SimpleNamespace(code=code, message="technical detail"))

    assert label in text
    assert code in text
    assert "technical detail" in text

def test_plot_panel_draws_raw_model_and_excluded_points_without_mutating_data(
    qtbot,
) -> None:
    mask = np.array([True, False, True, False])
    data = prepared_data(size=4, fit_mask=mask)
    candidate = _candidate(data)
    result = replace(final_fit_result(candidate), uncertainty=_uncertainty())
    before = (
        data.two_theta_deg.copy(),
        data.intensity_raw.copy(),
        data.fit_mask.copy(),
        candidate.model_normalized.copy(),
    )

    panel = _panel(qtbot, data=data, result=result)
    raw = panel.view("raw")

    np.testing.assert_allclose(_line_y(raw, "当前候选模型"), candidate.model_normalized * data.normalization)
    excluded = next(line for line in raw.axes.lines if line.get_label() == "排除点")
    np.testing.assert_allclose(excluded.get_xdata(), data.two_theta_deg[~mask])
    for actual, expected in zip(
        (data.two_theta_deg, data.intensity_raw, data.fit_mask, candidate.model_normalized),
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

def test_plot_panel_raw_markers_separate_included_and_excluded_fill(qtbot) -> None:
    data = prepared_data(size=4, fit_mask=np.array([True, False, True, False]))
    panel = _panel(qtbot, data=data, result=_result(data))

    lines = {line.get_label(): line for line in panel.view("raw").axes.lines}
    assert lines["拟合点"].get_marker() == "o"
    assert lines["拟合点"].get_markerfacecolor() != "none"
    assert lines["排除点"].get_marker() == "x"
    assert lines["排除点"].get_markerfacecolor() == "none"

def test_plot_panel_log_reflectivity_uses_display_floor_without_mutating_arrays(
    qtbot,
) -> None:
    raw = np.array([100.0, 1.0, 0.0, -2.0])
    data = prepared_data(size=4, intensity_raw=raw)
    before = data.intensity_normalized.copy()

    panel = _panel(qtbot, data=data)
    displayed = _line_y(panel.view("log"), "归一化数据")

    assert np.min(displayed) == data.r_floor
    np.testing.assert_array_equal(data.intensity_normalized, before)

def test_plot_panel_draws_qz4_weighted_residual_and_sld_from_candidate_arrays(
    qtbot,
) -> None:
    data = prepared_data(size=4)
    candidate = _candidate(
        data,
        weighted_residuals=np.array([0.5, np.nan, -0.25, 0.0]),
    )
    result = replace(final_fit_result(candidate), uncertainty=_uncertainty())
    panel = _panel(qtbot, data=data, result=result)

    qz4 = panel.view("qz4").axes.lines[0]
    residual = panel.view("residual").axes.lines[0]
    sld_lines = panel.view("sld").axes.lines
    np.testing.assert_array_equal(qz4.get_xdata(), candidate.qz_a_inv)
    np.testing.assert_array_equal(residual.get_xdata(), candidate.qz_a_inv)
    np.testing.assert_array_equal(residual.get_ydata(), candidate.weighted_residuals)
    np.testing.assert_allclose(sld_lines[0].get_xdata(), candidate.sld_depth_a / 10.0)
    np.testing.assert_allclose(sld_lines[0].get_ydata(), candidate.sld_profile_a2.real)
    np.testing.assert_allclose(sld_lines[1].get_ydata(), candidate.sld_profile_a2.imag)

def test_plot_panel_hides_uncertainty_owned_by_another_candidate(qtbot) -> None:
    data = prepared_data(size=4)
    result = replace(_result(data), uncertainty=_uncertainty("candidate-b"))

    panel = _panel(qtbot, data=data, result=result)
    view = panel.view("uncertainty")

    assert not view.axes.images
    assert "candidate-b" in "\n".join(text.get_text() for text in view.axes.texts)
    assert "candidate-a" in "\n".join(text.get_text() for text in view.axes.texts)

def test_plot_panel_uncertainty_tab_states_report_is_unavailable(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data, uncertainty=False))

    texts = tuple(
        text.get_text()
        for axes in panel.view("uncertainty").figure.axes
        for text in axes.texts
    )
    assert any("不可用" in text for text in texts)

def test_plot_panel_uses_independent_normalized_axis_for_heterogeneous_profiles(
    qtbot,
) -> None:
    profiles = (
        api.ParameterProfile(
            "component.0.thickness_a",
            np.array([20.0, 30.0, 60.0]),
            np.array([2.0, 1.0, 2.5]),
            True,
            False,
        ),
        api.ParameterProfile(
            "instrument.scale",
            np.array([0.7, 1.0, 1.1]),
            np.array([2.2, 1.0, 2.0]),
            False,
            True,
        ),
    )
    data = prepared_data(size=4)
    result = replace(_result(data), uncertainty=_uncertainty(profiles=profiles))

    panel = _panel(qtbot, data=data, result=result)
    profile_axes = panel.view("uncertainty").figure.axes[1]

    for line in profile_axes.lines:
        assert np.min(line.get_xdata()) == pytest.approx(0.0)
        assert np.max(line.get_xdata()) == pytest.approx(1.0)
    assert "独立归一化" in profile_axes.get_xlabel()

def test_plot_panel_draws_fixed_correlation_profile_interval_and_empty_batch_trend(
    qtbot,
) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))

    image = panel.view("uncertainty").axes.images[0]
    assert image.get_clim() == (-1.0, 1.0)
    assert "暂无批量趋势" in "\n".join(
        text.get_text() for text in panel.view("trend").axes.texts
    )

def test_plot_panel_draws_project_batch_trends_in_nm(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_batch_trends(("a", "b"), (30.0, 45.0), (100.0, 120.0))

    axes = panel.view("trend").axes
    np.testing.assert_allclose(axes.lines[0].get_ydata(), (3.0, 4.5))
    np.testing.assert_allclose(axes.lines[1].get_ydata(), (10.0, 12.0))
    assert axes.get_ylabel() == "长度 (nm)"

@pytest.mark.parametrize("dataset_ids", (("a", "a"), ("", "b")))
def test_plot_panel_rejects_invalid_batch_trend_dataset_ids_without_redraw(
    qtbot,
    dataset_ids,
) -> None:
    panel = _panel(qtbot)
    panel.set_batch_trends(("a", "b"), (10.0, 20.0), (30.0, 40.0))
    before = _artist_snapshot(panel)

    with pytest.raises(ValueError, match="dataset ids"):
        panel.set_batch_trends(dataset_ids, (10.0, 20.0), (30.0, 40.0))

    assert _artist_snapshot(panel) == before

def test_plot_panel_rejects_misaligned_batch_trend_columns_without_redraw(
    qtbot,
) -> None:
    panel = _panel(qtbot)
    before = _artist_snapshot(panel)

    with pytest.raises(ValueError, match="equal lengths"):
        panel.set_batch_trends(("a", "b"), (10.0,), (30.0, 40.0))

    assert _artist_snapshot(panel) == before

@pytest.mark.parametrize(
    ("thickness_a", "period_a"),
    (
        ((10.0, np.nan), (30.0, 40.0)),
        ((10.0, 20.0), (30.0, np.inf)),
    ),
)
def test_plot_panel_rejects_nonfinite_batch_trend_values_without_redraw(
    qtbot,
    thickness_a,
    period_a,
) -> None:
    panel = _panel(qtbot)
    before = _artist_snapshot(panel)

    with pytest.raises(ValueError, match="finite"):
        panel.set_batch_trends(("a", "b"), thickness_a, period_a)

    assert _artist_snapshot(panel) == before

def test_plot_panel_rejects_single_dataset_batch_trend_without_redraw(qtbot) -> None:
    panel = _panel(qtbot)
    before = _artist_snapshot(panel)

    with pytest.raises(ValueError, match="at least two"):
        panel.set_batch_trends(("a",), (10.0,), (30.0,))

    assert _artist_snapshot(panel) == before

def test_plot_panel_preserves_batch_trends_during_candidate_redraw(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    panel.set_batch_trends(("a", "b"), (10.0, 20.0), (30.0, 40.0))
    before = tuple(line.get_ydata().copy() for line in panel.view("trend").axes.lines)

    panel.set_result(_result(data), "candidate-b")

    for actual, expected in zip(panel.view("trend").axes.lines, before, strict=True):
        np.testing.assert_array_equal(actual.get_ydata(), expected)

def test_plot_panel_sets_interaction_mode_atomically(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_interaction_mode("mask")
    before = tuple(button.isChecked() for button in panel.mode_buttons().values())
    with pytest.raises(ValueError, match="unsupported"):
        panel.set_interaction_mode("paint")

    assert panel.interaction_mode() == "mask"
    assert tuple(button.isChecked() for button in panel.mode_buttons().values()) == before

def test_plot_panel_has_visible_accessible_interaction_modes(qtbot) -> None:
    panel = _panel(qtbot)

    buttons = {
        button.objectName(): button
        for button in panel.findChildren(QToolButton)
        if button.objectName().startswith("plotMode")
    }
    assert set(buttons) == {
        "plotModeView",
        "plotModeRange",
        "plotModeMask",
    }
    assert all(button.accessibleName() and button.toolTip() for button in buttons.values())

def test_plot_panel_emits_ordered_stored_fit_range(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    emitted: list[tuple[float, float]] = []
    panel.fit_range_requested.connect(lambda low, high: emitted.append((low, high)))
    panel.set_interaction_mode("range")

    panel.select_fit_range(2.5, 0.5)

    assert emitted == [(0.5, 2.5)]

def test_plot_panel_rejects_nonfinite_fit_range_without_signal(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    emitted: list[tuple[float, float]] = []
    panel.fit_range_requested.connect(lambda low, high: emitted.append((low, high)))
    panel.set_interaction_mode("range")

    with pytest.raises(ValueError, match="finite"):
        panel.select_fit_range(np.nan, 2.0)

    assert emitted == []

@pytest.mark.parametrize("mode", ("view", "mask"))
def test_plot_panel_nonrange_modes_cannot_select_fit_range(qtbot, mode) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    emitted: list[tuple[float, float]] = []
    panel.fit_range_requested.connect(lambda low, high: emitted.append((low, high)))
    panel.set_interaction_mode(mode)

    assert panel.select_fit_range(0.5, 2.5) is False
    assert emitted == []

def test_plot_panel_emits_prepared_index_for_direct_point_mask_request(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    emitted: list[int] = []
    panel.point_mask_requested.connect(emitted.append)
    panel.set_interaction_mode("mask")

    panel.request_point_mask(2)

    assert emitted == [2]

def test_plot_panel_rejects_out_of_range_point_mask_without_signal(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    emitted: list[int] = []
    panel.point_mask_requested.connect(emitted.append)
    panel.set_interaction_mode("mask")

    with pytest.raises(IndexError, match="point index"):
        panel.request_point_mask(4)

    assert emitted == []

@pytest.mark.parametrize("mode", ("view", "range"))
def test_plot_panel_nonmask_modes_cannot_request_point_mask(qtbot, mode) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    emitted: list[int] = []
    panel.point_mask_requested.connect(emitted.append)
    panel.set_interaction_mode(mode)

    assert panel.request_point_mask(2) is False
    assert emitted == []
