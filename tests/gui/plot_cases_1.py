"""Plot contract cases, partition 1; collected via test_plots.py."""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403

from xrr_fitter.gui import theme
from xrr_fitter.model.inference import CovarianceEvidence


def test_plot_panel_has_all_diagnostic_tabs(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.tab_titles() == TAB_TITLES
    # Every view stays owned; "sld" is the companion pane that sits beside the
    # tabs rather than in them.  "residual" is in both -- the fourth tab and the
    # card pinned below the strip -- so it keeps its place in the tab order.
    assert panel.view_keys() == (
        "log",
        "raw",
        "qz4",
        "residual",
        "candidates",
        "residual_map",
        "parameter_map",
        "uncertainty",
        "trend",
        "sld",
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

    np.testing.assert_allclose(_line_y(raw, "当前拟合模型"), candidate.model_normalized * data.normalization)
    np.testing.assert_allclose(_line_x(raw, "排除点"), data.two_theta_deg[~mask])
    for actual, expected in zip(
        (data.two_theta_deg, data.intensity_raw, data.fit_mask, candidate.model_normalized),
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_plot_panel_raw_markers_separate_included_and_excluded_fill(qtbot) -> None:
    data = prepared_data(size=4, fit_mask=np.array([True, False, True, False]))
    panel = _panel(qtbot, data=data, result=_result(data))

    raw = panel.view("raw")
    assert _marker(raw, "拟合点") == "o"
    assert _marker_filled(raw, "拟合点")
    assert _marker(raw, "排除点") == "x"
    assert not _marker_filled(raw, "排除点")


def test_plot_panel_log_reflectivity_uses_display_floor_without_mutating_arrays(
    qtbot,
) -> None:
    raw = np.array([100.0, 1.0, 0.0, -2.0])
    data = prepared_data(size=4, intensity_raw=raw)
    before = data.intensity_normalized.copy()

    panel = _panel(qtbot, data=data)
    displayed = _line_y(panel.view("log"), "观测数据")

    assert np.min(displayed) == data.r_floor
    np.testing.assert_array_equal(data.intensity_normalized, before)


def test_plot_labels_avoid_english_and_jargon_leaks(qtbot) -> None:
    """傻瓜式界面不该在中文标签里夹生词。

    对数视图曾把纵轴写成 "归一化 R (display floor=...)"，不确定性视图把子图
    标题写成 "profile likelihood 与区间"——两处都把英文术语直接漏给用户。标签
    应当整句中文（保留数字与单位原文），所以这里断言换成中文说法、且旧生词不再
    出现在任一坐标轴上。

    剖面那一格的抬头现在是 ``sld.PROFILE_TITLE``（字面值归
    ``tests/gui/test_profile_likelihood_readings.py`` 钉）。里头留着的 ``Profile`` 不是漏译：
    设计稿逐字这么写，它是方法名的专名那一半——和 ``MCMC``、``scale`` 一样，译过去读者反而对不
    上文献。这条测试防的是把整句英文漏出去，那件事仍钉在下面。
    """
    from dataclasses import replace

    from xrr_fitter.gui.plots.sld import PROFILE_TITLE

    data = prepared_data(size=4)
    candidate = _candidate(data)
    result = replace(final_fit_result(candidate), uncertainty=_uncertainty())
    panel = _panel(qtbot, data=data, result=result)

    log_ylabel = panel.view("log").axis_labels()[1]
    assert log_ylabel == "反射率 R（归一化 · 对数）"
    assert "display floor" not in log_ylabel

    titles = [axes.get_title() for axes in panel.view("uncertainty").figure.axes]
    assert any(title == PROFILE_TITLE for title in titles), titles
    assert not any("profile likelihood" in title.lower() for title in titles)


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

    sld_lines = panel.view("sld").axes.lines
    np.testing.assert_array_equal(_line_x(panel.view("qz4"), "观测数据"), candidate.qz_a_inv)
    np.testing.assert_array_equal(_line_x(panel.view("residual"), "加权残差"), candidate.qz_a_inv)
    np.testing.assert_array_equal(_line_y(panel.view("residual"), "加权残差"), candidate.weighted_residuals)
    np.testing.assert_allclose(sld_lines[0].get_xdata(), candidate.sld_depth_a / 10.0)
    np.testing.assert_allclose(sld_lines[0].get_ydata(), candidate.sld_profile_a2.real * SLD_DISPLAY_SCALE)
    np.testing.assert_allclose(sld_lines[1].get_ydata(), candidate.sld_profile_a2.imag * SLD_DISPLAY_SCALE)


def test_plot_panel_qz4_extreme_qz_uses_finite_scaled_display_values(qtbot) -> None:
    data = prepared_data(size=2)
    candidate = _candidate(
        data,
        qz_a_inv=np.array([1e80, 2e80]),
        model_normalized=np.array([1.0, 2.0]),
    )
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))

    qz4 = panel.view("qz4")
    for label in ("观测数据", "当前拟合模型"):
        assert np.all(np.isfinite(_line_y(qz4, label)))
    assert "归一化" in qz4.axis_labels()[1]


def test_plot_panel_sld_overlays_other_candidate_real_profiles_faintly(qtbot) -> None:
    data = prepared_data(size=4)
    other = _candidate(
        data,
        "candidate-b",
        objective=0.3,
        sld_depth_a=np.array([0.0, 30.0, 70.0]),
        sld_profile_a2=np.array([0.0 + 0.0j, 3e-5 + 0.0j, 1e-6 + 0.0j]),
    )
    result = final_fit_result(_candidate(data), other)
    panel = _panel(qtbot, data=data, result=result)

    lines = panel.view("sld").axes.lines
    by_label = {line.get_label(): line for line in lines}
    selected_real = by_label[CANDIDATE_REAL_LABEL]
    selected_imag = by_label["SLD 虚部"]
    overlay = by_label["candidate-b 实部"]
    # The selected candidate stays fully opaque; comparison profiles sit behind
    # it at reduced alpha so the active structure reads first.
    assert selected_real.get_alpha() in (None, 1.0)
    assert selected_imag.get_alpha() in (None, 1.0)
    assert overlay.get_alpha() is not None and overlay.get_alpha() < 1.0
    np.testing.assert_allclose(overlay.get_xdata(), other.sld_depth_a / 10.0)
    np.testing.assert_allclose(overlay.get_ydata(), other.sld_profile_a2.real * SLD_DISPLAY_SCALE)


def test_plot_panel_sld_depth_axis_follows_the_selected_candidate(qtbot) -> None:
    # A search that has not converged can hold a candidate whose stack is an order
    # of magnitude too thick.  Drawing it as an overlay let autoscale take the
    # union of every profile, so a selected structure that only occupies the first
    # few nm was squeezed into a sliver at the left edge of a 266 nm axis.
    data = prepared_data(size=4)
    selected = _candidate(
        data,
        sld_depth_a=np.array([0.0, 25.0, 50.0]),
        sld_profile_a2=np.array([0.0 + 0.0j, 2e-5 + 0.0j, 4e-6 + 0.0j]),
    )
    runaway = _candidate(
        data,
        "candidate-runaway",
        objective=0.9,
        sld_depth_a=np.array([0.0, 1330.0, 2660.0]),
        sld_profile_a2=np.array([0.0 + 0.0j, 3e-5 + 0.0j, 1e-6 + 0.0j]),
    )
    panel = _panel(qtbot, data=data, result=final_fit_result(selected, runaway))

    lower, upper = panel.view("sld").axes.get_xlim()
    # The overlay stays drawn for comparison, but it must not set the scale: the
    # selected profile ends at 5 nm, so the axis has to stay near that.
    assert upper < 2.0 * selected.sld_depth_a.max() / 10.0
    assert lower <= 0.0


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

    texts = tuple(text.get_text() for axes in panel.view("uncertainty").figure.axes for text in axes.texts)
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
    report = replace(_uncertainty(profiles=profiles), covariance_evidence=None, parameter_sigma=None)
    result = replace(_result(data), uncertainty=report)

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
    assert "暂无批量趋势" in "\n".join(text.get_text() for text in panel.view("trend").axes.texts)


def test_correlation_matrix_states_what_its_colours_mean(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    view = panel.view("uncertainty")

    keys = _colour_keys(view)

    assert len(keys) == 1
    assert keys[0].get_visible()


def test_correlation_matrix_takes_its_diverging_hues_from_the_theme(qtbot, monkeypatch) -> None:
    """The correlation matrix is a signed scale, so it follows the theme.

    ``imshow`` defaulted to a hardcoded Matplotlib map fixed at a light-background
    contrast, so on the dark palette the negative end sank toward the panel it is
    drawn on—exactly the failure the residual heatmap already guards against. The
    correlation matrix carries the same signed meaning, so it takes the same
    palette-derived diverging hues and flips with the appearance on the next draw.
    """
    from xrr_fitter.gui.plots import diagnostics

    palette = theme.DARK_PLOT_PALETTE
    monkeypatch.setattr(diagnostics, "current_plot_palette", lambda: palette)
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))

    colormap = panel.view("uncertainty").axes.images[0].get_cmap()

    assert np.allclose(colormap(0.0), palette.diverging_neg)
    assert np.allclose(colormap(1.0), palette.diverging_pos)
    assert np.allclose(colormap(0.5), palette.diverging_mid)


def test_repeated_uncertainty_redraws_keep_one_colour_key(qtbot) -> None:
    """Redrawing must not leave a second key, or a key's artists stacked.

    Each redraw rebuilds the matrix, and the key is attached from scratch as part
    of that. What has to hold across repaints is that the figure still carries
    exactly one key axes holding one bar's worth of artists.
    """
    data = prepared_data(size=4)
    result = _result(data)
    panel = _panel(qtbot, data=data, result=result)
    view = panel.view("uncertainty")
    view.canvas.draw()
    artists = len(_colour_keys(view)[0].collections)

    for _ in range(3):
        panel.set_result(result, "candidate-a")
        view.canvas.draw()

    keys = _colour_keys(view)
    assert len(keys) == 1
    assert len(keys[0].collections) == artists


def test_reflectivity_curve_carries_a_grid_to_read_values_against(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))

    plot_item = panel.view("log").plot_item

    # The pg pane carries its reference grid through showGrid on both axes, the
    # twin of the matplotlib gridlines the mpl draw_* functions drew.
    assert plot_item.ctrl.xGridCheck.isChecked()
    assert plot_item.ctrl.yGridCheck.isChecked()
    assert plot_item.getAxis("bottom").grid
    assert plot_item.getAxis("left").grid


def test_grid_skips_the_matrix_cells_and_the_colour_key_it_would_overdraw(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    view = panel.view("uncertainty")

    matrix = view.axes
    key = _colour_keys(view)[0]

    for axes in (matrix, key):
        assert not any(line.get_visible() for line in axes.xaxis.get_gridlines())
        assert not any(line.get_visible() for line in axes.yaxis.get_gridlines())


def test_categorical_trend_axis_gets_no_minor_ticks_between_datasets(qtbot) -> None:
    """Half a dataset is not a reading, so the dataset axis stays unsubdivided."""
    panel = _panel(qtbot)

    panel.set_batch_trends(("a", "b"), (30.0, 45.0), (100.0, 120.0))

    axes = panel.view("trend").axes
    assert len(axes.xaxis.get_minorticklocs()) == 0
    assert len(axes.yaxis.get_minorticklocs()) > 0


def test_empty_uncertainty_view_says_what_each_pane_will_show(qtbot) -> None:
    """Both panes get a caption; a silent blank pane reads as a failed draw."""
    panel = _panel(qtbot)
    view = panel.view("uncertainty")

    for axes in view.figure.axes:
        assert axes.get_title()
        assert tuple(text.get_text() for text in axes.texts)


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
    before = tuple(action.isChecked() for action in panel.mode_actions().values())
    with pytest.raises(ValueError, match="unsupported"):
        panel.set_interaction_mode("paint")

    assert panel.interaction_mode() == "mask"
    assert tuple(action.isChecked() for action in panel.mode_actions().values()) == before


def test_plot_panel_has_visible_accessible_interaction_modes(qtbot) -> None:
    """三档模式各自带着名字和说明——现在问的是 ``QAction``。

    设计稿的 ``.modebar`` 只给「范围」留了字形（``▭``），查看与掩膜从条的右键菜单进；
    模式的勾选状态因此住在 action 上，条上那一枚按钮靠 ``setDefaultAction`` 跟着它。
    """
    panel = _panel(qtbot)

    actions = panel.mode_actions()
    assert set(actions) == {"view", "range", "mask"}
    assert all(action.text() and action.toolTip() for action in actions.values())

    buttons = {
        button.objectName(): button
        for button in panel.findChildren(QToolButton)
        if button.objectName().startswith("plotMode")
    }
    assert set(buttons) == {"plotModeRange"}
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


def test_plot_panel_sld_legend_folds_comparison_curves_into_one_entry(qtbot) -> None:
    """Several kept candidates must not let the key crowd out the profiles."""
    data = prepared_data(size=4)
    others = tuple(
        _candidate(
            data,
            f"candidate-{suffix}",
            objective=0.3 + index * 0.05,
            sld_depth_a=np.array([0.0, 30.0, 70.0]),
            sld_profile_a2=np.array([0.0 + 0.0j, 3e-5 + 0.0j, 1e-6 + 0.0j]),
        )
        for index, suffix in enumerate(("b", "c", "d"))
    )
    panel = _panel(qtbot, data=data, result=final_fit_result(_candidate(data), *others))

    axes = panel.view("sld").axes
    labels = tuple(text.get_text() for text in axes.get_legend().get_texts())

    # 卡片抬头下面那排色标已经说了「当前 SLD 剖面」，轴里就不再重复；剩下的两样它说不到，
    # 而三条对比曲线折成一行计数，不是逐个列候选 id。
    assert labels == ("SLD 虚部", "其他候选 实部 ×3")
    # Every overlay curve keeps its own label, so selection still names it.
    line_labels = {line.get_label() for line in axes.lines}
    for other in others:
        assert f"{other.candidate_id} 实部" in line_labels


def test_uncertainty_pane_stays_legible_at_the_height_the_tab_actually_gets(qtbot) -> None:
    """两张子图的标题、刻度和图内文字，在发货尺寸下不许互相压字。

    尺寸取最小支持窗口（1272×693）里量到的画布 662×532：分析页选中时独占中栏，中栏那一
    段整段归它。这不是随手取的数——早先这条钉的是 835×210，那来自「四段常驻、分析页按
    2/5 分同一个画布」的旧世界，而两张子图现在上下叠、拿的是整段高度。835×210 的宽高比
    是 3.98:1，可达档恒在 1.24:1 ~ 0.74:1 之间，最小窗口的画布高度已经是 210 的两倍半，
    也就是说那个形状在应用里画不出来；把 figure 强钉到它上面量出来的「子图只剩 74px」是
    测试自己造的，不是产品的退化。

    竖排把风险换了个位置：矩阵那张的横轴刻度旋转 45° 贴在底边，下面紧接着就是 Profile
    那张的标题，两者之间没有别的东西隔开。所以量渲染后的包围盒，而不是相信
    ``constrained_layout`` 会兜底——它管子图之间的间距，不管标题和刻度溢出。
    """
    data = prepared_data(size=4)
    from tests.support.bootstrap_cases import bootstrap_evidence

    intervals = (("component.0.thickness_a", 35.0, 48.0), ("instrument.scale", 0.9, 1.1), ("roughness", 3.6, 4.9))
    names = tuple(name for name, *_ in intervals)
    report = replace(
        _uncertainty(),
        correlation_names=names,
        correlation_matrix=np.eye(3),
        covariance_evidence=CovarianceEvidence(names, np.eye(3), "gaussian_known_sigma", 3),
        parameter_sigma=np.ones(3),
        bootstrap_intervals=intervals,
        bootstrap_evidence=bootstrap_evidence(intervals),
    )
    result = replace(_result(data), uncertainty=report)
    panel = _panel(qtbot, data=data, result=result)
    view = panel.view("uncertainty")
    view.figure.set_size_inches(6.62, 5.32)
    view.figure.set_dpi(100)
    view.canvas.draw()
    renderer = view.figure.canvas.get_renderer()
    correlation, profile = view.figure.axes[0], view.figure.axes[1]

    titles = [axes.title.get_window_extent(renderer) for axes in (correlation, profile)]
    assert not titles[0].overlaps(titles[1]), [box.bounds for box in titles]

    # 矩阵底边那排斜刻度和下面那张的标题之间只隔着子图间距，先压到的就是这一对。
    for label in correlation.get_xticklabels():
        assert not label.get_window_extent(renderer).overlaps(titles[1]), label.get_text()

    # 图内文字（没有 profiles 时是那句「证据不可用」，有 profiles 时是图例）不许越到邻居的
    # 标题上。写成对两张都查，是因为哪一张缺证据取决于报告里带了什么。
    for axes, neighbour in ((correlation, titles[1]), (profile, titles[0])):
        for text in axes.texts:
            assert not text.get_window_extent(renderer).overlaps(neighbour), text.get_text()
