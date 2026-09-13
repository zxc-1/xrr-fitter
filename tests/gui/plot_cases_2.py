"""Plot contract cases, partition 2; collected via test_plots.py."""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403


def test_plot_panel_cancel_interaction_clears_active_range(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_interaction_mode("range")
    panel.show_range(0.5, 2.5)

    panel.cancel_interaction()

    assert panel.interaction_mode() == "view"
    assert panel.visible_range() is None


@pytest.mark.parametrize("value", (float("-inf"), float("inf"), float("nan")))
def test_plot_panel_labels_nonfinite_objective_as_inspection_only(value) -> None:
    from xrr_fitter.gui.plots.diagnostics import candidate_label

    candidate = SimpleNamespace(
        candidate_id="candidate-x",
        objective=value,
        ranking_objective=None,
        valid=True,
        stop_reason="converged",
    )

    assert "仅供检查" in candidate_label(candidate, selected=False)


def test_plot_panel_labels_invalid_archived_candidate_as_inspection_only() -> None:
    from xrr_fitter.gui.plots.diagnostics import candidate_label

    candidate = SimpleNamespace(
        candidate_id="candidate-x",
        objective=float("inf"),
        ranking_objective=None,
        valid=False,
        stop_reason="early_eliminated",
    )

    text = candidate_label(candidate, selected=True)
    assert "仅供检查" in text
    assert "早期淘汰" in text


def test_plot_panel_keeps_invalid_candidates_as_unselected_evidence(qtbot) -> None:
    data = prepared_data(size=4)
    valid = _candidate(data)
    invalid = _candidate(
        data,
        "candidate-invalid",
        objective=float("inf"),
        valid=False,
        stop_reason="invalid_model",
    )
    result = replace(final_fit_result(valid, invalid), uncertainty=_uncertainty())

    panel = _panel(qtbot, data=data, result=result)
    labels = tuple(line.get_label() for line in panel.view("candidates").axes.lines)

    assert any("candidate-invalid" in label and "仅供检查" in label for label in labels)


def test_plot_panel_rejects_misaligned_candidate_diagnostics_without_redraw(
    qtbot,
) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    before = _artist_snapshot(panel)
    bad = _candidate(
        prepared_data(size=3),
        qz_a_inv=np.array([0.1, 0.2, 0.3]),
        model_normalized=np.array([0.8, 0.4, 0.2]),
        log_residuals_decades=np.zeros(3),
        weighted_residuals=np.zeros(3),
    )

    with pytest.raises(ValueError, match="prepared point count"):
        panel.set_result(final_fit_result(bad), "candidate-a")

    assert _artist_snapshot(panel) == before


def test_plot_panel_invalid_candidate_index_preserves_previous_diagnostics(
    qtbot,
) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    before = _artist_snapshot(panel)

    with pytest.raises(KeyError, match="candidate"):
        panel.set_result(_result(data), "candidate-missing")

    assert _artist_snapshot(panel) == before


def test_plot_panel_select_candidate_updates_diagnostics_and_comparison_atomically(
    qtbot,
) -> None:
    data = prepared_data(size=4)
    result = _result(data)
    panel = _panel(qtbot, data=data, result=result)

    panel.set_result(result, "candidate-b")

    assert panel.selected_candidate_id() == "candidate-b"
    labels = tuple(line.get_label() for line in panel.view("candidates").axes.lines)
    assert any("candidate-b" in label and "查看中" in label for label in labels)


def test_plot_panel_set_dataset_clears_stale_candidate_diagnostics(qtbot) -> None:
    first = prepared_data(size=4)
    second = prepared_data(size=5)
    panel = _panel(qtbot, data=first, result=_result(first))

    panel.set_dataset("second", second)

    assert panel.selected_dataset_id() == "second"
    assert panel.selected_candidate_id() is None
    assert "暂无" in (panel.view("qz4").placeholder_text() or "")


def test_plot_rejects_unknown_dataset_without_mutating_active_state(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    before = panel.selected_dataset_id()

    with pytest.raises(KeyError, match="dataset"):
        panel.select_dataset("missing")

    assert panel.selected_dataset_id() == before


@pytest.mark.parametrize(
    "invalid_mask",
    (np.array([True]), np.array([True, False, True, False, True])),
)
def test_mask_plot_requires_one_mask_value_per_prepared_point(
    qtbot,
    invalid_mask,
) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    before = _artist_snapshot(panel)

    with pytest.raises(ValueError, match="mask"):
        panel.update_mask("curve", invalid_mask)

    assert _artist_snapshot(panel) == before


def test_plot_preserves_prepared_indices_when_nonfinite_points_are_filtered(
    qtbot,
) -> None:
    angles = np.array([0.1, np.nan, 0.3, 0.4])
    raw = np.array([100.0, 80.0, np.nan, 40.0])
    valid = np.array([True, False, False, True])
    data = prepared_data(
        size=4,
        two_theta_deg=angles,
        intensity_raw=raw,
        validation_mask=valid,
        fit_mask=valid,
    )
    panel = _panel(qtbot, data=data)

    assert panel.displayed_prepared_indices() == (0, 3)


def test_plots_package_initializer_is_empty() -> None:
    root = Path(__file__).resolve().parents[2]

    assert (root / "src/xrr_fitter/gui/plots/__init__.py").read_bytes() == b""


def test_active_dataset_selection_updates_plot_canvas(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    value = _project_with_curves(tmp_path)
    window = MainWindow(ProjectDocument(value))
    qtbot.addWidget(window)
    second = value.datasets[1]

    window.select_active_dataset(second.dataset_id)

    assert window.plot_panel.selected_dataset_id() == second.dataset_id
    raw_x = _line_x(window.plot_panel.view("raw"), "拟合点")
    assert raw_x[0] == pytest.approx(0.06)


def test_active_dataset_selection_rolls_back_when_plot_commit_fails(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    value = _project_with_curves(tmp_path)
    window = MainWindow(ProjectDocument(value))
    qtbot.addWidget(window)
    before = window.document.project
    before_plot = window.plot_panel.selected_dataset_id()
    second_id = value.datasets[1].dataset_id
    original = window.plot_panel.project_project

    def reject(project_value):
        if project_value.ui_state.active_dataset_id == second_id:
            raise RuntimeError("plot commit rejected")
        return original(project_value)

    monkeypatch.setattr(window.plot_panel, "project_project", reject)
    tree = window.data_panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    first_item = tree.topLevelItem(0)
    second_item = tree.topLevelItem(1)
    assert tree.currentItem() is first_item

    tree.setCurrentItem(second_item)

    assert window.document.project is before
    assert window.document.is_dirty is False
    assert window.plot_panel.selected_dataset_id() == before_plot
    assert tree.currentItem() is first_item


def test_diagnostic_selection_rollback_restores_previous_view(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project_with_curves(tmp_path, count=1)))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    original = window.plot_panel.project_project

    def reject(project_value):
        if project_value.ui_state.plot_tab_index == 1:
            raise RuntimeError("diagnostic selection rejected")
        return original(project_value)

    monkeypatch.setattr(window.plot_panel, "project_project", reject)

    window.plot_panel.tabs.setCurrentIndex(1)

    assert window.document.project.ui_state.plot_tab_index == 0
    assert window.plot_panel.current_view_key() == window.plot_panel.tab_keys()[0]
    assert window.document.is_dirty is False


def test_expert_projection_preserves_standard_selection_and_sld_canvas_state(qtbot) -> None:
    panel = _panel(qtbot)
    sld_canvas = panel.view("sld").canvas
    panel.select_view("qz4")

    panel.set_expert_mode(False)
    panel.set_expert_mode(True)

    assert panel.current_view_key() == "qz4"
    assert panel.view("sld").canvas is sld_canvas


def test_tab_selection_survives_expert_mode_round_trips(qtbot) -> None:
    """No tab is mode-gated now, so a selection is never displaced."""
    panel = _panel(qtbot)
    panel.select_view("raw")

    panel.set_expert_mode(False)
    assert panel.current_view_key() == "raw"
    panel.set_expert_mode(True)
    assert panel.current_view_key() == "raw"

    panel.select_view("log")
    panel.set_expert_mode(False)
    assert panel.current_view_key() == "log"


def test_import_plots_core_invalid_points_as_excluded(qtbot) -> None:
    data = prepared_data(
        size=4,
        validation_mask=np.array([True, False, True, True]),
        fit_mask=np.array([True, False, True, True]),
    )
    panel = _panel(qtbot, data=data)

    np.testing.assert_array_equal(_line_x(panel.view("raw"), "排除点"), data.two_theta_deg[[1]])


def test_main_window_connects_plot_range_and_point_mask_to_active_dataset(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project_with_curves(tmp_path, count=1)))
    qtbot.addWidget(window)
    dataset_id = window.document.active_dataset_id

    window.plot_panel.set_interaction_mode("range")
    window.plot_panel.select_fit_range(0.15, 0.45)
    assert window.document.project.datasets[0].fit_range_two_theta_deg == pytest.approx((0.15, 0.45))

    window.plot_panel.set_interaction_mode("mask")
    window.plot_panel.request_point_mask(10)
    assert window.document.project.datasets[0].dataset_id == dataset_id
    assert window.document.project.datasets[0].fit_mask[10] is False


def test_main_window_import_and_selection_updates_diagnostic_plot(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    paths = (
        _write_curve(tmp_path / "first.xy"),
        _write_curve(tmp_path / "second.xy", offset=0.01),
    )

    window.data_panel.add_paths(
        paths,
        beam=api.BeamSpec("monochromatic"),
        instrument=api.InstrumentSpec(),
    )
    assert window.plot_panel.selected_dataset_id() == "first"

    window.select_active_dataset("second")
    assert window.plot_panel.selected_dataset_id() == "second"
    assert _line_y(window.plot_panel.view("raw"), "拟合点").size == 32


def test_main_window_projects_parameter_expert_mode_to_sld_visibility(
    qtbot,
    tmp_path,
) -> None:
    """结构那一步的剖面两种模式下都在，专家开关不再是它的闸。

    设计稿帧③（专家 · 结构编辑）的画布就是层堆叠加 SLD 剖面这两张卡，而
    ``STEP_PLOT_PANES`` 里这一步的画布也只有 ``sld`` 一段。再压一层「显示高级选项」——
    出厂是关的——默认装机走到结构那一步会拿到一整块空白画布，而这一步的全部看点就是
    「改一层，剖面当场跟着动」。剖面归不归高级选项管，在没有步骤作用域的裸面板上仍然
    有效（见 ``plot_cases_4`` 与 ``plot_cases_5`` 的同名契约）。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project_with_curves(tmp_path, count=1)))
    qtbot.addWidget(window)
    pane = window.plot_panel.sld_pane

    assert window.plot_panel.canvas_pane_keys() == ("sld",)
    assert pane.isVisibleTo(window.plot_panel) is True
    window.parameters_panel.set_expert_mode(True)
    assert pane.isVisibleTo(window.plot_panel) is True


def test_plot_panel_zoom_to_range_focuses_angle_views_on_visible_region(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.show_range(0.8, 1.2)

    assert panel.zoom_to_range() is True

    raw_xrange = _view_xrange(panel.view("raw"))
    log_xrange = _view_xrange(panel.view("log"))
    assert raw_xrange == (0.8, 1.2)
    assert log_xrange == (0.8, 1.2)


def test_plot_panel_zoom_to_range_without_range_is_noop(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))

    # No fit range has been selected, so there is nothing to focus on.
    assert panel.zoom_to_range() is False


def test_plot_panel_reset_zoom_restores_autoscale(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.show_range(0.8, 1.2)
    panel.zoom_to_range()

    assert panel.reset_zoom() is True

    # Autoscale makes xlim span the full data extent again.
    raw_xrange = _view_xrange(panel.view("raw"))
    assert raw_xrange[0] < 0.8 and raw_xrange[1] > 1.2


def test_plot_toolbar_zoom_button_focuses_views_and_reset_restores(qtbot) -> None:
    """缩放到拟合范围 / 恢复完整视图 搬进条的右键菜单后，做的事一样。

    设计稿的 ``.modebar`` 只有四枚字形，这两条不在其中；它们仍是同一批 ``QAction``，
    从菜单触发和从前点按钮走的是同一条信号。
    """
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.show_range(0.8, 1.2)
    zoom = panel.toolbar.zoom_to_range_action
    reset = panel.toolbar.reset_zoom_action

    zoom.trigger()
    assert _view_xrange(panel.view("raw")) == (0.8, 1.2)
    # Zooming is an action, not a mode: the active mode is left untouched.
    assert panel.interaction_mode() == "view"

    reset.trigger()
    restored = _view_xrange(panel.view("raw"))
    assert restored[0] < 0.8 and restored[1] > 1.2


def _plot_texts(view):
    """互动面板画在图里的每一段文字。"""
    import pyqtgraph as pg

    return tuple(item.toPlainText() for item in view.plot_item.scene().items() if isinstance(item, pg.TextItem))


def test_interactive_views_leave_the_fit_quality_to_the_status_bar(qtbot) -> None:
    """设计稿把 J 放在状态栏和左栏管线上，图里不再有那行 ``J=… · 平均残差 …``。

    互动面板上有十字光标读数、拟合窗口说明、±1σ 说明三处文字，右下角再压一行拟合
    质量，读者要同时盯四处；而 J 是整份结果的属性，不是某一张图的，重复在四张图上
    只会让「哪张图的 J」变成一个不该存在的问题。matplotlib 的导出图仍旧带这行字，
    那是逐位不变的交付物，与屏幕上的排布无关。
    """
    data = prepared_data(size=4)
    candidate = _candidate(data, objective=0.25, log_residuals_decades=np.full(4, 0.1))
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))

    for key in ("log", "raw", "qz4", "residual"):
        view = panel.view(key)
        if not _is_live(view):
            continue
        assert not [text for text in _plot_texts(view) if "J=" in text], f"{key} view still captions J"
