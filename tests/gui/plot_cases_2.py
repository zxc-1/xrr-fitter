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
    assert "暂无" in "\n".join(text.get_text() for text in panel.view("qz4").axes.texts)

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
    raw_x = next(
        line.get_xdata()
        for line in window.plot_panel.view("raw").axes.lines
        if line.get_label() == "拟合点"
    )
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
    panel.select_view("residual")

    panel.set_expert_mode(False)
    assert panel.current_view_key() == "residual"
    panel.set_expert_mode(True)
    assert panel.current_view_key() == "residual"

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

    excluded = next(
        line for line in panel.view("raw").axes.lines if line.get_label() == "排除点"
    )

    np.testing.assert_array_equal(excluded.get_xdata(), data.two_theta_deg[[1]])

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
    assert window.document.project.datasets[0].fit_range_two_theta_deg == pytest.approx(
        (0.15, 0.45)
    )

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
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project_with_curves(tmp_path, count=1)))
    qtbot.addWidget(window)
    pane = window.plot_panel.sld_pane

    assert pane.isVisibleTo(window.plot_panel) is False
    window.parameters_panel.set_expert_mode(True)
    assert pane.isVisibleTo(window.plot_panel) is True


def test_plot_panel_zoom_to_range_focuses_angle_views_on_visible_region(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.show_range(0.8, 1.2)

    assert panel.zoom_to_range() is True

    raw_xlim = panel.view("raw").axes.get_xlim()
    log_xlim = panel.view("log").axes.get_xlim()
    assert raw_xlim == (0.8, 1.2)
    assert log_xlim == (0.8, 1.2)


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
    raw_xlim = panel.view("raw").axes.get_xlim()
    assert raw_xlim[0] < 0.8 and raw_xlim[1] > 1.2


def test_plot_toolbar_zoom_button_focuses_views_and_reset_restores(qtbot) -> None:
    from PySide6.QtWidgets import QToolButton
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.show_range(0.8, 1.2)
    zoom = panel.toolbar.findChild(QToolButton, "plotZoomToRange")
    reset = panel.toolbar.findChild(QToolButton, "plotResetZoom")

    zoom.click()
    assert panel.view("raw").axes.get_xlim() == (0.8, 1.2)
    # Zooming is an action, not a mode: the active mode is left untouched.
    assert panel.interaction_mode() == "view"

    reset.click()
    restored = panel.view("raw").axes.get_xlim()
    assert restored[0] < 0.8 and restored[1] > 1.2
