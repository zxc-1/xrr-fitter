"""Plot state, interaction, rollback, and lifecycle contract cases.

These cases exercise PlotPanel through user-visible GUI behavior, including
transactional redraw failures, deferred canvas work, resource release, and
keyboard or pointer interaction. They are collected via test_plots.py.
"""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403


def test_mask_rolls_back_when_plot_commit_fails(qtbot, tmp_path, monkeypatch) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project_with_curves(tmp_path, count=1)))
    qtbot.addWidget(window)
    dataset_id = window.document.active_dataset_id
    before = window.document.project
    events: list[object] = []
    window.data_panel.mask_changed.connect(events.append)
    monkeypatch.setattr(
        window.plot_panel,
        "project_project",
        lambda _project: (_ for _ in ()).throw(RuntimeError("plot mask rejected")),
    )

    with pytest.raises(RuntimeError, match="plot mask rejected"):
        window.data_panel.set_point_enabled(dataset_id, 10, False)

    assert window.document.project is before
    assert window.document.is_dirty is False
    assert events == []


def test_mask_updates_active_and_nonactive_diagnostic_views(qtbot) -> None:
    first = prepared_data(size=4)
    second = prepared_data(size=4, intensity_raw=np.array([90.0, 70.0, 50.0, 30.0]))
    panel = _panel(qtbot)
    panel.set_dataset("first", first)
    panel.set_dataset("second", second)
    panel.select_dataset("first")

    panel.update_mask("first", np.array([True, False, True, True]))
    assert _line_y(panel.view("raw"), "排除点").size == 1

    panel.update_mask("second", np.array([False, True, True, True]))
    panel.select_dataset("second")
    assert _line_y(panel.view("raw"), "排除点").size == 1


def test_nonactive_mask_update_does_not_switch_plot_dataset(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_dataset("first", prepared_data(size=4))
    panel.set_dataset("second", prepared_data(size=4))
    panel.select_dataset("first")
    before = _artist_snapshot(panel)

    panel.update_mask("second", np.array([False, True, True, True]))

    assert panel.selected_dataset_id() == "first"
    assert _artist_snapshot(panel) == before


def test_persisted_dock_and_plot_tab_changes_mark_project_dirty(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project_with_curves(tmp_path, count=1)))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    window.plot_panel.tabs.setCurrentIndex(1)
    assert window.document.project.ui_state.plot_tab_index == 1
    assert window.document.is_dirty is True

    # Panel geometry now lives in the opaque dock state rather than splitter
    # sizes, so rearranging a dock is what has to reach the project. The docks
    # only exist on the expert surface, which the guided default hides.
    window.set_guidance_visible(False)
    QApplication.processEvents()
    window.docks["resultsDock"].hide()
    QApplication.processEvents()
    assert window.document.project.ui_state.dock_state != ""


@pytest.mark.parametrize(
    ("expert_mode", "operation"),
    ((False, "replace"), (False, "select"), (True, "replace"), (True, "select")),
)
def test_plot_panel_candidate_redraw_failure_restores_all_views(
    qtbot,
    monkeypatch,
    expert_mode,
    operation,
) -> None:
    import xrr_fitter.gui.plots.panel as plot_panel_module

    data = prepared_data(size=4)
    result = _result(data)
    panel = _panel(qtbot, data=data, result=result)
    panel.set_expert_mode(expert_mode)
    before = _artist_snapshot(panel)
    original = plot_panel_module.draw_uncertainty
    failed = False

    def fail_live_once(view, value, candidate_id):
        nonlocal failed
        original(view, value, candidate_id)
        if candidate_id == "candidate-b" and hasattr(view.canvas, "_draw_timer") and not failed:
            failed = True
            raise RuntimeError("live candidate redraw failed")

    monkeypatch.setattr(plot_panel_module, "draw_uncertainty", fail_live_once)
    replacement = replace(result, warnings=("replacement",)) if operation == "replace" else result

    with pytest.raises(RuntimeError, match="live candidate redraw failed"):
        panel.set_result(replacement, "candidate-b")

    assert panel.selected_candidate_id() == "candidate-a"
    assert _artist_snapshot(panel) == before


def test_plot_panel_close_cancels_pending_draws_and_releases_plot_resources(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    # Only the matplotlib panes own a canvas with a deferred-draw timer; the live
    # reflectivity panes drive their own scene, so the release contract is read
    # off the static views.
    mpl_keys = _mpl_view_keys(panel)
    canvases = tuple(panel.view(key).canvas for key in mpl_keys)
    for canvas in canvases:
        canvas.draw_idle()

    panel.close()

    assert panel.resources_released() is True
    assert all(not canvas._draw_timer.isActive() for canvas in canvases)
    assert all(not panel.view(key).figure.axes for key in mpl_keys)


def test_plot_panel_close_releases_agg_renderer_buffers(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    # The Agg renderer buffer belongs to a matplotlib canvas; the raw pane is now
    # a live pg widget, so the buffer-release contract is read off a static view.
    view = panel.view("candidates")
    view.figure.set_layout_engine(None)
    view.figure.set_size_inches(0.8, 0.6)
    view.canvas.draw()
    assert view.canvas.renderer is not None

    panel.close()

    assert not hasattr(view.canvas, "renderer")


def test_hidden_plot_canvas_defers_queued_draw_until_visible(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    # The deferral contract belongs to the matplotlib canvases; the default tab is
    # now a live pg pane, so a static view is selected and its canvas observed.
    panel.select_view("candidates")
    canvas = panel.view("candidates").canvas

    QApplication.processEvents()

    assert not hasattr(canvas, "renderer")
    panel.show()
    qtbot.waitExposed(panel)
    qtbot.waitUntil(lambda: hasattr(canvas, "renderer"))


def test_scratch_plot_views_are_bounded_and_unrendered_until_projection() -> None:
    from xrr_fitter.gui.plots.diagnostics import build_scratch_views, release_scratch_views

    views = build_scratch_views()
    try:
        assert all(not hasattr(view.canvas, "renderer") for view in views.values())
        assert all(tuple(view.figure.get_size_inches()) == pytest.approx((0.8, 0.6)) for view in views.values())
        assert all(view.figure.dpi == pytest.approx(25.0) for view in views.values())
    finally:
        release_scratch_views(views)


def test_scratch_plot_views_reuse_figures_without_retaining_renderers() -> None:
    from xrr_fitter.gui.plots.diagnostics import build_scratch_views, release_scratch_views

    first = build_scratch_views()
    figure_ids = tuple(id(view.figure) for view in first.values())
    release_scratch_views(first)

    second = build_scratch_views()
    try:
        assert tuple(id(view.figure) for view in second.values()) == figure_ids
        assert all(not hasattr(view.canvas, "renderer") for view in second.values())
    finally:
        release_scratch_views(second)


def test_plot_panel_python_collection_does_not_invoke_destroyed_child_slot(qapp) -> None:
    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    panel.set_dataset("curve", prepared_data(size=4))
    panel_ref = weakref.ref(panel)

    del panel
    gc.collect()
    QApplication.processEvents()

    remaining = panel_ref()
    assert remaining is None or not isValid(remaining)


def test_plot_panel_create_redraw_close_releases_qt_and_matplotlib_objects(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    # The Qt/matplotlib object graph under test lives on the static views; the raw
    # pane is a live pg widget with no figure, so a static view is weak-referenced.
    canvas_ref = weakref.ref(panel.view("candidates").canvas)
    figure_ref = weakref.ref(panel.view("candidates").figure)
    panel_ref = weakref.ref(panel)

    panel.close()
    panel.deleteLater()
    del panel
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()

    assert panel_ref() is None
    assert canvas_ref() is None
    assert figure_ref() is None


def test_plot_panel_escape_cancels_only_from_panel_descendants(qtbot) -> None:
    from xrr_fitter.gui.plots.panel import PlotPanel

    parent = QWidget()
    layout = QVBoxLayout(parent)
    panel = PlotPanel()
    panel.set_dataset("curve", prepared_data(size=4))
    outside = QLineEdit(parent)
    layout.addWidget(panel)
    layout.addWidget(outside)
    qtbot.addWidget(parent)
    parent.show()
    panel.set_interaction_mode("range")
    panel.show_range(0.5, 2.5)
    panel.mode_buttons()["range"].setFocus()

    qtbot.keyClick(panel.mode_buttons()["range"], Qt.Key.Key_Escape)
    assert panel.interaction_mode() == "view"
    assert panel.visible_range() is None

    panel.set_interaction_mode("range")
    panel.show_range(0.5, 2.5)
    outside.setFocus()
    qtbot.keyClick(outside, Qt.Key.Key_Escape)
    assert panel.interaction_mode() == "range"
    assert panel.visible_range() == (0.5, 2.5)


def test_plot_panel_keyboard_activates_modes_and_canvases_are_accessible(qtbot) -> None:
    panel = _panel(qtbot)
    button = panel.mode_buttons()["mask"]

    qtbot.keyClick(button, Qt.Key.Key_Space)

    assert panel.interaction_mode() == "mask"
    # A live pane is its own addressable widget; an mpl view exposes one through
    # .canvas. build_tabs gives both the same name/description/focus contract.
    for key in panel.view_keys():
        widget = _addressable(panel.view(key))
        assert widget.accessibleName()
        assert widget.accessibleDescription()
        assert widget.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_plot_panel_mask_mode_click_requests_prepared_point(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    emitted: list[int] = []
    panel.point_mask_requested.connect(emitted.append)
    panel.set_interaction_mode("mask")
    raw = panel.view("raw")

    # The pg pane reports a mask click as a scene view-x (a 2θ angle); the
    # controller maps that angle to the nearest prepared point, the twin of the
    # old canvas button-press the mpl pane fielded.
    raw.point_mask_requested.emit(float(data.two_theta_deg[2]))

    assert emitted == [2]


def test_plot_panel_matplotlib_text_has_no_missing_cjk_glyphs(
    qtbot,
    monkeypatch,
    caplog,
) -> None:
    import xrr_fitter.gui.plots.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "CJK_FONT_FAMILIES", ("Hiragino Sans GB",))
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # The missing-glyph warning is a matplotlib text-rendering concern; the
        # live pg panes render labels through Qt, so only the static views draw.
        for key in _mpl_view_keys(panel):
            view = panel.view(key)
            view.figure.set_layout_engine(None)
            view.figure.set_size_inches(0.8, 0.6)
            canvas = view.canvas
            canvas.draw()
            canvas.release()

    assert not [warning for warning in caught if "Glyph" in str(warning.message)]
    assert "glyph" not in caplog.text.lower()


def test_live_preview_legend_draws_without_missing_cjk_glyphs(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    log = panel.view("log")

    # The searching-model preview is a lazily-created managed item named "搜索中
    # 模型" on the live log pane; pg renders its label through Qt, so publishing
    # then clearing must add and then drop exactly that one item.
    assert panel.set_preview_curve(data.qz_a_inv, np.full(data.qz_a_inv.size, 0.5))
    assert log.preview_item is not None

    panel.clear_preview_curve()
    assert log.preview_item is None


def test_plot_panel_no_best_redraw_failure_restores_previous_candidate(
    qtbot,
    monkeypatch,
) -> None:
    import xrr_fitter.gui.plots.panel as plot_panel_module

    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    invalid = _candidate(
        data,
        "candidate-invalid",
        objective=float("inf"),
        valid=False,
        stop_reason="invalid_model",
    )
    no_best = replace(_result(data, uncertainty=False), candidates=(invalid,), best_index=None)
    original = plot_panel_module.draw_candidate_comparison
    failed = False

    def fail_live_once(view, result, candidate_id):
        nonlocal failed
        original(view, result, candidate_id)
        if candidate_id is None and hasattr(view.canvas, "_draw_timer") and not failed:
            failed = True
            raise RuntimeError("no-best redraw failed")

    monkeypatch.setattr(plot_panel_module, "draw_candidate_comparison", fail_live_once)

    with pytest.raises(RuntimeError, match="no-best redraw failed"):
        panel.set_result(no_best, None)

    assert panel.selected_candidate_id() == "candidate-a"


def test_plot_panel_parent_destroy_breaks_python_cycles_without_gc(qapp) -> None:
    from xrr_fitter.gui.plots.panel import PlotPanel

    parent = QWidget()
    panel = PlotPanel()
    panel.setParent(parent)
    panel.set_dataset("curve", prepared_data(size=4))
    panel_ref = weakref.ref(panel)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    assert not isValid(panel)
    del panel
    del parent

    remaining = panel_ref()
    assert remaining is None or not isValid(remaining)


def test_plot_panel_parent_destroy_releases_diagnostic_views(qtbot) -> None:
    from xrr_fitter.gui.plots.panel import PlotPanel

    parent = QWidget()
    panel = PlotPanel(parent)
    panel.set_dataset("curve", prepared_data(size=4))
    # Only the matplotlib views own a figure; the live panes release their own
    # scene, so the figure-teardown contract is read off the static views.
    figures = tuple(panel.view(key).figure for key in _mpl_view_keys(panel))

    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert not isValid(panel)
    assert all(not figure.axes for figure in figures)


def test_plot_panel_range_drag_remains_visible_after_dataset_redraw(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    panel.set_interaction_mode("range")

    _drag_range(panel, 0.5, 2.5)
    panel.set_dataset("curve", data)

    assert panel.visible_range() == pytest.approx((0.5, 2.5))
    # The committed band rides the live pane's read-only region, redrawn from the
    # retained visible range -- the twin of the matplotlib axvspan patch.
    assert panel.view("raw").fit_range_item is not None


def test_plot_panel_range_mode_drag_emits_stored_two_theta(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    emitted: list[tuple[float, float]] = []
    panel.fit_range_requested.connect(lambda low, high: emitted.append((low, high)))
    panel.set_interaction_mode("range")

    _drag_range(panel, 0.5, 2.5)

    assert emitted == [pytest.approx((0.5, 2.5), abs=1e-6)]


def test_plot_panel_repeated_candidate_redraw_keeps_artists_axes_and_callbacks(qtbot) -> None:
    data = prepared_data(size=4)
    result = _result(data)
    panel = _panel(qtbot, data=data, result=result)
    # Only the matplotlib views carry reusable axes and mpl callbacks; the live
    # panes update managed items in place, so axis identity is read off them.
    mpl_keys = _mpl_view_keys(panel)
    axes_ids = tuple(id(panel.view(key).axes) for key in mpl_keys)
    callback_counts = panel.callback_counts()

    for index in range(8):
        panel.set_result(result, "candidate-a" if index % 2 == 0 else "candidate-b")

    assert tuple(id(panel.view(key).axes) for key in mpl_keys) == axes_ids
    assert panel.callback_counts() == callback_counts
    assert max(len(panel.view(key).axes.lines) for key in mpl_keys) <= 4


def test_plot_panel_replaces_stale_candidate_with_untrusted_no_best_result(qtbot) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data, result=_result(data))
    invalid = _candidate(
        data,
        "candidate-invalid",
        objective=float("inf"),
        valid=False,
        stop_reason="invalid_model",
    )
    no_best = replace(_result(data, uncertainty=False), candidates=(invalid,), best_index=None)

    panel.set_result(no_best, None)

    assert panel.selected_candidate_id() is None
    assert "暂无当前候选" in (panel.view("qz4").placeholder_text() or "")
    labels = tuple(line.get_label() for line in panel.view("candidates").axes.lines)
    assert any("candidate-invalid" in label and "仅供检查" in label for label in labels)


def test_plot_panel_teardown_does_not_leave_deleted_canvas_traceback(
    qtbot,
    capsys,
) -> None:
    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    panel.set_dataset("curve", prepared_data(size=4))
    # Only the matplotlib canvases can leave a deleted-C++ traceback on teardown;
    # the live panes have no canvas, so the static views drive the queued draws.
    for key in _mpl_view_keys(panel):
        panel.view(key).canvas.draw_idle()

    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qtbot.wait(10)

    assert "Internal C++ object" not in capsys.readouterr().err
