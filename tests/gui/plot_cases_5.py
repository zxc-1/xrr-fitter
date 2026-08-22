"""Plot contract cases, partition 5; collected via test_plots.py.

These cases pin the side-by-side reflectivity/SLD workspace. Judging a fit
requires reading curve agreement and the depth profile together, so the SLD
view is a permanent companion pane rather than one of the switchable
diagnostic tabs, and the default tab is the log view where reflectivity
structure is actually legible.
"""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403
from tests.support.model_cases import simple_structure as _simple_structure

from xrr_fitter.gui.theme import build_stylesheet


def _aligned_bands(label: str, marker: float = 0.0):
    depth = np.linspace(0.0, 40.0, 4)
    levels = (0.025, 0.16, 0.5, 0.84, 0.975)
    rows = (marker + np.arange(len(levels), dtype=float))[:, None] * 1e-6
    real = np.tile(rows, (1, depth.size))
    return api.SldUncertaintyBands(
        depth_a=depth,
        quantiles=levels,
        real=real,
        imaginary=real * 0.5,
        align_label=label,
        sample_count=4,
        total_samples=4,
        failure_rate=0.0,
    )


def _mcmc_report(marker: float = 0.0):
    samples = np.array([[20.0], [25.0], [30.0], [35.0]]) + marker
    return api.McmcReport(
        config=api.McmcConfig(walkers=4, burn_in=0, production_steps=4),
        child_seed=7,
        parameter_names=("component.0.thickness_a",),
        samples_physical=samples,
        log_probability=np.zeros(4),
        acceptance_fraction=np.full(4, 0.4),
        split_rhat=np.ones(1),
        effective_sample_size=np.full(1, 4.0),
        boundary_hits=(),
        candidate_id="candidate-a",
    )


def _sld_result(data, marker: float = 0.0):
    uncertainty = replace(
        _uncertainty(),
        mcmc=_mcmc_report(marker),
        sld_bands=_aligned_bands("基底界面", marker),
    )
    return replace(_result(data), uncertainty=uncertainty)


def _sld_project(tmp_path, name: str, marker: float = 0.0):
    source = tmp_path / f"{name}.xy"
    source.write_text("0.1 100\n0.2 50\n0.3 25\n0.4 12\n", encoding="utf-8")
    project = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id=f"{name}-instrument"),
    )
    dataset_id = project.datasets[0].dataset_id
    project = api.set_structure(project, dataset_id, _simple_structure())
    result = _sld_result(prepared_data(size=4), marker)
    project = replace(
        project,
        datasets=(replace(project.datasets[0], last_valid_result=result),),
    )
    return project, result


def _projected_sld_panel(qtbot, tmp_path, name: str = "curve", marker: float = 0.0):
    from xrr_fitter.gui.plots.panel import PlotPanel

    project, result = _sld_project(tmp_path, name, marker)
    panel = PlotPanel()
    qtbot.addWidget(panel)
    panel.project_project(project)
    return panel, result


def _surface_replay(monkeypatch):
    calls: list[str] = []

    def replay(_structure, _report, *, wavelength_a, align):
        assert wavelength_a > 0.0
        calls.append(align)
        label = "表面界面" if align == "surface" else "基底界面"
        return _aligned_bands(label, 20.0 + len(calls))

    monkeypatch.setattr(api, "sld_uncertainty_bands", replay)
    return calls


def _fail_next_live_uncertainty(monkeypatch, message: str) -> None:
    import xrr_fitter.gui.plots.panel as plot_panel_module

    original = plot_panel_module.draw_uncertainty
    failed = False

    def fail_live_once(view, result, candidate_id):
        nonlocal failed
        original(view, result, candidate_id)
        if hasattr(view.canvas, "_draw_timer") and not failed:
            failed = True
            raise RuntimeError(message)

    monkeypatch.setattr(plot_panel_module, "draw_uncertainty", fail_live_once)


def test_sld_is_a_permanent_pane_outside_the_diagnostic_tabs(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_expert_mode(True)

    assert "sld" not in panel.tab_keys()
    assert "SLD 深度剖面" not in panel.tab_titles()
    # The view itself survives; only its placement changed.
    assert panel.view("sld") is not None
    assert panel.sld_pane.isVisibleTo(panel) is True


def test_reflectivity_and_sld_are_visible_together(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.show()

    reflectivity = _addressable(panel.view(panel.current_view_key()))
    sld = panel.view("sld").canvas

    assert reflectivity.isVisibleTo(panel) is True
    assert sld.isVisibleTo(panel) is True
    # Stacked vertically, so neither pane occludes the other.
    assert panel.plot_splitter.orientation() == Qt.Orientation.Vertical


def test_log_view_is_the_default_selection(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.tab_keys()[0] == "log"
    assert panel.current_view_key() == "log"


def test_remaining_tabs_keep_their_order_after_sld_leaves(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.tab_keys() == (
        "log",
        "raw",
        "qz4",
        "residual",
        "candidates",
        "residual_map",
        "parameter_map",
        "uncertainty",
        "trend",
    )


def test_every_tab_reads_in_full_at_the_documented_window_width(qtbot) -> None:
    """No diagnostic may hide behind a scroll arrow or an ellipsis at 1280.

    Qt's default answer to nine labels wanting more room than the stack has is to
    park the overflow behind scroll arrows, and a diagnostic a user never sees is
    one they will not know exists.  Turning the arrows off is only half an
    answer: at the general 12px tab padding the labels then elide down to two
    characters, where 加权残差 / 残差热图 / 参数热图 all read as the same trimmed
    stub.  The narrower padding this bar asks for buys back the characters, so
    the contract is the strong one -- every label drawn whole.

    Qt elides exactly when a tab's rect is narrower than its size hint, so
    ``rect >= hint`` is the assertion for "drawn without an ellipsis", and it
    implies the weaker "inside the bar" check it replaces.
    """
    panel = _panel(qtbot, data=prepared_data(size=4))
    # _panel builds the widget directly, bypassing apply_theme, so the padding
    # rule under test would not otherwise be in play.
    panel.setStyleSheet(build_stylesheet(panel.palette()))
    panel.show()
    qtbot.waitExposed(panel)
    panel.resize(568, 600)
    qtbot.wait(20)

    bar = panel.tabs.tabBar()
    assert bar.usesScrollButtons() is False
    assert bar.elideMode() == Qt.TextElideMode.ElideRight
    assert bar.width() <= 568, "the bar got more room than the documented layout leaves it"
    for index in range(bar.count()):
        label = bar.tabText(index)
        rect = bar.tabRect(index)
        assert rect.x() + rect.width() <= bar.width(), f"tab {label!r} overflows the bar"
        assert rect.width() >= bar.tabSizeHint(index).width(), f"tab {label!r} is drawn elided"


def test_each_tab_names_itself_in_full_through_its_tooltip(qtbot) -> None:
    """Labels elide at narrower widths, so the full name lives in the per-tab tip."""
    panel = _panel(qtbot)
    tabs = panel.tabs

    for index in range(tabs.count()):
        tip = tabs.tabToolTip(index)
        assert tabs.tabText(index) in tip
        # A tip that only repeats the label adds nothing over reading the tab.
        assert tip != tabs.tabText(index)


def test_sld_pane_is_expert_only(qtbot) -> None:
    """The depth profile is expert evidence, so standard mode keeps it hidden."""
    panel = _panel(qtbot, data=prepared_data(size=4))

    panel.set_expert_mode(False)
    assert panel.sld_pane.isVisibleTo(panel) is False

    panel.set_expert_mode(True)
    assert panel.sld_pane.isVisibleTo(panel) is True


def test_number_shortcuts_cover_every_remaining_tab(qtbot) -> None:
    from PySide6.QtGui import QKeySequence

    panel = _panel(qtbot, data=prepared_data(size=4))
    keys = [shortcut.key() for shortcut in panel.view_shortcuts]

    assert len(keys) == 9
    assert keys[0] == QKeySequence("Alt+1")
    assert keys[8] == QKeySequence("Alt+9")


def test_selecting_sld_as_a_tab_is_rejected(qtbot) -> None:
    """``select_view`` addresses tabs; the companion pane is not one."""
    panel = _panel(qtbot)

    with pytest.raises(KeyError, match="sld"):
        panel.select_view("sld")


def test_sld_band_toggle_is_disabled_without_sampling(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    assert panel.sld_bands_toggle.isEnabled() is False
    assert panel.sld_bands_toggle.toolTip() != ""


def test_sld_band_toggle_enables_when_bands_exist(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4), bands=_zero_width_bands())
    assert panel.sld_bands_toggle.isEnabled() is True
    assert panel.sld_bands_toggle.isChecked() is True


def test_sld_draw_without_bands_matches_the_bandless_element_sequence(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    axes = panel.view("sld").axes
    assert not axes.collections


def test_sld_caption_matches_the_export_caption(qtbot) -> None:
    bands = _zero_width_bands()
    panel = _panel(qtbot, data=prepared_data(size=4), bands=bands)
    assert bands.caption() in panel.view("sld").axes.get_title()


def test_sld_band_x_matches_curve_x_scale(qtbot) -> None:
    bands = _zero_width_bands()
    panel = _panel(qtbot, data=prepared_data(size=4), bands=bands)
    axes = panel.view("sld").axes
    line_x = max(float(np.max(line.get_xdata())) for line in axes.lines)
    band_x = max(
        float(np.max(path.vertices[:, 0])) for collection in axes.collections for path in collection.get_paths()
    )
    # depth_a is stored in Angstrom while the pane plots nm; a band that skipped
    # the /10 conversion would sit ~10x further out than the curve it annotates.
    assert band_x <= line_x * 2.0


def test_sld_band_legends_use_en_dashes(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4), bands=_zero_width_bands())
    labels = tuple(text.get_text() for text in panel.view("sld").axes.get_legend().get_texts())

    assert "16–84%" in labels
    assert "2.5–97.5%" in labels
    assert "16-84%" not in labels
    assert "2.5-97.5%" not in labels


def test_surface_aligned_bands_survive_show_range_redraw(qtbot, tmp_path, monkeypatch) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    calls = _surface_replay(monkeypatch)

    panel.sld_align_selector.setCurrentIndex(1)
    panel.show_range(0.1, 0.3)

    assert calls == ["surface"]
    assert "对齐 表面界面" in panel.view("sld").axes.get_title()


def test_surface_aligned_bands_survive_toggle_off_and_on(qtbot, tmp_path, monkeypatch) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    calls = _surface_replay(monkeypatch)

    panel.sld_align_selector.setCurrentIndex(1)
    panel.sld_bands_toggle.setChecked(False)
    assert not panel.view("sld").axes.collections

    panel.sld_bands_toggle.setChecked(True)

    assert calls == ["surface"]
    assert "对齐 表面界面" in panel.view("sld").axes.get_title()


def test_hidden_band_alignment_choice_is_replayed_when_reenabled(qtbot, tmp_path, monkeypatch) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    calls = _surface_replay(monkeypatch)

    panel.sld_bands_toggle.setChecked(False)
    panel.sld_align_selector.setCurrentIndex(1)
    assert calls == []

    panel.sld_bands_toggle.setChecked(True)

    assert calls == ["surface"]
    assert "对齐 表面界面" in panel.view("sld").axes.get_title()


def test_same_report_and_alignment_reuse_one_view_only_replay(qtbot, tmp_path, monkeypatch) -> None:
    panel, result = _projected_sld_panel(qtbot, tmp_path)
    calls = _surface_replay(monkeypatch)

    panel.sld_align_selector.setCurrentIndex(1)
    panel.set_result(result, "candidate-a")
    panel.show_range(0.1, 0.3)

    assert calls == ["surface"]


def test_sld_alignment_cache_keeps_the_report_object_not_its_integer_id(qtbot, tmp_path, monkeypatch) -> None:
    panel, result = _projected_sld_panel(qtbot, tmp_path)
    _surface_replay(monkeypatch)

    panel.sld_align_selector.setCurrentIndex(1)

    assert getattr(panel._sld_band_cache, "report", None) is result.uncertainty.mcmc


def test_new_report_invalidates_surface_cache_and_resets_to_persisted_backing(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, result = _projected_sld_panel(qtbot, tmp_path)
    calls = _surface_replay(monkeypatch)
    panel.sld_align_selector.setCurrentIndex(1)

    replacement = replace(
        result,
        uncertainty=replace(
            result.uncertainty,
            mcmc=_mcmc_report(10.0),
            sld_bands=_aligned_bands("基底界面", 10.0),
        ),
    )
    panel.set_result(replacement, "candidate-a")

    assert calls == ["surface"]
    assert panel.sld_align_selector.currentIndex() == 0
    assert "对齐 基底界面" in panel.view("sld").axes.get_title()


def test_new_dataset_invalidates_surface_cache_and_resets_to_persisted_backing(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path, "first")
    calls = _surface_replay(monkeypatch)
    panel.sld_align_selector.setCurrentIndex(1)
    second, _second_result = _sld_project(tmp_path, "second", 10.0)

    panel.project_project(second)

    assert calls == ["surface"]
    assert panel.selected_dataset_id() == "second"
    assert panel.sld_align_selector.currentIndex() == 0
    assert "对齐 基底界面" in panel.view("sld").axes.get_title()


def test_failed_alignment_replay_keeps_previous_projection_and_does_not_raise_to_qt(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    _surface_replay(monkeypatch)
    panel.sld_align_selector.setCurrentIndex(1)
    before_title = panel.view("sld").axes.get_title()
    before_collections = len(panel.view("sld").axes.collections)

    def fail(*_args, **_kwargs):
        raise ValueError("bad replay")

    monkeypatch.setattr(api, "sld_uncertainty_bands", fail)
    panel.sld_align_selector.setCurrentIndex(0)
    panel.show_range(0.1, 0.3)

    assert panel.sld_align_selector.currentIndex() == 1
    assert panel.view("sld").axes.get_title() == before_title
    assert len(panel.view("sld").axes.collections) == before_collections


def test_failed_alignment_live_draw_restores_the_previous_replay_state(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    _surface_replay(monkeypatch)
    before_title = panel.view("sld").axes.get_title()
    before_collections = len(panel.view("sld").axes.collections)
    _fail_next_live_uncertainty(monkeypatch, "alignment redraw failed")

    panel.sld_align_selector.setCurrentIndex(1)

    assert panel._sld_band_cache is None
    assert panel.sld_align_selector.currentIndex() == 0
    assert panel.view("sld").axes.get_title() == before_title
    assert len(panel.view("sld").axes.collections) == before_collections


def test_new_report_live_failure_restores_surface_sld_projection(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, result = _projected_sld_panel(qtbot, tmp_path)
    _surface_replay(monkeypatch)
    panel.sld_align_selector.setCurrentIndex(1)
    before_cache = panel._sld_band_cache
    before_title = panel.view("sld").axes.get_title()
    before_collections = len(panel.view("sld").axes.collections)
    replacement = replace(
        result,
        uncertainty=replace(
            result.uncertainty,
            mcmc=_mcmc_report(10.0),
            sld_bands=_aligned_bands("基底界面", 10.0),
        ),
    )
    _fail_next_live_uncertainty(monkeypatch, "new report redraw failed")

    with pytest.raises(RuntimeError, match="new report redraw failed"):
        panel.set_result(replacement, "candidate-a")

    assert panel._sld_band_cache is before_cache
    assert panel.sld_align_selector.currentIndex() == 1
    assert panel.view("sld").axes.get_title() == before_title
    assert len(panel.view("sld").axes.collections) == before_collections


def test_first_band_result_live_failure_restores_disabled_unchecked_controls(
    qtbot,
    monkeypatch,
) -> None:
    data = prepared_data(size=4)
    panel = _panel(qtbot, data=data)
    result = _sld_result(data)
    before_title = panel.view("sld").axes.get_title()
    _fail_next_live_uncertainty(monkeypatch, "first band redraw failed")

    with pytest.raises(RuntimeError, match="first band redraw failed"):
        panel.set_result(result, "candidate-a")

    assert panel._result is None
    assert panel.sld_bands_toggle.isEnabled() is False
    assert panel.sld_bands_toggle.isChecked() is False
    assert panel.sld_align_selector.isEnabled() is False
    assert panel.view("sld").axes.get_title() == before_title


def test_new_project_live_failure_restores_surface_sld_projection_and_dataset(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel, result = _projected_sld_panel(qtbot, tmp_path, "first")
    _surface_replay(monkeypatch)
    panel.sld_align_selector.setCurrentIndex(1)
    before_cache = panel._sld_band_cache
    before_title = panel.view("sld").axes.get_title()
    before_collections = len(panel.view("sld").axes.collections)
    second, _second_result = _sld_project(tmp_path, "second", 10.0)
    _fail_next_live_uncertainty(monkeypatch, "new project redraw failed")

    with pytest.raises(RuntimeError, match="new project redraw failed"):
        panel.project_project(second)

    assert panel.selected_dataset_id() == "first"
    assert panel._result is result
    assert panel._sld_band_cache is before_cache
    assert panel.sld_align_selector.currentIndex() == 1
    assert panel.view("sld").axes.get_title() == before_title
    assert len(panel.view("sld").axes.collections) == before_collections


def _structure_only_panel(qtbot, tmp_path, name: str = "curve", *, structure=None):
    """A projected panel that carries a structure but no fitted result.

    The companion pane must show the current structure's nominal profile and its
    draggable interface handles even before a fit exists, so a user can shape the
    stack by hand and then fit.  ``structure`` overrides the default single film
    so a case can pin what a stack that is not drag-editable looks like.
    """
    from xrr_fitter.gui.plots.panel import PlotPanel

    source = tmp_path / f"{name}.xy"
    source.write_text("0.1 100\n0.2 50\n0.3 25\n0.4 12\n", encoding="utf-8")
    project = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id=f"{name}-instrument"),
    )
    dataset_id = project.datasets[0].dataset_id
    project = api.set_structure(project, dataset_id, _simple_structure() if structure is None else structure)
    panel = PlotPanel()
    qtbot.addWidget(panel)
    panel.project_project(project)
    return panel


def test_sld_pane_draws_nominal_structure_and_interface_handles(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    axes = panel.view("sld").axes
    labels = {line.get_label() for line in axes.lines}

    assert "结构标称 实部" in labels
    # A single film gives exactly one draggable interface, at its backing side.
    handles = [line for line in axes.lines if line.get_label() == "_interface_0"]
    assert len(handles) == 1
    assert float(np.asarray(handles[0].get_xdata())[0]) == pytest.approx(2.0, abs=0.05)


def test_sld_interface_handles_stay_out_of_the_legend(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    axes = panel.view("sld").axes
    legend = axes.get_legend()
    labels = () if legend is None else tuple(text.get_text() for text in legend.get_texts())

    assert all(not label.startswith("_interface_") for label in labels)


def test_sld_interface_drag_edits_layer_thickness(qtbot, tmp_path) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    panel.set_expert_mode(True)
    edits: list[object] = []
    panel.structure_edit_requested.connect(edits.append)

    # simple_structure has a 20 A film -> a backing interface at 2.0 nm; dragging
    # it to 3.0 nm asks for a 30 A film.
    _drag_sld_interface(panel, 2.0, 3.0)

    assert len(edits) == 1
    assert edits[0].components[0].thickness_a == pytest.approx(30.0, abs=0.5)


def _nominal_level_at(axes, depth_nm: float) -> float:
    """Sample the nominal real SLD the pane draws at a depth, in A^-2."""
    nominal = next(line for line in axes.lines if line.get_label() == "结构标称 实部")
    depths = np.asarray(nominal.get_xdata(), dtype=float)
    levels = np.asarray(nominal.get_ydata(), dtype=float)
    return float(levels[int(np.argmin(np.abs(depths - float(depth_nm))))])


def test_sld_pane_draws_a_level_handle_per_layer(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    axes = panel.view("sld").axes
    handles = [line for line in axes.lines if line.get_label() == "_level_0"]

    assert len(handles) == 1
    # The handle spans the layer it stands for, so a grab anywhere across the
    # film raises or lowers that film and not a neighbour.
    xdata = np.asarray(handles[0].get_xdata(), dtype=float)
    assert float(xdata.min()) == pytest.approx(0.0, abs=0.05)
    assert float(xdata.max()) == pytest.approx(2.0, abs=0.05)
    # And it sits at the level the pane actually draws mid-film, so what the
    # reader grabs is what the reader sees.
    midpoint = _nominal_level_at(axes, 1.0)
    ydata = np.asarray(handles[0].get_ydata(), dtype=float)
    assert float(ydata.min()) == pytest.approx(midpoint, rel=1e-6)
    assert float(ydata.max()) == pytest.approx(midpoint, rel=1e-6)


def test_sld_level_handles_stay_out_of_the_legend(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    legend = panel.view("sld").axes.get_legend()
    labels = () if legend is None else tuple(text.get_text() for text in legend.get_texts())

    assert all(not label.startswith("_level_") for label in labels)


def test_sld_level_drag_edits_layer_density_scale(qtbot, tmp_path) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    panel.set_expert_mode(True)
    edits: list[object] = []
    panel.structure_edit_requested.connect(edits.append)

    # The SLD is exactly linear in density_scale, so halving the drawn level
    # halves the scale.  The grab is mid-film, clear of the interface handle at
    # 2.0 nm, so the vertical gesture cannot be read as a thickness edit.
    _drag_sld_level(panel, 0, 1.0, 0.5)

    assert len(edits) == 1
    assert edits[0].components[0].density_scale == pytest.approx(0.5, abs=0.02)
    assert edits[0].components[0].thickness_a == pytest.approx(20.0, abs=0.05)


def test_sld_level_drag_leaves_a_gradient_stack_alone(qtbot, tmp_path) -> None:
    """A stack that has no single per-layer level offers nothing to grab.

    A gradient layer's SLD varies across its own thickness, so there is no one
    plateau a handle could stand for; the pane still draws the nominal profile
    but offers no level handle, the same way it offers no interface handle.
    """
    gradient = api.GradientLayerSpec("ramp", 10e-6 + 1e-6j, 30e-6 + 3e-6j, 20.0, microslab_max_a=2.0)
    structure = replace(_simple_structure(), components=(gradient,))
    panel = _structure_only_panel(qtbot, tmp_path, structure=structure)
    axes = panel.view("sld").axes
    labels = [line.get_label() for line in axes.lines]

    assert "结构标称 实部" in labels
    assert not [label for label in labels if label.startswith("_level_")]
    assert not [label for label in labels if label.startswith("_interface_")]


def test_sld_pane_draws_a_roughness_handle_per_editable_interface(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    axes = panel.view("sld").axes
    handles = [line for line in axes.lines if line.get_label() == "_roughness_0"]

    assert len(handles) == 1
    # The whisker starts at the backing interface (2.0 nm) and runs outward by the
    # interface's roughness; simple_structure's backing roughness is 3.0 A = 0.3 nm.
    xdata = np.asarray(handles[0].get_xdata(), dtype=float)
    assert float(xdata.min()) == pytest.approx(2.0, abs=0.05)
    assert float(xdata.max()) == pytest.approx(2.3, abs=0.05)


def test_sld_roughness_handles_stay_out_of_the_legend(qtbot, tmp_path) -> None:
    panel = _structure_only_panel(qtbot, tmp_path)
    legend = panel.view("sld").axes.get_legend()
    labels = () if legend is None else tuple(text.get_text() for text in legend.get_texts())

    assert all(not label.startswith("_roughness_") for label in labels)


def test_sld_roughness_drag_edits_backing_roughness(qtbot, tmp_path) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    panel.set_expert_mode(True)
    edits: list[object] = []
    panel.structure_edit_requested.connect(edits.append)

    # The sole backing-edge handle governs the deepest interface, so its width
    # commits backing_roughness_a, not a component's roughness.  Widening the
    # whisker to 0.6 nm asks for a 6.0 A interface, well under the 9.8 A ceiling.
    _drag_sld_roughness(panel, 0, 0.6)

    assert len(edits) == 1
    assert edits[0].backing_roughness_a == pytest.approx(6.0, abs=0.1)
    assert edits[0].components[0].thickness_a == pytest.approx(20.0, abs=0.05)


def test_sld_roughness_drag_clamps_below_the_domain_ceiling(qtbot, tmp_path) -> None:
    panel, _result_value = _projected_sld_panel(qtbot, tmp_path)
    panel.set_expert_mode(True)
    edits: list[object] = []
    panel.structure_edit_requested.connect(edits.append)

    # The stack rejects a roughness at or above 0.49 x min(neighbour thickness);
    # the 20 A film puts that ceiling at 9.8 A.  Dragging far past it (2.0 nm =
    # 20 A) must land strictly below the ceiling, not at or beyond it.
    _drag_sld_roughness(panel, 0, 2.0)

    assert len(edits) == 1
    assert edits[0].backing_roughness_a < 9.8
    assert edits[0].backing_roughness_a == pytest.approx(9.8, abs=0.01)
    # The clamp must land strictly below the ceiling the stack itself enforces, so
    # expanding the committed structure raises nothing (roughness validation is
    # wavelength-independent, so any positive wavelength proves it).
    from xrr_fitter.physics.stack import expand_structure

    expand_structure(edits[0], wavelength_a=1.5406)


def test_sld_roughness_handle_skips_a_layer_whose_transition_sets_the_width(qtbot, tmp_path) -> None:
    """A layer whose interface width is a transition offers no roughness whisker.

    The stack forbids a nonzero roughness on a layer that carries a transition, so
    the interface backing onto such a layer must not present a grabbable width; the
    deepest interface, backing onto the substrate, still does.
    """
    base = _simple_structure()
    graded = api.LayerSpec(
        "graded",
        base.components[0].material,
        30.0,
        transition=api.InterfaceTransition((api.TransitionBranch("erf", 1.0, 8.0),), microslab_max_a=2.0),
    )
    structure = replace(base, components=(base.components[0], graded))
    panel = _structure_only_panel(qtbot, tmp_path, structure=structure)
    labels = [line.get_label() for line in panel.view("sld").axes.lines]

    # Handle 0 backs onto the graded layer, whose width the transition owns; handle
    # 1 backs onto the substrate and keeps its width grabbable.
    assert "_roughness_0" not in labels
    assert "_roughness_1" in labels
