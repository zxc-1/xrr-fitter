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

    reflectivity = panel.view(panel.current_view_key()).canvas
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
        "uncertainty",
        "trend",
    )


def test_sld_pane_is_expert_only(qtbot) -> None:
    """The depth profile is expert evidence, so standard mode keeps it hidden."""
    panel = _panel(qtbot, data=prepared_data(size=4))

    panel.set_expert_mode(False)
    assert panel.sld_pane.isVisibleTo(panel) is False

    panel.set_expert_mode(True)
    assert panel.sld_pane.isVisibleTo(panel) is True


def test_number_shortcuts_cover_the_seven_remaining_tabs(qtbot) -> None:
    from PySide6.QtGui import QKeySequence

    panel = _panel(qtbot, data=prepared_data(size=4))
    keys = [shortcut.key() for shortcut in panel.view_shortcuts]

    assert len(keys) == 7
    assert keys[0] == QKeySequence("Alt+1")
    assert keys[6] == QKeySequence("Alt+7")


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
