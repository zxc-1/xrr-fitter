"""Real rendered multi-parameter inference summaries stay bounded and readable."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QPlainTextEdit
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.plots.diagnostics import DiagnosticView, _axes
from xrr_fitter.gui.plots.parameter_labels import short_labels
from xrr_fitter.gui.plots.sld import draw_uncertainty
from xrr_fitter.gui.results.inference_text import interval_metadata
from xrr_fitter.gui.results.panel import ResultsPanel
from xrr_fitter.gui.results.uncertainty import McmcControls, UncertaintyView
from xrr_fitter.model.bootstrap import BootstrapResult
from xrr_fitter.model.inference import CovarianceEvidence

PARAMETER_NAMES = (
    "component.0.thickness_a",
    "component.0.sld_real_a2",
    "component.0.sld_imag_a2",
    "component.0.roughness_a",
    "backing.roughness_a",
    "instrument.scale",
    "instrument.background",
    "instrument.angle_offset_deg",
)


def _report(kind):
    profiles = ()
    if kind != "bootstrap":
        profiles = tuple(
            api.ParameterProfile(name, np.array([0.8, 1.0, 1.2]), np.array([0.3, 0.1, 0.4]), True, True)
            for name in PARAMETER_NAMES
        )
    bootstrap = None
    intervals = ()
    if kind != "profiles":
        intervals = tuple((name, 0.8, 1.2) for name in PARAMETER_NAMES)
        bootstrap = BootstrapResult(
            PARAMETER_NAMES,
            np.linspace(np.full(8, 0.75), np.full(8, 1.25), 400),
            intervals,
            0.0,
            400,
            method="parametric_gaussian",
        )
    return api.UncertaintyReport(
        PARAMETER_NAMES,
        np.eye(8),
        profiles,
        intervals,
        0.0,
        (),
        (),
        None,
        (),
        candidate_id="candidate-a",
        covariance_evidence=CovarianceEvidence(PARAMETER_NAMES, np.eye(8), "gaussian_known_sigma", 8),
        parameter_sigma=np.ones(8),
        bootstrap_performed=bootstrap is not None,
        bootstrap_evidence=bootstrap,
    )


def _render(report):
    figure = Figure(figsize=(10, 5), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    left = _axes(figure, "uncertainty")
    view = DiagnosticView(figure, canvas, left)
    draw_uncertainty(view, SimpleNamespace(uncertainty=report), "candidate-a")
    canvas.draw()
    return view


@pytest.fixture(params=("profiles", "bootstrap", "both"))
def rendered(request, qapp):
    report = _report(request.param)
    view = _render(report)
    yield report, view
    view.figure.clear()


def _annotation_artists(axes):
    artists = list(axes.texts)
    legend = axes.get_legend()
    if legend is not None:
        artists.extend((legend, *legend.get_texts()))
        if legend.get_title().get_text():
            artists.append(legend.get_title())
    return artists


def _assert_inside(outer, inner):
    assert outer.x0 <= inner.x0, (outer.bounds, inner.bounds)
    assert outer.y0 <= inner.y0, (outer.bounds, inner.bounds)
    assert inner.x1 <= outer.x1, (outer.bounds, inner.bounds)
    assert inner.y1 <= outer.y1, (outer.bounds, inner.bounds)


def _annotation_text(axes):
    return "\n".join(artist.get_text() for artist in _annotation_artists(axes) if hasattr(artist, "get_text"))


def _full_evidence_text(qtbot, report):
    result = replace(final_fit_result(fit_candidate("candidate-a")), uncertainty=report)
    full_view = UncertaintyView()
    qtbot.addWidget(full_view)
    full_view.set_result(result, "candidate-a")
    return full_view.text()


def test_multi_parameter_annotations_fit_all_four_canvas_edges(rendered):
    _report_value, view = rendered
    renderer = view.canvas.get_renderer()
    for axes in view.figure.axes:
        artists = (*_annotation_artists(axes), axes.title, axes.xaxis.label, axes.yaxis.label)
        for artist in artists:
            if artist.get_visible() and (not hasattr(artist, "get_text") or artist.get_text()):
                _assert_inside(view.figure.bbox, artist.get_window_extent(renderer))


def test_multi_parameter_summary_has_readable_nonoverlapping_text(rendered):
    _report_value, view = rendered
    axes = view.figure.axes[1]
    renderer = view.canvas.get_renderer()
    assert axes.bbox.width >= 0.25 * view.figure.bbox.width
    allocation = axes.get_subplotspec().get_position(view.figure)
    assert axes.bbox.height >= 0.5 * allocation.height * view.figure.bbox.height
    for artist in _annotation_artists(axes):
        _assert_inside(axes.bbox, artist.get_window_extent(renderer))
        if hasattr(artist, "get_fontsize"):
            assert artist.get_fontsize() >= theme.FONT_PT_SM
    legend = axes.get_legend()
    if legend is not None:
        key = legend.get_window_extent(renderer)
        assert all(not key.overlaps(note.get_window_extent(renderer)) for note in axes.texts)


def test_multi_parameter_summary_keeps_all_profile_curves(rendered):
    report, view = rendered
    axes = view.figure.axes[1]
    assert len(axes.lines) == len(report.profiles)
    labels = short_labels(tuple(profile.name for profile in report.profiles), ())
    for line, profile, label in zip(axes.lines, report.profiles, labels, strict=True):
        assert label in line.get_label()
        np.testing.assert_array_equal(line.get_ydata(), profile.objectives - np.min(profile.objectives))
    if report.profiles:
        assert len(axes.get_legend().get_texts()) == len(report.profiles)


def test_multi_parameter_summary_links_to_complete_preserved_evidence(rendered, qtbot):
    report, view = rendered
    # The current layout keeps the graph uncluttered and puts complete metadata
    # in the owned evidence view, reachable from the read-only evidence action.
    assert "95%" not in _annotation_text(view.figure.axes[1])
    full_text = _full_evidence_text(qtbot, report)
    for profile in report.profiles:
        assert profile.name in full_text
        assert interval_metadata(profile) in full_text
    if report.bootstrap_evidence is not None:
        assert interval_metadata(report.bootstrap_evidence) in full_text
        assert all(name in full_text for name, _lower, _upper in report.bootstrap_intervals)


def test_multi_parameter_bootstrap_summary_shows_saved_sampling_evidence(rendered, qtbot):
    report, view = rendered
    summary = _full_evidence_text(qtbot, report)
    if report.bootstrap_evidence is None:
        assert "Bootstrap：未执行" in summary
    else:
        assert "400/400" in summary
        assert "Bootstrap 失败率：0" in summary
        assert all(name in summary for name, _lower, _upper in report.bootstrap_intervals)
        assert "95%" in summary


def _nonuniform_profiles(kind):
    report = _report("profiles")
    if kind == "mixed_intervals":
        first = replace(
            report.profiles[0],
            interval_kind="likelihood_ratio",
            confidence_level=0.95,
            method="chi_square_1df",
            delta_total=3.841458820694124,
        )
        return replace(report, profiles=(first, *report.profiles[1:]))
    profiles = tuple(
        replace(
            profile,
            interval_kind="unavailable",
            method="likelihood_ratio",
            unavailable_reason="rank_deficient" if index % 2 else "boundary_optimum",
        )
        for index, profile in enumerate(report.profiles)
    )
    return replace(report, profiles=profiles)


@pytest.mark.parametrize("kind", ("mixed_intervals", "unavailable_reasons"))
def test_nonuniform_profile_metadata_is_not_one_shared_calibration(qtbot, kind):
    report = _nonuniform_profiles(kind)
    view = _render(report)
    try:
        summary = _annotation_text(view.figure.axes[1])
        assert "95%" not in summary
        full_text = _full_evidence_text(qtbot, report)
        for profile in report.profiles:
            assert profile.name in full_text
            assert interval_metadata(profile) in full_text
    finally:
        view.figure.clear()


def _results_panel(qtbot, report, expert_mode, tmp_path):
    result = replace(final_fit_result(fit_candidate("candidate-a")), uncertainty=report)
    value = replace(project(dataset_project(result=result)), base_directory=str(tmp_path))
    value = api.select_active_dataset(value, "curve")
    value = api.set_expert_mode(value, expert_mode)
    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    panel.show()
    return panel


def _assert_body_evidence_is_visible(panel, report):
    evidence = panel.findChild(QPlainTextEdit, "uncertaintyEvidence")
    assert evidence is panel.uncertainty.evidence
    assert evidence.isVisibleTo(panel)
    assert evidence.accessibleName() == "候选解不确定度证据"
    assert (
        panel.secondary.layout().indexOf(panel.uncertainty) > panel.secondary.layout().indexOf(panel.clear_button) >= 0
    )
    text = evidence.toPlainText()
    assert all(name in text for name in report.correlation_names)
    assert "[0.08, 0.12] nm" in text
    assert "[0.8, 1.2]" in text


def _assert_dialog_keeps_the_same_owned_evidence(panel):
    dialog = panel.open_uncertainty_dialog()
    assert dialog.findChild(McmcControls) is panel.mcmc_group
    assert dialog.findChild(QPlainTextEdit, "uncertaintyEvidence") is panel.uncertainty.evidence
    assert panel.uncertainty.evidence.isVisibleTo(dialog)


@pytest.mark.parametrize("expert_mode", (False, True), ids=("ordinary", "expert"))
def test_evidence_pointer_names_the_real_results_body_in_both_modes(qtbot, expert_mode, tmp_path):
    report = _report("both")
    panel = _results_panel(qtbot, report, expert_mode, tmp_path)
    view = _render(report)
    try:
        _assert_body_evidence_is_visible(panel, report)
        assert panel.uncertainty_button.isVisibleTo(panel) is expert_mode
        _assert_dialog_keeps_the_same_owned_evidence(panel)
        summary = _annotation_text(view.figure.axes[1])
        assert panel.uncertainty_button.text() not in summary
    finally:
        view.figure.clear()
