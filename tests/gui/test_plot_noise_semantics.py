"""Both plot backends label the exact candidate evidence they display."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from tests.gui.plot_support import _candidate, _panel, _result
from tests.support.model_cases import final_fit_result, prepared_data

import xrr_fitter.api as api
from xrr_fitter.gui.plots.diagnostics import build_scratch_views, release_scratch_views
from xrr_fitter.gui.plots.reflectivity import _prepared_dataset, draw_residual
from xrr_fitter.gui.plots.sld import draw_uncertainty
from xrr_fitter.model.bootstrap import BootstrapResult


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_live_and_static_residual_axes_use_candidate_units(qtbot, mode):
    data = prepared_data(size=24)
    candidate = _candidate(data, noise_model=mode)
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))
    view = panel.view("residual")
    label = view.plot_item.getAxis("left").labelText
    assert candidate.residual_name in label
    assert candidate.residual_unit in label
    np.testing.assert_array_equal(view.observed_item.getData()[1], candidate.weighted_residuals)
    scratch = build_scratch_views()
    try:
        draw_residual(scratch["residual"], candidate)
        assert scratch["residual"].axes.get_ylabel() == label
    finally:
        release_scratch_views(scratch)


@pytest.mark.parametrize("mode", ("gaussian", "poisson"))
def test_quality_caption_reports_actual_mode_not_a_recomputed_log_statistic(qtbot, mode):
    data = prepared_data(size=24)
    candidate = _candidate(data, noise_model=mode)
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))
    caption = panel.view("log").quality_caption_text()
    assert mode in caption
    assert f"J={candidate.objective:.4g}" in caption
    assert "平均残差" not in caption
    assert "decade" not in caption


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_residual_heatmap_colour_key_retains_the_candidate_units(qtbot, mode):
    data = prepared_data(size=24)
    candidate = _candidate(data, noise_model=mode)
    panel = _panel(qtbot, data=data, result=final_fit_result(candidate))
    view = panel.view("residual_map")
    label = view.figure.axes[1].get_ylabel()
    assert candidate.residual_name in label
    assert candidate.residual_unit in label
    np.testing.assert_array_equal(view.axes.images[0].get_array()[0], candidate.weighted_residuals)


def test_interval_metadata_does_not_collapse_the_diagnostic_axes(qtbot):
    data = prepared_data(size=24)
    panel = _panel(qtbot, data=data, result=_result(data))
    view = panel.view("uncertainty")
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        view.canvas.draw()
    assert not any("axes sizes collapsed" in str(item.message) for item in observed)
    renderer = view.canvas.get_renderer()
    assert all(note.get_window_extent(renderer).x1 <= view.figure.bbox.x1 for note in view.figure.axes[1].texts)


def test_project_plot_preparation_respects_gaussian_zero_and_negative_observations(tmp_path):
    source = tmp_path / "gaussian.xy"
    source.write_text("\n".join(f"{0.05 + i * 0.03} {i - 2} 1" for i in range(40)) + "\n", encoding="utf-8")
    value = replace(api.new_project(), fit_config=replace(api.FitConfig.fast(73), noise_model="gaussian"))
    value = api.add_dataset(
        value, source, api.InstrumentSpec(), column_mapping=api.DataColumnMapping(intensity_sigma=2)
    )

    data, _mask = _prepared_dataset(value, value.datasets[0])

    np.testing.assert_array_equal(data.intensity_raw[:3], [-2, -1, 0])
    assert np.all(data.validation_mask[:3])


def test_profile_plot_labels_exploratory_evidence_without_likelihood_claims():
    profile = api.ParameterProfile("scale", np.array([0.8, 1, 1.2]), np.array([0.3, 0.1, 0.4]), True, True)
    bootstrap = BootstrapResult(
        ("scale",),
        np.ones((8, 1)),
        (),
        0,
        8,
        method="wild_block_residual",
        unavailable_reason="insufficient_successful_samples",
    )
    report = api.UncertaintyReport(
        ("scale",),
        np.ones((1, 1)),
        (profile,),
        (),
        0,
        (),
        (),
        None,
        (),
        candidate_id="candidate-a",
        bootstrap_performed=True,
        bootstrap_evidence=bootstrap,
    )
    data = prepared_data(size=24)
    result = replace(final_fit_result(_candidate(data)), uncertainty=report)
    views = build_scratch_views()
    try:
        draw_uncertainty(views["uncertainty"], result, "candidate-a")
        axes = views["uncertainty"].figure.axes[1]
        annotations = tuple(axes.texts) + tuple(axes.get_legend().get_texts())
        text = "\n".join(item.get_text() for item in annotations)
        assert "loss_support" in text
        assert "objective_tolerance" in text
        assert "exploratory_bootstrap" in text
        assert "insufficient_successful_samples" in text.replace("\n", "")
        assert "95%" not in text
        assert "似然" not in axes.get_title()
    finally:
        release_scratch_views(views)
