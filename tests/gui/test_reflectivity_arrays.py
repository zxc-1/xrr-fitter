"""reflectivity_pane_arrays feeds the pyqtgraph panes the exact data matplotlib drew.

The wholesale pyqtgraph migration routes the four interactive reflectivity panes
(log / raw / qz⁴ / residual) through ``LiveReflectivityPlot.show_*`` instead of the
matplotlib ``draw_*`` functions. This pure projection helper is the seam: it turns a
``PreparedData`` + mask + candidate into the backend-neutral arrays each ``show_*``
wants. Pinning those arrays against what ``draw_*`` plots proves the swap is faithful
before any panel wiring changes, so a reader sees identical curves whichever backend
drew the tab.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.support.model_cases import fit_candidate, prepared_data

from xrr_fitter.gui.plots.diagnostics import build_scratch_views
from xrr_fitter.gui.plots.reflectivity import (
    draw_log,
    draw_qz4,
    draw_raw,
    draw_residual,
    reflectivity_pane_arrays,
)


def _candidate(data):
    size = data.two_theta_deg.size
    return replace(
        fit_candidate("candidate-a", 0.2),
        qz_a_inv=np.linspace(0.015, 0.25, size),
        model_normalized=np.geomspace(0.9, 2e-5, size),
        log_residuals_decades=np.linspace(-0.2, 0.2, size),
        weighted_residuals=np.linspace(-1.0, 1.0, size),
        sld_depth_a=np.array([0.0, 20.0, 50.0]),
        sld_profile_a2=np.array([0.0 + 0.0j, 2e-5 + 1e-7j, 4e-6 + 0.0j]),
    )


def _line(view, label):
    return next(line for line in view.axes.lines if line.get_label() == label)


def test_log_pane_arrays_match_draw_log():
    data = prepared_data(size=24)
    candidate = _candidate(data)
    arrays = reflectivity_pane_arrays(data, data.fit_mask, candidate)
    views = build_scratch_views()
    draw_log(views["log"], data, candidate)
    observed = _line(views["log"], "归一化数据").get_ydata()
    model = _line(views["log"], "当前候选模型").get_ydata()
    # draw_log floors for display; the struct carries the unfloored values and the
    # widget floors on show, so parity compares the floored projection.
    assert arrays.r_floor == data.r_floor
    assert np.allclose(np.maximum(arrays.log_observed, arrays.r_floor), observed)
    assert arrays.log_model is not None
    assert np.allclose(np.maximum(arrays.log_model, arrays.r_floor), model)
    # The bottom-right J=… · 平均残差 … note the pg pane will set as its quality
    # caption must read byte-identically to the matplotlib overlay draw_log wrote.
    assert arrays.quality_caption == views["log"].axes.texts[-1].get_text()


def test_raw_pane_arrays_match_draw_raw():
    data = prepared_data(size=24)
    candidate = _candidate(data)
    arrays = reflectivity_pane_arrays(data, data.fit_mask, candidate)
    views = build_scratch_views()
    draw_raw(views["raw"], data, data.fit_mask, candidate)
    angles, raw, keep = arrays.raw_angles, arrays.raw_intensity, arrays.raw_mask
    finite = np.isfinite(angles) & np.isfinite(raw)
    assert np.allclose(_line(views["raw"], "拟合点").get_ydata(), raw[finite & keep])
    assert np.allclose(_line(views["raw"], "排除点").get_ydata(), raw[finite & ~keep])
    assert arrays.raw_model is not None
    assert np.allclose(_line(views["raw"], "当前候选模型").get_ydata(), arrays.raw_model)


def test_qz4_pane_arrays_match_draw_qz4():
    data = prepared_data(size=24)
    candidate = _candidate(data)
    arrays = reflectivity_pane_arrays(data, data.fit_mask, candidate)
    assert arrays.qz4 is not None
    data_qz, data_values, model_qz, model_values = arrays.qz4
    views = build_scratch_views()
    draw_qz4(views["qz4"], data, candidate)
    obs = _line(views["qz4"], "归一化数据")
    mod = _line(views["qz4"], "当前候选模型")
    assert np.allclose(obs.get_xdata(), data_qz) and np.allclose(obs.get_ydata(), data_values)
    assert np.allclose(mod.get_xdata(), model_qz) and np.allclose(mod.get_ydata(), model_values)
    # The pg pane takes its dynamic y-label from the arrays; it must equal the axis
    # label draw_qz4 chose, including the overflow-scaling reference when present.
    assert arrays.qz4_ylabel == views["qz4"].axes.get_ylabel()


def test_residual_pane_arrays_match_draw_residual():
    data = prepared_data(size=24)
    candidate = _candidate(data)
    arrays = reflectivity_pane_arrays(data, data.fit_mask, candidate)
    assert arrays.residual is not None
    qz, weighted = arrays.residual
    views = build_scratch_views()
    draw_residual(views["residual"], candidate)
    line = _line(views["residual"], "加权残差")
    assert np.allclose(line.get_xdata(), qz) and np.allclose(line.get_ydata(), weighted)


def test_without_candidate_dependent_panes_are_empty():
    data = prepared_data(size=16)
    arrays = reflectivity_pane_arrays(data, data.fit_mask, None)
    # A missing candidate empties every pane that depicts a fit, matching how the
    # matplotlib qz⁴ and residual views fall back to their placeholder state.
    assert arrays.log_model is None
    assert arrays.raw_model is None
    assert arrays.qz4 is None
    assert arrays.residual is None
    # The fit-quality annotations are candidate-derived, so a bare dataset carries
    # neither the qz⁴ label nor the quality caption and the panes stay unannotated.
    assert arrays.qz4_ylabel is None
    assert arrays.quality_caption is None
    # The observed series each pane keeps regardless of a candidate stays present.
    assert np.allclose(arrays.log_observed, data.intensity_normalized)
    assert np.allclose(arrays.raw_intensity, data.intensity_raw)
    assert np.array_equal(arrays.raw_mask, np.asarray(data.fit_mask, dtype=bool))
