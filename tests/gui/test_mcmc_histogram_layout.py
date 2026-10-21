"""Keep all posterior marginals readable inside a small evidence viewport."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QScrollArea
from tests.gui.test_uncertainty_dialog import _mcmc_report, _result, _uncertainty

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.parameter_labels import result_definitions, titled_labels
from xrr_fitter.gui.plots.posterior import McmcHistogramPlot

PARAMETER_NAMES = (
    "component.0.thickness_a",
    "component.0.density_scale",
    "component.0.roughness_a",
    "backing.roughness_a",
    "instrument.angle_offset_deg",
    "instrument.scale",
    "instrument.background",
    "instrument.relative_sigma",
)


def _result_with_marginals(count=8):
    scales = np.array([173.0, 0.98, 3.0, 4.05, 0.005, 0.991, 2.183456e-8, 0.015])[:count]
    samples = np.linspace(0.9, 1.1, 128)[:, None] * scales
    chain = _mcmc_report(
        config=api.McmcConfig(walkers=32, burn_in=2, production_steps=4),
        parameter_names=PARAMETER_NAMES[:count],
        samples_physical=samples,
        log_probability=np.zeros(128),
        acceptance_fraction=np.full(32, 0.3),
        split_rhat=np.full(count, 1.01),
        effective_sample_size=np.full(count, 120.0),
    )
    return replace(_result(), uncertainty=replace(_uncertainty(), mcmc=chain))


def _shown_plot(qtbot, size=(400, 202), *, result=None):
    plot = McmcHistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(*size)
    plot.show_report(_result_with_marginals() if result is None else result, "candidate-a")
    plot.show()
    qtbot.waitExposed(plot)
    plot.canvas.draw()
    return plot


def _assert_panel_readable(plot, axes, decorated, renderer):
    assert axes.get_window_extent(renderer).height / plot.canvas.devicePixelRatioF() >= 80
    assert plot.figure.bbox.contains(decorated.x0, decorated.y0), axes.get_title()
    assert plot.figure.bbox.contains(decorated.x1, decorated.y1), axes.get_title()
    assert axes.title.get_fontsize() == theme.FONT_PT_SM
    assert axes.xaxis.label.get_fontsize() == theme.FONT_PT_SM


def _assert_readable(plot):
    renderer = plot.canvas.get_renderer()
    panels = plot.figure.axes
    bounds = [axes.get_tightbbox(renderer) for axes in panels]
    for upper, lower in zip(bounds, bounds[1:], strict=False):
        assert upper.y0 >= lower.y1 + 4, f"Adjacent decorations overlap by {lower.y1 - upper.y0:.2f} px"
    for axes, decorated in zip(panels, bounds, strict=True):
        _assert_panel_readable(plot, axes, decorated, renderer)


@pytest.mark.parametrize("size", ((400, 202), (370, 540), (720, 360)))
def test_all_marginals_stay_readable_without_enlarging_the_viewport(qtbot, size):
    plot = _shown_plot(qtbot, size)
    assert (plot.width(), plot.height()) == size
    # 每个参数一格、按顺序；标题用证据图共用的短名（F26p），不是机器路径。
    titles = titled_labels(PARAMETER_NAMES, result_definitions(_result_with_marginals()))
    assert [axes.get_title() for axes in plot.figure.axes] == list(titles)
    _assert_readable(plot)


def test_scrolling_reaches_the_last_marginal_without_losing_samples(qtbot):
    plot = _shown_plot(qtbot)
    scroll = plot.findChild(QScrollArea, "mcmcHistogramScroll")
    assert scroll is not None, "The complete marginal canvas needs an internal scroll viewport"
    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0
    qtbot.keyClick(bar, Qt.Key.Key_End)
    assert bar.value() == bar.maximum()
    bottom = plot.canvas.mapTo(scroll.viewport(), QPoint(0, plot.canvas.height()))
    assert bottom.y() <= scroll.viewport().height()
    samples = _result_with_marginals().uncertainty.mcmc.samples_physical
    for index, axes in enumerate(plot.figure.axes):
        assert len(axes.patches) == 24
        assert sum(patch.get_height() for patch in axes.patches) == 128
        drawn = sorted(float(line.get_xdata()[0]) for line in axes.lines)
        assert np.allclose(drawn, np.quantile(samples[:, index], (0.16, 0.5, 0.84)))


def _assert_axis_readouts_are_separate(axes, renderer):
    readout = axes.xaxis.label.get_window_extent(renderer)
    lower, upper = axes.get_xlim()
    labels = [text for text in axes.get_xticklabels() if lower <= text.get_position()[0] <= upper]
    for text in (*labels, axes.xaxis.get_offset_text()):
        if text.get_visible() and text.get_text():
            assert not readout.overlaps(text.get_window_extent(renderer)), axes.get_title()
    for left, right in zip(labels, labels[1:], strict=False):
        assert not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer))


def test_quantile_readouts_do_not_overlap_ticks_or_scientific_scale_factors(qtbot):
    plot = _shown_plot(qtbot, (370, 202))
    renderer = plot.canvas.get_renderer()
    for axes in plot.figure.axes:
        _assert_axis_readouts_are_separate(axes, renderer)


def _result_with_narrow_marginal(index, low, high):
    result = _result_with_marginals()
    report = result.uncertainty
    samples = report.mcmc.samples_physical.copy()
    samples[:, index] = np.linspace(low, high, samples.shape[0])
    chain = replace(report.mcmc, samples_physical=samples)
    return replace(result, uncertainty=replace(report, mcmc=chain))


@pytest.mark.parametrize("size", ((370, 202), (720, 360)))
@pytest.mark.parametrize(
    ("index", "low", "high"),
    ((0, 173.01, 173.03), (5, 0.991001, 0.991009), (6, 2.183451e-8, 2.183459e-8)),
    ids=("thickness", "scale", "background"),
)
def test_narrow_marginals_keep_distinct_readable_ticks(qtbot, index, low, high, size):
    plot = _shown_plot(qtbot, size, result=_result_with_narrow_marginal(index, low, high))
    axes = plot.figure.axes[index]
    lower, upper = axes.get_xlim()
    labels = [text.get_text() for text in axes.get_xticklabels() if lower <= text.get_position()[0] <= upper]

    assert len(labels) >= 2
    assert len(set(labels)) == len(labels), labels
    _assert_readable(plot)
    renderer = plot.canvas.get_renderer()
    for panel in plot.figure.axes:
        _assert_axis_readouts_are_separate(panel, renderer)


def test_resizing_keeps_all_marginals_readable(qtbot):
    plot = _shown_plot(qtbot)
    for size in ((720, 600), (370, 202)):
        plot.resize(*size)
        qtbot.wait(1)
        plot.canvas.draw()
        assert (plot.width(), plot.height()) == size
        _assert_readable(plot)


def test_wheel_over_the_read_only_canvas_scrolls_its_viewport(qtbot, qapp):
    plot = _shown_plot(qtbot)
    scroll = plot.findChild(QScrollArea, "mcmcHistogramScroll")
    before = scroll.verticalScrollBar().value()
    limits = [axes.get_xlim() for axes in plot.figure.axes]
    event = QWheelEvent(
        QPointF(80, 80),
        QPointF(plot.canvas.mapToGlobal(QPoint(80, 80))),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    qapp.sendEvent(plot.canvas, event)
    assert scroll.verticalScrollBar().value() > before
    assert [axes.get_xlim() for axes in plot.figure.axes] == limits


@pytest.mark.parametrize("replacement", ("one_parameter", "empty", "other_candidate"))
def test_less_evidence_restores_a_compact_canvas_without_stale_panels(qtbot, replacement):
    plot = _shown_plot(qtbot)
    if replacement == "one_parameter":
        plot.show_report(_result_with_marginals(1), "candidate-a")
    elif replacement == "empty":
        plot.show_report(None, None)
    else:
        plot.show_report(_result_with_marginals(), "candidate-b")
    qtbot.wait(1)
    plot.canvas.draw()
    assert len(plot.figure.axes) == 1
    axes = plot.figure.axes[0]
    assert axes.get_window_extent(plot.canvas.get_renderer()).height >= 80
    assert plot.canvas.height() <= plot.height()
    if replacement != "one_parameter":
        assert not axes.patches
        assert axes.texts
