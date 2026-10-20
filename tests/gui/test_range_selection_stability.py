"""Range selection must not rescale the coordinate system under the pointer."""

from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.gui.plots.live import LiveReflectivityPlot


def _plot(qtbot):
    plot = LiveReflectivityPlot()
    qtbot.addWidget(plot)
    plot.set_observed(np.array([2.0, 3.0, 4.0]), np.array([1.0, 0.5, 0.2]))
    plot.resize(800, 400)
    plot.show()
    qtbot.waitExposed(plot)
    qtbot.wait(1)
    return plot


@pytest.mark.parametrize("committed", (None, (2.2, 3.8)))
def test_range_selector_starts_from_current_window_without_committing(qtbot, committed):
    plot = _plot(qtbot)
    if committed is not None:
        plot.show_fit_range(*committed)
        qtbot.wait(1)
    before = np.array(plot.plot_item.vb.viewRange())
    received = []
    plot.fit_range_selected.connect(lambda *args: received.append(args))

    plot.enable_range_selection(True)
    qtbot.wait(1)

    assert plot.range_item.getRegion() == pytest.approx(committed or tuple(before[0]))
    assert np.array(plot.plot_item.vb.viewRange()) == pytest.approx(before)
    assert received == []


def test_range_selector_motion_does_not_participate_in_auto_range(qtbot):
    plot = _plot(qtbot)
    plot.show_fit_range(2.2, 3.8)
    plot.enable_range_selection(True)
    qtbot.wait(1)
    before = np.array(plot.plot_item.vb.viewRange())

    plot.range_item.setRegion((-5.0, 15.0))
    qtbot.wait(1)

    assert np.array(plot.plot_item.vb.viewRange()) == pytest.approx(before)
    assert plot.plot_item.vb.autoRangeEnabled() == [True, True]


def test_reentering_range_mode_uses_latest_committed_window(qtbot):
    plot = _plot(qtbot)
    plot.show_fit_range(2.2, 3.8)
    plot.enable_range_selection(True)
    plot.enable_range_selection(False)
    plot.show_fit_range(2.5, 3.5)

    plot.enable_range_selection(True)

    assert plot.range_item.getRegion() == pytest.approx((2.5, 3.5))
