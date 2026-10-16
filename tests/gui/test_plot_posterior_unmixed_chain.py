"""链没混合时，「MCMC 后验」页不给分位：不写 P16/P50/P84，也不画三条分位线，改写一句为什么。

审计实测：右栏分位卡在 split-R̂ ≥ 1.10 时已经不给分位（F13），中栏这一页却照旧在每格下面念 P16/P50/P84、
画三条分位线。R̂ 41 的链里那三个数只是各 walker 停在哪里，换一处摆出来，读者照样会当成可信区间抄走。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.gui.plot_support import _uncertainty
from tests.gui.test_plot_axes_readings import NAMES, _two_parameter_result

import xrr_fitter.api as api
from xrr_fitter.gui.plots.posterior import McmcHistogramPlot
from xrr_fitter.gui.results.convergence import unmixed_chain_reason

QUANTILE_MARKS = ("P16", "P50", "P84")


def _chain(split_rhat: tuple[float, float]) -> api.McmcReport:
    return api.McmcReport(
        config=api.McmcConfig(walkers=4, burn_in=0, production_steps=4),
        child_seed=7,
        parameter_names=NAMES,
        samples_physical=np.array([[10.0, 3.0], [11.0, 3.2], [12.0, 2.9], [13.0, 3.1]]),
        log_probability=np.zeros(4),
        acceptance_fraction=np.full(4, 0.4),
        split_rhat=np.asarray(split_rhat),
        effective_sample_size=np.full(2, 400.0),
        boundary_hits=(),
        candidate_id="candidate-a",
    )


def _posterior(qtbot, chain: api.McmcReport) -> McmcHistogramPlot:
    result = replace(_two_parameter_result(), uncertainty=replace(_uncertainty(), mcmc=chain))
    plot = McmcHistogramPlot()
    qtbot.addWidget(plot)
    plot.show_report(result, "candidate-a")
    return plot


def _page_text(plot: McmcHistogramPlot) -> str:
    """页上写着的每一行字：每格的标题与横轴名、轴里的注记、整张图的抬头。"""
    parts = [text.get_text() for text in plot.figure.texts]
    for axes in plot.figure.axes:
        parts += [axes.get_title(), axes.get_xlabel(), *(text.get_text() for text in axes.texts)]
    return "\n".join(parts)


def test_an_unmixed_chain_gets_no_quantiles_and_says_why(qtbot) -> None:
    chain = _chain((1.002, 41.13))
    reason = unmixed_chain_reason(chain)
    assert reason is not None, "前提：这条链没混合"

    plot = _posterior(qtbot, chain)
    text = _page_text(plot)

    assert not any(mark in text for mark in QUANTILE_MARKS)
    assert not any(axes.lines for axes in plot.figure.axes), "分位线不写数也是在报分位"
    assert reason in text


def test_a_mixed_chain_still_reads_out_its_quantiles(qtbot) -> None:
    chain = _chain((1.002, 1.004))
    assert unmixed_chain_reason(chain) is None, "前提：这条链混合了"

    plot = _posterior(qtbot, chain)
    panels = [axes for axes in plot.figure.axes if axes.patches]

    assert len(panels) == len(NAMES)
    for axes in panels:
        assert all(mark in axes.get_xlabel() for mark in QUANTILE_MARKS)
        assert len(axes.lines) == len(QUANTILE_MARKS)
