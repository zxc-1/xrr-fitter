"""拟合卡与结果检视器的文字只说中文，机器串不上屏。

审计实测：进度卡详情行「C 3/10 · DE generation (seed 1) · best=0.0125」把阶段键、worker 的英文
消息和 ``best=`` 原样摆出来；walkers 不够时报「walkers must be even and at least 2*nfree+2」，
不说到底要几个；MCMC 结果被判过期时报「stale MCMC result rejected…」；无量纲参数的单位写成
「core unit」；结果面板的几道守卫报的也是英文。
"""

from __future__ import annotations

import pytest
from tests.gui.test_results import _panel, _project_with_result, _two_candidate_result

import xrr_fitter.api as api


def _english_words(text: str) -> list[str]:
    """Latin words of three or more letters; units and symbols such as nm, Å, J pass."""
    import re

    return re.findall(r"[A-Za-z]{3,}", text)


def test_the_progress_detail_line_carries_no_worker_english(qtbot) -> None:
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_progress(api.FitProgress("curve", "E", 2, 4, 0.0125, "DE generation (seed 2)", iteration=7))

    detail = view.detail_label.text()
    assert _english_words(detail) == []
    assert "迭代 7" in detail
    assert "0.0125" in detail
    view.freeze()


@pytest.mark.parametrize("walkers", (3, 10))
def test_too_few_walkers_says_how_many_are_needed(qtbot, walkers) -> None:
    """五个自由参数至少要 12 条链，而且是偶数：报错要把这个数说出来。"""
    from xrr_fitter.gui.results.uncertainty import McmcControls

    controls = McmcControls()
    qtbot.addWidget(controls)
    controls.walkers.setValue(walkers)

    with pytest.raises(ValueError) as caught:
        controls.validated_config(5)

    message = str(caught.value)
    assert "12" in message and "偶数" in message
    assert "nfree" not in message and "must" not in message


def test_an_invalid_step_setting_is_explained_in_chinese(qtbot) -> None:
    from xrr_fitter.gui.results.uncertainty import McmcControls

    controls = McmcControls()
    qtbot.addWidget(controls)
    controls.walkers.setValue(12)
    controls.production.setValue(2)

    with pytest.raises(ValueError) as caught:
        controls.validated_config(5)

    assert _english_words(str(caught.value)) == []


def test_a_dimensionless_interval_has_no_made_up_unit() -> None:
    from xrr_fitter.gui.results.uncertainty import _interval_text

    assert _interval_text(("component.0.density_scale", 0.9, 1.1)) == "component.0.density_scale [0.9, 1.1]"


def test_the_mcmc_guards_speak_chinese(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))

    with pytest.raises(ValueError, match="高级模式"):
        panel.start_mcmc()

    panel.controller._job = object()
    try:
        with pytest.raises(RuntimeError) as caught:
            panel.clear_results()
    finally:
        panel.controller._job = None
    assert "MCMC" in str(caught.value)
    assert _english_words(str(caught.value).replace("MCMC", "")) == []
