"""切一次视图只报一次；面板释放之后，留下来的 tab 与工具栏再动也碰不到它。

审计实测：``select_view`` 自己发一次 ``view_changed``，``setCurrentIndex`` 换页时 tab 的回调又发一次，主窗口据此
把工作区写两遍；而页没换时只发一次——同一个动作报几次，取决于读者原先停在哪一页。``release_resources``
之后控制器仍挂在 tab 与工具栏的信号上：换一下 tab，就去已经释放的面板上找 ``tab_keys``。
"""

from __future__ import annotations

from tests.gui.plot_support import _panel
from tests.support.model_cases import prepared_data


def _record(signal) -> list[int]:
    seen: list[int] = []
    signal.connect(seen.append)
    return seen


def test_select_view_announces_the_view_exactly_once_whether_or_not_the_page_changes(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    seen = _record(panel.view_changed)

    panel.select_view("raw")
    assert len(seen) == 1
    panel.select_view("raw")
    assert len(seen) == 2
    panel.select_view("uncertainty")
    assert len(seen) == 3


def test_a_tab_click_still_announces_the_view_once(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    seen = _record(panel.view_changed)

    panel.reflectivity_tabs.setCurrentIndex(1)

    assert len(seen) == 1


def test_tabs_and_toolbar_left_behind_by_a_released_panel_reach_nothing(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    seen = _record(panel.view_changed)
    log = panel.view("log")
    tabs, analysis, toolbar = panel.reflectivity_tabs, panel.analysis_tabs, panel.toolbar

    panel.release_resources()
    tabs.setCurrentIndex(1)
    analysis.setCurrentIndex(2)
    toolbar.set_mode("range")

    assert seen == []
    assert log.range_item is None
