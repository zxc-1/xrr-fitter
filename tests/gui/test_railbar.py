"""设计稿的 ``.railbar``：一个数字旁边写出它被允许落在哪一段里。

输入框只说了「现在是 48.7」，没说这个数能走多远。拟合器动的正是这个数，而它的活动
范围是声明里写死的——读者要判断「48.7 是不是快贴边了」，就得离开这张卡去参数表里翻
那一行的上下界。设计稿把这段话直接印在框旁边：``40 ≤ 48.7 ≤ 60``，并用填充长度画出
它落在哪儿。

这一条量的是那根条：文本怎么写、填充算多长、没有界限时它闭嘴。
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")


def _rail(qtbot):
    from xrr_fitter.gui.railbar import RailBar

    rail = RailBar("probeRail")
    qtbot.addWidget(rail)
    return rail


def test_the_rail_writes_the_span_the_way_the_design_writes_it(qtbot) -> None:
    """``40 ≤ 48.7 ≤ 60``：下界、当前值、上界，中间隔着数学上的「不大于」。

    写成 ``范围 40–60`` 会漏掉最要紧的一件事——当前值坐在哪儿。三个数排成一行，读者
    一眼就知道 48.7 离哪一端近。
    """
    rail = _rail(qtbot)

    rail.set_span(40.0, 48.7, 60.0)

    assert rail.text() == "40 ≤ 48.7 ≤ 60"
    assert rail.fraction() == pytest.approx(0.435)


def test_trailing_zeros_are_dropped_so_the_seventy_pixel_rail_still_fits_three_numbers(qtbot) -> None:
    """``2 ≤ 2.28 ≤ 2.5``，不是 ``2.00 ≤ 2.28 ≤ 2.50``。

    这根条在设计稿里只有 70px 的下限宽度，三个数加两个符号全挤在里面。固定两位小数
    会让一半的字符是没有信息的零，而先被挤掉的恰好是右端的上界。
    """
    rail = _rail(qtbot)

    rail.set_span(2.0, 2.28, 2.5)

    assert rail.text() == "2 ≤ 2.28 ≤ 2.5"


def test_a_value_outside_its_declared_bounds_stays_inside_the_rail(qtbot) -> None:
    """初值可以落在界限外，填充却不能画到框外。

    导入的项目、手敲进输入框的数字都可能越界，而越界正是这根条最该说清楚的时刻。
    填充钳在两端，文本照实写出那个越界的数——画面不失真，读数不撒谎。
    """
    rail = _rail(qtbot)

    rail.set_span(40.0, 80.0, 60.0)

    assert rail.text() == "40 ≤ 80 ≤ 60"
    assert rail.fraction() == pytest.approx(1.0)

    rail.set_span(40.0, 10.0, 60.0)

    assert rail.fraction() == pytest.approx(0.0)


def test_a_span_with_no_width_does_not_divide_by_zero(qtbot) -> None:
    """上下界重合的量（钉死的那种）没有「落在哪儿」可言，填充给满。

    这一步不判断该不该有这种声明，只保证画它的时候不会拿 0 去除。
    """
    rail = _rail(qtbot)

    rail.set_span(2.33, 2.33, 2.33)

    assert rail.fraction() == pytest.approx(1.0)
    assert rail.text() == "2.33 ≤ 2.33 ≤ 2.33"


def test_a_rail_with_nothing_to_report_says_nothing(qtbot) -> None:
    """没有选中层、或这个量根本没有声明时，条是空的。

    留着上一层的 ``40 ≤ 48.7 ≤ 60`` 会读成「当前这一层的厚度界限」，而这正是它此刻
    唯一不能表示的意思。
    """
    rail = _rail(qtbot)
    rail.set_span(40.0, 48.7, 60.0)

    rail.clear()

    assert rail.text() == ""
    assert rail.fraction() is None


def test_the_rail_keeps_the_design_s_height_and_minimum_width(qtbot) -> None:
    """``height:16px;min-width:70px``：它嵌在一行里，不能跟着输入框长高。

    条和 96px 的输入框并排站在同一行上，高度一旦跟着字体走，这一行就会比相邻两行
    高出一截，四个字段的基线全部错开。
    """
    from xrr_fitter.gui.railbar import RailBar

    rail = _rail(qtbot)

    assert rail.height() == RailBar.HEIGHT_PX
    assert rail.minimumWidth() == RailBar.MINIMUM_WIDTH_PX
