"""设计稿的 ``.badge``：一句只读的状态，画成带边框的胶囊。

普通一行 ``先验：无（均匀）`` 和它周围的说明文字长得一样，读者得逐行读过去才知道哪句
是「这个量此刻的状态」。设计稿给这类句子一圈边框和一个颜色，让它在一屏里先被认出来是
状态、再被读内容。

配色规则 ``QLabel[badge="true"]`` 早就在主题里了（``results/verdict.py`` 在用），这里
只是把「加属性、换类别要重刷、没话说就不占地方」这套零件收成一个控件，省得每处再抄一遍。
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")


def test_a_badge_says_its_text_and_carries_the_styling_property(qtbot) -> None:
    """文字和类别是两件事：类别只决定配色，读出来的永远是那句话。

    ``badge`` 这个属性就是主题里那条胶囊规则的选择器；漏掉它，这行字会退回成普通正文。
    """
    from xrr_fitter.gui.badge import Badge

    badge = Badge("priorBadge")
    qtbot.addWidget(badge)

    badge.setText("先验：无（均匀）")

    assert badge.text() == "先验：无（均匀）"
    assert badge.objectName() == "priorBadge"
    assert badge.property("badge") is True


def test_the_neutral_kind_is_the_absence_of_a_status_colour(qtbot) -> None:
    """设计稿的 ``.badge.mut`` 就是主题里那条不带 ``statusKind`` 的基础规则。

    「未共享」这类结论不能借用四种状态色里的任何一种，否则「这项没设」会被读成
    「这项通过」或「这项有问题」。
    """
    from xrr_fitter.gui.badge import Badge

    badge = Badge("sharingBadge", kind="mut")
    qtbot.addWidget(badge)

    assert badge.kind() == "mut"
    assert badge.property("statusKind") == ""


def test_changing_the_kind_restyles_without_touching_the_text(qtbot) -> None:
    """同一句话在不同状态下换颜色，句子本身不该被改写。

    动态属性改了之后 Qt 不会自动重算样式，得走 ``theme.set_status_kind`` 那条重刷；
    不重刷的话属性变了、颜色没变，屏幕上说的是上一个状态。
    """
    from xrr_fitter.gui.badge import Badge

    badge = Badge("sharingBadge", kind="mut")
    qtbot.addWidget(badge)
    badge.setText("共享：跨 3 集共享厚度")

    badge.set_kind("info")

    assert badge.kind() == "info"
    assert badge.property("statusKind") == "info"
    assert badge.text() == "共享：跨 3 集共享厚度"


def test_an_unknown_kind_is_refused_rather_than_silently_unstyled(qtbot) -> None:
    """拼错的类别当场报错，而不是安静地退回中性色。

    退回中性色的话，一个本该是警告的结论会长得和「无」一模一样。
    """
    from xrr_fitter.gui.badge import Badge

    badge = Badge("priorBadge")
    qtbot.addWidget(badge)

    with pytest.raises(ValueError):
        badge.set_kind("muted")


def test_a_badge_with_nothing_to_say_takes_up_no_room(qtbot) -> None:
    """没有状态可报时整枚消失，而不是留一个空胶囊。

    空边框会被读成「有个状态，但没加载出来」——那正是它此刻不能表示的意思。
    """
    from xrr_fitter.gui.badge import Badge

    badge = Badge("priorBadge")
    qtbot.addWidget(badge)
    badge.setText("先验：正态")
    assert not badge.isHidden()

    badge.setText("")
    assert badge.isHidden()

    badge.setText("先验：正态")
    assert not badge.isHidden()


def test_the_badge_is_a_pill_not_a_rectangle(qtbot) -> None:
    """``border-radius:999px`` 配固定高度才是胶囊；高度跟着字体浮动会成圆角矩形。

    这枚东西和相邻的普通文字并排，高度自己长一截就会把那一行整体撑高。
    """
    from xrr_fitter.gui.badge import Badge

    badge = Badge("priorBadge")
    qtbot.addWidget(badge)
    badge.setText("先验：无（均匀）")

    assert badge.minimumHeight() == Badge.HEIGHT_PX
    assert badge.maximumHeight() == Badge.HEIGHT_PX
