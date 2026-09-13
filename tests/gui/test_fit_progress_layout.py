"""帧④「总进度」卡在中栏那点高度里的排版契约。

这里问的不是「显示了什么」（那在 `test_fit_progress.py`），而是「显示出来的还读得全吗」：
卡里装的横幅、四行文字和九行阶梯加起来比中栏高，Qt 在没有滚动容器时会把每一件都压到它
自己要的高度以下，连 ``minimumSizeHint`` 都不保。被压掉的像素落在哪一句话上，屏上就永久
少半句——所以这几条按中栏的真实尺寸立视图，并套上应用发货的那份样式表来量。
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea

import xrr_fitter.api as api

# 中栏在 1400×900 给这只视图的宽高，以及窗口窄一档（1100×760）时的那一份。两档都要成立：
# 「够宽时装得下」不算通过，屏上真实的高度就是装不下。
COLUMN_SIZES = ((794, 400), (640, 300))


def _joint_layout(*names: str):
    """一份把 ``names`` 里所有数据集绑在一起共享同一个厚度的联合布局。"""
    return api.JointFitLayout(
        tuple(names),
        (
            api.SharingRule(
                "aSi_thickness",
                tuple(api.ParameterReference(name, "component.0.thickness_a") for name in names),
            ),
        ),
    )


def _joint_view(qtbot, width: int, height: int):
    """按中栏的真实尺寸立起帧④ 的进度视图，并套上应用发货的那份样式表。

    ``fixture`` 不走 :func:`theme.apply_theme`，而横幅那一圈 ``padding: 10px 14px`` 只写在样式表
    里；不套样式表量到的就不是应用实际发出去的那只框，它要多高也跟着不对。
    """
    from xrr_fitter.gui import theme
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    view.setStyleSheet(theme.build_stylesheet(view.palette()))
    view.set_joint_layout(_joint_layout("curve-0", "curve-1", "curve-2"))
    view.resize(width, height)
    view.show()
    qtbot.waitExposed(view)
    return view


def _cropped(label: QLabel) -> int:
    """这只折行标签比它排完版所需的高度少了多少像素。"""
    return label.heightForWidth(label.width()) - label.height()


@pytest.mark.parametrize(("width", "height"), COLUMN_SIZES)
def test_the_joint_banner_says_its_whole_sentence_on_screen(qtbot, width: int, height: int) -> None:
    """联合横幅那句话必须整句留在屏上，不能被卡片的高度预算裁掉半句。

    这句话管着同屏其它每一个数字怎么读：读者以为在看一条曲线的拟合时，进度条上那
    ``620 / 1000`` 其实是三条一起跑出来的，右边解出来的厚度也是三条共用的一个。折行后要
    两行，卡片只给一行时先被裁掉的正是「共享参数：…」那半句——剩下的半句读起来像是在说
    「联合拟合，数据集三个」，而共享了什么没说。
    """
    view = _joint_view(qtbot, width, height)
    banner = view.joint_label

    assert banner.isHidden() is False
    assert _cropped(banner) <= 4, f"横幅被裁 {_cropped(banner)}px：{banner.text()!r}"


def test_the_progress_card_scrolls_when_it_outgrows_the_column(qtbot) -> None:
    """卡片比中栏高时靠滚动去够，而不是把卡里每一件按比例压扁。

    这张卡装的是横幅、四行文字和九行阶梯，加起来比 1400×900 的中栏还高。没有滚动容器时
    Qt 只能压——而且连 ``minimumSizeHint`` 都不保（横幅实测被压到 13px），于是矮下去的不是
    最不重要的那一件，是所有件。有了滚动容器，放不下的部分是「要滚一下才看见」，不再是
    「屏上永远读不到」。
    """
    view = _joint_view(qtbot, *COLUMN_SIZES[0])
    scroll = view.findChild(QScrollArea, "fitProgressScroll")

    assert scroll is not None, "「总进度」卡没有放进可滚容器"
    card = scroll.widget()
    assert card is not None and card.objectName() == "fitProgressCard"
    assert card.height() >= card.minimumSizeHint().height(), "卡片仍被压在它的最小高度以下"
    assert scroll.verticalScrollBar().maximum() > 0, "卡片高过视口，却滚不动"
    # 横向不滚：一栏的宽度是给定的预算，卡片该在这个宽度里排版，而不是靠左右拖动去读。
    assert scroll.horizontalScrollBarPolicy() is Qt.ScrollBarPolicy.ScrollBarAlwaysOff


@pytest.mark.parametrize(("width", "height"), COLUMN_SIZES)
def test_nothing_else_on_the_progress_card_pays_for_the_banner(qtbot, width: int, height: int) -> None:
    """横幅要回的那几像素不能从同卡别人身上扣。

    卡里 ``fitProgressDetail`` 也会折行，写的是当前阶段正在做什么。只把横幅一处钉住，差额
    会被摊到它和九行阶梯上——换来的是另一句读不全的话，而不是一屏读得全的卡。
    """
    view = _joint_view(qtbot, width, height)
    card = view.findChild(QFrame, "fitProgressCard")
    assert card is not None

    cropped = {
        label.objectName() or label.text(): _cropped(label)
        for label in card.findChildren(QLabel)
        if label.wordWrap() and label.isVisibleTo(card) and _cropped(label) > 4
    }
    assert cropped == {}
