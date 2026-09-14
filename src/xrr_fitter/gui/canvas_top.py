"""设计稿 ``.canvas-top``：画布列顶上那一行——几个 tab，紧跟着这一页的动作。

设计稿四帧画布都以同一行开头（帧① 反射率四视图 + 模式条、帧③ 样品结构三视图 + 三个结构
命令、帧④ 进度三视图 + 刷新徽标、帧⑤ 不确定度四视图），所以它是构件而不是某一帧的装饰。

两条骨架规则决定了这里不能直接用 ``QTabWidget.setCornerWidget``：

* 动作紧跟 tab，不贴右边框。设计稿的 ``.spring`` 只在 ``.cmdbar`` 与 ``.statusbar`` 里带
  ``flex:1``（HTML 90 与 216 行），``.canvas-top``（148 行）里它是零宽的——右侧留白落在
  行尾，而不是插在 tab 和动作之间。``setCornerWidget`` 只会把动作贴到最右。
* tab 与动作共用一段高度。帧③此前是八个按钮排成两行外加一条氧化层建议条，光行头就吃掉
  画布三段高度中的一段。
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QTabBar, QTabWidget, QVBoxLayout, QWidget

from xrr_fitter.gui import theme


def canvas_tab_bar(parent: QWidget | None, name: str, titles: Iterable[str]) -> QTabBar:
    """设计稿 ``.tabs``：一排贴着画布上沿的 tab，选中那个戴 2px 强调色下划线。

    ``setDrawBase(False)``：QTabBar 自带的基线会和这一行的下边框叠成两道，画布上沿凭空
    粗一档。``setExpanding(False)``：默认会把 tab 撑满整条宽度，那正是设计稿不要的
    「动作被推到另一头」。
    """
    tabs = QTabBar(parent)
    tabs.setObjectName(name)
    tabs.setProperty("canvasTab", True)
    tabs.setDrawBase(False)
    tabs.setExpanding(False)
    tabs.setUsesScrollButtons(False)
    tabs.setElideMode(Qt.TextElideMode.ElideRight)
    for title in titles:
        tabs.addTab(title)
    return tabs


def canvas_top(
    parent: QWidget | None = None,
    *,
    name: str = "",
    tabs_name: str = "",
    titles: Iterable[str] = (),
) -> tuple[QWidget, QTabBar, QHBoxLayout]:
    """Build one ``.canvas-top``: tabs, then this page's actions, then the slack.

    Returns the row, its tab bar, and the layout callers append action widgets to.
    """
    row = QWidget(parent)
    row.setObjectName(name)
    row.setProperty("canvasTop", True)
    # QSS 要给 QWidget 子类画下边框，得先声明它有样式化背景，否则规则会被静默丢掉。
    row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    layout = QHBoxLayout(row)
    layout.setContentsMargins(theme.SPACE_MD, theme.SPACE_SM, theme.SPACE_MD, theme.SPACE_SM)
    layout.setSpacing(theme.SPACE_XS)
    tabs = canvas_tab_bar(row, tabs_name, titles)
    layout.addWidget(tabs)

    actions = QHBoxLayout()
    actions.setContentsMargins(0, 0, 0, 0)
    actions.setSpacing(theme.SPACE_XS)
    layout.addLayout(actions)
    layout.addStretch(1)
    return row, tabs, actions


def canvas_tab_group(
    pages: QTabWidget,
    *,
    parent: QWidget | None = None,
    name: str = "",
    row_name: str = "",
    tabs_name: str = "",
) -> tuple[QWidget, QTabBar, QHBoxLayout]:
    """给一个现成的 ``QTabWidget`` 换上设计稿那条行头：扁 tab 条在上，页面在下。

    ``QTabWidget`` 自带的 tab 是一只带边框的小盒子，且它唯一的「行内附加控件」入口
    ``setCornerWidget`` 只有贴右边框这一种摆法——两条都和设计稿相反。所以容器退回去只当
    页面壳（自己那条 tab 条藏起来），露在外面的是 ``canvas_top`` 那一行。

    两个方向都接：条上换页要真的翻页，而换视图的代码（菜单「视图」、Alt+N、恢复工作区）
    走的是容器，条得跟着亮。``setCurrentIndex`` 在序号没变时不发信号，所以两边互接不会
    递归。
    """
    group = QWidget(parent)
    group.setObjectName(name)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    titles = tuple(pages.tabText(index) for index in range(pages.count()))
    row, tabs, actions = canvas_top(group, name=row_name, tabs_name=tabs_name, titles=titles)
    layout.addWidget(row)
    pages.tabBar().hide()
    layout.addWidget(pages, 1)

    tabs.setCurrentIndex(pages.currentIndex())
    tabs.currentChanged.connect(pages.setCurrentIndex)
    pages.currentChanged.connect(tabs.setCurrentIndex)
    return group, tabs, actions
