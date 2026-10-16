"""The design's four-glyph plot mode bar and its right-click menu.

设计稿 ``.modebar`` 那一条：平移 / 框选放大 / 复位三枚导航字形加「框出拟合范围」，其余模式与一次性
命令收进条自己的右键菜单。条只管「摆了哪几枚、勾着哪一档」；模式点下去之后图上发生什么归
:mod:`xrr_fitter.gui.plots.interactions` 的控制器，两边分开放，复杂度门禁才能各自衡量。
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QHBoxLayout, QToolButton, QWidget

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.plot_icons import plot_icon

MODE_SPECS = (
    ("view", "plotModeView", "查看", "查看和缩放诊断图"),
    ("range", "plotModeRange", "范围", "选择拟合角度范围"),
    ("mask", "plotModeMask", "掩膜", "切换单个预处理点的掩膜"),
)

# 三档模式里只有「范围」在设计稿的条上留了字形（``▭``）；查看与掩膜从条的右键菜单进。
# 决定谁上条的是这一行，不是 ``MODE_SPECS`` 的顺序。
BAR_MODES = ("range",)

# Deliberately not prefixed "plotMode": these are not members of the exclusive
# mode group -- they answer "how do I move around this plot", not "what does
# clicking it do" -- and the panel tests read the "plotMode" prefix as "the mode
# glyph the design kept on the bar".
NAVIGATION_SPECS = (
    ("pan", "plotNavPan", "平移", "按住左键拖动图像，按住右键拖动缩放"),
    ("zoom", "plotNavZoom", "框选放大", "拖出一个矩形，放大到该区域"),
    ("home", "plotNavHome", "复位", "恢复当前图刚绘制时的坐标范围"),
)

# 设计稿 ``.modebar`` 容器的 title 把条上那四枚念了一遍，所以这一句既是条的说明，也是
# 「这一条摆了哪四枚」的凭据。
MODEBAR_TOOLTIP = "平移 / 缩放 / 复位 / 选择拟合范围"

# The glyph painters lay their strokes out on a 16px grid, so drawing at 16
# applies a scale of exactly 1 and every stroke lands on a whole pixel; any
# other size resamples them and softens the edges.
GLYPH_PX = 16


def _wear_glyph(button: QToolButton, glyph: str) -> None:
    """Wear the painted glyph alone, with the name kept for hover and screen readers.

    These controls share the plot's tab row, the way every peer charting tool puts
    its mode bar on the chart's own header, and that row has to hold four view
    tabs beside them.  Labels are what made the bar wide, so the glyph carries the
    meaning and the words move to the tooltip and the accessible name.
    """
    button.setIcon(plot_icon(glyph, size=GLYPH_PX))
    _wear_bar_chrome(button)


def _wear_bar_chrome(button: QToolButton) -> None:
    """条上一枚字形的穿法，图标由谁给不管。

    ``setDefaultAction`` 会替按钮抄来 action 的图标，但抄不走 iconSize、按钮样式、
    autoRaise 和光标，所以这四样单独留在按钮上。
    """
    button.setIconSize(QSize(GLYPH_PX, GLYPH_PX))
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setAutoRaise(True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)


class PlotInteractionToolbar(QWidget):
    """Own one exclusive, programmatically validated plot mode.

    设计稿的 ``.modebar`` 是四枚字形——``✥ ⤢ ⌂ ▭``，平移 / 框选放大 / 复位，加上
    「框出拟合范围」。此前这里排了九枚，和四个 tab 抢同一行。收窄不是砍功能：查看 ·
    掩膜 · 缩放到拟合范围 · 恢复完整视图 · 叠加对比 变成这一条自己的右键菜单（同一批
    ``QAction`` 也挂进 视图 菜单），落点选在条上而不是图里，因为图 body 的右键归
    pyqtgraph 的 ViewBox——那里有它自己的菜单和右键拖动缩放。
    """

    mode_changed = Signal(str)
    zoom_to_range_requested = Signal()
    reset_zoom_requested = Signal()
    navigation_requested = Signal(str)
    overlay_toggled = Signal(bool)
    sld_profile_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotInteractionToolbar")
        self.setAccessibleName("绘图交互模式")
        self.setToolTip(MODEBAR_TOOLTIP)
        # The theme gives this bar the design's grouping pill -- a background and a
        # 1px border -- but Qt only auto-styles the background of a plain QWidget;
        # measured on the built window, the border stroke was absent (row 0 and
        # row 1 both alpha 12) until this attribute made the widget paint its own
        # background, after which they read 50 against 23.  Without it the four
        # glyphs sit loose on the tab row instead of inside one pill.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # 条自己就是那五条搬走的命令的落点。
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        # 先把模式记成 view，再去勾那条 QAction：勾选会走 ``_mode_triggered``，而它要读
        # ``self._mode``——次序反了就会在构造中途发一次 mode_changed，那时谁都还没接线。
        self._mode = "view"
        self._mode_actions: dict[str, QAction] = {}
        self._buttons: dict[str, QToolButton] = {}
        self._group = QActionGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        # Padding of its own, because the bar sits in the tab row's corner slot
        # rather than in a body row that already carries the panel's margins.
        layout.setContentsMargins(theme.SPACE_XS, theme.SPACE_XS, theme.SPACE_XS, theme.SPACE_XS)
        layout.setSpacing(2)
        self._install_mode_actions()
        # 设计稿里这四枚紧挨着，一个 pill 里不再分组——所以没有组间空隙。也没有尾部
        # stretch：条与视图 tab 同处一行，stretch 会让它吃掉 tab 剩下的每一个像素。
        self._install_navigation_buttons(layout)
        self._install_mode_buttons(layout)
        self._install_zoom_actions()
        self._mode_actions["view"].setChecked(True)

    def _install_mode_actions(self) -> None:
        """三档交互模式做成一组互斥 ``QAction``，而不是一组按钮。

        状态存在 action 上，条上的 ``▭`` 与菜单里的「范围」才不会各记一份勾选，出现
        「条上亮着、菜单里没勾」这种自相矛盾的画面。
        """
        for mode, _name, text, description in MODE_SPECS:
            action = QAction(text, self)
            action.setObjectName(f"plotToolAction:{mode}")
            action.setCheckable(True)
            action.setToolTip(f"{text}：{description}")
            action.setIcon(plot_icon(mode, size=GLYPH_PX))
            action.triggered.connect(lambda _checked=False, key=mode: self._mode_triggered(key))
            self._group.addAction(action)
            self.addAction(action)
            self._mode_actions[mode] = action

    def _install_mode_buttons(self, layout: QHBoxLayout) -> None:
        """把留在条上的那一档摆成一枚字形（设计稿里是 ``▭``）。

        用 ``setDefaultAction`` 而不是另建一颗独立按钮：勾选、可用性、字形都由 action
        一处决定，两个入口天然同步。``accessibleName`` 不在 Qt 的同步清单里，得在按钮
        上单独说一遍。
        """
        for mode, name, text, _description in MODE_SPECS:
            if mode not in BAR_MODES:
                continue
            button = QToolButton(self)
            button.setObjectName(name)
            button.setDefaultAction(self._mode_actions[mode])
            button.setAccessibleName(text)
            button.setProperty("plotMode", mode)
            _wear_bar_chrome(button)
            self._buttons[mode] = button
            layout.addWidget(button)

    def _install_navigation_buttons(self, layout: QHBoxLayout) -> None:
        """Add the pan, box-zoom and reset controls onto the same row.

        Pan and box zoom latch, so they are checkable, but they are not part of
        the exclusive mode group: they answer "how do I move around this plot",
        not "what does clicking it do", and the panel makes the two families
        mutually exclusive through the canvas widget lock instead.
        """
        self._navigation: dict[str, QToolButton] = {}
        for action, name, text, description in NAVIGATION_SPECS:
            button = QToolButton(self)
            button.setObjectName(name)
            button.setCheckable(action != "home")
            button.setAccessibleName(text)
            button.setToolTip(f"{text}：{description}")
            _wear_glyph(button, action)
            button.setProperty("plotNavigation", action)
            button.clicked.connect(self._navigation_clicked)
            self._navigation[action] = button
            layout.addWidget(button)

    def _navigation_clicked(self) -> None:
        self.navigation_requested.emit(str(self.sender().property("plotNavigation")))

    def navigation_buttons(self) -> dict[str, QToolButton]:
        return dict(self._navigation)

    def show_navigation_mode(self, mode: str) -> None:
        """Mirror the canvas's navigation state onto the latching buttons.

        The state lives on the navigator, not on the buttons: pan also ends when
        a range drag takes the widget lock, or when the user switches to a view
        whose navigator is idle.  Reflecting it back keeps a button from staying
        pressed over a canvas that is no longer panning.
        """
        for action, button in self._navigation.items():
            if button.isCheckable():
                button.setChecked(action == mode)

    def _menu_action(self, key: str, text: str, tooltip: str, glyph: str) -> QAction:
        """一条只在菜单里露面的命令：字形留着，因为 视图 菜单也认它。

        ``plotToolAction:`` 这个前缀和 视图 菜单里视图项的 ``plotViewAction:`` 是同一套
        约定——菜单栏那份「这一栏只该有这几条」的断言靠前缀把借来的命令认出来。
        """
        action = QAction(text, self)
        action.setObjectName(f"plotToolAction:{key}")
        action.setToolTip(tooltip)
        action.setIcon(plot_icon(glyph, size=GLYPH_PX))
        self.addAction(action)
        return action

    def _install_zoom_actions(self) -> None:
        """Add non-exclusive zoom actions distinct from the mode action group.

        Zooming to the fit range is a one-shot action, not a persistent mode,
        so these stay outside the exclusive group; pressing one must not clear
        the active view/range/mask mode the user is working in.  它们在菜单里排在
        三档模式后面，用分隔线隔开——一次性动作和「点下去会一直是这样」的模式不是一类。
        """
        separator = QAction(self)
        separator.setSeparator(True)
        self.addAction(separator)
        self.zoom_to_range_action = self._menu_action(
            "zoom_to_range", "缩放到拟合范围", "缩放拟合区：将反射率视图缩放到当前拟合角度范围", "zoom_to_range"
        )
        self.zoom_to_range_action.triggered.connect(lambda _checked=False: self.zoom_to_range_requested.emit())
        self.reset_zoom_action = self._menu_action(
            "reset_zoom", "恢复完整视图", "全览：恢复到完整角度范围", "reset_zoom"
        )
        self.reset_zoom_action.triggered.connect(lambda _checked=False: self.reset_zoom_requested.emit())
        trailing = QAction(self)
        trailing.setSeparator(True)
        self.addAction(trailing)
        self.overlay_action = self._menu_action(
            "overlay", "叠加对比", "叠加对比：显示所有数据集的反射率曲线", "overlay"
        )
        self.overlay_action.setCheckable(True)
        self.overlay_action.toggled.connect(self.overlay_toggled.emit)
        # 结果步的画布是设计稿帧① 的两张卡，拟合出来的 SLD 剖面不在上面；这一条把它加回画布。
        # 只在结果那两步可用：结构与参数两步画布上本来就是剖面（画的是正在编辑的标称结构）。
        self.sld_profile_action = self._menu_action(
            "sld_profile",
            "显示拟合 SLD 剖面",
            "显示拟合 SLD 剖面：在结果页的反射率与残差下方加一张拟合候选的 SLD 深度剖面（仅结果步可用）",
            "sld_profile",
        )
        self.sld_profile_action.setCheckable(True)
        self.sld_profile_action.setEnabled(False)
        self.sld_profile_action.toggled.connect(self.sld_profile_toggled.emit)

    def tool_actions(self) -> tuple[QAction, ...]:
        """条上这一份右键菜单，按菜单里的先后顺序。"""
        return tuple(self.actions())

    def mode_actions(self) -> dict[str, QAction]:
        return dict(self._mode_actions)

    def buttons(self) -> dict[str, QToolButton]:
        """留在条上的模式字形——设计稿收窄之后只剩 ``▭``。"""
        return dict(self._buttons)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in self._mode_actions:
            raise ValueError(f"unsupported plot interaction mode: {mode}")
        if mode == self._mode:
            return
        # ``setChecked`` 不发 triggered，所以这里不会绕回 ``_mode_triggered`` 再发一次。
        self._mode_actions[mode].setChecked(True)
        self._mode = mode
        self.mode_changed.emit(mode)

    def _mode_triggered(self, mode: str) -> None:
        # 互斥组里那条勾着的 action 点不掉（Qt6 直接拒绝取消勾选），所以点重复的一档就是
        # 什么都没变——早退，别发一次空的 mode_changed。
        if mode == self._mode:
            return
        self._mode = mode
        self.mode_changed.emit(mode)
