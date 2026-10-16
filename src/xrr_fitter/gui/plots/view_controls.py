"""View addressing and interaction entry points of the plot panel.

读者换视图的几条路——tab、Alt+N、菜单「视图」——以及窗口与菜单借面板去驱动模式条、交互控制器的
那几个入口，都只是把请求转给 toolbar 或 ``PlotInteractionController``。它们从
:mod:`xrr_fitter.gui.plots.panel` 分出来，面板模块只留投影与绘制事务，复杂度门禁才能各自衡量。

:class:`ViewControlsMixin` 混入 ``PlotPanel``：运行时 ``self`` 就是面板。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

from xrr_fitter.gui.plots.diagnostics import ANALYSIS_KEYS, REFLECTIVITY_KEYS, TAB_SPECS, VIEW_SPECS


class ViewControlsMixin:
    """Address views by key or visible position and forward mode requests."""

    #
    # Install view shortcuts
    #
    def _install_view_shortcuts(self) -> None:
        """Bind one Alt+N per diagnostic tab, addressed by visible position.

        Users switch among the diagnostic plots constantly; clicking or cycling
        with Ctrl+Tab is slow.  The keys count the tabs, not the views: the SLD
        profile is a companion pane that never leaves the screen, so it is not
        addressable and consumes no number.  The weighted residual counts,
        because it is both -- a tab of its own *and* the card pinned under the
        strip; the one card moves between the two homes.
        Numbering by visible position rather than by fixed view key means a
        hidden tab shifts the keys up instead of leaving a key that lands on
        nothing and a gap in the sequence.
        """
        self.view_shortcuts: list[QShortcut] = []
        for position in range(len(self.tab_keys())):
            shortcut = QShortcut(QKeySequence(f"Alt+{position + 1}"), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setProperty("viewPosition", position)
            # Bind the position through the sender rather than a lambda closing
            # over self; a self-capturing closure held by the shortcut's signal
            # forms a cycle PySide cannot break, leaking the panel on teardown.
            shortcut.activated.connect(self._view_shortcut_activated)
            self.view_shortcuts.append(shortcut)

    #
    # View shortcut activated
    #
    def _view_shortcut_activated(self) -> None:
        shortcut = self.sender()
        if shortcut is not None:
            self.select_visible_view(int(shortcut.property("viewPosition")))

    #
    # Select visible view
    #
    def select_visible_view(self, position: int) -> bool:
        """Select the Nth (0-based) currently-visible diagnostic view."""
        visible: list[str] = []
        for i in range(self.reflectivity_tabs.count()):
            if self.reflectivity_tabs.isTabVisible(i):
                visible.append(REFLECTIVITY_KEYS[i])
        for i in range(self.analysis_tabs.count()):
            if self.analysis_tabs.isTabVisible(i):
                visible.append(ANALYSIS_KEYS[i])
        if not 0 <= position < len(visible):
            return False
        self.select_view(visible[position])
        return True

    #
    # Tab titles
    #
    def tab_titles(self) -> tuple[str, ...]:
        ref = tuple(self.reflectivity_tabs.tabText(i) for i in range(self.reflectivity_tabs.count()))
        ana = tuple(self.analysis_tabs.tabText(i) for i in range(self.analysis_tabs.count()))
        return ref + ana

    #
    # Tab keys
    #
    def tab_keys(self) -> tuple[str, ...]:
        """The switchable diagnostic tabs, in tab-bar order (both groups)."""
        return tuple(key for key, _title, _description in TAB_SPECS)

    #
    # Reflectivity tab keys
    #
    def reflectivity_tab_keys(self) -> tuple[str, ...]:
        """Tab keys belonging to the reflectivity group."""
        return REFLECTIVITY_KEYS

    #
    # Analysis tab keys
    #
    def analysis_tab_keys(self) -> tuple[str, ...]:
        """Tab keys belonging to the analysis/diagnostic group."""
        return ANALYSIS_KEYS

    #
    # View keys
    #
    def view_keys(self) -> tuple[str, ...]:
        """Every owned view, including the companion pane outside the tab bar."""
        return tuple(key for key, _title, _description in VIEW_SPECS)

    #
    # Current view key
    #
    def current_view_key(self) -> str:
        return self._interactions.current_view_key()

    #
    # Select view
    #
    def select_view(self, key: str) -> None:
        self._interactions.select_view(key)

    #
    # Set expert mode
    #
    def set_expert_mode(self, enabled: bool) -> None:
        self._interactions.set_expert_mode(enabled)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Lock scientific inputs without disabling zoom, pan, or inspection."""
        self._interactions.set_editing_enabled(enabled)

    #
    # Apply workspace
    #
    def apply_workspace(self, *, expert_mode: bool, tab_index: int, analysis_tab_index: int = 0) -> None:
        self._interactions.apply_workspace(expert_mode, tab_index, analysis_tab_index)

    #
    # Mode buttons
    #
    def mode_buttons(self) -> dict[str, object]:
        return self.toolbar.buttons()

    #
    # Mode actions
    #
    def mode_actions(self) -> dict[str, object]:
        """三档交互模式的 ``QAction``——状态住在这里，不在按钮上。

        设计稿的 ``.modebar`` 只给「范围」留了字形，查看与掩膜从条的右键菜单进，所以问
        「模式现在是哪一档」得问 action，``mode_buttons()`` 只剩留在条上的那一枚。
        """
        return self.toolbar.mode_actions()

    #
    # Navigation buttons
    #
    def navigation_buttons(self) -> dict[str, object]:
        return self.toolbar.navigation_buttons()

    #
    # Navigation mode
    #
    def navigation_mode(self) -> str:
        return self._interactions.navigation_mode()

    #
    # Navigators
    #
    def navigators(self) -> dict[str, object]:
        return self._interactions.navigators()

    #
    # Interaction mode
    #
    def interaction_mode(self) -> str:
        return self.toolbar.mode()

    #
    # Set interaction mode
    #
    def set_interaction_mode(self, mode: str) -> None:
        self.toolbar.set_mode(mode)

    #
    # Select fit range
    #
    def select_fit_range(self, first: float, second: float) -> bool:
        return self._interactions.select_fit_range(first, second)

    #
    # Request point mask
    #
    def request_point_mask(self, index: int) -> bool:
        return self._interactions.request_point_mask(index)

    #
    # Cancel interaction
    #
    def cancel_interaction(self) -> None:
        self._interactions.cancel()

    #
    # Callback counts
    #
    def callback_counts(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return self._interactions.callback_counts()
