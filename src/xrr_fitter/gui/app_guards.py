"""Application-wide event filters the main window installs for its lifetime.

Both guards watch events that no single widget gets to veto in time: the quit that
macOS sends to the application, and a wheel notch that lands on an editor the reader
was only scrolling past.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox

# 只有读者点进去的数值框和下拉框才吃滚轮。没点进去时滚轮该滚外面的面板，而不是悄悄改掉
# 一层厚度或噪声模式——那一下会作废已有的拟合结果，读者却以为自己只是在翻页。
WHEEL_GUARDED_EDITORS = (QAbstractSpinBox, QComboBox)


class QuitGuard(QObject):
    """Ask the window before an application quit closes it.

    ⌘Q, the Dock's 退出 and logging out arrive as ``QEvent::Quit`` on the
    application, and Qt answers it with ``closeAllWindows()``. Those close events
    are not spontaneous, so to the window they look like a programmatic close,
    which must stay silent. The question therefore has to be asked here, before Qt
    starts closing; ignoring the event cancels the quit, and on macOS it becomes
    ``NSTerminateCancel``.
    """

    def __init__(self, parent: QObject, confirm: Callable[[], bool]) -> None:
        super().__init__(parent)
        self._confirm = confirm

    def eventFilter(self, _watched: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Quit or self._confirm():
            return False
        event.ignore()
        return True


class WheelGuard(QObject):
    """Keep the mouse wheel off editors that do not have keyboard focus.

    ``QAbstractSpinBox`` steps on every wheel notch whether or not it is focused,
    and its default ``WheelFocus`` hands it focus before any filter sees the event,
    so the policy is narrowed to ``StrongFocus`` when the editor is polished. An
    unfocused editor then ignores the notch and Qt passes it on to the scroll area
    behind it. A click or Tab still focuses the editor, after which the wheel steps
    it as before.
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind == QEvent.Type.Polish:
            if isinstance(watched, WHEEL_GUARDED_EDITORS) and watched.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                watched.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            return False
        if kind == QEvent.Type.Wheel and isinstance(watched, WHEEL_GUARDED_EDITORS) and not watched.hasFocus():
            event.ignore()
            return True
        return False


def install_app_guards(window: QObject, confirm_quit: Callable[[], bool]) -> tuple[QObject, ...]:
    """Install both guards on the application, owned by ``window``.

    Call this before the window builds its children: an editor polished before the
    wheel guard exists keeps ``WheelFocus``.
    """
    application = QCoreApplication.instance()
    guards = (QuitGuard(window, confirm_quit), WheelGuard(window))
    for guard in guards:
        application.installEventFilter(guard)
    return guards


def remove_app_guards(guards: tuple[QObject, ...]) -> None:
    application = QCoreApplication.instance()
    if application is None:
        return
    for guard in guards:
        application.removeEventFilter(guard)
