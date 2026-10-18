"""A closed main window must be collectable, along with the menus its plots built.

pyqtgraph builds each plot's context menus without a Qt parent, so they live exactly as long as
the plot's Python objects.  If anything keeps a closed window's Python side reachable, every one
of those menus stays a live top-level widget -- and an appearance switch re-polishes and walks
every widget the application still has.
"""

import gc
import weakref

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMenu
from shiboken6 import getCppPointer


def _top_level_menus() -> set[int]:
    return {getCppPointer(widget)[0] for widget in QApplication.topLevelWidgets() if isinstance(widget, QMenu)}


def test_a_closed_window_is_released_with_the_plot_menus_it_built(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    before = _top_level_menus()
    window = MainWindow()
    window.show()
    qtbot.waitExposed(window)
    assert _top_level_menus() - before, "the window's plots build parentless menus"
    released = weakref.ref(window)

    window.close()
    QApplication.processEvents()
    window.deleteLater()
    del window
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()

    assert released() is None
    assert not _top_level_menus() - before
