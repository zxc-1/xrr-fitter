"""Structural layout invariants for the content-sized GUI widgets under extreme
window sizes and an enlarged font.

The shipped MainWindow is a QDockWidget layout, so the QSplitter checks in
test_workspace.py exercise a synthetic widget tree that no longer mirrors it.
These contracts instead pin the size behaviour of the widgets this overhaul
added -- the content-sized trees (sizing.ContentSizedTree) and the stack
diagram (structure.stack) -- at the size extremes and font scale where a
regression would clip content or starve a band rather than merely look off.

Font scale stands in for a non-default display DPI: offscreen cannot change the
screen's real DPI, but DPI's chief layout consequence is that font-driven
metrics grow, and ContentSizedTree.sizeHint is defined in those metrics. The
stack diagram sizes in raw pixels, so it is pinned across window heights rather
than font scale.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTreeWidgetItem

import xrr_fitter.api as api
from xrr_fitter.gui.sizing import ContentSizedTree

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _tree_with_rows(qtbot, count: int) -> ContentSizedTree:
    """A bare content-sized tree holding ``count`` top-level rows."""
    tree = ContentSizedTree()
    qtbot.addWidget(tree)
    tree.setHeaderHidden(True)
    for index in range(count):
        tree.addTopLevelItem(QTreeWidgetItem([f"row {index}"]))
    return tree
