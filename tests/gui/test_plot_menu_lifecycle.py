"""Plot controls leave no orphan menus after their owning widget is destroyed."""

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from xrr_fitter.gui.plots.live import LiveReflectivityPlot, ObjectiveTracePlot


def _deliver_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("plot_type", (LiveReflectivityPlot, ObjectiveTracePlot))
@pytest.mark.parametrize("destroy_parent", (False, True))
def test_plot_destruction_releases_both_control_menus(qapp, plot_type, destroy_parent):
    owner = QWidget()
    plot = plot_type(owner)
    menus = (plot.plotItem.ctrlMenu, plot.plotItem.vb.menu)
    assert all(isValid(menu) and menu.actions() for menu in menus)

    try:
        if destroy_parent:
            owner.deleteLater()
        else:
            plot.close()
            plot.deleteLater()
        _deliver_deletes()

        assert not any(isValid(menu) for menu in menus), "plot menus outlive their destroyed owner"
    finally:
        for widget in (*menus, owner):
            if isValid(widget):
                widget.deleteLater()
        _deliver_deletes()
