"""Plot contract cases, partition 5; collected via test_plots.py.

These cases pin the side-by-side reflectivity/SLD workspace. Judging a fit
requires reading curve agreement and the depth profile together, so the SLD
view is a permanent companion pane rather than one of the switchable
diagnostic tabs, and the default tab is the log view where reflectivity
structure is actually legible.
"""

from __future__ import annotations

from tests.gui.plot_support import *  # noqa: F403


def test_sld_is_a_permanent_pane_outside_the_diagnostic_tabs(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.set_expert_mode(True)

    assert "sld" not in panel.tab_keys()
    assert "SLD 深度剖面" not in panel.tab_titles()
    # The view itself survives; only its placement changed.
    assert panel.view("sld") is not None
    assert panel.sld_pane.isVisibleTo(panel) is True


def test_reflectivity_and_sld_are_visible_together(qtbot) -> None:
    panel = _panel(qtbot, data=prepared_data(size=4))
    panel.show()

    reflectivity = panel.view(panel.current_view_key()).canvas
    sld = panel.view("sld").canvas

    assert reflectivity.isVisibleTo(panel) is True
    assert sld.isVisibleTo(panel) is True
    # Stacked vertically, so neither pane occludes the other.
    assert panel.plot_splitter.orientation() == Qt.Orientation.Vertical


def test_log_view_is_the_default_selection(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.tab_keys()[0] == "log"
    assert panel.current_view_key() == "log"


def test_remaining_tabs_keep_their_order_after_sld_leaves(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.tab_keys() == (
        "log",
        "raw",
        "qz4",
        "residual",
        "candidates",
        "uncertainty",
        "trend",
    )


def test_sld_pane_is_expert_only(qtbot) -> None:
    """The depth profile is expert evidence, so standard mode keeps it hidden."""
    panel = _panel(qtbot, data=prepared_data(size=4))

    panel.set_expert_mode(False)
    assert panel.sld_pane.isVisibleTo(panel) is False

    panel.set_expert_mode(True)
    assert panel.sld_pane.isVisibleTo(panel) is True


def test_number_shortcuts_cover_the_seven_remaining_tabs(qtbot) -> None:
    from PySide6.QtGui import QKeySequence

    panel = _panel(qtbot, data=prepared_data(size=4))
    keys = [shortcut.key() for shortcut in panel.view_shortcuts]

    assert len(keys) == 7
    assert keys[0] == QKeySequence("Alt+1")
    assert keys[6] == QKeySequence("Alt+7")


def test_selecting_sld_as_a_tab_is_rejected(qtbot) -> None:
    """``select_view`` addresses tabs; the companion pane is not one."""
    panel = _panel(qtbot)

    with pytest.raises(KeyError, match="sld"):
        panel.select_view("sld")
