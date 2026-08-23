"""One icon per command, resolved by the command's identity rather than a widget.

A command reaches the user twice: as a toolbar ``QPushButton`` and as a menu
``QAction``.  Those are separate objects, so an icon hung on one never reaches
the other, and "新建" was a labelled glyph on the toolbar but bare text in the
menu - the same command learned twice.  The icon belongs to the command, so both
surfaces look it up here by the callback that defines the command; painting from
the active palette's text colour keeps every glyph legible in both themes.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon

from xrr_fitter.gui.plots.plot_icons import PAINTERS, plot_icon

# The commands that carry a painted glyph.  Tests iterate this to assert that
# every registered command resolves to a distinct non-null icon.
COMMAND_PAINTERS: dict[str, object] = {
    name: fn
    for name, fn in PAINTERS.items()
    if name
    in {
        "new_project_dialog",
        "open_project_dialog",
        "save_project_dialog",
        "save_project_as_dialog",
        "reload_source_dialog",
        "relink_source_dialog",
        "export_results_dialog",
        "start_fit",
        "cancel_fit",
        "import_files",
        "import_folder",
        "force_stop",
    }
}

# Backwards-compatible name used by tests.
COMMAND_PIXMAPS = COMMAND_PAINTERS


def command_icon(command: str) -> QIcon:
    """The glyph for a command, or a null icon when the command has none.

    Every caller renders through the shared painter registry so a button and its
    menu twin compare equal byte for byte, which is the whole point of a single
    map.
    """
    if command not in PAINTERS:
        return QIcon()
    return plot_icon(command)
