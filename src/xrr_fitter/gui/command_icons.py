"""One icon per command, resolved by the command's identity rather than a widget.

A command reaches the user twice: as a toolbar ``QPushButton`` and as a menu
``QAction``.  Those are separate objects, so an icon hung on one never reaches
the other, and "新建" was a labelled glyph on the toolbar but bare text in the
menu - the same command learned twice.  The icon belongs to the command, so both
surfaces look it up here by the callback that defines the command; system theme
icons are preferred for standard operations, with custom painted fallbacks for
domain-specific controls.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle

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

# Map command names to (freedesktop theme name, QStyle fallback).
# Each entry MUST map to a DISTINCT pixmap — commands sharing the same visual
# must NOT appear here (they keep their custom painter instead).
_SYSTEM_ICONS: dict[str, tuple[str, QStyle.StandardPixmap | None]] = {
    "new_project_dialog": ("document-new", QStyle.StandardPixmap.SP_FileIcon),
    "open_project_dialog": ("document-open", QStyle.StandardPixmap.SP_DirOpenIcon),
    "save_project_dialog": ("document-save", QStyle.StandardPixmap.SP_DialogSaveButton),
    "reload_source_dialog": ("view-refresh", QStyle.StandardPixmap.SP_BrowserReload),
    "start_fit": ("media-playback-start", QStyle.StandardPixmap.SP_MediaPlay),
    "cancel_fit": ("media-playback-stop", QStyle.StandardPixmap.SP_MediaStop),
    "force_stop": ("process-stop", QStyle.StandardPixmap.SP_BrowserStop),
}


def _system_icon(command: str) -> QIcon | None:
    """Try to resolve a system icon for the command."""
    entry = _SYSTEM_ICONS.get(command)
    if entry is None:
        return None
    theme_name, style_fallback = entry
    # Try freedesktop theme first (Linux with KDE/GNOME).
    icon = QIcon.fromTheme(theme_name)
    if not icon.isNull():
        return icon
    # Fall back to QStyle (works on macOS/Windows).
    if style_fallback is not None:
        app = QApplication.instance()
        if app is not None:
            style = app.style()
            if style is not None:
                icon = style.standardIcon(style_fallback)
                if not icon.isNull():
                    return icon
    return None


def command_icon(command: str) -> QIcon:
    """The glyph for a command, or a null icon when the command has none.

    Prefers system theme icons for standard file/media operations; falls back to
    the custom painter registry for domain-specific commands.
    """
    sys_icon = _system_icon(command)
    if sys_icon is not None:
        return sys_icon
    if command not in PAINTERS:
        return QIcon()
    return plot_icon(command)
