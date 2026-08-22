"""One icon per command, resolved by the command's identity rather than a widget.

A command reaches the user twice: as a toolbar ``QPushButton`` and as a menu
``QAction``.  Those are separate objects, so an icon hung on one never reaches
the other, and "新建" was a labelled glyph on the toolbar but bare text in the
menu - the same command learned twice.  The icon belongs to the command, so both
surfaces look it up here by the callback that defines the command; identical keys
resolve to the identical rendered glyph because they share one ``QStyle``.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle

# Keyed by the window callback that names the command (see project.actions and
# window_layout specs).  import_files/import_folder/force_stop have no toolbar
# twin, but a menu that ices only some of its rows reserves the icon column and
# indents the rest, so they carry a glyph too.
COMMAND_PIXMAPS: dict[str, QStyle.StandardPixmap] = {
    "new_project_dialog": QStyle.StandardPixmap.SP_FileIcon,
    "open_project_dialog": QStyle.StandardPixmap.SP_DirOpenIcon,
    "save_project_dialog": QStyle.StandardPixmap.SP_DialogSaveButton,
    # SP_DialogSaveAllButton renders null under this style, so 另存为 borrows the
    # drive glyph: a save aimed at a chosen location rather than the open file.
    "save_project_as_dialog": QStyle.StandardPixmap.SP_DriveFDIcon,
    "reload_source_dialog": QStyle.StandardPixmap.SP_BrowserReload,
    # Not SP_FileLinkIcon: the native macOS style has no "linked file" pixmap and
    # falls back to the plain SP_FileIcon, so 重新链接 rendered pixel-identical to
    # 新建项目.  A forward arrow keeps it in the same family as its sibling 重新加载
    # (a reload arrow) while reading as "re-point the source at a new file".
    "relink_source_dialog": QStyle.StandardPixmap.SP_ArrowForward,
    "export_results_dialog": QStyle.StandardPixmap.SP_FileDialogDetailedView,
    "start_fit": QStyle.StandardPixmap.SP_MediaPlay,
    "cancel_fit": QStyle.StandardPixmap.SP_BrowserStop,
    "import_files": QStyle.StandardPixmap.SP_FileDialogContentsView,
    # Not SP_DirLinkIcon: like the file-link glyph above it degrades to the bare
    # folder the native style also hands 打开项目 (SP_DirOpenIcon), so the two were
    # indistinguishable.  SP_FileDialogNewFolder is the one folder-flavoured pixmap
    # this style renders distinctly, and a menu of look-alike folders tells its
    # commands apart by nothing.
    "import_folder": QStyle.StandardPixmap.SP_FileDialogNewFolder,
    "force_stop": QStyle.StandardPixmap.SP_MediaStop,
}


def command_icon(command: str) -> QIcon:
    """The glyph for a command, or a null icon when the command has none.

    Every caller renders through the application style so a button and its menu
    twin compare equal byte for byte, which is the whole point of a single map.
    """
    pixmap = COMMAND_PIXMAPS.get(command)
    if pixmap is None:
        return QIcon()
    return QApplication.instance().style().standardIcon(pixmap)
