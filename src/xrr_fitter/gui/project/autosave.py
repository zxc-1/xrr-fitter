"""Timed draft autosave beside the saved project for crash recovery.

The workspace already prompts before a user-initiated close discards unsaved
changes.  That guard cannot help an unexpected exit (crash, power loss), so
this controller periodically writes the current project to a sidecar draft
(``<path>.autosave``) whenever it is dirty.  The draft never overwrites the
user's own file, and a successful manual save discards it — so a draft left on
disk means the previous session ended uncleanly and is worth offering back.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument

DRAFT_SUFFIX = ".autosave"

# Two minutes between drafts: frequent enough to bound lost work, rare enough
# that serialising a large project never intrudes on interaction.
AUTOSAVE_INTERVAL_MS = 120_000


class AutosaveController(QObject):
    """Own the draft-autosave timer and the sidecar draft file lifecycle."""

    draft_saved = Signal(object)

    def __init__(
        self,
        document: ProjectDocument,
        *,
        interval_ms: int = AUTOSAVE_INTERVAL_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._document = document
        self._timer = QTimer(self)
        self._timer.setObjectName("autosaveTimer")
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.autosave_now)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def draft_path(self) -> Path | None:
        """Return the sidecar draft path, or None when the project is unsaved."""
        path = self._document.path
        return None if path is None else Path(str(path) + DRAFT_SUFFIX)

    def autosave_now(self) -> bool:
        """Write a draft if the project is dirty and has a saved location."""
        draft = self.draft_path()
        if draft is None or not self._document.is_dirty:
            return False
        api.save_project(self._document.project, draft)
        self.draft_saved.emit(draft)
        return True

    def discard_draft(self) -> None:
        """Delete the draft once a real save has superseded it."""
        draft = self.draft_path()
        if draft is not None:
            draft.unlink(missing_ok=True)

    def has_recoverable_draft(self) -> bool:
        """A draft on disk means the last session never saved-and-discarded."""
        draft = self.draft_path()
        return draft is not None and draft.exists()

    def recover(self) -> api.XrrProject:
        """Load the draft project so the caller can adopt the recovered work."""
        draft = self.draft_path()
        if draft is None or not draft.exists():
            raise FileNotFoundError("no autosave draft to recover")
        return api.load_project(draft)
