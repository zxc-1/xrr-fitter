from __future__ import annotations

from pathlib import Path

import xrr_fitter.api as api


def _saved_document(tmp_path: Path):
    from xrr_fitter.gui.document import ProjectDocument

    document = ProjectDocument()
    path = tmp_path / "proj.xrrproj.json"
    document.save(path)
    return document, path


def _controller(document):
    from xrr_fitter.gui.project.autosave import AutosaveController

    return AutosaveController(document, interval_ms=60_000)


def test_autosave_writes_a_sidecar_draft_only_when_dirty(qtbot, tmp_path) -> None:
    document, path = _saved_document(tmp_path)
    controller = _controller(document)
    original = path.read_bytes()

    # A clean project has nothing to protect, so no draft is written.
    assert controller.autosave_now() is False
    assert not controller.draft_path().exists()

    # Once dirty, a draft appears beside the project file and the user's own
    # file is never touched by the autosave.
    document.mark_dirty()
    assert controller.autosave_now() is True
    draft = controller.draft_path()
    assert draft == Path(str(path) + ".autosave")
    assert draft.exists()
    assert path.read_bytes() == original


def test_autosave_is_a_noop_without_a_saved_path(qtbot) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    document = ProjectDocument()
    document.mark_dirty()
    controller = _controller(document)

    # An unsaved new project has no obvious draft location; the close prompt
    # guards it instead, so autosave stays a no-op rather than guessing a path.
    assert controller.draft_path() is None
    assert controller.autosave_now() is False


def test_recoverable_draft_survives_until_discarded(qtbot, tmp_path) -> None:
    document, path = _saved_document(tmp_path)
    controller = _controller(document)

    assert controller.has_recoverable_draft() is False

    document.mark_dirty()
    controller.autosave_now()

    # A lingering draft means the last session never saved-and-discarded, so it
    # is offered for recovery and loads back into a real project.
    assert controller.has_recoverable_draft() is True
    recovered = controller.recover()
    assert isinstance(recovered, api.XrrProject)

    # A successful save supersedes the draft, so discarding clears the offer.
    controller.discard_draft()
    assert controller.has_recoverable_draft() is False
    assert not controller.draft_path().exists()
