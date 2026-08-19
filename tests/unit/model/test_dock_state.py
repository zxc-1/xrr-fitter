"""Persistence of the dockable workspace layout.

``QMainWindow.saveState()`` returns an opaque, Qt-version-dependent byte string,
so the project stores it as base64 text and treats it as untrusted on the way
back in. An empty value means "use the default arrangement", which is what a
project written before docks existed decodes to.
"""

from __future__ import annotations

import pytest

from xrr_fitter.model.project import ProjectUiState


def test_dock_state_defaults_to_empty() -> None:
    assert ProjectUiState().dock_state == ""


def test_dock_state_round_trips_through_the_model() -> None:
    state = ProjectUiState(dock_state="AAAA/wAAAAD9AAAA")

    assert state.dock_state == "AAAA/wAAAAD9AAAA"


def test_dock_state_rejects_a_non_string() -> None:
    with pytest.raises((TypeError, ValueError)):
        ProjectUiState(dock_state=b"AAAA")


def test_setting_dock_state_leaves_other_ui_fields_untouched() -> None:
    from tests.support.model_cases import project

    from xrr_fitter.services.projects import set_dock_state

    original = project()
    updated = set_dock_state(original, "AAAA/wAAAAD9AAAA")

    assert updated.ui_state.dock_state == "AAAA/wAAAAD9AAAA"
    assert original.ui_state.dock_state == ""
    assert updated.ui_state.expert_mode == original.ui_state.expert_mode
    assert updated.ui_state.plot_tab_index == original.ui_state.plot_tab_index


def test_setting_the_same_dock_state_is_identity() -> None:
    """An unchanged layout must not mark the project dirty."""
    from tests.support.model_cases import project

    from xrr_fitter.services.projects import set_dock_state

    original = set_dock_state(project(), "AAAA")

    assert set_dock_state(original, "AAAA") is original
