"""Dock-layout round-trip and backward compatibility of the project codec.

A project written before the dockable layout existed has no ``dock_state`` key.
Reading it must succeed and yield the empty default rather than raising, because
the alternative is that upgrading the application makes existing files
unopenable.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.support.model_cases import project

from xrr_fitter.io.project_codec import project_from_bytes, project_to_bytes
from xrr_fitter.model.project import ProjectUiState


DOCK_STATE = "AAAA/wAAAAD9AAAAAgAAAAAAAAEA"


def test_dock_state_survives_an_encode_decode_round_trip() -> None:
    from dataclasses import replace

    original = replace(
        project(),
        ui_state=ProjectUiState(dock_state=DOCK_STATE),
    )

    restored = project_from_bytes(project_to_bytes(original))

    assert restored.ui_state.dock_state == DOCK_STATE


def test_project_without_dock_state_key_decodes_to_the_default() -> None:
    payload = json.loads(project_to_bytes(project()).decode("utf-8"))
    del payload["ui_state"]["dock_state"]

    restored = project_from_bytes(json.dumps(payload).encode("utf-8"))

    assert restored.ui_state.dock_state == ""


def test_reference_projects_still_load_without_a_dock_state() -> None:
    """The checked-in schema examples predate the field."""
    root = Path(__file__).resolve().parents[2] / "fixtures" / "reference_inputs"
    for stem in ("single-layer", "mo-si-periodic"):
        path = root / f"{stem}.xrrproj.json"
        if not path.exists():
            continue
        loaded = project_from_bytes(path.read_bytes())
        assert loaded.ui_state.dock_state == ""
