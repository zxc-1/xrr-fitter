"""Guidance mode and pipeline navigation coordination.

M3 verifies that the two orthogonal modes (guidance view-layer toggle vs
expert_mode document-persistent flag) stay independent while both driving
the same pipeline nav state.
"""

from __future__ import annotations

from pathlib import Path

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path, *, structured: bool = False):
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "curve.xy"),
        api.InstrumentSpec(),
    )
    if structured:
        structure = api.StructureSpec(
            AIR,
            (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
            SI,
        )
        project = api.set_structure(project, "curve", structure)
    return api.select_active_dataset(project, "curve")


def _window(qtbot, project=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    document = ProjectDocument() if project is None else ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(1)
    return window


def test_guidance_toggle_does_not_alter_pipeline_nav_step(qtbot, tmp_path) -> None:
    """Toggling guidance visible/hidden is a view concern, not project state."""
    window = _window(qtbot, _project(tmp_path, structured=True))
    nav = window.pipeline_nav
    step_before = nav.current_step_index()

    window.set_guidance_visible(True)
    qtbot.wait(1)

    assert nav.current_step_index() == step_before

    window.set_guidance_visible(False)
    qtbot.wait(1)

    assert nav.current_step_index() == step_before


def test_expert_mode_flag_is_orthogonal_to_guidance_visibility(qtbot, tmp_path) -> None:
    """expert_mode lives on the document; guidance visibility lives on the view."""
    window = _window(qtbot, _project(tmp_path, structured=True))

    window.set_guidance_visible(True)
    assert window.guidance.isVisibleTo(window) is True

    window.parameters_panel.set_expert_mode(True)
    assert window.document.project.ui_state.expert_mode is True
    assert window.guidance.isVisibleTo(window) is True


def test_pipeline_nav_accessible_name(qtbot) -> None:
    """The pipeline stepper carries an accessible name for screen readers.

    It used to sit inside a QDockWidget that announced itself; on the design's
    left rail it is a plain child widget, so the name has to be its own.
    """
    window = _window(qtbot)
    nav = window.pipeline_nav
    assert nav.accessibleName() == "管线导航面板"


def test_guidance_step_indicator_has_accessible_name(qtbot) -> None:
    """The guidance panel itself carries an accessible name."""
    window = _window(qtbot)
    assert window.guidance.accessibleName() == "引导流程"
