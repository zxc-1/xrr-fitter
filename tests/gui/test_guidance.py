"""Step-by-step guidance mode.

Expert mode used to be the only working surface: standard mode merely hid some
controls, leaving a newcomer facing the full dock workspace. Guidance is a
separate surface that walks import to result, showing only what the current step
needs. Every gate is answered by existing public API reads, so the mode adds no
api surface.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QAbstractButton, QComboBox, QLineEdit, QSpinBox

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)

STEP_NAMES = ("importStep", "structureStep", "fitStep", "resultStep")

INTERACTIVE = (QAbstractButton, QComboBox, QSpinBox, QLineEdit)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path: Path, *, structured: bool = False):
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
    window.resize(1280, 760)
    window.show()
    qtbot.wait(1)
    return window


def test_guidance_is_the_default_surface_for_a_new_project(qtbot) -> None:
    """A newcomer meets the guided flow, not the full dock workspace."""
    window = _window(qtbot)

    assert window.guidance.isVisibleTo(window) is True
    assert window.docks["parametersDock"].isVisibleTo(window) is False


def test_guidance_declares_the_four_workflow_steps(qtbot) -> None:
    window = _window(qtbot)

    assert tuple(window.guidance.step_names()) == STEP_NAMES


def test_each_step_stays_within_the_control_budget(qtbot, tmp_path) -> None:
    """The whole point is a small surface, so the budget is asserted."""
    window = _window(qtbot, _project(tmp_path, structured=True))

    for name in STEP_NAMES:
        window.guidance.show_step(name)
        qtbot.wait(1)
        visible = [
            widget for kind in INTERACTIVE for widget in window.guidance.findChildren(kind) if widget.isVisible()
        ]
        assert len(visible) <= 10, f"{name} shows {len(visible)} controls"


def test_a_step_is_blocked_until_its_precondition_holds(qtbot) -> None:
    """An empty project cannot advance past import."""
    window = _window(qtbot)

    assert window.guidance.step_is_available("importStep") is True
    assert window.guidance.step_is_available("structureStep") is False
    assert window.guidance.step_is_available("fitStep") is False


def test_importing_data_unblocks_the_structure_step(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path))

    assert window.guidance.step_is_available("structureStep") is True
    assert window.guidance.step_is_available("fitStep") is False


def test_defining_a_structure_unblocks_the_fit_step(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path, structured=True))

    assert window.guidance.step_is_available("fitStep") is True


def test_switching_to_expert_reveals_the_dock_workspace(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path, structured=True))

    window.set_guidance_visible(False)

    assert window.guidance.isVisibleTo(window) is False
    assert window.docks["parametersDock"].isVisibleTo(window) is True


def test_switching_back_and_forth_keeps_project_state(qtbot, tmp_path) -> None:
    """The surfaces are two views of one project, not two projects."""
    window = _window(qtbot, _project(tmp_path, structured=True))
    before = window.document.project

    window.set_guidance_visible(False)
    window.set_guidance_visible(True)

    assert window.document.project is before
    assert window.guidance.step_is_available("fitStep") is True


def test_guidance_toggle_lives_in_the_view_menu(qtbot) -> None:
    window = _window(qtbot)
    action = window.chrome_actions["guidanceModeAction"]

    assert action.isCheckable() is True
    assert action.isChecked() is True

    action.setChecked(False)

    assert window.guidance.isVisibleTo(window) is False
