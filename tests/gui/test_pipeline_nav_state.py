"""Pipeline navigation state synchronization with the project document.

The navigation panel reflects pipeline progress: steps illuminate as the
project acquires data, structure, and results. The guidance panel and the
pipeline nav share the same project state but express it independently —
the nav highlights via dot styling, guidance gates via step availability.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

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


def test_pipeline_nav_highlights_import_step_for_empty_project(qtbot) -> None:
    """An empty project has no data — only the import step is reached."""
    window = _window(qtbot)
    nav = window.pipeline_nav

    assert nav.current_step_index() == 0


def test_pipeline_nav_advances_to_structure_when_data_imported(qtbot, tmp_path) -> None:
    """After importing data, the structure step becomes current."""
    window = _window(qtbot, _project(tmp_path))
    nav = window.pipeline_nav

    assert nav.current_step_index() >= 1


def test_pipeline_nav_advances_to_parameters_when_structured(qtbot, tmp_path) -> None:
    """After defining structure, the parameters step becomes current."""
    window = _window(qtbot, _project(tmp_path, structured=True))
    nav = window.pipeline_nav

    assert nav.current_step_index() >= 2


def test_pipeline_nav_updates_on_project_change(qtbot, tmp_path) -> None:
    """The nav refreshes when the project state changes."""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = _project(tmp_path)
    document = ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(1)

    nav = window.pipeline_nav
    assert nav.current_step_index() >= 1

    structured = api.set_structure(
        project,
        "curve",
        api.StructureSpec(
            AIR,
            (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
            SI,
        ),
    )
    document.replace_project(structured)
    qtbot.wait(1)

    assert nav.current_step_index() >= 2


def test_pipeline_nav_dot_marks_completed_steps(qtbot, tmp_path) -> None:
    """Steps before the current one trade their ordinal for a check."""
    window = _window(qtbot, _project(tmp_path, structured=True))
    nav = window.pipeline_nav
    current = nav.current_step_index()

    for i in range(current):
        dot = nav.findChild(QLabel, f"pipelineDot_{nav.step_titles()[i]}")
        assert dot.text() == "✓", f"step {i} should be completed"
