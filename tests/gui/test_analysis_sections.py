"""Collapsible analysis-column sections.

Stacking the parameter, fit, and result cards in a scroll area pushed the
column's content well past the viewport at the documented minimum window size,
so the lower sections were reachable only by scrolling. Each section now has a
header that toggles its body, which lets the column fit while still allowing two
sections to stay open together when the user wants to compare them.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QScrollArea

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)

SECTION_NAMES = ("parametersSection", "fitSection", "resultsSection")


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path: Path, *, expert: bool = False):
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "curve.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
        SI,
    )
    project = api.set_structure(project, "curve", structure)
    if expert:
        project = api.set_expert_mode(project, True)
    return api.select_active_dataset(project, "curve")


def _window(qtbot, tmp_path: Path, *, expert: bool = False, size=(1280, 760)):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path, expert=expert)))
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    qtbot.wait(1)
    return window


def test_every_analysis_section_is_collapsible(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path)

    for name in SECTION_NAMES:
        section = window.analysis_sections[name]
        assert section.toggle.isCheckable() is True


def test_collapsing_a_section_hides_only_its_body(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path)
    section = window.analysis_sections["parametersSection"]

    section.set_expanded(False)

    assert section.body.isVisibleTo(window) is False
    # The header stays reachable so the section can be brought back.
    assert section.toggle.isVisibleTo(window) is True
    assert window.analysis_sections["fitSection"].body.isVisibleTo(window) is True


def test_sections_are_independent_rather_than_mutually_exclusive(
    qtbot,
    tmp_path,
) -> None:
    """Watching progress against the parameter table requires two open at once."""
    window = _window(qtbot, tmp_path)

    for name in ("parametersSection", "fitSection"):
        window.analysis_sections[name].set_expanded(True)

    assert window.analysis_sections["parametersSection"].body.isVisibleTo(window)
    assert window.analysis_sections["fitSection"].body.isVisibleTo(window)


def test_default_open_sections_fit_the_documented_minimum_window(
    qtbot,
    tmp_path,
) -> None:
    """At 1280x760 the column's default content must not overflow its viewport."""
    window = _window(qtbot, tmp_path, expert=True)

    scroll = window.findChild(QScrollArea, "analysisScroll")
    assert scroll.widget().sizeHint().height() <= scroll.viewport().height()
