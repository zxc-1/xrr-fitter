"""Analysis docks hold their panel directly.

The cards that once let a shared analysis column collapse its sections became
redundant when each panel moved into its own tabbed dock: the card header
repeated the dock's own title verbatim, so reaching the parameter table meant
passing a dock tab labelled 参数, then a card header labelled 参数, then the
table's own tab. The docks now hold their panels directly, which costs the
per-section collapsing the shared column needed and requires every panel to
report a size hint its dock can actually satisfy.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QScrollArea, QWidget

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)

DOCK_PANELS = (
    ("parametersDock", "parameters_panel"),
    ("fitDock", "fit_panel"),
    ("resultsDock", "result_panel"),
)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
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
    # The sections live in docks, which the guided surface hides by design.
    window.set_guidance_visible(False)
    qtbot.wait(1)
    return window


def test_each_analysis_dock_holds_its_panel_without_an_intervening_card(
    qtbot,
    tmp_path,
) -> None:
    """A dock titled 参数 must not wrap its panel in a card titled 参数 too."""
    window = _window(qtbot, tmp_path)

    for dock_name, panel_name in DOCK_PANELS:
        scroll = window.findChild(QScrollArea, f"{dock_name}Scroll")
        assert scroll is not None, dock_name
        assert scroll.widget() is getattr(window, panel_name), dock_name


def test_no_analysis_panel_is_reachable_only_through_a_redundant_header(
    qtbot,
    tmp_path,
) -> None:
    """No widget may repeat its own dock's title on the way to the panel."""
    window = _window(qtbot, tmp_path)

    for dock_name, _ in DOCK_PANELS:
        dock = window.docks[dock_name]
        titles = [
            child.text()
            for child in dock.findChildren(QWidget)
            if hasattr(child, "text") and child.text() == dock.windowTitle()
        ]
        assert titles == [], f"{dock_name} repeats {dock.windowTitle()!r}"


def test_each_analysis_dock_fits_the_documented_minimum_window(
    qtbot,
    tmp_path,
) -> None:
    """At 1280x760 no analysis dock's content may overflow its own viewport."""
    window = _window(qtbot, tmp_path, expert=True)

    for name in ("parametersDock", "fitDock", "resultsDock"):
        scroll = window.findChild(QScrollArea, f"{name}Scroll")
        assert scroll is not None, name
        overflow = scroll.widget().sizeHint().height() - scroll.viewport().height()
        assert overflow <= 0, f"{name} overflows by {overflow}px"
