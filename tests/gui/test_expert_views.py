from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QCheckBox

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


def _project(tmp_path):
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
        SI,
    )
    return api.set_structure(project, "sample", structure)


def test_expert_mode_toggle_updates_only_project_ui_state(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    document = ProjectDocument(_project(tmp_path))
    panel = ParametersPanel(document)
    qtbot.addWidget(panel)
    original = document.project
    updated = api.set_expert_mode(original, True)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "set_expert_mode",
        lambda project, enabled: (
            calls.append((project, enabled)),
            updated,
        )[1],
    )

    panel.set_expert_mode(True)

    assert calls == [(original, True)]
    assert document.project is updated
    assert document.project.datasets == original.datasets


def test_expert_mode_projects_expert_only_parameter_rows(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    standard = panel.row_names

    panel.set_expert_mode(True)

    assert len(panel.row_names) > len(standard)
    assert "instrument.absolute_sigma_a_inv" in panel.row_names
    assert panel.expert_mode is True


def test_main_window_composes_data_structure_and_parameter_panels(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)

    assert window.data_panel.document is window.document
    assert window.structure_panel.document is window.document
    assert window.parameters_panel.document is window.document
    assert window.structure_panel.structure == window.document.project.datasets[0].structure
    assert "component.0.thickness_a" in window.parameters_panel.row_names


def test_expert_toggle_control_has_visible_name_and_reflects_document(qtbot, tmp_path) -> None:
    """The inspector's depth switch cannot borrow the name of the surface switch.

    Two different things were both called 专家模式.  The command bar's segment picks
    which surface is on screen -- ``WORKSPACE_MODE_SPECS``' 引导 / 专家 -- and the
    guided surface's own closing line sends the reader there by name:
    「需要精细控制每个参数的边界、先验或跨数据集共享？切换到专家模式即可。」  Obeying
    that hint landed on a workspace holding a *second*, unchecked control with the
    identical label, so the app said 专家模式 was on and off at once.

    This toggle switches depth, not surface, and it reveals more than parameters:
    the expert-only declarations, the SLD pane and the diagnostic tabs, 结果 panel's
    uncertainty entry, and the data panel's preset command.  So it is named for what
    it uncovers rather than for who is presumed to be looking.
    """
    from xrr_fitter.gui.chrome import WORKSPACE_MODE_SPECS
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    toggle = window.findChild(QCheckBox, "expertModeToggle")

    assert toggle is not None
    assert toggle.text() == "显示高级选项"
    assert toggle.accessibleName() == "切换高级选项"
    # The word stays with the segment that owns it in the design.
    assert "专家" in [label for _key, _name, label, _tip in WORKSPACE_MODE_SPECS]
    assert "专家" not in toggle.text()
    assert "专家" not in toggle.accessibleName()

    toggle.setChecked(True)

    assert window.document.project.ui_state.expert_mode is True


def test_every_surface_that_drives_the_depth_toggle_calls_it_by_its_own_name(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    """A menu entry and a refusal message both name this toggle; both must agree with it.

    The 视图 menu's entry drives ``expert_toggle.setChecked`` directly, and the
    refusal shown when a gated diagnostic tab is picked names the control the
    reader has to go turn on.  While all three read 专家模式 the app had two
    controls under one word -- the command bar's 引导↔专家 segment picks a surface,
    this one picks a depth -- so following either pointer led to the wrong switch.
    Renaming the checkbox alone would have been worse than leaving it: the menu and
    the message would then point at a名字 nothing on screen answers to.
    """
    from xrr_fitter.gui.chrome import _select_view
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    toggle = window.findChild(QCheckBox, "expertModeToggle")
    entry = window.chrome_actions["expertModeAction"]

    # The menu entry is the same switch by another route, so it reads as one.
    assert entry.text() == toggle.text()

    # No tab is hidden by depth any more -- ``_apply_tabs`` moved that projection
    # onto the SLD companion pane and ``setTabVisible`` is called nowhere -- so the
    # refusal is reached by raising what a hidden tab would raise.  The wording is
    # still the reader's only pointer if the gate ever comes back.
    def _refuse(_key: str) -> None:
        raise ValueError("diagnostic view is hidden: residual_map")

    monkeypatch.setattr(window.plot_panel, "select_view", _refuse)
    _select_view(window, "residual_map")

    message = window.statusBar().currentMessage()
    assert toggle.text() in message, message
    assert "专家模式" not in message, message
