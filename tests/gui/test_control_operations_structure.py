"""Structure mutations driven by visible MainWindow controls, not setters."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialogButtonBox
from tests.gui.control_operation_driver import controls as controls
from tests.gui.control_operation_driver import setup_project

import xrr_fitter.api as api


def _structure_step(driver, window):
    driver.named_click(window, "pipelineStep_结构")
    assert window.pipeline_nav.current_step_index() == 1


def _select_component(driver, window, index):
    tree = window.structure_panel.editor.tree
    item = tree.topLevelItem(index + 1)
    tree.scrollToItem(item)
    driver.reach(tree.viewport(), tree.visualItemRect(item).center())
    driver.qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(item).center())
    assert tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == index
    return tree


def _row_menu(driver, tree, action):
    driver.context_menu(tree.viewport(), tree.visualItemRect(tree.currentItem()).center(), action)


def test_initialize_and_add_layer_through_modal_fields(controls, tmp_path):
    window = controls.window(setup_project(tmp_path, layers=0))
    _structure_step(controls, window)
    controls.named_click(window, "initializeStructureButton")
    assert window.document.project.datasets[0].structure.backing.formula == "Si"
    controls.record("initializeStructureButton", "click suggested initialization", "Air/Si bare structure commits")
    _structure_step(controls, window)

    def configure(dialog):
        controls.type(dialog.name_editor, "manual-layer")
        controls.type(dialog.formula_editor, "SiO2")
        controls.type(dialog.density_editor, "2.2")
        controls.type(dialog.thickness_editor, "4.8")
        controls.type(dialog.roughness_editor, "0.3")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(lambda: controls.named_click(window, "addLayerButton"), [configure])
    layer = window.document.project.datasets[0].structure.components[0]
    assert layer == api.LayerSpec("manual-layer", api.MaterialSpec("SiO2", "SiO2", 2.2), 48.0, roughness_a=3.0)
    controls.record(
        (
            "addLayerButton",
            "layerNameInput",
            "layerFormulaInput",
            "layerDensityInput",
            "layerThicknessInput",
            "layerRoughnessInput",
            "layerDialogButtons:Ok",
        ),
        "click Add layer; keyboard-fill all ordinary fields; click OK",
        "complete immutable LayerSpec commits in angstrom units",
    )


def test_layer_transition_table_and_cancel_inputs(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    _structure_step(controls, window)
    original = window.document.project

    def cancel(dialog):
        controls.type(dialog.name_editor, "discarded")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Cancel)

    controls.modals(lambda: controls.named_click(window, "addLayerButton"), [cancel])
    assert window.document.project is original
    controls.record("layerDialogButtons:Cancel", "type layer name then Cancel", "project remains identical")

    def transition(dialog):
        controls.type(dialog.name_editor, "transition")
        controls.type(dialog.formula_editor, "SiO2")
        controls.type(dialog.density_editor, "2.2")
        controls.type(dialog.thickness_editor, "5")
        controls.type(dialog.roughness_editor, "0.3")
        controls.click(dialog.transition_toggle)
        assert not dialog.roughness_editor.isEnabled() and dialog.roughness_editor.value() == 0
        controls.click(dialog.transition_toggle)
        assert dialog.roughness_editor.isEnabled()
        controls.click(dialog.transition_toggle)
        controls.type(dialog.microslab_editor, "0.1")
        for row, fields in enumerate((("erf", "2", "1"), ("linear", "2", "0.5"))):
            for column, value in enumerate(fields):
                controls.table_text(dialog.branch_table, row, column, value)
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(lambda: controls.named_click(window, "addLayerButton"), [transition])
    layer = window.document.project.datasets[0].structure.components[-1]
    assert layer.roughness_a == 0
    assert tuple(branch.weight for branch in layer.transition.branches) == (0.5, 0.5)
    assert tuple(branch.thickness_a for branch in layer.transition.branches) == (10, 5)
    assert layer.transition.microslab_max_a == 1
    controls.record(
        ("layerTransitionToggle", "layerMicroslabInput", "layerTransitionTable"),
        "toggle off/on; type microslab; double-click/edit six table cells",
        "transition normalizes weights and replaces roughness",
    )


def _periodic_control_ids(kind):
    identities = [
        "addPeriodicBlockButton",
        "periodicNameInput",
        "periodicRepeatsInput",
        "periodicLayerTable",
        "periodicDriftKind",
        "periodicDialogButtons:Ok",
    ]
    if kind is not None:
        identities += ["periodicDriftTarget", "periodicDriftAmount"]
    if kind == "sine":
        identities += ["periodicDriftPeriod", "periodicDriftPhase"]
    if kind == "random":
        identities += ["periodicDriftSeed"]
    return identities


@pytest.mark.parametrize("kind", (None, "linear", "sine", "random"))
def test_periodic_dialog_all_drift_modes(controls, tmp_path, kind):
    window = controls.window(setup_project(tmp_path))
    _structure_step(controls, window)

    def configure(dialog):
        controls.type(dialog.name_editor, "Mo-Si")
        controls.type(dialog.repeats_editor, "3")
        for row, fields in enumerate((("Mo", "Mo", "10.28", "2.5", "0.1"), ("Si", "Si", "2.329", "4", "0.1"))):
            for column, value in enumerate(fields):
                controls.table_text(dialog.table, row, column, value)
        controls.select(dialog.drift_kind, dialog.drift_kind.findData(kind))
        if kind is not None:
            controls.select(dialog.drift_target, dialog.drift_target.findData("roughness"))
            assert dialog.drift_target.currentData() == "roughness"
            controls.select(dialog.drift_target, dialog.drift_target.findData("thickness"))
            controls.type(dialog.drift_amount, "0.01")
        if kind == "sine":
            controls.type(dialog.drift_period, "4")
            controls.type(dialog.drift_phase, "0.2")
        if kind == "random":
            controls.type(dialog.drift_seed, "123")
            assert "用户编辑 = 123" in dialog.drift_seed_source.text()
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(lambda: controls.named_click(window, "addPeriodicBlockButton"), [configure])
    block = window.document.project.datasets[0].structure.components[-1]
    assert isinstance(block, api.PeriodicBlock) and block.repeats == 3
    assert [layer.thickness_a for layer in block.layers] == [25, 40]
    assert (None if block.drift is None else block.drift.kind) == kind
    if kind == "sine":
        assert block.drift.period == 4 and block.drift.phase == pytest.approx(0.2)
    if kind == "random":
        assert block.drift.seed == 123
    identities = _periodic_control_ids(kind)
    controls.record(
        identities,
        f"Add periodic block; keyboard-fill cells; choose drift {kind}; click OK",
        "ordered cell and exact drift parameters commit to project",
    )


def test_selected_layer_inline_material_and_geometry_inputs(controls, tmp_path):
    window = controls.window(setup_project(tmp_path))
    _structure_step(controls, window)
    _select_component(controls, window, 0)
    controls.record("structureTree:selection", "click actual layer row", "selected layer inspector binds that layer")
    for name, text, attribute, expected in (
        ("selectedLayerThicknessInput", "5.2", "thickness_a", 52.0),
        ("selectedLayerRoughnessInput", "0.4", "roughness_a", 4.0),
    ):
        controls.type(controls.widget(window, name), text)
        assert getattr(window.document.project.datasets[0].structure.components[0], attribute) == pytest.approx(
            expected
        )
        controls.record(
            name, "Ctrl/Meta+A, type number, Tab", "selected layer geometry commits with nm-to-A conversion"
        )
    controls.type(controls.widget(window, "selectedLayerDensityInput"), "2.3")
    assert window.document.project.datasets[0].structure.components[0].material.bulk_density_g_cm3 == pytest.approx(2.3)
    controls.type(controls.widget(window, "selectedLayerFormulaInput"), "Al2O3")
    assert window.document.project.datasets[0].structure.components[0].material.formula == "Al2O3"
    controls.record(
        ("selectedLayerDensityInput", "selectedLayerFormulaInput"),
        "keyboard replacement and Tab",
        "material formula and density commit independently",
    )
    for formula in ("Si", "SiO2", "Si3N4", "TaN", "Zr"):

        def choose(menu, formula=formula):
            action = next(action for action in menu.actions() if action.text() == formula)
            controls.qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())

        controls.popup(lambda: controls.named_click(window, "selectedLayerMaterialMenuButton"), choose)
        assert window.document.project.datasets[0].structure.components[0].material.formula == formula
        controls.record(
            f"materialMenu:{formula}",
            "open material popup and click named item",
            "selected layer formula changes to chosen material",
        )
    controls.record(
        "selectedLayerMaterialMenuButton",
        "open actual popup five times",
        "all five materials commit without losing row selection",
    )


def test_structure_context_menu_backing_and_component_reorder(controls, tmp_path):
    window = controls.window(setup_project(tmp_path, layers=2))
    _structure_step(controls, window)
    tree = _select_component(controls, window, 1)
    _row_menu(controls, tree, "moveComponentUpAction")
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film1", "film0"]
    _select_component(controls, window, 0)
    _row_menu(controls, tree, "moveComponentDownAction")
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film0", "film1"]
    controls.record(
        ("structureTree:contextMenu", "moveComponentUpAction", "moveComponentDownAction"),
        "QWindow right-click then pointer menu items",
        "two layers exchange order and restore",
    )

    def configure(dialog):
        controls.type(dialog.formula_editor, "Al2O3")
        controls.type(dialog.density_editor, "3.95")
        controls.type(dialog.roughness_editor, "0.5")
        controls.buttons(dialog, QDialogButtonBox.StandardButton.Ok)

    controls.modals(lambda: _row_menu(controls, tree, "editBackingAction"), [configure])
    structure = window.document.project.datasets[0].structure
    assert structure.backing == api.MaterialSpec("Al2O3", "Al2O3", 3.95)
    assert structure.backing_roughness_a == 5
    controls.record(
        (
            "editBackingAction",
            "backingFormulaInput",
            "backingDensityInput",
            "backingRoughnessInput",
            "backingDialogButtons:Ok",
        ),
        "context menu Edit backing; keyboard fields; OK",
        "backing material and roughness commit",
    )
    _select_component(controls, window, 0)
    _row_menu(controls, tree, "removeComponentAction")
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film1"]
    controls.record("removeComponentAction", "context-menu Delete", "selected component alone is removed")


def test_structure_shortcuts_move_and_remove_selected_component(controls, tmp_path):
    window = controls.window(setup_project(tmp_path, layers=2))
    _structure_step(controls, window)
    tree = _select_component(controls, window, 1)
    QTest.keySequence(tree, QKeySequence("Ctrl+Up"))
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film1", "film0"]
    _select_component(controls, window, 0)
    QTest.keySequence(tree, QKeySequence("Ctrl+Down"))
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film0", "film1"]
    _select_component(controls, window, 1)
    QTest.keySequence(tree, window.structure_panel.editor.remove_action.shortcut())
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film0"]
    controls.record(
        ("structureShortcut:Ctrl+Up", "structureShortcut:Ctrl+Down", "structureShortcut:Delete"),
        "send actual shortcuts with tree focus",
        "reorder and delete affect selected component only",
    )


@pytest.mark.parametrize("accept", (True, False))
def test_structure_oxide_accept_and_refuse_are_real_actions(controls, tmp_path, accept):
    window = controls.window(setup_project(tmp_path, layers=0))
    controls.named_click(window, "initializeStructureButton")
    _structure_step(controls, window)
    if accept:
        controls.named_click(window, "oxideSuggestionButton")
    else:
        tree = window.structure_panel.editor.tree
        item = tree.topLevelItem(tree.topLevelItemCount() - 1)
        controls.qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(item).center())
        _row_menu(controls, tree, "oxideSuggestionRefuseAction")
    dataset = window.document.project.datasets[0]
    assert dataset.oxide_decisions[-1].accepted is accept
    assert len(dataset.structure.components) == int(accept)
    controls.record(
        "oxideSuggestionButton" if accept else "oxideSuggestionRefuseAction",
        "click suggestion or right-click/refuse",
        "oxide decision persists; only acceptance inserts a layer",
    )


def test_structure_shortcut_reorder_keeps_same_component_selected(controls, tmp_path):
    window = controls.window(setup_project(tmp_path, layers=2))
    _structure_step(controls, window)
    tree = _select_component(controls, window, 1)
    QTest.keySequence(tree, QKeySequence("Ctrl+Up"))
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film1", "film0"]
    assert tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == 0, (
        "reordering must keep film1 selected, not the old numerical row"
    )
    QTest.keySequence(tree, QKeySequence("Ctrl+Down"))
    assert [layer.name for layer in window.document.project.datasets[0].structure.components] == ["film0", "film1"]
