from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
)

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path, scale: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _project(tmp_path):
    project = api.new_project()
    for name, scale in (("first", 1000.0), ("second", 800.0)):
        project = api.add_dataset(
            project,
            _write_curve(tmp_path / f"{name}.xy", scale),
            api.InstrumentSpec(instrument_id="shared"),
        )
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
        SI,
    )
    for dataset_id in ("first", "second"):
        project = api.set_structure(project, dataset_id, structure)
    return api.set_batch_mode(project, "joint")


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    return panel


def _rule(key="shared-thickness") -> api.SharingRule:
    return api.SharingRule(
        key,
        (
            api.ParameterReference("first", "component.0.thickness_a"),
            api.ParameterReference("second", "component.0.thickness_a"),
        ),
    )


def test_sharing_change_adds_member_validates_and_invalidates_all_affected_datasets(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    rule = _rule()
    updated = api.set_sharing_rules(original, (rule,))
    calls: list[tuple[object, ...]] = []

    def validate(project, rules):
        calls.append(("validate", project, rules))
        return tuple(rules)

    def commit(project, rules):
        calls.append(("commit", project, rules))
        return updated

    monkeypatch.setattr(api, "validate_sharing_rules", validate)
    monkeypatch.setattr(api, "set_sharing_rules", commit)

    panel.apply_sharing_rules((rule,))

    assert calls == [
        ("validate", original, (rule,)),
        ("commit", original, (rule,)),
    ]
    assert panel.document.project is updated
    assert panel.sharing_rules == (rule,)


def test_duplicate_sharing_key_error_is_visible_and_preserves_project(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    before = panel.document.project
    events: list[object] = []
    panel.sharing_changed.connect(events.append)

    with pytest.raises(ValueError, match="sharing_key values must be unique"):
        panel.apply_sharing_rules((_rule("duplicate"), _rule("duplicate")))

    assert panel.document.project is before
    assert panel.sharing_error_text() == "sharing_key values must be unique"
    assert events == []


def test_sharing_change_removes_two_member_rule_when_requested(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.apply_sharing_rules((_rule(),))

    changed = panel.remove_sharing_rule("shared-thickness")

    assert changed is True
    assert panel.document.project.sharing_rules == ()


def test_sharing_failure_keeps_rules_and_project_atomic(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    before = panel.document.project
    monkeypatch.setattr(
        api,
        "set_sharing_rules",
        lambda *_args: (_ for _ in ()).throw(ValueError("sharing rejected")),
    )

    with pytest.raises(ValueError, match="sharing rejected"):
        panel.apply_sharing_rules((_rule(),))

    assert panel.document.project is before
    assert panel.sharing_rules == ()


def test_parameter_table_offers_sharing_only_for_common_free_rows(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)

    names = panel.eligible_sharing_names(("first", "second"))

    assert "component.0.thickness_a" in names
    assert "instrument.linear_background_per_a_inv" not in names


def _constraint_rule(target_ds="first", source_ds="second") -> api.ConstraintRule:
    return api.ConstraintRule(
        api.ParameterReference(target_ds, "component.0.thickness_a"),
        api.ConstraintNode(
            "mul",
            operands=(
                api.ConstraintNode("const", value=2.0),
                api.ConstraintNode(
                    "ref",
                    reference=api.ParameterReference(source_ds, "component.0.thickness_a"),
                ),
            ),
        ),
    )


def test_constraint_change_validates_commits_and_emits_persisted_rules(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    rule = _constraint_rule()
    updated = api.set_constraint_rules(original, (rule,))
    calls: list[tuple[object, ...]] = []
    events: list[object] = []
    panel.constraints_changed.connect(events.append)

    def validate(project, rules):
        calls.append(("validate", project, rules))
        return tuple(rules)

    def commit(project, rules):
        calls.append(("commit", project, rules))
        return updated

    monkeypatch.setattr(api, "validate_constraint_rules", validate)
    monkeypatch.setattr(api, "set_constraint_rules", commit)

    panel.apply_constraint_rules((rule,))

    assert calls == [
        ("validate", original, (rule,)),
        ("commit", original, (rule,)),
    ]
    assert panel.document.project is updated
    assert panel.constraint_rules == (rule,)
    assert events == [updated.constraint_rules]


def test_constraint_tree_shows_root_leaf_payload(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    source = api.ParameterReference("second", "component.0.thickness_a")
    rule = api.ConstraintRule(
        api.ParameterReference("first", "component.0.thickness_a"),
        api.ConstraintNode("ref", reference=source),
    )

    panel.apply_constraint_rules((rule,))

    item = panel.constraint_editor.tree.topLevelItem(0)
    assert item is not None
    assert item.text(1) == "second:component.0.thickness_a"


def test_unknown_constraint_parameter_error_is_visible_and_preserves_project(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    before = panel.document.project
    events: list[object] = []
    panel.constraints_changed.connect(events.append)
    unknown = api.ConstraintRule(
        api.ParameterReference("first", "component.0.thickness_a"),
        api.ConstraintNode(
            "ref",
            reference=api.ParameterReference("first", "does.not.exist"),
        ),
    )

    with pytest.raises(ValueError, match="constraint references unknown parameter"):
        panel.apply_constraint_rules((unknown,))

    assert panel.document.project is before
    assert "does.not.exist" in panel.constraint_error_text()
    assert events == []


def test_constraint_dialog_keeps_empty_expression_error_open(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.parameters.constraints import ConstraintDialog

    panel = _panel(qtbot, tmp_path)
    targets = (api.ParameterReference("first", "component.0.thickness_a"),)
    sources = (api.ParameterReference("second", "component.0.thickness_a"),)
    dialog = ConstraintDialog(targets, sources, parent=panel)
    qtbot.addWidget(dialog)
    dialog.show()

    buttons = dialog.findChild(QDialogButtonBox, "constraintDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    error = dialog.findChild(QLabel, "constraintDialogError")
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.isVisible()
    assert error.isVisible()
    assert error.text()


def test_constraint_dialog_builds_expression_and_commits_rule(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.parameters.constraints import ConstraintDialog

    committed: list[api.ConstraintRule] = []
    targets = (api.ParameterReference("first", "component.0.thickness_a"),)
    sources = (api.ParameterReference("second", "component.0.thickness_a"),)
    dialog = ConstraintDialog(targets, sources, commit_rule=committed.append)
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.findChild(QDoubleSpinBox, "constraintConstantInput").setValue(2.0)
    qtbot.mouseClick(dialog.findChild(QPushButton, "constraintAddConstantButton"), Qt.LeftButton)
    qtbot.mouseClick(dialog.findChild(QPushButton, "constraintAddParameterButton"), Qt.LeftButton)
    dialog.findChild(QComboBox, "constraintOperatorInput").setCurrentText("mul")
    qtbot.mouseClick(dialog.findChild(QPushButton, "constraintAddOperatorButton"), Qt.LeftButton)
    buttons = dialog.findChild(QDialogButtonBox, "constraintDialogButtons")
    qtbot.mouseClick(buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.LeftButton)

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert committed == [_constraint_rule()]
    assert dialog.rule() == _constraint_rule()


def test_constraint_editor_targets_exclude_already_driven_parameters(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    panel.apply_constraint_rules((_constraint_rule(),))

    first_targets = tuple(reference.parameter_name for reference in panel.eligible_constraint_targets("first"))
    second_targets = tuple(reference.parameter_name for reference in panel.eligible_constraint_targets("second"))
    sources = tuple(reference.parameter_name for reference in panel.eligible_constraint_sources("second"))

    assert "component.0.thickness_a" not in first_targets
    assert "component.0.thickness_a" in second_targets
    assert "component.0.thickness_a" in sources


def test_constraint_editor_targets_exclude_integer_parameters(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = api.describe_parameters(panel.document.project, "first")
    integer_name = "component.0.thickness_a"
    definitions = tuple(
        replace(definition, integer=True, locked=False) if definition.name == integer_name else definition
        for definition in original
    )
    monkeypatch.setattr(api, "describe_parameters", lambda *_args: definitions)

    targets = tuple(reference.parameter_name for reference in panel.eligible_constraint_targets("first"))

    assert integer_name not in targets


def test_constraint_editor_exposes_working_add_and_delete_commands(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.parameters import constraints as module

    panel = _panel(qtbot, tmp_path)
    editor = panel.constraint_editor
    rule = _constraint_rule()
    opened: list[tuple[tuple[object, ...], tuple[object, ...]]] = []

    class AcceptedDialog:
        def __init__(self, targets, sources, *_args, **_kwargs):
            opened.append((tuple(targets), tuple(sources)))

        def exec(self):
            return QDialog.DialogCode.Accepted

        def rule(self):
            return rule

    monkeypatch.setattr(module, "ConstraintDialog", AcceptedDialog)
    qtbot.mouseClick(
        editor.findChild(QPushButton, "constraintAddRuleButton"),
        Qt.LeftButton,
    )

    assert opened
    assert panel.constraint_rules == (rule,)

    editor.tree.setCurrentItem(editor.tree.topLevelItem(0))
    qtbot.mouseClick(
        editor.findChild(QPushButton, "constraintDeleteRuleButton"),
        Qt.LeftButton,
    )

    assert panel.constraint_rules == ()


def test_constraint_add_command_validates_inside_the_dialog(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.parameters import constraints as module

    panel = _panel(qtbot, tmp_path)
    callbacks: list[object] = []

    class RejectedDialog:
        def __init__(self, *_args, commit_rule=None, **_kwargs):
            callbacks.append(commit_rule)

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(module, "ConstraintDialog", RejectedDialog)
    qtbot.mouseClick(
        panel.constraint_editor.findChild(
            QPushButton,
            "constraintAddRuleButton",
        ),
        Qt.LeftButton,
    )

    assert len(callbacks) == 1
    assert callable(callbacks[0])


def test_constraint_add_command_surfaces_validation_errors_without_mutation(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.parameters import constraints as module

    panel = _panel(qtbot, tmp_path)
    rule = _constraint_rule()
    panel.apply_constraint_rules((rule,))
    before = panel.document.project

    class DuplicateDialog:
        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def rule(self):
            return replace(
                rule,
                expression=api.ConstraintNode("const", value=80.0),
            )

    monkeypatch.setattr(module, "ConstraintDialog", DuplicateDialog)
    qtbot.mouseClick(
        panel.constraint_editor.findChild(
            QPushButton,
            "constraintAddRuleButton",
        ),
        Qt.LeftButton,
    )

    assert panel.document.project is before
    assert "unique" in panel.constraint_error_text()
