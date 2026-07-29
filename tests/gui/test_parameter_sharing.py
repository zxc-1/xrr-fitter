from __future__ import annotations

from pathlib import Path

import pytest

import xrr_fitter.api as api


AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path, scale: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}"
            for index in range(64)
        )
        + "\n",
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
