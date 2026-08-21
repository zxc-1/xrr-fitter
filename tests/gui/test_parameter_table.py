from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget

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
        api.InstrumentSpec(instrument_id="parameter-gui"),
    )
    structure = api.StructureSpec(
        AIR,
        (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),),
        SI,
    )
    return api.set_structure(project, "sample", structure)


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    return panel


def test_parameter_table_shows_required_fields(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")

    assert table is not None
    assert [table.horizontalHeaderItem(index).text() for index in range(table.columnCount())] == [
        "参数",
        "初值",
        "下限",
        "上限",
        "单位",
        "锁定",
        "先验",
    ]
    assert "component.0.thickness_a" in panel.row_names
    assert "instrument.scale" in panel.row_names
    assert all(not definition.expert_only for definition in panel.visible_definitions)


def test_length_parameters_display_nm_but_emit_angstrom_settings(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    name = "component.0.thickness_a"
    displayed = panel.display_values(name)

    panel.set_display_parameter(
        name,
        initial=4.5,
        lower=2.0,
        upper=10.0,
        locked=False,
    )

    setting = panel.document.project.datasets[0].parameter_settings[0]
    assert displayed[0] == pytest.approx(4.0)
    assert panel.display_unit(name) == "nm"
    assert setting == api.ParameterSetting(name, 45.0, 20.0, 100.0)


def test_user_edit_in_parameter_table_commits_display_value_and_lock(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    name = "component.0.thickness_a"
    row = panel.row_names.index(name)

    table.item(row, 1).setText("4.5")
    table.item(row, 5).setCheckState(Qt.CheckState.Checked)

    setting = next(value for value in panel.document.project.datasets[0].parameter_settings if value.name == name)
    assert setting.initial == pytest.approx(45.0)
    assert setting.locked is True


def test_parameter_commit_routes_only_through_set_parameter_settings(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    original = panel.document.project
    definition = next(item for item in panel.definitions if item.name == "instrument.scale")
    setting = api.ParameterSetting(
        definition.name,
        1.5,
        definition.lower,
        definition.upper,
    )
    updated = api.set_parameter_settings(original, "sample", (setting,))
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "set_parameter_settings",
        lambda project, dataset_id, settings: (
            calls.append((project, dataset_id, settings)),
            updated,
        )[1],
    )

    panel.set_parameter(
        definition.name,
        initial=setting.initial,
        lower=setting.lower,
        upper=setting.upper,
        locked=False,
    )

    assert calls == [(original, "sample", (setting,))]
    assert panel.document.project is updated


def test_parameter_failure_preserves_project_table_and_signal(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, tmp_path)
    name = "instrument.scale"
    before = panel.document.project
    before_values = panel.display_values(name)
    events: list[object] = []
    panel.settings_changed.connect(events.append)
    monkeypatch.setattr(
        api,
        "set_parameter_settings",
        lambda *_args: (_ for _ in ()).throw(ValueError("settings rejected")),
    )

    with pytest.raises(ValueError, match="settings rejected"):
        panel.set_parameter(name, initial=2.0, lower=0.1, upper=10.0, locked=False)

    assert panel.document.project is before
    assert panel.display_values(name) == before_values
    assert events == []


def test_parameter_refresh_and_noop_commit_preserve_exact_setting_bytes(
    qtbot,
    tmp_path,
) -> None:
    panel = _panel(qtbot, tmp_path)
    definition = next(item for item in panel.definitions if item.name == "instrument.scale")
    panel.set_parameter(
        definition.name,
        initial=definition.initial,
        lower=definition.lower,
        upper=definition.upper,
        locked=definition.locked,
    )
    persisted = panel.document.project
    setting = persisted.datasets[0].parameter_settings[0]
    events: list[object] = []
    panel.settings_changed.connect(events.append)

    changed = panel.set_parameter(
        setting.name,
        initial=setting.initial,
        lower=setting.lower,
        upper=setting.upper,
        locked=setting.locked,
    )

    assert changed is False
    assert panel.document.project is persisted
    assert panel.document.project.datasets[0].parameter_settings == (setting,)
    assert events == []


def test_unstructured_active_dataset_clears_parameter_projection(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "empty.xy"),
        api.InstrumentSpec(),
    )
    panel = ParametersPanel(ProjectDocument(project))
    qtbot.addWidget(panel)

    assert panel.definitions == ()
    assert panel.row_names == ()


def test_bounds_problem_names_first_inconsistency() -> None:
    from xrr_fitter.gui.parameters.panel import bounds_problem

    assert bounds_problem(5.0, 1.0, 10.0) is None  # a consistent triple passes
    assert bounds_problem(5.0, 10.0, 1.0) == "下限不能大于上限"
    assert bounds_problem(0.5, 1.0, 10.0) == "初值不能小于下限"
    assert bounds_problem(15.0, 1.0, 10.0) == "初值不能大于上限"


def test_inconsistent_bound_edit_flags_cell_and_keeps_entry(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    name = "component.0.thickness_a"
    row = panel.row_names.index(name)
    project_before = panel.document.project

    # Drive the lower bound above the upper bound in display units.
    table.item(row, 2).setText("9999")

    # The edit is rejected without a project mutation, the typed value stays on
    # screen for correction, the cell is flagged, and the reason is shown.
    assert panel.document.project is project_before  # no commit happened
    assert table.item(row, 2).text() == "9999"  # entry preserved, not reverted
    assert table.item(row, 2).toolTip() == "下限不能大于上限"
    assert panel.status_label.text() == "下限不能大于上限"


def test_reset_parameter_removes_override_and_restores_default(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    name = "component.0.thickness_a"

    # Establish a user override, then confirm it is persisted.
    panel.set_display_parameter(name, initial=6.0, lower=2.0, upper=10.0, locked=True)
    assert any(value.name == name for value in panel.document.project.datasets[0].parameter_settings)

    # Resetting drops the persisted setting so the declared default reasserts.
    assert panel.reset_parameter(name) is True
    assert not any(value.name == name for value in panel.document.project.datasets[0].parameter_settings)
    # Resetting an already-default parameter is a no-op, not an error.
    assert panel.reset_parameter(name) is False


def _definition(name: str, **changes) -> api.ParameterDefinition:
    base = {
        "name": name,
        "display_name": name,
        "unit": "Å",
        "category": "structure",
        "initial": 40.0,
        "lower": 1.0,
        "upper": 100.0,
        "transform": "linear",
        "locked": False,
    }
    base.update(changes)
    return api.ParameterDefinition(**base)


def _table(qtbot, *definitions: api.ParameterDefinition):
    from xrr_fitter.gui.parameters.table import ParameterTable

    table = ParameterTable()
    qtbot.addWidget(table)
    table.load(definitions, expert_mode=True)
    return table


def test_empty_prior_column_yields_its_surplus_to_the_name_column(qtbot) -> None:
    # Most projects configure no priors, leaving column 6 blank.  A blank column
    # that still absorbs surplus width steals it from the name column, which then
    # elides the parameter names down to an unreadable "幂律背..." stub -- and the
    # two distinct power-law background parameters then render identically.
    definitions = (
        _definition("instrument.background.power_law_amplitude", display_name="幂律背景幅值 B₂"),
        _definition("instrument.background.power_law_exponent", display_name="幂律背景指数 p"),
        _definition("instrument.resolution.relative_sigma", display_name="相对分辨率 σq/q"),
    )
    table = _table(qtbot, *definitions)
    table.resize(380, 200)  # the width parametersDock actually gets on screen

    # An empty column has no content to show, so it must not hold a share of the
    # width comparable to the names it is starving.
    assert table.columnWidth(6) < table.columnWidth(0) / 2
    assert table.columnWidth(0) >= table.sizeHintForColumn(0)


def test_populated_prior_column_does_not_starve_the_name_column(qtbot) -> None:
    # A soft_range summary is long enough to claim the whole dock width on its
    # own.  The parameter name identifies the row and cannot be traded away for
    # a bound summary, so the prior column gives up its overflow to a tooltip.
    definitions = (
        _definition(
            "instrument.background.power_law_amplitude",
            display_name="幂律背景幅值 B₂",
            prior=api.PriorSpec("soft_range", (1.0, 9.0, 0.5)),
        ),
        _definition(
            "instrument.background.power_law_exponent",
            display_name="幂律背景指数 p",
            prior=api.PriorSpec("normal", (2.0, 0.3)),
        ),
    )
    table = _table(qtbot, *definitions)
    table.resize(380, 200)

    assert table.columnWidth(0) >= table.sizeHintForColumn(0)
    # Nothing is lost: the full summary stays reachable on the cell itself.
    assert "soft_range" in table.item(0, 6).toolTip()


def test_prior_column_header_and_readonly(qtbot) -> None:
    table = _table(qtbot, _definition("component.0.thickness_a"))

    assert table.horizontalHeaderItem(6).text() == "先验"
    prior_cell = table.item(0, 6)
    assert prior_cell is not None
    assert not (prior_cell.flags() & Qt.ItemFlag.ItemIsEditable)


def test_prior_column_renders_summary(qtbot) -> None:
    with_prior = _definition(
        "instrument.scale",
        unit="1",
        initial=1.0,
        lower=0.5,
        upper=1.5,
        prior=api.PriorSpec("normal", (1.0, 0.2)),
    )
    without_prior = _definition("instrument.scale.other", unit="1", initial=1.0, lower=0.5, upper=1.5)
    table = _table(qtbot, with_prior, without_prior)

    assert "normal" in table.item(0, 6).text()
    assert table.item(1, 6).text() == ""


def test_prior_column_respects_nm_toggle(qtbot) -> None:
    # A length parameter shows its Å-space center (40.0) as nm (4.0) so the
    # prior summary agrees with the initial/lower/upper columns above it.
    definition = _definition(
        "component.0.thickness_a",
        prior=api.PriorSpec("normal", (40.0, 5.0)),
    )
    table = _table(qtbot, definition)

    text = table.item(0, 6).text()
    assert "4" in text
    assert "40" not in text


def test_roughness_fraction_prior_summary_remains_an_unscaled_fraction(qtbot) -> None:
    definition = _definition(
        "component.0.roughness_a",
        initial=3.0,
        lower=0.0,
        upper=50.0,
        transform="roughness_fraction",
        prior=api.PriorSpec("normal", (0.5, 0.1)),
    )
    table = _table(qtbot, definition)

    assert table.item(0, 6).text() == "normal(μ=0.5, σ=0.1)"


def test_constraint_driven_row_locks_value_columns_and_check(qtbot) -> None:
    driven = _definition("component.0.thickness_a", constrained=True)
    table = _table(qtbot, driven)

    # 初值/下限/上限 join the always-read-only display-name/unit columns: a
    # constraint-driven value is computed, so the user may not type over it.
    for column in (0, 1, 2, 3, 4):
        assert not (table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable)
    # The lock checkbox degrades to a read-only indicator: still visible and
    # selectable, no longer user-checkable, and annotated with the reason.
    locked = table.item(0, 5)
    assert locked.flags() & Qt.ItemFlag.ItemIsEnabled
    assert locked.flags() & Qt.ItemFlag.ItemIsSelectable
    assert not (locked.flags() & Qt.ItemFlag.ItemIsUserCheckable)
    assert locked.checkState() == Qt.CheckState.Checked
    assert "约束" in locked.toolTip()


def test_constraint_driven_row_disables_mutating_context_actions(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    target = "component.0.density_scale"
    rule = api.ConstraintRule(
        api.ParameterReference("sample", target),
        api.ConstraintNode(
            "mul",
            operands=(
                api.ConstraintNode(
                    "ref",
                    reference=api.ParameterReference(
                        "sample",
                        "component.0.thickness_a",
                    ),
                ),
                api.ConstraintNode("const", value=0.01),
            ),
        ),
    )
    panel.apply_constraint_rules((rule,))

    menu = panel._row_context_menu(target)
    actions = {action.objectName(): action for action in menu.actions()}

    for name in ("resetParameterAction", "editPriorAction", "clearPriorAction"):
        assert actions[name].isEnabled() is False
        assert "约束" in actions[name].toolTip()


def test_unconstrained_row_matches_head_editability(qtbot) -> None:
    plain = _definition("component.0.thickness_a")  # constrained defaults to False

    table = _table(qtbot, plain)

    # Value columns stay editable exactly as before the constraint feature.
    for column in (1, 2, 3):
        assert table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable
    # Display-name and unit columns remain read-only, unchanged from HEAD.
    for column in (0, 4):
        assert not (table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable)
    # The lock cell keeps its interactive checkbox and carries no tooltip.
    locked = table.item(0, 5)
    assert locked.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert locked.toolTip() == ""


def test_computed_bound_shows_few_digits_and_keeps_full_value_reachable(qtbot) -> None:
    """A twelve-digit bound used to stretch the column past the dock width."""
    definition = _definition("component.0.roughness_a", lower=0.37167741227456, upper=266.0914692)

    table = _table(qtbot, definition)

    # Å→nm scaling puts these at 0.037167741227456 and 26.60914692.
    assert table.item(0, 2).text() == "0.0371677"
    assert table.item(0, 3).text() == "26.6091"
    # Rounding hides digits, so the cell carries the exact value in its tooltip.
    assert table.item(0, 2).toolTip() == repr(0.037167741227456)


def test_editing_one_cell_persists_untouched_bounds_at_full_precision(
    qtbot,
    tmp_path,
) -> None:
    """The commit path reads the whole row, so rounding must not reach storage."""
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    name = "component.0.roughness_a"

    # Give the row a bound that does not survive six significant digits.
    panel.set_display_parameter(name, initial=0.3, lower=0.037167741227456, upper=2.0, locked=False)
    row = panel.row_names.index(name)

    # Touch only the initial value; the two bounds are left exactly as rendered.
    table.item(row, 1).setText("0.4")

    setting = next(value for value in panel.document.project.datasets[0].parameter_settings if value.name == name)
    assert setting.initial == pytest.approx(4.0)
    # The untouched bound keeps every digit instead of the rounded 0.0371677.
    assert setting.lower == pytest.approx(0.37167741227456, rel=0, abs=1e-15)


def _grouped_project(tmp_path):
    """A structure with two layers, so the table holds more than one group."""
    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "grouped.xy"),
        api.InstrumentSpec(instrument_id="parameter-groups"),
    )
    structure = api.StructureSpec(
        AIR,
        (
            api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),
            api.LayerSpec("cap", SIO2, 12.0, roughness_a=2.0),
        ),
        SI,
    )
    return api.set_structure(project, "grouped", structure)


def _grouped_panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_grouped_project(tmp_path)))
    qtbot.addWidget(panel)
    return panel


def _group_rows(table) -> dict[int, str]:
    """Rows that caption a group rather than declaring a parameter."""
    return {
        row: table.item(row, 0).text()
        for row in range(table.rowCount())
        if table.item(row, 0) is not None and table.item(row, 0).data(Qt.ItemDataRole.UserRole) is None
    }


def test_every_parameter_row_sits_under_a_caption_naming_its_owner(qtbot, tmp_path) -> None:
    """Ten of seventeen rows are instrument parameters; nothing on screen says so.

    The declarations already arrive clustered -- film's thickness/density/roughness
    are adjacent, then cap's, then the backing, then ten instrument rows -- but the
    table renders all seventeen as identical adjacent rows, so the clustering is
    invisible and a user cannot tell which layer 厚度 belongs to without opening
    the tooltip on every row.
    """
    panel = _grouped_panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    captions = _group_rows(table)

    assert captions, "no group captions rendered"
    # Every parameter row must be reachable by scanning up to a caption, and that
    # caption must name the layer, the backing or the instrument that owns it.
    owners = {
        "component.0": "film",
        "component.1": "cap",
        "backing": "基底",
        "instrument": "仪器",
    }
    current = None
    seen: dict[str, list[str]] = {}
    for row in range(table.rowCount()):
        if row in captions:
            current = captions[row]
            continue
        name = str(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        assert current is not None, f"{name} precedes every caption"
        prefix = "component.0" if name.startswith("component.0") else name.split(".")[0]
        prefix = "component.1" if name.startswith("component.1") else prefix
        seen.setdefault(prefix, []).append(current)
    for prefix, expected in owners.items():
        assert prefix in seen, prefix
        assert all(expected in caption for caption in seen[prefix]), (prefix, seen[prefix])


def test_group_caption_is_not_mistaken_for_a_parameter(qtbot, tmp_path) -> None:
    """A caption carries no value, so it must not read or write as a row."""
    panel = _grouped_panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    captions = _group_rows(table)

    assert captions, "no group captions rendered"
    for row in captions:
        caption = table.item(row, 0)
        assert not (caption.flags() & Qt.ItemFlag.ItemIsEditable)
        assert not (caption.flags() & Qt.ItemFlag.ItemIsSelectable)
        # The numeric columns hold nothing to edit or commit.
        for column in (1, 2, 3, 5):
            assert table.item(row, column) is None, (row, column)
    # Captions stay out of the declaration projection entirely.
    assert len(panel.visible_definitions) == table.rowCount() - len(captions)
    assert all(name != "" for name in panel.row_names if name)


def test_row_names_stay_aligned_with_physical_rows_across_captions(qtbot, tmp_path) -> None:
    """Callers locate a row by name, so the index must survive the captions."""
    panel = _grouped_panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")

    # Without captions the index is trivially aligned, so the guarantee only
    # means something once rows have been inserted between the parameters.
    assert _group_rows(table), "no group captions rendered"
    for name in ("component.0.thickness_a", "component.1.roughness_a", "instrument.scale"):
        row = panel.row_names.index(name)
        assert str(table.item(row, 0).data(Qt.ItemDataRole.UserRole)) == name


def test_single_group_is_left_uncaptioned(qtbot) -> None:
    """A caption naming the only group present separates nothing."""
    table = _table(
        qtbot,
        _definition("instrument.scale", category="instrument"),
        _definition("instrument.background", category="instrument"),
    )

    assert _group_rows(table) == {}
    assert table.rowCount() == 2
