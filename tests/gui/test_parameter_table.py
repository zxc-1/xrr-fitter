from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget

import xrr_fitter.api as api
from xrr_fitter.gui.parameters.table import CONSTRAINT_DRIVEN_TOOLTIP, PRIOR_COLUMN

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
    """Five columns, because a 340px inspector cannot hold seven.

    单位 and 锁定 were columns of their own and each cost 44px to show a 25px
    glyph, which is a tenth of the column the design gives the whole inspector.
    They now ride in the name cell exactly where the design puts them --
    ``密度 ρ <span class="lock on">`` in frame ①'s grid and ``厚度 d（nm）`` in
    frame ③'s field editor -- so the four columns that carry digits keep their
    width.
    """
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")

    assert table is not None
    assert [table.horizontalHeaderItem(index).text() for index in range(table.columnCount())] == [
        "参数",
        "初值",
        "下限",
        "上限",
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
        freedom=api.ParameterFreedom.FREE,
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
    table.item(row, 0).setCheckState(Qt.CheckState.Checked)

    setting = next(value for value in panel.document.project.datasets[0].parameter_settings if value.name == name)
    assert setting.initial == pytest.approx(45.0)
    assert setting.locked is True


def test_name_cell_check_walks_all_three_gears_and_redisplays_each_one(
    qtbot,
    tmp_path,
) -> None:
    """名字格那个勾走满三档，每一档都存成对应的 ``freedom``，刷新之后还停在原处。

    第三档没有自己的列，它就是这个勾的半选态；``ItemIsUserTristate`` 不在，点击只在两态之间
    翻，「仅范围」从参数表里根本点不出来——而这张表是唯一逐行可见的入口，读者会以为这个量只有
    开关两种状态。回显同样要成立：档位只存在 setting 里，映射不回勾的话，刚点出来的「仅范围」
    会在下一次刷新时画成「自由」，而拟合器按的仍是仅范围。
    """
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    name = "component.0.thickness_a"
    row = panel.row_names.index(name)
    assert table.item(row, 0).flags() & Qt.ItemFlag.ItemIsUserTristate

    walked = []
    # 从声明那一档（自由）出发把一圈走完，最后一步回到自由——原地设同一个值不会发信号，
    # 所以起手先点半选。
    for state in (Qt.CheckState.PartiallyChecked, Qt.CheckState.Checked, Qt.CheckState.Unchecked):
        table.item(row, 0).setCheckState(state)

        setting = next(value for value in panel.document.project.datasets[0].parameter_settings if value.name == name)
        walked.append((setting.freedom, table.item(row, 0).checkState()))

    assert walked == [
        (api.ParameterFreedom.RANGE_ONLY, Qt.CheckState.PartiallyChecked),
        (api.ParameterFreedom.FIXED, Qt.CheckState.Checked),
        (api.ParameterFreedom.FREE, Qt.CheckState.Unchecked),
    ]


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
        freedom=api.ParameterFreedom.FREE,
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
        panel.set_parameter(name, initial=2.0, lower=0.1, upper=10.0, freedom=api.ParameterFreedom.FREE)

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
        freedom=api.ParameterFreedom.from_locked(definition.locked),
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
        freedom=setting.freedom,
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
    panel.set_display_parameter(name, initial=6.0, lower=2.0, upper=10.0, freedom=api.ParameterFreedom.FIXED)
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
    # Most projects configure no priors, leaving the last column blank.  A blank
    # column that still absorbs surplus width steals it from the name column, which
    # then elides the parameter names down to an unreadable "幂律背..." stub -- and
    # the two distinct power-law background parameters then render identically.
    #
    # These three are declared dimensionless (an amplitude, an exponent and a
    # relative sigma), so their names carry no ``（unit）`` suffix -- which is what
    # the measurement below is about: the name alone, not a unit riding with it.
    definitions = (
        _definition("instrument.background.power_law_amplitude", display_name="幂律背景幅值 B₂", unit=""),
        _definition("instrument.background.power_law_exponent", display_name="幂律背景指数 p", unit=""),
        _definition("instrument.resolution.relative_sigma", display_name="相对分辨率 σq/q", unit=""),
    )
    table = _table(qtbot, *definitions)
    table.show()
    table.resize(380, 200)
    qtbot.waitUntil(lambda: table.viewport().width() > 0, timeout=1000)

    # An empty column has no content to show, so it must not hold a share of the
    # width comparable to the names it is starving.
    assert table.columnWidth(PRIOR_COLUMN) < table.columnWidth(0) / 2
    assert table.columnWidth(0) >= table.sizeHintForColumn(0)


def test_populated_prior_column_does_not_starve_the_name_column(qtbot) -> None:
    # A soft_range summary is long enough to claim the whole dock width on its
    # own.  The parameter name identifies the row and cannot be traded away for
    # a bound summary, so the prior column gives up its overflow to a tooltip.
    definitions = (
        _definition(
            "instrument.background.power_law_amplitude",
            display_name="幂律背景幅值 B₂",
            unit="",
            prior=api.PriorSpec("soft_range", (1.0, 9.0, 0.5)),
        ),
        _definition(
            "instrument.background.power_law_exponent",
            display_name="幂律背景指数 p",
            unit="",
            prior=api.PriorSpec("normal", (2.0, 0.3)),
        ),
    )
    table = _table(qtbot, *definitions)
    table.show()
    table.resize(380, 200)
    qtbot.waitUntil(lambda: table.viewport().width() > 0, timeout=1000)

    assert table.columnWidth(0) >= table.sizeHintForColumn(0)
    # Nothing is lost: the full summary stays reachable on the cell itself.
    assert "soft_range" in table.item(0, PRIOR_COLUMN).toolTip()


def test_prior_column_header_and_readonly(qtbot) -> None:
    table = _table(qtbot, _definition("component.0.thickness_a"))

    assert table.horizontalHeaderItem(PRIOR_COLUMN).text() == "先验"
    prior_cell = table.item(0, PRIOR_COLUMN)
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

    assert "normal" in table.item(0, PRIOR_COLUMN).text()
    assert table.item(1, PRIOR_COLUMN).text() == ""


def test_prior_column_respects_nm_toggle(qtbot) -> None:
    # A length parameter shows its Å-space center (40.0) as nm (4.0) so the
    # prior summary agrees with the initial/lower/upper columns above it.
    definition = _definition(
        "component.0.thickness_a",
        prior=api.PriorSpec("normal", (40.0, 5.0)),
    )
    table = _table(qtbot, definition)

    text = table.item(0, PRIOR_COLUMN).text()
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

    assert table.item(0, PRIOR_COLUMN).text() == "normal(μ=0.5, σ=0.1)"


def test_constraint_driven_row_locks_value_columns_and_check(qtbot) -> None:
    driven = _definition("component.0.thickness_a", constrained=True)
    table = _table(qtbot, driven)

    # 初值/下限/上限 join the always-read-only name and 先验 columns: a
    # constraint-driven value is computed, so the user may not type over it.
    for column in (0, 1, 2, 3, PRIOR_COLUMN):
        assert not (table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable)
    # The lock degrades to a read-only indicator: still visible and selectable,
    # no longer user-checkable, and annotated with the reason.  It lives in the
    # name cell now, so those are the flags of the cell that also carries the
    # parameter's name.
    locked = table.item(0, 0)
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
    # The name and 先验 columns remain read-only, unchanged from HEAD.
    for column in (0, PRIOR_COLUMN):
        assert not (table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable)
    # The name cell keeps its interactive lock checkbox, and says nothing about a
    # constraint -- it still carries the identifying tooltip every name cell has.
    locked = table.item(0, 0)
    assert locked.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert CONSTRAINT_DRIVEN_TOOLTIP not in locked.toolTip()


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
    panel.set_display_parameter(
        name, initial=0.3, lower=0.037167741227456, upper=2.0, freedom=api.ParameterFreedom.FREE
    )
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
        # The lock shares column 0 with the caption now, so a caption has to leave
        # CheckStateRole unset: setting it to Unchecked would draw an empty box
        # beside 表面氧化层 · SiO₂ and invite a click that locks nothing.
        assert caption.data(Qt.ItemDataRole.CheckStateRole) is None, row
        # The numeric columns hold nothing to edit or commit.
        for column in (1, 2, 3, PRIOR_COLUMN):
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


def test_the_parameter_grid_has_no_row_number_gutter(qtbot, tmp_path) -> None:
    """``<table class="grid">`` numbers nothing down its left edge.

    Qt shows a vertical header by default, and here it measured 35px of a 213px
    viewport -- a sixth of the table spent restating an ordinal the design does not
    draw and that no other surface refers to.  Rows are addressed by the parameter
    name in column 0, which is the same identifier the commit path reads.
    """
    panel = _panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")

    assert table.verticalHeader().isVisible() is False


def test_the_lock_is_a_glyph_in_the_name_cell_not_a_column_of_its_own(qtbot) -> None:
    """Frame ① draws ``<td>密度 ρ <span class="lock on"></span></td>``.

    The lock was a 44px column carrying one checkbox, which is the widest thing on
    the table per pixel of information.  In the design it is a glyph inside the
    name cell, and a table item's own check indicator is exactly that: it sits at
    the head of the cell, before the text, and is clicked in place.
    """
    unlocked = _definition("component.0.thickness_a")
    locked = _definition("component.0.roughness_a", locked=True)
    table = _table(qtbot, unlocked, locked)

    assert "锁定" not in [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    assert table.item(0, 0).checkState() == Qt.CheckState.Unchecked
    assert table.item(1, 0).checkState() == Qt.CheckState.Checked
    assert table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable
    # A lock is toggled by clicking the box, never by typing into the name.
    assert not (table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable)


def test_the_unit_rides_with_the_quantity_it_measures(qtbot) -> None:
    """Frame ③ writes ``<label>厚度 d（nm）</label>``, not a 单位 column.

    A unit column held at most four characters and cost 44px to do it; the design
    never gives the unit a column of its own, in either the read-only grid or the
    field editor.  Fullwidth parens are the design's own punctuation.

    A dimensionless declaration gets no parenthetical at all: frame ① renders its
    unit cell as ``—``, and ``（）`` around nothing would read as a missing value
    rather than as an absent dimension.
    """
    length = _definition("component.0.thickness_a")  # declared in Å, displayed in nm
    angle = _definition("instrument.offset", unit="°")
    dimensionless = _definition("instrument.scale", unit="")
    table = _table(qtbot, length, angle, dimensionless)
    # Two groups means captions, so physical rows and declarations no longer line
    # up; row_names is what keeps an index honest.
    rows = {name: index for index, name in enumerate(table.row_names)}

    assert "单位" not in [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    assert table.item(rows["component.0.thickness_a"], 0).text().endswith("（nm）")
    assert table.item(rows["instrument.offset"], 0).text().endswith("（°）")
    assert "（" not in table.item(rows["instrument.scale"], 0).text()


def test_a_captioned_row_does_not_repeat_its_owners_name(qtbot, tmp_path) -> None:
    """Frame ①'s grouprow carries the owner; its rows carry only the quantity.

    ``<tr class="grouprow"><td colspan="4">表面氧化层 · SiO₂</td></tr>`` is followed
    by ``<td>厚度 d</td>``, not by ``<td>表面氧化层 厚度 d</td>``.  The declarations
    arrive prefixed -- a layer named film contributes "film 厚度" -- so once the
    caption above says film, the prefix is spent width restating it on every row.

    基底那一行不是靠剥前缀短下来的：声明写的是「基底连接界面粗糙度」，一个整词，没有前缀
    可剥（剥「基底」二字会剩下「连接界面粗糙度」）。它读成「粗糙度 σ」靠的是整名改写
    （``grouping.ROW_ALIASES``），因为上一行的分组行已经说明这是哪一处界面。全名仍在悬停
    提示里，改写没有把「哪一处」丢掉。
    """
    panel = _grouped_panel(qtbot, tmp_path)
    table = panel.findChild(QTableWidget, "parameterTable")
    captions = _group_rows(table)

    assert "film" in captions.values(), captions
    thickness = panel.row_names.index("component.0.thickness_a")
    text = table.item(thickness, 0).text()
    assert not text.startswith("film"), text
    assert "厚度" in text
    # Nothing is lost: the tooltip still identifies the row in full.
    assert "film" in table.item(thickness, 0).toolTip()

    backing = panel.row_names.index("backing.roughness_a")
    assert table.item(backing, 0).text().startswith("粗糙度 σ"), table.item(backing, 0).text()
    assert "基底连接界面粗糙度" in table.item(backing, 0).toolTip()


def test_an_uncaptioned_table_keeps_its_names_whole(qtbot) -> None:
    """With one group there is no caption to carry the owner, so nothing is stripped.

    ``_row_layout`` emits no captions for a single group, and a row whose owner is
    named nowhere above it has to name it itself.
    """
    table = _table(
        qtbot,
        _definition("component.0.thickness_a", display_name="film 厚度"),
        _definition("component.0.roughness_a", display_name="film 粗糙度"),
    )

    assert _group_rows(table) == {}
    assert table.item(0, 0).text() == "film 厚度 d（nm）"
    assert table.item(1, 0).text() == "film 粗糙度 σ（nm）"


def test_initial_column_wears_a_value_position_delegate(qtbot) -> None:
    """The 初值 cell carries a faint fill bar, so a railed value shows at a glance.

    Every other column stays plain: only the initial value is placed against its
    own bounds, and a bar drawn behind a bound column would have no bounds of its
    own to sit between.
    """
    from xrr_fitter.gui.parameters.table import ValuePositionDelegate

    table = _table(qtbot, _definition("component.0.thickness_a"))

    assert isinstance(table.itemDelegateForColumn(1), ValuePositionDelegate)
    assert not isinstance(table.itemDelegateForColumn(2), ValuePositionDelegate)
    assert not isinstance(table.itemDelegateForColumn(3), ValuePositionDelegate)


def test_value_position_bar_places_the_value_between_its_bounds(qtbot) -> None:
    """The fill fraction is (initial - lower) / (upper - lower)."""
    from xrr_fitter.gui.parameters.table import VALUE_POSITION_ROLE

    table = _table(qtbot, _definition("instrument.scale", initial=40.0, lower=1.0, upper=100.0))

    fraction = table.item(0, 1).data(VALUE_POSITION_ROLE)
    assert fraction == pytest.approx((40.0 - 1.0) / (100.0 - 1.0))


def test_value_position_bar_flags_a_value_railed_against_a_bound(qtbot) -> None:
    """A value pinned to a bound reads as an empty or a full bar, not a middling one."""
    from xrr_fitter.gui.parameters.table import VALUE_POSITION_ROLE

    at_floor = _definition("instrument.scale", initial=1.0, lower=1.0, upper=100.0)
    at_ceiling = _definition("instrument.background", initial=100.0, lower=1.0, upper=100.0)
    table = _table(qtbot, at_floor, at_ceiling)

    assert table.item(0, 1).data(VALUE_POSITION_ROLE) == pytest.approx(0.0)
    assert table.item(1, 1).data(VALUE_POSITION_ROLE) == pytest.approx(1.0)


def test_value_position_bar_is_absent_when_bounds_have_no_width(qtbot) -> None:
    """A pinned parameter (lower == upper) has no interval to place a value in."""
    from xrr_fitter.gui.parameters.table import VALUE_POSITION_ROLE

    pinned = _definition("instrument.scale", initial=5.0, lower=5.0, upper=5.0)
    table = _table(qtbot, pinned)

    assert table.item(0, 1).data(VALUE_POSITION_ROLE) is None


def test_value_position_bar_is_scale_invariant_across_the_nm_toggle(qtbot) -> None:
    """A length row scales all three columns to nm together, so the fraction holds.

    The bar answers "where does the value sit between its bounds", which is the
    same question in Å or nm; scaling initial, lower and upper by one factor must
    leave the fraction unchanged rather than shifting the fill.
    """
    from xrr_fitter.gui.parameters.table import VALUE_POSITION_ROLE

    length = _definition("component.0.thickness_a", initial=40.0, lower=10.0, upper=90.0)
    table = _table(qtbot, length)

    assert table.item(0, 1).data(VALUE_POSITION_ROLE) == pytest.approx((40.0 - 10.0) / (90.0 - 10.0))


def test_value_position_delegate_paints_both_bar_and_barless_rows(qtbot) -> None:
    """Rendering must survive a mix of fraction-bearing and pinned (bar-less) rows.

    The delegate reads VALUE_POSITION_ROLE and paints a fill only when it is set;
    grabbing the table forces that paint path over a normal row and a pinned row at
    once, so a regression in the override surfaces as a raised paint error here
    rather than only in a running window.
    """
    ranged = _definition("instrument.scale", initial=40.0, lower=1.0, upper=100.0)
    pinned = _definition("instrument.background", initial=5.0, lower=5.0, upper=5.0)
    table = _table(qtbot, ranged, pinned)
    table.resize(400, 200)

    pixmap = table.grab()

    assert not pixmap.isNull()


def test_a_long_name_elides_instead_of_wrapping_past_its_row(qtbot) -> None:
    """名字放不下时省略，不许换行——行高是单行的，第二行会被裁掉。

    ``QTableView`` 默认开启 wordWrap，所以「入射侧 粗糙度」在窄下来的参数列里折成
    两行，而行高仍按单行算，截图里第二行只剩半个字。同列的其他单元格却是省略号，
    两种行为混在一张表里。这里量渲染需要的行高，而不是相信默认值。
    """
    table = _table(qtbot, _definition("component.0.roughness_a", display_name="入射侧 粗糙度"))
    table.resize(340, 200)
    table.setColumnWidth(0, 60)

    assert table.sizeHintForRow(0) <= table.rowHeight(0), (table.sizeHintForRow(0), table.rowHeight(0))


def test_a_reload_keeps_the_reader_on_the_same_quantity(qtbot) -> None:
    """重填之后当前行还停在同一个量上——按名字认，不按行号认。

    改一格数、锁一个量都会让整张表重填，而重填把当前行清成 -1。跟着当前行走的东西
    （设计稿帧③ 的抬头「参数化 · 厚度 d」和它下面那三档）于是在每一次编辑之后归零：
    改完一个数想接着改它的档位，得先回表里把同一行再点一次。

    行号不能用：重填的原因往往正是行的构成变了（换数据集、切高级选项），第 n 行装的
    会是另一份声明。名字标识的是量本身，位置变了也还是它。

    这里拿两个仪器量来换位置，而不是厚度和粗糙度：``grouping.QUANTITY_ORDER`` 会把同一
    归属下的厚度排到粗糙度前面，交换入参根本换不动行号，断言就成了永真。仪器那两个量
    都不在那份次序里，同一名次下保留入参顺序，交换于是真的把行换了过来。
    """
    first = _definition("instrument.angle_offset_deg", display_name="入射角零点偏移", unit="°")
    second = _definition("instrument.footprint_spill_angle_deg", display_name="足迹满斑角 θ_fp", unit="°")
    table = _table(qtbot, first, second)
    table.setCurrentCell(1, 0)
    assert table.current_name() == "instrument.footprint_spill_angle_deg"

    table.load((second, first), expert_mode=True)

    assert table.current_name() == "instrument.footprint_spill_angle_deg"
    assert table.currentRow() == 0


def test_a_quantity_that_is_gone_leaves_no_current_row(qtbot) -> None:
    """读者点的那个量不在了，就没有当前行——而不是把光标落到顶上那一行。

    落到顶上会让跟着走的抬头改口说另一个量，而它读起来和「读者自己点了这一行」一模一样。
    """
    first = _definition("component.1.thickness_a", display_name="厚度")
    second = _definition("component.1.roughness_a", display_name="粗糙度")
    table = _table(qtbot, first, second)
    table.setCurrentCell(1, 0)

    table.load((first,), expert_mode=True)

    assert table.current_name() is None
