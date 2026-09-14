"""设计稿帧③「参数化 · 厚度 d」那一段：当前这个量此刻怎么被对待。

三档说的是拟合器拿它怎么办——自由地推、钉死不动、还是优先待在这一行填的区间内（越界不是
一堵墙，按 ``soft_range`` 先验渐进受罚，声明的上下限仍是硬边界）。两枚徽章说的是另外两件
同样只属于这一个量的事：它有没有先验，以及它是不是和别的数据集绑在一起。

这几样此前都散着：锁在参数表名字格的勾里，先验在表的最后一列，共享要切到另一个标签
页去看。散着的问题不是找不到，是读者没法一眼回答「我现在改的这个数，拟合时到底算什么」。
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

import xrr_fitter.api as api

pytest.importorskip("pytestqt")


def _definition(**overrides) -> api.ParameterDefinition:
    fields = {
        "name": "component.1.thickness_a",
        "display_name": "a-Si 厚度",
        "unit": "A",
        "category": "structure",
        "initial": 487.0,
        "lower": 100.0,
        "upper": 900.0,
        "transform": "linear",
        "locked": False,
    }
    fields.update(overrides)
    return api.ParameterDefinition(**fields)


def _row(qtbot):
    from xrr_fitter.gui.parameters.disposition import ParameterDisposition

    widget = ParameterDisposition()
    qtbot.addWidget(widget)
    return widget


def test_a_free_parameter_reads_as_free(qtbot) -> None:
    """没锁也没被约束驱动，就是拟合器可以随便推的那一档。"""
    widget = _row(qtbot)

    widget.show_definition(_definition(), quantity="厚度 d")

    assert widget.free_button.isChecked()
    assert widget.free_button.isEnabled()
    assert widget.fixed_button.isEnabled()


def test_a_locked_parameter_reads_as_fixed(qtbot) -> None:
    """锁了就是「固定」：拟合器不动它，但读者随时可以解锁。"""
    widget = _row(qtbot)

    widget.show_definition(_definition(locked=True), quantity="厚度 d")

    assert widget.fixed_button.isChecked()
    assert widget.fixed_button.isEnabled()
    assert widget.free_button.isEnabled()


def test_a_constraint_driven_parameter_reads_as_fixed_and_cannot_be_switched(qtbot) -> None:
    """被约束驱动的量落在「固定」上，而且三档全点不动。

    亮在「固定」是因为拟合器确实不动它：编译出来的声明 ``locked`` 为真。它不落在「仅范围」
    上——那一档现在专指「附一条 ``soft_range`` 先验、越界渐进受罚」这个处置，和「值是别处算
    出来的」是两件事，共用一个圆点会让读者以为自己填的区间在起作用。

    三档必须一起点不动：从这里出去意味着删掉那条约束，那是约束页的事，在这里点一下就悄悄
    删掉一条规则，读者不会预期。禁用之外还得有那句 tooltip，否则「固定」会被读成读者自己选的。
    """
    from xrr_fitter.gui.parameters.disposition import CONSTRAINT_DRIVEN_TOOLTIP

    widget = _row(qtbot)

    widget.show_definition(_definition(constrained=True), quantity="厚度 d")

    assert widget.fixed_button.isChecked()
    assert not widget.free_button.isEnabled()
    assert not widget.fixed_button.isEnabled()
    assert not widget.range_button.isEnabled()
    assert widget.fixed_button.toolTip() == CONSTRAINT_DRIVEN_TOOLTIP
    assert widget.range_button.toolTip() == CONSTRAINT_DRIVEN_TOOLTIP


def test_the_setting_gear_wins_over_the_declaration(qtbot) -> None:
    """传了 ``freedom`` 就按它亮，声明里的两态只是没 setting 时的回落。

    「仅范围」只存在 setting 里——声明只有 ``locked``，两态装不下它。这一段要是仍从声明读，
    切到一个「仅范围」的量上会亮成「自由」：读者看到的处置和拟合器实际用的处置不是一个。
    """
    widget = _row(qtbot)

    widget.show_definition(_definition(), quantity="厚度 d", freedom=api.ParameterFreedom.RANGE_ONLY)

    assert widget.range_button.isChecked()
    assert not widget.free_button.isChecked()
    assert widget.range_button.isEnabled()


def test_nothing_selected_leaves_the_row_disabled_and_silent(qtbot) -> None:
    """没有当前行时三档全灰、徽章全不占地方。

    留一档亮着会被读成「当前那个量是自由的」，而此刻根本没有当前那个量。
    """
    widget = _row(qtbot)
    widget.show_definition(_definition(), quantity="厚度 d")

    widget.show_definition(None, quantity=None)

    assert not widget.free_button.isEnabled()
    assert not widget.fixed_button.isEnabled()
    assert not widget.range_button.isEnabled()
    assert widget.prior_badge.isHidden()
    assert widget.sharing_badge.isHidden()


def test_the_prior_badge_names_the_uniform_default_rather_than_going_blank(qtbot) -> None:
    """没有先验也是一句话：设计稿写「先验：无（均匀）」。

    留空会被读成「先验没加载出来」；而「无」在这里有确切含义——上下限之内一律等权。
    """
    widget = _row(qtbot)

    widget.show_definition(_definition(), quantity="厚度 d")

    assert widget.prior_badge.text() == "先验：无（均匀）"
    assert widget.prior_badge.kind() == "mut"


def test_a_configured_prior_is_spelled_out_the_way_the_table_spells_it(qtbot) -> None:
    """有先验时报的是同一份摘要，和参数表最后一列逐字一致。

    两处各写一套的话，同一个先验会在卡上说一个形状、在表里说另一个。
    """
    from xrr_fitter.gui.parameters.table import prior_summary

    widget = _row(qtbot)
    definition = _definition(prior=api.PriorSpec("normal", (487.0, 20.0)))

    widget.show_definition(definition, quantity="厚度 d")

    assert widget.prior_badge.text() == f"先验：{prior_summary(definition)}"
    assert widget.prior_badge.kind() == "info"


def test_an_unshared_parameter_says_nothing_about_sharing(qtbot) -> None:
    """没共享就没有这枚徽章：一排结论里，「未共享」和「共享」一样重会喧宾夺主。"""
    widget = _row(qtbot)

    widget.show_definition(_definition(), quantity="厚度 d")

    assert widget.sharing_badge.isHidden()


def test_a_shared_parameter_counts_the_datasets_it_is_tied_across(qtbot) -> None:
    """设计稿写「共享：跨 3 集共享厚度」——数的是数据集，不是成员条目。

    联合拟合里一个数据集可能贡献多个成员，报成员数就会把「跨 3 集」说成「跨 5 项」。
    """
    widget = _row(qtbot)
    rule = api.SharingRule(
        "thickness",
        (
            api.ParameterReference("a", "component.1.thickness_a"),
            api.ParameterReference("b", "component.1.thickness_a"),
            api.ParameterReference("c", "component.1.thickness_a"),
        ),
    )

    widget.show_definition(_definition(sharing_key="thickness"), quantity="厚度 d", sharing_rules=(rule,))

    assert widget.sharing_badge.text() == "共享：跨 3 集共享厚度 d"
    assert widget.sharing_badge.kind() == "info"


def _project(tmp_path) -> api.Project:
    curve = tmp_path / "curve.xy"
    curve.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    structure = api.StructureSpec(
        api.MaterialSpec("Air", "N", 0.0012),
        (
            api.LayerSpec("SiO2", api.MaterialSpec("SiO2", "SiO2", 2.19), 34.2, roughness_a=5.1),
            api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),
        ),
        api.MaterialSpec("c-Si", "Si", 2.33),
        backing_roughness_a=3.0,
    )
    project = api.add_dataset(api.new_project(), curve, api.InstrumentSpec())
    return api.select_active_dataset(api.set_structure(project, "curve", structure), "curve")


def test_the_panel_follows_its_own_table_row(qtbot, tmp_path) -> None:
    """整机接线：在参数表里走一行，这一段说的就是那一行。

    抬头（「参数化 · 厚度 d」）和这几档必须跟同一行走。各跟各的，抬头会点着一个量、
    档位说着另一个量的事，而两处都没标注自己说的是哪一行。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    table = panel.parameter_table
    row = next(
        index
        for index in range(table.rowCount())
        if (item := table.item(index, 0)) is not None
        and item.data(Qt.ItemDataRole.UserRole) == "component.1.thickness_a"
    )

    table.setCurrentCell(row, 0)

    assert panel.disposition.free_button.isChecked()
    assert panel.disposition.prior_badge.text() == "先验：无（均匀）"


def test_choosing_fixed_locks_the_parameter_for_real(qtbot, tmp_path) -> None:
    """点「固定」要真的把这个量锁进项目，而不只是把圆点点亮。

    走的是参数表名字格那个勾原本那条提交路径——两条路各提交各的，锁上之后表里的勾会
    和这里的档位对不上。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    panel = ParametersPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)
    table = panel.parameter_table
    row = next(
        index
        for index in range(table.rowCount())
        if (item := table.item(index, 0)) is not None
        and item.data(Qt.ItemDataRole.UserRole) == "component.1.thickness_a"
    )
    table.setCurrentCell(row, 0)

    panel.disposition.fixed_button.click()

    definition = next(item for item in panel.definitions if item.name == "component.1.thickness_a")
    assert definition.locked
    assert panel.disposition.fixed_button.isChecked()
