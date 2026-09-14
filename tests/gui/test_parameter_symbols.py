"""帧① 参数行的符号标注。

设计稿的行名是「厚度 d」「粗糙度 σ」「密度 ρ」——量名后面跟这个量的通用符号。
实测是「厚度」「入射侧粗糙度」「相对密度」，没有符号。差别不是修辞：读者手上的
文献、实验记录和另一套软件的输出全用 d/σ/ρ，行名不带符号时要读者自己在中文量名
和公式里的符号之间做一次翻译。

这不是给这三行开的特例——声明里的仪器量本来就写成「幂律背景指数 p」「相对分辨率
σq/q」「足迹满斑角 θ_fp」，「量名 + 符号」是既有约定，缺的正是设计稿标注的这三类。

规则放在 ``parameters.grouping``：设置表与结果表共用它，同一个厚度在两张表里必须
读出同一个行名。
"""

from __future__ import annotations

import pytest

import xrr_fitter.api as api
from xrr_fitter.gui.parameters.grouping import annotate_symbol, row_name


def _definition(name: str, display_name: str, unit: str = "", **changes) -> api.ParameterDefinition:
    values = {
        "name": name,
        "display_name": display_name,
        "unit": unit,
        "category": "structure",
        "initial": 1.0,
        "lower": 0.0,
        "upper": 100.0,
        "transform": "linear",
        "locked": False,
    }
    values.update(changes)
    return api.ParameterDefinition(**values)


@pytest.mark.parametrize(
    ("display_name", "expected"),
    [
        ("厚度", "厚度 d"),
        ("相对密度", "相对密度 ρ"),
        ("入射侧粗糙度", "入射侧粗糙度 σ"),
        ("顶界面粗糙度", "顶界面粗糙度 σ"),
        ("基底连接界面粗糙度", "基底连接界面粗糙度 σ"),
    ],
)
def test_the_quantities_the_design_annotates_carry_their_symbol(display_name: str, expected: str) -> None:
    """三个量各有多个写法，符号只认后缀。

    一层有「入射侧粗糙度」，周期块有「顶界面粗糙度」，基底那一处叫「基底连接界面
    粗糙度」——三者是同一个 σ。限定语留着不删：一个周期块同时排出两种粗糙度时，
    只剩「粗糙度 σ」的两行会读成重复行。
    """
    assert annotate_symbol(display_name) == expected


@pytest.mark.parametrize(
    "display_name",
    [
        "SLD 实部",
        "SLD 吸收部",
        "上侧 SLD 吸收部",
        "重复次数",
        "漂移标度",
        "微薄片上限",
        "常数背景",
        "线性背景",
        "入射角零点偏移",
        "尺度",
    ],
)
def test_a_quantity_the_design_gives_no_symbol_for_is_left_alone(display_name: str) -> None:
    """没有公认符号的量不硬造一个：造出来的符号比没有符号更难对照。"""
    assert annotate_symbol(display_name) == display_name


@pytest.mark.parametrize(
    "display_name",
    [
        "幂律背景幅值 B₂",
        "幂律背景指数 p",
        "相对分辨率 σq/q",
        "足迹满斑角 θ_fp",
        "绝对分辨率 σq,0",
        "角域分辨率 σθ",
    ],
)
def test_a_name_that_already_carries_a_symbol_is_left_alone(display_name: str) -> None:
    """已带符号的量一个字都不改：「角域分辨率 σθ σ」读起来像两个符号打架。"""
    assert annotate_symbol(display_name) == display_name


def test_a_copied_quantity_keeps_the_copy_marker_at_the_end() -> None:
    """周期块展开出的副本行以「副本2」收尾，符号不插到它后面去。

    ``{base} 副本{k}`` 是这一行与哪一次重复对应的唯一线索，而「厚度 副本2 d」把
    符号挂在了副本号后面，读起来像副本自己有个符号。后缀不匹配就不标注，这一行
    因此原样留下。
    """
    assert annotate_symbol("厚度 副本2") == "厚度 副本2"


def test_the_symbol_lands_after_the_owner_is_dropped() -> None:
    """标注加在去掉归属之后：分组行已经写了层名，行里只剩量名加符号。"""
    assert row_name("SiO2 厚度", "SiO2") == "厚度 d"
    assert row_name("SiO2 相对密度", "SiO2") == "相对密度 ρ"


def test_a_row_without_a_caption_keeps_its_owner_and_still_gets_the_symbol() -> None:
    """只有一个分组时表里不开标题行，行名仍带着层名——符号照样加在末尾。"""
    assert row_name("film 厚度", None) == "film 厚度 d"


def test_both_tables_read_the_same_row_name() -> None:
    """设置表与结果表共用这条规则：同一个厚度不许在两张表里读出两个名字。

    设置表在符号后面再挂单位，写成设计稿的 ``厚度 d（nm）``；结果表的单位另有一列,
    所以到符号为止。两者的量名部分必须逐字相同。
    """
    from xrr_fitter.gui.parameters.table import _name_text

    definition = _definition("component.0.thickness_a", "SiO2 厚度", "Å")

    assert _name_text(definition, caption="SiO2") == "厚度 d（nm）"
    assert row_name(definition.display_name, "SiO2") == "厚度 d"


def test_an_auto_added_oxide_group_reads_as_the_design_writes_it() -> None:
    """帧③ 参数化表格的分组行也走结构列表那套名字。

    自动加入的氧化层带着 ``services.structures`` 生成的占位名进来，分组行原样显示它，
    中文表格里就冒出一行英文（截断后成了 ``SiO2 nativ…``）。分组行与层列表说的是同一层，
    名字不该有两种写法。
    """
    from xrr_fitter.gui.parameters.grouping import component_captions

    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0j),
        (api.LayerSpec("SiO2 native oxide", api.MaterialSpec("SiO2", "SiO2", 2.19), 34.2),),
        api.MaterialSpec("Si", "Si", 2.329),
    )

    assert component_captions(structure) == {"component.0": "SiO₂ · 表面氧化层"}


def test_the_translated_caption_still_drops_the_owner_the_declaration_carries() -> None:
    """声明带的前缀是生成的原名，分组行显示的是译名——剥离仍要认得出是同一层。

    不认的话行名会退回成 ``SiO2 native oxide 厚度 d``，比没翻译还长。
    """
    assert row_name("SiO2 native oxide 厚度", "SiO₂ · 表面氧化层") == "厚度 d"
