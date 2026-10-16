"""结构编辑被拒时屏上那句话：哪一格、为什么、界限是多少。

服务层与物理层的拒绝是给机器读的英文（``interface.2.roughness_a must be below 9.8 A``），
里面的界面序号数的是展开后的切片，单位是 Å，读者对着层堆叠找不到它说的是哪一道界面。
这里只翻认得的那几类；认不出的原样上屏——编一个原因比露出原文更糟。
"""

from __future__ import annotations

import re

import xrr_fitter.api as api

FORMULA_PATTERN = re.compile(r"invalid material formula (?P<formula>'[^']*'|\"[^\"]*\")")
ELEMENT_PATTERN = re.compile(r"unknown element (?P<element>\S+)")
ROUGHNESS_PATTERN = re.compile(r"interface\.(?P<interface>\d+)\.roughness_a must be below (?P<limit>[0-9.eE+-]+) A")

FORMULA_TEXT = "化学式 {formula} 无法识别：元素符号区分大小写（如 SiO2、Al2O3、Si3N4）。"
ELEMENT_TEXT = "化学式 {formula} 无法识别：没有名为 {element} 的元素（元素符号区分大小写）。"
ROUGHNESS_TEXT = "{lower}的入射侧粗糙度（与{upper}之间的界面）须小于 {limit:g} nm。"
INTERFACE_TEXT = "从空气侧数第 {number} 道界面的粗糙度须小于 {limit:g} nm。"


def is_formula_error(error: BaseException) -> bool:
    return FORMULA_PATTERN.search(str(error)) is not None


def _formula_text(message: str) -> str | None:
    match = FORMULA_PATTERN.search(message)
    if match is None:
        return None
    formula = match.group("formula")[1:-1]
    element = ELEMENT_PATTERN.search(message)
    if element is not None:
        return ELEMENT_TEXT.format(formula=formula, element=element.group("element"))
    return FORMULA_TEXT.format(formula=formula)


def _sides(structure: api.StructureSpec | None) -> tuple[str, ...] | None:
    """一道界面上下两侧的名字，只在界面序号能一一对上层堆叠时才给。

    界面序号数的是展开后的切片：周期块按重复次数、梯度层按切片数、带过渡的层按微切片展开，
    那时第 N 道界面对不上堆叠里的第 N 行，只能报序号。
    """
    if structure is None:
        return None
    if not all(
        isinstance(component, api.LayerSpec) and component.transition is None for component in structure.components
    ):
        return None
    names = [component.name for component in structure.components]
    return ("空气", *names, structure.backing.name.strip() or "基底")


def _roughness_text(message: str, structure: api.StructureSpec | None) -> str | None:
    match = ROUGHNESS_PATTERN.search(message)
    if match is None:
        return None
    interface = int(match.group("interface"))
    limit = float(match.group("limit")) / 10.0
    sides = _sides(structure)
    if sides is None or interface + 1 >= len(sides):
        return INTERFACE_TEXT.format(number=interface + 1, limit=limit)
    return ROUGHNESS_TEXT.format(upper=sides[interface], lower=sides[interface + 1], limit=limit)


def structure_error_text(error: BaseException, structure: api.StructureSpec | None = None) -> str:
    """把一次结构拒绝写成读者能照着改的一句话。

    ``structure`` 是被拒的那一份候选结构；给了它，粗糙度那一类才说得出是哪两层之间。
    """
    message = str(error)
    return _formula_text(message) or _roughness_text(message, structure) or message
