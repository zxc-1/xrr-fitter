"""层名的显示形态：翻译生成的占位名，并按模式选分隔符。

自动加入的表面氧化层没有人给它起过名字——``services.structures`` 按
``f"{formula} native oxide"`` 生成一个，中文界面上原样显示它等于这一行没有名字。
翻译放在显示层而不是服务层：那个串同时是 ``services.materials`` 认领这一层的钥匙，
也已经写进了存档，改它要动领域模型和持久化。

分隔符随模式变。设计稿同一层在专家模式写「SiO₂ · 表面氧化层」，在引导模式写
「SiO₂ 表面氧化层」——三帧十二行没有例外，所以这是一条规则而不是笔误：专家列表用
间隔点把「叫什么」和「是什么」分开，引导列表面对的是新手，少一个符号少一层解析。
"""

from __future__ import annotations

# ``LayerSpec.name`` 里化学式是 ASCII 的（``SiO2``），设计稿一律排成下标（``SiO₂``）。
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
PLAIN_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

NATIVE_OXIDE_SUFFIX = " native oxide"
OXIDE_ROLE = "表面氧化层"

# 专家列表的「名字 · 说明」分隔符，引导列表把它收成一个空格。
SEPARATOR = " · "


def subscript_formula(formula: str) -> str:
    """把化学式里的数字降为下标：``SiO2`` → ``SiO₂``。"""
    return formula.translate(SUBSCRIPT_DIGITS)


def expert_name(name: str) -> str:
    """专家列表里这一层叫什么。

    只认服务层生成的那一种占位名。用户自己起的名字原样留着——哪怕它碰巧也以
    ``native oxide`` 结尾，那也是他写的字，不该被改写。
    """
    if name.endswith(NATIVE_OXIDE_SUFFIX):
        formula = name[: -len(NATIVE_OXIDE_SUFFIX)]
        if formula:
            return f"{subscript_formula(formula)}{SEPARATOR}{OXIDE_ROLE}"
    return name


def inline_name(name: str) -> str:
    """一行之内提到这一层时它叫什么：专家写法去掉间隔点。

    引导列表和检视器抬头是同一件事的两处：那里的间隔点是外层结构的分隔符（一格一层、
    「选中层 · 」接层名），层名自己再带一个同样的符号，两个符号就分不出哪个是界。
    """
    return expert_name(name).replace(SEPARATOR, " ")


def grouped_name(display: str) -> str:
    """结果值表的分组行里这一层叫什么：专家写法的两半互换。

    设计稿对同一层写过相反的两种顺序，两处都不是笔误。层列表回答「这一层是什么材料」，
    所以材料在前（``SiO₂ · 表面氧化层``）；结果值表的分组行回答「下面这几个数属于样品的
    哪一部分」，所以角色在前（``表面氧化层 · SiO₂``）。

    只有一半的分组（``仪器``、``基底``）没有可换的东西，原样返回。
    """
    head, separator, tail = display.partition(SEPARATOR)
    return f"{tail}{separator}{head}" if separator else display


def generated_name(display: str) -> str | None:
    """``expert_name`` 的逆：显示名是译出来的就还原成生成的原名，否则 ``None``。

    参数表需要这个方向。分组行显示的是译名，而参数声明带的前缀是生成的原名
    （``"SiO2 native oxide 厚度"``）——两边对不上，剥离就会失败，行名反而更长。
    重写只有这一种形态，逆映射因此是确定的，不必再多存一份原名。
    """
    role = f"{SEPARATOR}{OXIDE_ROLE}"
    if not display.endswith(role):
        return None
    formula = display[: -len(role)].translate(PLAIN_DIGITS)
    return f"{formula}{NATIVE_OXIDE_SUFFIX}" if formula else None
