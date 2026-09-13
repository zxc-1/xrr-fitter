"""参数按归属分组的规则，供设置表与结果表共用。

声明到达时已经按归属聚在一起——一层的全部参数相邻，然后下一层、基底、仪器——
但十七行同样式的相邻行把这个结构完全藏了起来，而其中十行属于仪器而不是样品。
每一簇前面开一行标题行，用户不必悬停就能看出某个「厚度」属于哪一层。改按
``category`` 分组会把同一层的厚度、密度、粗糙度打散到三个相距很远的块里，比它
所替代的平表更难读。

长度量的显示换算也放在这里：声明是 Å，屏幕上是 nm。设置表拿它换算初值与上下限，
结果表拿它换算结果值与 1σ，两处必须同一套规则，否则同一个厚度会在两张表里读出
差十倍的数。
"""

from __future__ import annotations

from collections.abc import Mapping

import xrr_fitter.api as api
from xrr_fitter.gui.structure import naming

BACKING_CAPTION = "基底"
INSTRUMENT_CAPTION = "仪器"

# 量名后面跟这个量的通用符号，写成帧① 的 ``厚度 d``。读者手上的文献、实验记录和
# 另一套软件的输出都用 d/σ/ρ，行名不带符号时要在中文量名和公式里的符号之间自己做
# 一次翻译。声明里的仪器量本来就写成「幂律背景指数 p」「相对分辨率 σq/q」,「量名 +
# 符号」是既有约定，缺的正是这三类。
#
# 认后缀而不认整名：一层的粗糙度叫「入射侧粗糙度」，周期块的叫「顶界面粗糙度」,
# 基底那一处叫「基底连接界面粗糙度」——三者是同一个 σ，而限定语要留着，一个周期块
# 同时排出两种粗糙度时只剩「粗糙度 σ」的两行会读成重复行。后缀不匹配就不标注，
# 「厚度 副本2」这种以副本号收尾的行因此原样留下：符号挂在副本号后面会读成副本
# 自己有个符号。
SYMBOLS: tuple[tuple[str, str], ...] = (
    ("厚度", "d"),
    ("粗糙度", "σ"),
    ("密度", "ρ"),
)

# 帧① 的行名读的是物理量，声明读的是这个量在模型里的位置。同一个 σ 在声明里按位置分成
# 「入射侧粗糙度」和「基底连接界面粗糙度」，可是行上方的分组行已经写明是哪一层、哪块基底，
# 位置说到第二遍就只是把行名撑长；「尺度」和「常数背景」则是两个谁都认得的仪器量在这套
# 声明里的内部叫法，文献和另一套软件写的是 scale 与 background。
#
# 认整名而不认后缀：周期块的「顶界面粗糙度」和块内子层的「W 入射侧粗糙度」都不是整名匹配，
# 于是原样留下——一个周期块同时排出块顶与子层两种粗糙度时，把两者都改写成「粗糙度」会读成
# 重复行。声明本身一个字都不改：``display_name`` 落在检查点的参数指纹里（``fit.checkpoint``），
# 改一个字就让所有旧存档续跑不上。
ROW_ALIASES: tuple[tuple[str, str], ...] = (
    ("入射侧粗糙度", "粗糙度"),
    ("基底连接界面粗糙度", "粗糙度"),
    ("尺度", "强度标度 scale"),
    ("常数背景", "本底 background"),
)

# 声明里的密度是相对体密度的倍率，帧① 写的是能跟文献对照的绝对密度。倍率乘上材料自己的
# 体密度就是绝对值，所以只有拿到体密度时这一行才改名改单位；直接给 SLD 的材料没有体密度，
# 那里仍写倍率。
RELATIVE_DENSITY = "相对密度"
ABSOLUTE_DENSITY = "密度"
DENSITY_UNIT = "g·cm⁻³"
DENSITY_SUFFIX = ".density_scale"

# 组内顺序：厚度 → 粗糙度 → 密度。声明按编译顺序交出来（厚度 → 密度 → 粗糙度），而帧①
# 三层都按「多厚 · 多糙 · 多密」读——先是这层有多厚，再是它与上一层的界面有多糙，最后才是
# 它是什么料。排序只在同一个归属内部生效，归属之间的先后不动。
QUANTITY_ORDER: tuple[str, ...] = (".thickness_a", ".roughness_a", DENSITY_SUFFIX)


def uses_nm(definition: api.ParameterDefinition) -> bool:
    return definition.name.endswith((".thickness_a", ".roughness_a"))


def _is_density(definition: api.ParameterDefinition) -> bool:
    return definition.name.endswith(DENSITY_SUFFIX)


def display_scale(definition: api.ParameterDefinition, bulk_density: float | None = None) -> float:
    if uses_nm(definition):
        return 0.1
    if bulk_density is not None and _is_density(definition):
        return bulk_density
    return 1.0


def display_unit(definition: api.ParameterDefinition, bulk_density: float | None = None) -> str:
    if uses_nm(definition):
        return "nm"
    if bulk_density is not None and _is_density(definition):
        return DENSITY_UNIT
    return definition.unit


def group_key(definition: api.ParameterDefinition) -> str:
    """Identify the layer, the backing or the instrument owning a declaration.

    Component parameters are keyed on ``component.{index}`` rather than on the
    leading segment alone, so a periodic block's per-layer and per-repeat rows stay
    with the block that produced them.
    """
    head, _, _ = definition.name.partition(".")
    if head != "component":
        return head
    return ".".join(definition.name.split(".")[:2])


def caption_text(key: str, captions: Mapping[str, str]) -> str:
    """Name a group the way the structure editor names it.

    A component's caption cannot be recovered from its rows: a periodic block named
    ML contributes rows reading "W 厚度" and "Si 厚度", which share no token with
    the block.  The owner therefore supplies the component names and the key is
    only a fallback for callers that hold no structure.

    调用方给的名字优先于这里的兜底词：基底那一组在帧① 上写的是「晶硅基底 · c-Si」而不是
    「基底」，那个名字只有结构自己知道，而没有结构可读的调用方仍读到「基底」。
    """
    if key in captions:
        return captions[key]
    if key == "instrument":
        return INSTRUMENT_CAPTION
    if key == "backing":
        return BACKING_CAPTION
    return key


def group_caption(key: str, captions: Mapping[str, str]) -> str:
    """Name a group the way the results table heads it.

    与 `caption_text` 同一个名字，两半的顺序相反：设置区那张表跟着层列表读「材料 ·
    角色」，结果值表读「角色 · 材料」。差别只在显示，剥离行名用的仍是原名，所以两个
    入口都从这里出去，调用处不必各自记得该换哪一半。
    """
    return naming.grouped_name(caption_text(key, captions))


def component_captions(structure: api.StructureSpec | None) -> dict[str, str]:
    """Name each component group the way the structure editor names it.

    A component's own name is not recoverable from its parameter rows, so the
    structure is read wherever it is available and the tables render what they
    are handed.  Both the settings table and the result table need the same
    names, and a project holding no structure yet contributes none.
    """
    if structure is None:
        return {}
    return {
        f"component.{index}": naming.expert_name(component.name) for index, component in enumerate(structure.components)
    }


def group_captions(structure: api.StructureSpec | None) -> dict[str, str]:
    """Name every group a result table heads, the backing included.

    帧① 的基底那一组写的是它自己的名字（「晶硅基底 · c-Si」），跟上面两层一个写法；只写
    「基底」时读者读不出衬底是什么料，而料号正是他要拿来和层密度对照的东西。

    只有名字里带着 ``naming.SEPARATOR`` 才当成「材料 · 角色」用：一块只叫 ``Si`` 的基底
    经过两半互换仍是 ``Si``，那一行就成了一个孤立的化学式，反而比「基底」更难读。
    """
    captions = component_captions(structure)
    if structure is None:
        return captions
    backing = naming.expert_name(structure.backing.name)
    if naming.SEPARATOR in backing:
        captions["backing"] = backing
    return captions


def bulk_densities(structure: api.StructureSpec | None) -> dict[str, float]:
    """Pair each declaration owner with the bulk density its material was given.

    密度声明是倍率，帧① 写绝对值，两者之间差的就是这张表：键是声明名去掉最后一段后的归属
    前缀（``component.0``、``component.0.layer.1``、``backing``），值是那份材料的体密度。
    直接给 SLD 的材料没有体密度，梯度层根本没有材料，两者都不进表——取不到就仍按倍率显示。
    """
    if structure is None:
        return {}
    densities: dict[str, float] = {}
    for index, component in enumerate(structure.components):
        prefix = f"component.{index}"
        _record_density(densities, prefix, getattr(component, "material", None))
        layers = tuple(getattr(component, "layers", ()))
        for position, layer in enumerate(layers):
            material = getattr(layer, "material", None)
            _record_density(densities, f"{prefix}.layer.{position}", material)
            for repeat in range(int(getattr(component, "repeats", 0) or 0)):
                _record_density(densities, f"{prefix}.repeat.{repeat}.layer.{position}", material)
    _record_density(densities, "backing", structure.backing)
    return densities


def _record_density(densities: dict[str, float], prefix: str, material: object | None) -> None:
    density = None if material is None else getattr(material, "bulk_density_g_cm3", None)
    if density is not None:
        densities[prefix] = float(density)


def owner_prefix(name: str) -> str:
    """Which layer, repeat, basement or instrument a declaration hangs off.

    ``group_key`` 把周期块的全部子层归到块自己名下（分组行只写块名），而单位换算与组内
    排序要认到子层：块内两个子层各有一份厚度，按块排序会把两份厚度排到一起，子层就散了。
    """
    head, _, _ = name.rpartition(".")
    return head or name


def strip_caption(text: str, caption: str | None) -> str:
    """Drop the owner from a display name when a caption above already names it.

    Frame ①'s ``<tr class="grouprow">`` carries 表面氧化层 · SiO₂ and its rows carry
    only ``厚度 d``, while the declarations arrive prefixed: a layer named film
    contributes "film 厚度".  The match is exact and includes the separating space,
    so 基底's group -- whose single row reads 基底连接界面粗糙度, one word rather
    than a prefixed quantity -- is left whole.

    A caption the structure editor translated is matched by its generated
    preimage too: the declaration was built from the raw name, so an oxide group
    captioned SiO₂ · 表面氧化层 still has to shed the "SiO2 native oxide " its rows
    carry, or the translation would leave the rows longer than it found them.
    """
    if not caption:
        return text
    for owner in (caption, naming.generated_name(caption)):
        if owner is not None and text.startswith(f"{owner} "):
            return text[len(owner) + 1 :]
    return text


def annotate_symbol(text: str) -> str:
    """Append the symbol frame ① writes after a quantity that has one.

    Only the three the design annotates are recognised, and only as a suffix -- see
    ``SYMBOLS``.  A name that already ends in its own symbol is left whole because
    it cannot match: 角域分辨率 σθ ends in the symbol, not in 粗糙度.
    """
    for suffix, symbol in SYMBOLS:
        if text.endswith(suffix):
            return f"{text} {symbol}"
    return text


def alias_quantity(text: str, bulk_density: float | None = None) -> str:
    """Rewrite a stripped quantity into the word frame ① writes for it.

    整名匹配，见 ``ROW_ALIASES``。密度那一条另算：只有拿到体密度、行上的数确实是绝对密度时
    才去掉「相对」二字，否则「密度 2.19」和「密度 0.995」会读成同一个量的两次测量。
    """
    if text == RELATIVE_DENSITY:
        return ABSOLUTE_DENSITY if bulk_density is not None else text
    for source, target in ROW_ALIASES:
        if text == source:
            return target
    return text


def row_name(display_name: str, caption: str | None, bulk_density: float | None = None) -> str:
    """What a parameter row calls itself, in both tables.

    The owner is dropped first and the symbol appended after, so a row under 表面
    氧化层 · SiO₂ reads ``厚度 d`` rather than ``SiO2 厚度 d``: the symbol belongs to
    the quantity, and the quantity is what remains once the caption above has named
    the owner.

    改名夹在两者之间：前缀还在时整名对不上，符号已经挂上时也对不上。
    """
    return annotate_symbol(alias_quantity(strip_caption(display_name, caption), bulk_density))


def _quantity_rank(definition: api.ParameterDefinition) -> int:
    for rank, suffix in enumerate(QUANTITY_ORDER):
        if definition.name.endswith(suffix):
            return rank
    return len(QUANTITY_ORDER)


def sort_rows(definitions: tuple[api.ParameterDefinition, ...]) -> tuple[api.ParameterDefinition, ...]:
    """Order each owner's own rows 厚度 → 粗糙度 → 密度, leaving the owners in place.

    排序稳定，键里的归属取的是第一次出现的次序，所以归属之间一动不动；仪器那一组三个量都
    不在 ``QUANTITY_ORDER`` 里，同一个名次下原序保留。
    """
    owners = tuple(dict.fromkeys(owner_prefix(definition.name) for definition in definitions))
    order = {owner: index for index, owner in enumerate(owners)}
    return tuple(sorted(definitions, key=lambda value: (order[owner_prefix(value.name)], _quantity_rank(value))))


def row_layout(
    definitions: tuple[api.ParameterDefinition, ...],
    captions: Mapping[str, str],
) -> tuple[tuple[str, api.ParameterDefinition | None], ...]:
    """Interleave group captions with the declarations they introduce.

    A caption naming the only group present separates nothing, so a table holding
    one group is left exactly as it was before grouping existed.

    组内顺序在这里定下（``sort_rows``），两张表因此读出同一套先后。
    """
    ordered = sort_rows(tuple(definitions))
    keys = tuple(dict.fromkeys(group_key(definition) for definition in ordered))
    if len(keys) < 2:
        return tuple((group_key(value), value) for value in ordered)
    rows: list[tuple[str, api.ParameterDefinition | None]] = []
    current: str | None = None
    for definition in ordered:
        key = group_key(definition)
        if key != current:
            rows.append((key, None))
            current = key
        rows.append((key, definition))
    return tuple(rows)
