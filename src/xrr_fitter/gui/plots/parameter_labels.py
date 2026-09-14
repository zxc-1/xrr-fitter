"""参数在图上叫什么——把 ``component.0.thickness_a`` 压成读者扫得动的 ``d·ox``。

参数的机器路径是给代码用的：它要唯一、要能拼、要在存盘后仍指向同一个数。相关矩阵要的是
另一件事——17×17 的格子里来回对照行列，名字每读一次都得读到第三段才知道是谁，扫视就断了。
设计稿因此把它写成「符号·归属」：``d·ox``、``ρ·aSi``。

这里只收有出处的符号：厚度 d、相对密度 ρ、粗糙度 σ 与设计稿逐字写过的 ``scale``，以及
``docs/algorithm.md`` 里 ``θ = 2θ / 2 + Δθ`` 的那个 ``Δθ``。仪器量另有六项的 ``display_name``
末尾本来就写着符号（``相对分辨率 σq/q``），轴上只留那一段——那不是新造记号，读者在参数表里
见过它。剩下的（``常数背景``、``线性背景``）没有出处，宁可长一点也照原样写：为图上好看凭空
造一套符号，代价是同一个参数在表里一个名、图上另一个名，读者得先学一遍对照关系。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# 三个在设计稿上露过脸的符号。键是参数路径的末段——同一个末段在每一层上都是同一个物理量，
# 所以映射按末段走，而不必为每层各写一遍。
SYMBOLS = {"thickness_a": "d", "density_scale": "ρ", "roughness_a": "σ"}

# 归属名不从 ``display_name`` 取的两处。``backing`` 的 display_name 是「基底连接界面粗糙度」
# 整句，切首段会得到整句本身；这里给它一个词。
OWNERS = {"backing": "基底"}

# 逐字写死的两个短名。``instrument.scale`` 的 display_name 是「尺度」，但设计稿上写的是
# ``scale``——同屏还有一个「相对密度」，两个都读作「尺度/密度」时分不开。``angle_offset_deg``
# 的 ``Δθ`` 出自 ``docs/algorithm.md`` 的 ``θ = 2θ / 2 + Δθ``，那里定义的就是这个量。
LITERALS = {"instrument.scale": "scale", "instrument.angle_offset_deg": "Δθ"}

SEPARATOR = "·"


def short_label(name: str, display_name: str | None = None) -> str:
    """``name`` 在图上的短名；``display_name`` 缺席时退回路径本身。

    退回是有意的：图上宁可出现一个长名字，也不能出现一个猜出来的名字。一个参数没在
    ``parameter_definitions`` 里被声明过，就说明这里没有它的中文名可用。
    """
    if name in LITERALS:
        return LITERALS[name]
    if not display_name:
        return name
    prefix, _, leaf = name.rpartition(".")
    symbol = SYMBOLS.get(leaf)
    if symbol is None or not prefix:
        return _trailing_symbol(display_name) or display_name
    owner = OWNERS.get(prefix) or _leading_word(display_name)
    if owner is None:
        return display_name
    return f"{symbol}{SEPARATOR}{owner}"


def _trailing_symbol(display_name: str) -> str | None:
    """``display_name`` 末尾已经是符号时把它单独取出——「相对分辨率 σq/q」里的 ``σq/q``。

    只在尾段不含汉字时才当符号：「ox 厚度」的尾段是「厚度」，取了它就丢掉归属，两层会撞成
    同名；而矩阵上正需要靠归属区分同一物理量的不同层。
    """
    _, separator, tail = display_name.rpartition(" ")
    if not separator or not tail or _has_han(tail):
        return None
    return tail


def _has_han(text: str) -> bool:
    """``text`` 里有没有汉字——判「这一段是符号还是词」的那道线。"""
    return any("一" <= character <= "鿿" for character in text)


def _leading_word(display_name: str) -> str | None:
    """``display_name`` 的首段——「ox 厚度」里的 ``ox``。

    没有空格时返回 ``None`` 而不是整串：那说明这个 display_name 不是「归属 + 物理量」的写法，
    照切会把整句话搬到符号后面（``σ·基底连接界面粗糙度``），比不缩写更难读。
    """
    head, separator, _ = display_name.partition(" ")
    return head if separator and head else None


def label_map(definitions: Iterable[object]) -> Mapping[str, str]:
    """参数路径到短名的映射，直接喂给 ``set_xticks`` / ``set_yticks``。

    取 ``definitions`` 而不是 project 或 structure：``FitResult.parameter_definitions`` 就挂在
    每个结果上，画图的一侧手里已经有它，不必为一行标签把工程对象逐层传下来。
    """
    return {name: short_label(name, display) for name, display in _display_map(definitions).items()}


def _display_map(definitions: Iterable[object]) -> Mapping[str, str | None]:
    """参数路径到 ``display_name`` 的映射——短名与量名都从它派生。"""
    displays: dict[str, str | None] = {}
    for definition in definitions:
        name = getattr(definition, "name", None)
        if isinstance(name, str):
            displays[name] = getattr(definition, "display_name", None)
    return displays


def short_labels(names: Iterable[str], definitions: Iterable[object]) -> tuple[str, ...]:
    """``names`` 按 ``definitions`` 翻成短名，顺序照原样——轴刻度靠位置对齐数据。"""
    labels = label_map(definitions)
    return tuple(labels.get(name, short_label(name)) for name in names)


def quantity_name(display_name: str | None) -> str:
    """``display_name`` 尾段里的中文量名——「aSi 相对密度」里的「相对密度」；取不到是空串。

    只在尾段含汉字时才算量名：「相对分辨率 σq/q」的尾段是记号不是量名，而「尺度」没有分段，
    整串就是量名本身——但那时归属没处放，图例写「尺度 scale」是把同一个词说两遍。
    """
    if not display_name:
        return ""
    _, separator, tail = display_name.rpartition(" ")
    if not separator or not _has_han(tail):
        return ""
    return tail


def quantity_names(names: Iterable[str], definitions: Iterable[object]) -> tuple[str, ...]:
    """``names`` 各自的中文量名，顺序照原样；取不到的位置是空串。

    图例上要用：短名里那个 ``d`` / ``ρ`` / ``σ`` 只对读过矩阵刻度的人成立，而图例常常是一条
    曲线第一次出现的地方。量名从 ``display_name`` 取而不另立一张表——同一个参数在参数表里叫
    什么，图上就叫什么，否则读者得先学一遍对照关系。
    """
    displays = _display_map(definitions)
    return tuple(quantity_name(displays.get(name)) for name in names)
