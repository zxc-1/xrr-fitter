"""Correlation matrices and their responsive coefficient, legend, and summary rendering.

SLD/profile page composition stays with its owning view; this module owns only
how correlation evidence is ranked and drawn on a supplied Matplotlib axes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from matplotlib.artist import Artist
from matplotlib.text import Text
from matplotlib.ticker import Formatter
from matplotlib.transforms import Bbox, offset_copy

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.diagnostics import diverging_colormap
from xrr_fitter.gui.plots.parameter_labels import short_labels

# 抬头点明是谁之间的相关。同屏还有残差自相关、SLD 可信带这些同样带「相关」字样的证据，
# 光写「相关矩阵」读者得先猜是哪一种。
CORRELATION_TITLE = "参数相关矩阵"

# 色标回答的是「这个颜色代表什么量」，答案是 Pearson ρ。此前写的「固定范围 [-1, 1]」说的是
# 量程，而量程由两端刻度自己说得清——那一行于是占着位置没回答读者站在这里的问题。
COLOUR_KEY_LABEL = "Pearson 相关系数 ρ"

# 色标上只写两端和中性点。收窄到方阵实宽（约 250px）之后这不是取舍而是算术：matplotlib 默认
# 给九档（±1 / ±0.75 / ±0.5 / ±0.25 / 0），``-1.00`` 一个标签约 30px，摊下来要 270px。三档也
# 够——具体是多少不在这里读，每格里都印着系数本身；正号照写，与格内 ``+0.55`` 同一口径。
COLOUR_KEY_TICKS = (-1.0, 0.0, 1.0)
COLOUR_KEY_TICK_LABELS = ("−1", "0", "+1")

# 矩阵旁那栏读数的名字。它只有文字没有数据，``apply_axes_grid`` 的 ``has_data`` 一条就把它
# 跳过了；这个名字给的是「哪一格是那一栏」的抓取名，与 ``COLORBAR_AXES_LABEL`` 同一条理由。
SUMMARY_AXES_LABEL = "<correlation-summary>"

# 逐格加粗、格内字色、以及「最强 / 次强」两行读数共用的那条界。
#
# 它不是 ``ConfidenceThresholds.strong_correlation``（默认 0.95）。那条是判定拟合可信度的硬
# 门：|ρ| 到 0.95 时两个参数已基本不可分辨，整份结果降级。图上这条界回答的是另一个问题
# ——从哪里起，逐参数的 ±1σ 就不该再单独读。|ρ|=0.6 时联合不确定区的长短轴比已到约 2:1，
# 也就是「沿坐标轴切出来的 ±1σ 明显窄于真实区间」开始成立的地方，正是该把读者引向下半那张
# Profile 似然的那一点。两条界各自诚实，所以这里不去借用那一条。
NOTABLE_CORRELATION = 0.6

# 印在格子里的数需要多宽。``+0.55`` 是 5 个字符，``FONT_PT_SM``（9pt）下每个数字宽约 0.55em
# ≈ 6.6px，连左右呼吸约 38px。窄过这个数就不印：17 参数的实测里一格只有 6px（见
# ``test_correlation_labels``），五个字符在那儿必然互相压掉，读到的不是数而是墨。
MINIMUM_LABELLED_CELL_PIXELS = 38.0

# 横轴那排名字斜排的角度。斜排是为了让「符号·归属」这种两三个字符的名字排得下 17 列，代价是
# 相邻两条标签成了两条平行的文本行：刻度间距 d 对应的基线垂距只有 ``d·sin45°``，也就是间距的
# 七成。这个角度同时被 ``set_xticks`` 和 ``_ColumnLabels`` 用——一处改另一处的算式就错了。
MATRIX_TICK_ROTATION = 45.0

# 读数栏在矩阵框右边的位置与宽度，按矩阵自己的 axes 坐标给。矩阵是方阵，在半栏高的格子里
# 宽度由高度定死，右侧本就空着一片——这一栏站的正是那片空白，所以它跟着矩阵框走而不去占
# 布局的一格。
SUMMARY_PANEL_BOUNDS = (1.06, 0.0, 0.62, 1.0)

# 这一栏窄到装不下一行短名对（``d·aSi ↔ ρ·aSi`` 在 9pt 下约 78px，判读句折行还要八九个字
# 的宽度）时就不画。对话框那一页 400px 宽的画布留给它只有约 68px，而那一页的用处本来就是
# 「把矩阵放大到看得见每一格的数」，读数栏在中栏那一屏。窄没窄过这条线要等排完版才知道，
# 所以那个判断在 ``_SummaryStack.draw`` 里，不在构图时。
MINIMUM_SUMMARY_PANEL_PIXELS = 110.0

# 行距：字号换成像素再乘这个数。9pt ≈ 12px，一行连行距约 17px。
SUMMARY_LINE_SPACING = 1.45

# 一行读数连行距占多少点（9pt × 1.45）。段间隙按它的倍数给（``SUMMARY_BLOCK_GAP``），比到像素上
# 要乘 dpi/72。各段自己往下走多远不用它：那是按每次绘制实测的段高摊的（``_SummaryStack``）。
# 不论谁来改，这个量都不能折成栏高的比例（axes 分数）——栏高要等 ``constrained_layout`` 排完版
# 才知道，构图时取到的是上一次排版留下的值，画布一改尺寸同一个分数对应的像素间距跟着改。实测
# 中栏被压扁时画一次、再把窗口拉大到 794×738，五段里有两段被推到栏外一二百像素，落进下方
# Profile 那一格（``Text`` 的 ``clip_on`` 默认是关的，越界既不报错也不截断）。
SUMMARY_LINE_POINTS = theme.FONT_PT_SM * SUMMARY_LINE_SPACING

# 两组读数之间空出的那口气，按行高的倍数给。这是**上限**：栏矮到装不下时先压它（见
# ``_SummaryStack``）——四段内容一个字都不能少（那半句物理解释正是这对参数为什么纠缠），字号
# 9pt 已是全图最小档，间隙是这里唯一可让的东西。
SUMMARY_BLOCK_GAP = 0.6

# 压间隙时留给栏底的那一像素。余量吃干净时末段的下沿正好压在栏底线上，而「整段在栏内」这条
# 判据没有容差——四段实测高度累加出来的零点几纳米误差往下落就算越界。一像素肉眼看不出，判据
# 上却是实打实的余量。
SUMMARY_FIT_MARGIN_PIXELS = 1.0

# 折行时从哪个码位起按整宽算。0x2E80 往上是 CJK 部首、汉字、假名和全角标点，往下是拉丁、
# 希腊（``ρ``、``σ``）和数学符号（``±``）这些窄字符。
FULL_WIDTH_CODEPOINT = 0x2E80

# 窄字符占一个整宽的多少。9pt 下汉字约 12px、数字约 6.6px，正是这个比例。
NARROW_CHARACTER_EM = 0.55

# 一个整宽在屏上合多少像素，与 ``theme`` 里的字号换算同一套。这是构图时的口径（等价 96 dpi）；
# 排完版之后按 figure 自己的 dpi 重算（``_SummaryStack.draw``），两者在 100 dpi 下差 0.5px。
SUMMARY_EM_PIXELS = theme.FONT_PT_SM * 4.0 / 3.0

# 折行时给栏右侧留出的余量，按整宽计。整宽是个估算（窄字符按 0.55 摊），一行里窄字符多寡不同、
# 累出来的误差就往右顶；余量少留一个整宽，末尾那个字才不会压在栏的右边线上被裁掉半边。
SUMMARY_WRAP_SLACK_EM = 1.0

# 还没排版时按哪个宽度折。折行要用栏的实宽，而栏宽要等 ``constrained_layout`` 跑完才知道，所以
# 每次绘制都按当帧的栏宽重折（``_SummaryStack.draw``）。这个常量只是尚未绘制过时手里那份：取
# 「窄过它就不画这一栏」的下限（``MINIMUM_SUMMARY_PANEL_PIXELS``），按它折出来的行在任何画得出
# 这一栏的尺寸下都排得进栏宽。
SUMMARY_COLUMNS = MINIMUM_SUMMARY_PANEL_PIXELS / SUMMARY_EM_PIXELS


def _cell_side_pixels(axes: object, count: int) -> float:
    """一格在屏上的边长。

    矩阵是等比方阵，格边长由 axes 的短边定；``apply_aspect`` 得先跑一次，否则量到的还是布局
    分配的那个长方框，宽高都不是矩阵实得的尺寸。
    """
    if count <= 0:
        return 0.0
    axes.apply_aspect()
    corner = axes.transData.transform((0.0, 0.0))
    unit = axes.transData.transform((1.0, 1.0))
    return float(min(abs(unit[0] - corner[0]), abs(unit[1] - corner[1])))


class _CellLabel(Text):
    """印在格心的那个系数，画不画由它自己在每次 draw 时决定。

    「格子够不够宽」随画布尺寸变，而画布会被拖动：拉窄之后 17 个参数的格子只剩几个像素，
    建 artist 那一刻算出的结论就过期了，而 resize 只重排布局、不重走一遍绘制。判断挪进
    ``draw`` 之后，尺寸一变结论跟着变，上层什么都不用管。
    """

    def __init__(self, *args: object, cells: int, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._cells = int(cells)

    def draw(self, renderer: object) -> None:
        side = _cell_side_pixels(self.axes, self._cells) if self.axes is not None else 0.0
        self.set_visible(side >= MINIMUM_LABELLED_CELL_PIXELS)
        super().draw(renderer)


def _display_width(text_value: str) -> float:
    """一串字有多少个 em 宽：方块字算一个整宽，拉丁字母和数字算 ``NARROW_CHARACTER_EM``。

    ``d·aSi`` 五个字符比 ``常数背景`` 四个字符窄得多，所以「哪个名字最长」按字符数是数不出来的。
    """
    return sum(1.0 if ord(char) >= FULL_WIDTH_CODEPOINT else NARROW_CHARACTER_EM for char in text_value)


class _ColumnLabels(Formatter):
    """横轴那排名字，隔几列写一个由它自己在每次 draw 时算。

    45° 斜排下相邻两条标签是两条平行的文本行，基线垂距 ``pitch·sin45°`` 只有刻度间距的七成，
    而它必须大于一行字的高度才不互相压。17 个参数摊在 1400×900 主窗口那档中栏上，垂距只有
    10.1px 而字高 12.5px——后一条压在前一条的上半截里，屏上读到的是一片斜向的墨。够不够只能
    看画布：同样 17 个参数在 1272×1060 那档垂距 13.4px，反倒读得出。所以这不是「参数多了就
    撤掉」，是每次绘制时按当下的间距决定隔几列写一个。

    ``隔几列`` 而不是「全撤掉」：矩阵的列序与行序相同，名字在纵轴上有全套一份，但读者要在
    17×17 的格子间对照行列时，横轴上每隔一格有个锚点比光秃秃的一排刻度线好找得多。

    跳过谁有讲究：最宽的那个名字必须留着。刻度带有多高由**最宽的可见标签**的对角投影定，而带高
    吃掉画布高度、方阵（``aspect='equal'``）的边长跟着缩、间距于是变小——如果跳过的正好是最宽的
    那个，带高会缩、间距会变大、算出来的步长又回到 1、最宽的那个于是回来、带高再涨，一次 draw
    一个结论地来回翻。把最宽的那个钉在可见集合里，带高与步长就互不影响，结论稳定。
    """

    def __init__(self, labels: Iterable[str]) -> None:
        self._labels = tuple(labels)
        self._widest = max(range(len(self._labels)), key=lambda index: _display_width(self._labels[index]), default=0)

    def __call__(self, value: float, position: int | None = None) -> str:
        index = int(round(float(value)))
        if not 0 <= index < len(self._labels):
            return ""
        step = self._step()
        if step > 1 and (index - self._widest) % step:
            return ""
        return self._labels[index]

    def _step(self) -> int:
        """按当下的刻度间距，隔几列写一个才让相邻两条读得开。"""
        axes = getattr(self.axis, "axes", None)
        if axes is None:
            return 1
        clearance = _cell_side_pixels(axes, len(self._labels)) * math.sin(math.radians(MATRIX_TICK_ROTATION))
        height = theme.FONT_PT_SM * float(axes.figure.dpi) / 72.0
        if clearance <= 0.0 or height <= 0.0:
            return 1
        return max(1, math.ceil(height / clearance))


def _label_cells(axes: object, matrix: np.ndarray, palette: theme.PlotPalette) -> None:
    """把系数印进各自那一格。

    颜色只给「大概多强、什么符号」；读者要带走的是数本身——「这一对是 −0.72」这句话应当能从
    矩阵上直接抄下来，而不是先目测色深再回头对色标。主对角线只印一个 ``1``：它恒等于 1，印
    ``+1.00`` 是拿四个字符换零信息。

    字色跟着格子的底色走，而不是跟着「重要不重要」走：diverging 色标两端深、中间浅，所以过了
    ``NOTABLE_CORRELATION`` 的格子底色已经足够深，白字才读得出；淡格子上则要用前景色。
    """
    size = int(matrix.shape[0])
    for row in range(size):
        for column in range(size):
            value = float(matrix[row, column])
            strong = row == column or abs(value) >= NOTABLE_CORRELATION
            label = _CellLabel(
                column,
                row,
                "1" if row == column else f"{value:+.2f}",
                cells=size,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=theme.FONT_PT_SM,
                fontweight="bold" if strong else "normal",
                color="#FFFFFF" if strong else palette.foreground,
            )
            label.set_transform(axes.transData)
            axes.add_artist(label)


def _ranked_pairs(matrix: np.ndarray) -> tuple[tuple[int, int, float], ...]:
    """非对角的参数对，按 |ρ| 降序。

    ``report.strong_correlations`` 排不出这个序：它按索引顺序产出，又按配置的判定阈值过滤，
    于是「次强」那一行常常整个不在里面——|ρ|=0.55 的一对够不上任何判定门，却正是读者扫完矩阵
    之后想确认的第二格。
    """
    size = int(matrix.shape[0])
    pairs = [(row, column, float(matrix[row, column])) for row in range(size) for column in range(row + 1, size)]
    pairs.sort(key=lambda item: -abs(item[2]))
    return tuple(pairs)


def _is_thickness_density_pair(first: str, second: str) -> bool:
    """这一对是不是「厚度 × 密度」——判读句里那条物理解释只对这一对成立。"""
    names = (first.lower(), second.lower())
    return any("thickness" in name for name in names) and any("density" in name for name in names)


def _correlation_hint(first: str, second: str, labels: tuple[str, str], value: float) -> str:
    """最强那一对意味着什么。

    「−0.72」本身不告诉读者下一步做什么。两个参数强相关时，协方差派生的 ±1σ 是沿各自坐标轴
    切出来的，而真实的联合不确定区是斜的椭圆，于是逐参数的 ±1σ 系统性偏窄——这正是同屏下半
    那张 Profile 似然存在的理由。物理那半句只在这一对确实是厚度与密度时给：换成别的组合，
    「电子密度与厚度难以同时唯一确定」就是一句不成立的话。
    """
    if abs(value) < NOTABLE_CORRELATION:
        return f"✓ 无强相关（最强 |ρ|={abs(value):.2f} < {NOTABLE_CORRELATION:g}）：逐参数 ±1σ 可各自读。"
    parts = [f"{labels[0]} 与 {labels[1]} 强{'负' if value < 0.0 else '正'}相关。"]
    if _is_thickness_density_pair(first, second):
        parts.append("薄层的电子密度与厚度难以同时唯一确定；")
    parts.append("单看 ±1σ 会低估真实不确定度，需结合 Profile 似然判读。")
    return "⚠ " + "".join(parts)


def _draw_correlation_summary(
    axes: object,
    names: tuple[str, ...],
    matrix: np.ndarray,
    definitions: Iterable[object] = (),
) -> None:
    """矩阵旁那栏读数：最强的一对是谁、系数多少、次强是谁，以及这对纠缠意味着什么。

    n² 个数印进格子解决了「这一格是多少」，没解决「该看哪一格」——读者仍要把整片数扫一遍再
    自己排序。这一栏写的就是排序的结果。名字用矩阵刻度上的同一套短名，读者拿着这行回矩阵上
    就能找到那一格，不必再翻译一次。
    """
    ranked = _ranked_pairs(matrix)
    if not ranked:
        return
    panel = axes.inset_axes(SUMMARY_PANEL_BOUNDS)
    panel.set_label(SUMMARY_AXES_LABEL)
    panel.set_axis_off()
    # 这一栏整块站在矩阵框外面（``SUMMARY_PANEL_BOUNDS`` 的 x 从 1.06 起），而 constrained_layout
    # 量的是父 axes 连子 axes 在内的整个 tight bbox：不摘出来的话，「右边多出一栏」会被当成
    # 「矩阵需要的地方变宽了」，于是布局反过来把方阵压小——读数栏越宽，它想说明的那张矩阵越小。
    panel.set_in_layout(False)
    palette = theme.current_plot_palette()
    labels = short_labels(names, definitions)
    first, second, value = ranked[0]
    entries = [
        (f"最强相关\n{labels[first]} ↔ {labels[second]}", palette.foreground),
        (f"系数 ρ\n{value:+.2f}", palette.foreground),
    ]
    if len(ranked) > 1:
        third, fourth, runner_up = ranked[1]
        entries.append((f"次强\n{labels[third]} ↔ {labels[fourth]} {runner_up:+.2f}", palette.foreground))
    # 判定门那句不在这里：它讲的是整张图怎么读，与逐条读数不是一个层级，归卡抬头右端
    # （``_draw_correlation_heading``）。搬走一段就是给这一栏让出约 54px——它一直差的正是那点高度。
    entries.append(
        (_correlation_hint(names[first], names[second], (labels[first], labels[second]), value), palette.muted)
    )
    _stack_summary_lines(panel, entries)


def _draw_correlation_heading(axes: object) -> None:
    """卡抬头那一行：主标题靠左，判定门那句副题靠右。

    设计稿的 ``.ph`` 是 ``display:flex; justify-content:space-between``——两段同排、各贴一端。
    副题写的是判读规则（哪条线之后算强相关），它得和被判读的那张图同屏，否则逐格加粗和「最强
    相关」那几行都是没有出处的断言；但它也不该排进读数栏，那栏码的是「最强那对是谁」这类逐条
    读数，夹一句规则进去等于让读者在两种语境间来回切。

    副题的右端对齐读数栏的右端，而不是方阵的右端：设计稿那一行横跨整张卡，而这张图里「卡」是
    方阵加右侧读数栏两块。只对齐方阵的话，窄档（方阵 174px）上两段字合起来 245px 上下，必然
    压在一起。代价是这段字站在方阵框外，于是要和读数栏同样 ``set_in_layout(False)``——不摘出去
    的话 constrained_layout 会把「右边多出一段字」算成「矩阵要的地方变宽了」，反过来压小方阵。
    """
    axes.set_title(CORRELATION_TITLE, loc="left")
    subtitle = axes.set_title(
        f"Pearson ρ · |ρ|≥{NOTABLE_CORRELATION:g} 视为强相关",
        loc="right",
        fontsize=theme.FONT_PT_SM,
        color=theme.current_plot_palette().muted,
    )
    # ``_update_title_position`` 每次 draw 只重算 y（把抬头推到斜列名带的上方），x 照原样留着。
    subtitle.set_x(SUMMARY_PANEL_BOUNDS[0] + SUMMARY_PANEL_BOUNDS[2])
    subtitle.set_in_layout(False)


def _wrap_units(text_value: str, columns: float) -> Iterable[str]:
    """把一串字切成折行时不该再拆开的那些单元。

    方块字各自一个单元——中文长句整句没有空白，只能逐字折。拉丁字母、数字和 ``·``/``±``/``↔``
    这些窄字符连成一片时算一个单元：``+0.55`` 从中间断开会被读成 ``+0.5``，``Profile`` 断开就
    认不出它指的是同屏那张图。空白单独成一个单元，好让折行落在词与词之间。

    宽过一整行的单元没有不硬切的走法（切了至少读得到，不切会一路伸出画布），所以这里就把它拆成
    单字符交出去——单个字符必然装得下。
    """
    unit = ""
    for char in text_value:
        narrow = not char.isspace() and ord(char) < FULL_WIDTH_CODEPOINT
        if unit and not narrow:
            yield from _split_oversized(unit, columns)
            unit = ""
        if narrow:
            unit += char
            continue
        yield char
    if unit:
        yield from _split_oversized(unit, columns)


def _split_oversized(unit: str, columns: float) -> Iterable[str]:
    """一个单元装得下就整块交出去，宽过一整行才拆成单字符。"""
    if _display_width(unit) > columns:
        yield from unit
    else:
        yield unit


def _wrap_to_width(text_value: str, columns: float) -> str:
    """按显示宽度折行，一个字符也不增不减。

    ``textwrap`` 在这里用不上：它按空白分词，而中文长句整句没有空白，于是要么不折（一路
    伸出画布），要么在硬切处把行内词用单空格重新连起来——那就动了原文。这一栏的读数是要被
    读出来的句子，折行只该决定它在哪儿换行，不该改写它。

    宽度按 em 计：方块字算一个整宽，拉丁字母和数字算 0.55——这正是 ``d·aSi`` 与「厚度」在同
    一个字号下的实际比例。断点落在 ``_wrap_units`` 切出的单元之间，不落在单元内部：逐字符累宽
    会把 ``+0.55`` 切成 ``+0.5`` 和 ``5``，读者扫过去读到的是另一个数。

    行末的空白留在本行，不推到下一行开头：推过去的话每个折行处都会多出一个前导空格，读起来像
    缩进了一格，而它其实只是断点的残留。
    """
    lines: list[str] = []
    current = ""
    width = 0.0
    for unit in _wrap_units(text_value, columns):
        if unit == "\n":
            lines.append(current)
            current = ""
            width = 0.0
            continue
        step = _display_width(unit)
        if current and not unit.isspace() and width + step > columns:
            lines.append(current)
            current = ""
            width = 0.0
        current += unit
        width += step
    lines.append(current)
    return "\n".join(lines)


class _SummaryStack(Artist):
    """读数栏那几段怎么排，由它在每次 draw 时按当下的栏框定：栏太窄整栏让位，折行按实宽，栏矮了压间隙。

    三个决定以前都在构图时下，而那时读到的栏框是**上一次**排版留下的——``constrained_layout``
    要等 draw 才跑，这一栏又整块挂在矩阵的 axes 坐标上（``SUMMARY_PANEL_BOUNDS``）。于是「够不够
    宽」在中栏被压扁那一档判成了「够」：栏实宽 62.8px、门是 110px，四段照样画出来，每段只剩最左
    边一两个字，其余被栏框裁掉。而「间隙多大」按行高的固定倍数给：1400×900 那档四段内容 253.8px
    装得进 261.5px 的栏，三个段间隙 32.6px 却把总跨顶到 286.4px，末段那句「需结合 Profile 似然
    判读」被栏底裁掉——读者要带走的下一步动作正是这半句。

    折行同理。原先按那条「窄过它就不画」的下限（110px）折，而 1400×900 那档栏实宽 161px：每行只
    用掉栏宽的七成，行数因此多出三成，高度也就多出三成——把栏顶爆的那几十像素有一半是这么来的。
    这里按当帧实宽重折，宽回来的那 51px 就用上了；拖窗口时也跟着重折，而 ``_wrap_to_width`` 往
    文本里插的是真换行符，构图时折死的话画布再怎么变都是那几行。

    挪进 ``draw`` 与 ``_CellLabel``、``_ColumnLabels``、``_pin_colour_key`` 是同一条路：结论依赖
    排版结果，就在每次绘制时按当下的框重算，画布怎么拖都跟得上。用一只垫底的 artist 而不是让每段
    自己算，是因为间隙要按四段合起来还差多少来压，单独一段看不到别人有多高；``Artist`` 的 zorder
    默认 0，比 ``Text`` 的 3 小，同一次 draw 里排在它们之前，位置改完文本才画。``draw_event`` 回调
    不行：它在整张图画完之后才跑，那时文本已经画在旧位置上，要再触发一次重绘而重绘又触发回调。

    栏太窄时收的是各段的可见性，不是栏本身：``Axes`` 一旦不可见就不再 ``apply_aspect``，它的框会
    冻在上一帧，下次拖宽了也就再判不出「宽回来了」，这一栏于是永久消失。
    """

    def __init__(self, panel: object, lines: Iterable[tuple[Text, str]]) -> None:
        super().__init__()
        self._panel = panel
        # 每段连它折行前的原文一起存：折行按当帧栏宽重来，而折过的文本折不回去（插进去的是真
        # 换行符，栏窄下来那一行会连着上一行的残段再折一次，越折越碎）。
        self._lines = tuple(lines)

    def draw(self, renderer: object) -> None:
        box = self._panel.get_window_extent(renderer)
        readable = box.width >= MINIMUM_SUMMARY_PANEL_PIXELS
        for line, _ in self._lines:
            line.set_visible(readable)
        if not readable:
            return
        # 整宽按 figure 自己的 dpi 换算（模块常量那份是 96 dpi 的口径），这样折出来的行宽与量到的
        # 栏宽是同一套单位。
        em = theme.FONT_PT_SM * float(self._panel.figure.dpi) / 72.0
        columns = max(box.width / em - SUMMARY_WRAP_SLACK_EM, 1.0)
        for line, source in self._lines:
            line.set_text(_wrap_to_width(source, columns))
        # 段高要在段可见、且文本按本帧重折之后才量得准（``Text.get_window_extent`` 对不可见的段返回
        # 单位框，行数也是刚才那一步才定的）。量实高而不按行数乘行距推：推出来的值差着零点几像素，
        # 而余量本来就只有七八像素，差出来的那点正好够让末段压在栏底线上。
        heights = [line.get_window_extent(renderer).height for line, _ in self._lines]
        spare = box.height - sum(heights) - SUMMARY_FIT_MARGIN_PIXELS
        ceiling = SUMMARY_BLOCK_GAP * SUMMARY_LINE_POINTS * float(self._panel.figure.dpi) / 72.0
        gap = min(ceiling, max(spare, 0.0) / max(len(self._lines) - 1, 1))
        cursor = 0.0
        for (line, _), height in zip(self._lines, heights, strict=False):
            # 推的距离按像素（``units="dots"``）给：这一帧的栏框和段高都是像素量出来的，换成点数
            # 再让 matplotlib 折回像素等于在两套单位之间往返一趟，而余量只有几像素。
            line.set_transform(offset_copy(self._panel.transAxes, self._panel.figure, y=-cursor, units="dots"))
            cursor += height + gap


def _stack_summary_lines(panel: object, entries: list[tuple[str, str]]) -> None:
    """把那几段读数码进读数栏，长句折行，越出栏外的部分裁掉。

    Matplotlib 不给文本自动换行，而这一栏只有一百多像素宽：判读句原样画出去会一路伸出画布
    右界，所以得折。折成几行按栏的实宽算，而栏宽要等 ``constrained_layout`` 跑完才知道，所以每次
    绘制时按当帧的栏宽重折（``_SummaryStack``）——这里建的 ``Text`` 先按 ``SUMMARY_COLUMNS`` 折一份，
    作为尚未绘制过时手里那个保底，原文一并交给 stack 以便重折。

    每段先都站在栏的左上角（``transAxes`` 的 ``(0, 1)``）：往下推多远、以及这一栏画不画，两件事
    都要看栏排完版有多大，同样归 ``_SummaryStack`` 在每次 draw 时定。

    ``clip_on`` 必须显式打开：``Axes.text`` 已经把 ``clip_path`` 设成了这一栏的框，但 ``Text``
    的 ``clip_on`` 默认是关的，于是超出栏底的段照样画出来，落进下方 Profile 那一格。间隙压到零
    仍码不下时，宁可让末尾几段被栏底裁掉——判读句在右栏证据清单里另有一份（``results/
    uncertainty.py`` 的 ``STRONG_CORRELATION_NOTE``），画到别人的格子上则是读错图。
    """
    lines = [
        (
            panel.text(
                0.0,
                1.0,
                _wrap_to_width(text_value, SUMMARY_COLUMNS),
                transform=panel.transAxes,
                ha="left",
                va="top",
                fontsize=theme.FONT_PT_SM,
                color=colour,
                linespacing=SUMMARY_LINE_SPACING,
                clip_on=True,
            ),
            text_value,
        )
        for text_value, colour in entries
    ]
    panel.add_artist(_SummaryStack(panel, lines))


def _pin_colour_key(key_axes: object, axes: object, *, horizontal: bool) -> None:
    """把色标那条收到方阵的实边长上，另一个方向仍归排版。

    ``figure.colorbar(ax=axes)`` 按**分配框**排：横躺的占整宽，竖立的占整高。而方阵是
    ``aspect='equal'``，实边长由分配框的**短边**定死——两者于是差着好几倍，差在哪个方向取决于
    分配框是扁还是竖长。实测横躺那档（中栏 1400×900）方阵 246.8px、色标 723.5px，左端伸到
    x=53.3 而方阵左端在 290.6；实测竖立那档（「不确定度分析…」对话框拉到 320×1400）方阵
    215.7px、色标 517.8px，上下各伸出 151.0px。两次都是屏上一条比矩阵本身还醒目的长条，而读者
    没有任何线索把它和那个小方块对上。

    收窄写不成静态比例（``shrink``）：方阵占分配框多少是排版结果，1272×1060 那档占 0.55、
    1400×900 只占 0.34。所以挂在 ``set_axes_locator`` 上——它每次 draw 被调用来定位，拿到的
    正是这一帧排完版的方阵框，与 ``_ColumnLabels`` 依赖当下间距是同一条理由。原 locator
    （colorbar 自己那个）要包住而不是换掉：色带厚度、以及不受约束那个方向上的位置仍该由它说。
    """
    base = key_axes.get_axes_locator()

    def locate(target: object, renderer: object) -> object:
        box = base(target, renderer) if base is not None else target.get_position(original=True)
        # 方阵实框要 ``apply_aspect`` 之后才是方的；这一步在 matrix 自己 draw 时也会跑，这里
        # 先跑一次是因为 locator 可能在它之前被问到。
        axes.apply_aspect()
        host = axes.get_position()
        if horizontal:
            return Bbox.from_bounds(host.x0, box.y0, host.width, box.height)
        return Bbox.from_bounds(box.x0, host.y0, box.width, host.height)

    key_axes.set_axes_locator(locate)


def _draw_correlation(
    figure: object,
    axes: object,
    report: object,
    definitions: Iterable[object] = (),
    *,
    wide_layout: bool = True,
) -> None:
    """The signed correlation matrix plus the colour key that states its scale.

    A signed scale, like the residual heatmap: the palette's diverging hues
    resolve per draw so the matrix flips with the appearance, where a built-in map
    would stay fixed at a light-background contrast and sink into the dark panel.
    The neutral midpoint keeps a near-zero correlation from reading as a weak
    signal of one sign.  Without the key the matrix is a field of colours with no
    stated scale, so a cell only reads as "reddish" rather than as a correlation
    near +1.

    刻度写短名（``d·ox``）而不是参数路径：17 个参数的实测里 16 个 ytick 越出画布左界，最长的
    ``instrument.powerlaw_background_amplitude`` 越界 155px——比矩阵本身还宽，``constrained_layout``
    随之整个放弃排版。名字长度在这张图上不是观感问题，是矩阵还剩多少地方的问题。

    色标横躺在矩阵下方，方阵一并靠左——这两件事同出一个前提：横向有富余。矩阵是方阵，在半栏高
    的格子里它的宽度由高度定死，右侧本就空着一片；色标竖在右边等于把那片空白再切一刀，而矩阵
    自己一点没变宽。躺到下面之后横向整条都归它，−1 / 0 / +1 三个刻度也读得开，右侧那片连续的
    空白则整块让给读数栏。``wide_layout=False`` 的扁框里这个前提不成立，见
    ``draw_correlation_page``。
    """
    matrix = np.asarray(report.correlation_matrix, dtype=float)
    palette = theme.current_plot_palette()
    image = axes.imshow(matrix, vmin=-1.0, vmax=1.0, cmap=diverging_colormap())
    image.set_clim(-1.0, 1.0)
    if wide_layout:
        # 靠左：不定 anchor 时 matplotlib 把等比方阵摆在分配框正中，右侧那片空白被劈成两半，
        # 读数栏于是没有一块连续的地方可站。扁框里没有富余可分，居中即是唯一选择。
        axes.set_anchor("W")
    key = figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal" if wide_layout else "vertical",
        label=COLOUR_KEY_LABEL,
    )
    key.set_ticks(COLOUR_KEY_TICKS, labels=COLOUR_KEY_TICK_LABELS)
    _pin_colour_key(key.ax, axes, horizontal=wide_layout)
    if wide_layout:
        # 「这条尺量的是什么」得排在「这一头是多少」之前，可 ``colorbar(label=...)`` 把标题放在
        # 刻度那一侧，读者先撞上三个数字、再往下找它们是什么的 −1。色标又是贴着矩阵下沿的
        # （``_pin_colour_key``），上方那几像素本来就闲着，标题挪上去不多占一行高度——留在下方则
        # 要在刻度之后再排一行，而这张图和 Profile 那格共享一栏高度。
        key.ax.xaxis.set_label_position("top")
    positions = tuple(range(len(report.correlation_names)))
    labels = short_labels(report.correlation_names, definitions)
    # 刻度字号显式给到和全图其余文字同一档：不给的话走 rcParams 的 10pt，比图上任何一处文字都
    # 大，而这排字恰恰是挤得最凶的一处——9pt 一项就让 1272×1060 那档的垂距从互压 0.5px 转为读得出。
    axes.tick_params(labelsize=theme.FONT_PT_SM)
    # 列名写在矩阵上方，把下边整条让给色标。色标已经收到方阵实宽，横躺在下方时它和这排斜标签
    # 争的是同一条边（17 参数那档标签带高 36px 上下），色标被整条推远，读者要跨过一片斜向的字
    # 才找到那把尺。挪到上方之后两者各占一边，中间只隔着矩阵自己。
    #
    # 锚点跟着换：45° 是向右上伸展，``ha="left"`` 让文本起点落在刻度上；``"right"`` 把锚点当成
    # 文本末端，整排字于是从刻度往左下铺开，压进矩阵里。
    axes.set_xticks(positions, labels, rotation=MATRIX_TICK_ROTATION, ha="left")
    axes.xaxis.set_ticks_position("top")
    # 文本改由 formatter 逐次 draw 给出（旋转和对齐仍是上一行设好的 artist 属性，formatter 不碰）：
    # 隔几列写一个取决于当下的刻度间距，而间距要到排完版才知道。改 tick 上的 ``Text`` 是没用的，
    # formatter 每次 draw 都会把文本重设一遍。
    axes.xaxis.set_major_formatter(_ColumnLabels(labels))
    axes.set_yticks(positions, labels)
    _draw_correlation_heading(axes)
    _label_cells(axes, matrix, palette)
    _draw_correlation_summary(axes, tuple(report.correlation_names), matrix, definitions)
