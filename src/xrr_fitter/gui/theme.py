"""Palette-aware application theme built on a 4px spacing grid.

The stylesheet styles controls only; window backgrounds stay native so the
application follows the platform light and dark appearance.  Semantic state
colors are exposed through dynamic properties (``statusKind``, ``primary``,
``sectionCard``, ``sectionHeader``, ``commandBar``) so widgets opt in without
hand-written inline stylesheets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from xrr_fitter.gui.sizing import ElidingLabel


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    accent: str
    accent_hover: str
    accent_text: str
    surface: str
    surface_border: str
    border_strong: str
    muted_text: str
    faint_text: str
    ok: str
    info: str
    warn: str
    error: str
    selection_bg: str


LIGHT_TOKENS = ThemeTokens(
    accent="#2F6BD8",
    accent_hover="#255CC0",
    accent_text="#FFFFFF",
    surface="rgba(0, 0, 0, 12)",
    surface_border="#D9DCE3",
    border_strong="#C2C7D2",
    muted_text="#5B6270",
    faint_text="#8A909C",
    ok="#2E7D32",
    info="#1565C0",
    warn="#9A6700",
    error="#B3261E",
    selection_bg="rgba(47, 107, 216, 36)",
)

DARK_TOKENS = ThemeTokens(
    accent="#5A8DEE",
    accent_hover="#6E9CF2",
    accent_text="#101418",
    surface="rgba(255, 255, 255, 14)",
    surface_border="rgba(255, 255, 255, 42)",
    border_strong="rgba(255, 255, 255, 60)",
    muted_text="rgba(255, 255, 255, 150)",
    faint_text="rgba(255, 255, 255, 115)",
    ok="#7BC67E",
    info="#8AB4F8",
    warn="#E3B341",
    error="#F28B82",
    selection_bg="rgba(90, 141, 238, 70)",
)

# The 4px grid the module docstring refers to.  Layouts import these instead of
# writing literals so one dock's density stays comparable to the next; the
# right-hand docks scroll inside a fixed viewport, so raising a step here costs
# vertical budget in every stacked row at once.
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24
SPACE_XXL = 32

# ``.statusbar .dot{width:8px;height:8px;border-radius:50%}``：直径与半径都要给样式表，
# 定尺归 ``status_bar`` 用，半径归下面的 QSS 用——一个来源，免得两处各改一半成了椭圆。
STATUS_DOT_PX = 8
STATUS_DOT_RADIUS_PX = STATUS_DOT_PX // 2

# Qt matches ``font-family`` against installed families only -- it does not read
# the CSS generic ``monospace`` -- and no desktop ships a family under that name.
# Asking for it therefore lands on the proportional UI font while Qt logs a
# "missing font family" warning, which is how a monospaced label can look
# monospaced in review and not be.  So the stack names the families each platform
# actually installs, and keeps the generic last as a documented intent.
MONO_FONT_STACK = '"SF Mono", "Cascadia Code", "Menlo", "Consolas", "DejaVu Sans Mono", monospace'

# Point sizes for text drawn outside the stylesheet (Matplotlib artists) and for
# the few labels that need to depart from the widget default.
FONT_PT_SM = 9
FONT_PT_MD = 11
FONT_PT_LG = 14

# One interactive-control height for buttons, tool buttons and single-line
# inputs alike.  These used to carry three different minimums (22, 20, and
# unset), which is what made neighbouring controls in one row look mismatched.
# 设计稿 `.btn{height:30px}`（HTML line 90）——所有常规按钮、工具按钮和输入框的统一高度。
CONTROL_MIN_H = 30

# 设计稿把命令条里的按钮单列一档（``.btn.sm``，26px，比独立的 ``.btn`` 矮 4px）。矮下来
# 不是为了省地方：这一档按钮总是跟 tab 或分段控件并排站在同一条 chrome 带里，一旦比旁边
# 的 tab 高，那条带的高度就由按钮说了算，行头凭空粗一圈。``.canvas-top`` 正是这么一条带，
# 而画布列一共只有三段高度可分。QSS 的 min/max-height 量的是内容矩形，两侧各 1px 边框要
# 自己扣掉。
COMMAND_BUTTON_H = 26

# 设计稿 ``.btn.lg``（``height:44px;padding:0 22px;font-size:14px;border-radius:8px``）：帧②
# 引导页那一行 CTA 单列一档，比应用里其他按钮高一倍多。高一档不是装饰——那两枚按钮是整屏
# 唯一的出路，一行里只有它们，所以它们自己定这一行的高度。两枚必须同高，否则并排站着会差
# 出一两像素，看着像没对齐（``.ctarow`` 是 ``align-items:center`` 的 flex 行）。
LARGE_BUTTON_H = 44
LARGE_BUTTON_PAD_PX = 22

# 设计稿 ``.wizhead .wh .n``（HTML 491）：向导抬头那一格的记号是 30px 的实心圆。它比左栏管线
# 和拟合梯子的小圆点大一档——分格卡片里它是那一格唯一的图形，20px 摆在 14px 的粗标题旁边会被
# 读成标题的附注。圆角必须是直径的一半，所以两处（这里定尺寸、QSS 里定圆角）只能引同一个数：
# 分开写死过一次，30px 的方块配 10px 圆角画出来是圆角方块。
STEP_DOT_PX = 30
LARGE_BUTTON_FONT_PX = 14
LARGE_BUTTON_RADIUS_PX = 8

# The side of one cell in the plot's mode bar.  That bar is the one place a
# button holds a glyph and nothing else, so it opts out of CONTROL_MIN_H and the
# general button padding: a square cell a little larger than the glyph keeps the
# bar short enough to share the plot's tab row instead of doubling its height.
GLYPH_CELL_PX = 24

# Unlike the tokens above, these do not follow the appearance.  A series that
# changed hue when the desktop switched to dark would invalidate the reference a
# user has already built ("the blue curve is my data"), so each hue is instead
# chosen to survive both canvas backgrounds.  Okabe-Ito throughout, which keeps
# the set separable under the common colour vision deficiencies.
DATA_OBSERVED = "#0072B2"
DATA_CANDIDATE = "#D55E00"
DATA_PREVIEW = "#009E73"
DATA_RANGE = "#E69F00"

# Cycled per component in the structure diagram so neighbouring layers separate
# at a glance.  The first three are the series hues above: a reader who has
# learnt "blue is the data" loses nothing by meeting blue again as a layer,
# whereas a second unrelated six-colour set would be six more things to learn.
DATA_SEQUENCE = (DATA_OBSERVED, DATA_CANDIDATE, DATA_PREVIEW, "#CC79A7", DATA_RANGE, "#56B4E9")

# Air and the substrate are semi-infinite, so they are not layers and must not
# borrow a sequence hue.
DATA_NEUTRAL = "#9AA0A6"

# Confidence double-encoding: shape + colour, never colour alone.
# Each verdict maps to a distinct Unicode glyph so the signal survives
# greyscale printing, colour vision deficiency, and screenshots without context.
CONFIDENCE_GLYPHS: dict[str, str] = {
    "可信": "●",
    "可用但相关": "◆",
    "多解": "▲",
    "不可信": "■",
}

# "不可用" is not a classification, so it stays out of the table above, but it is
# still a verdict the panel paints and therefore needs a shape authored here like
# the other four.  A hollow ring reads as "nothing measured yet" rather than as a
# fifth grade of confidence.
CONFIDENCE_FALLBACK_GLYPH = "○"

# The colour half of the double-encoding: four verdicts on four distinct kinds, so
# no two rows share a hue.  This is the only verdict-to-colour map in the app --
# every surface that paints a verdict (the badge, the result card, the candidate
# row marker, the status bar) reads it here rather than restating a variant, so a
# 可用但相关 cannot be blue in one corner of the screen and amber in another.
CONFIDENCE_COMPARISON_KINDS: dict[str, str] = {
    "可信": "ok",
    "可用但相关": "info",
    "多解": "warn",
    "不可信": "error",
}

# The kind that pairs with CONFIDENCE_FALLBACK_GLYPH: "not measured yet" is not a
# grade, so it is drawn in the muted text colour rather than in any status hue.
# Every kind above is a ThemeTokens field name, which lets a painter resolve any
# of them with one ``getattr(tokens, kind)`` instead of a per-kind branch.
CONFIDENCE_FALLBACK_KIND = "muted_text"

# Captions drawn on top of a fill.  Two are needed because the fills span most
# of the luminance range: white reaches only 2.25:1 on the orange and black only
# 4.05:1 on the deep blue, so neither one works for the whole sequence.
BAND_LABEL_ON_DARK = "#FFFFFF"
BAND_LABEL_ON_LIGHT = "#101010"


# A swatch wide enough to read as a line and tall enough to read as a box, at
# the caption's own scale so the key does not tower over its text.
LEGEND_SWATCH_W = 14
LEGEND_SWATCH_H = 10
LEGEND_LINE_H = 3
LEGEND_DOT_D = 9
# A dash swatch is one gap short of a line: the model curve is drawn dashed, and
# a solid bar in the key would describe a curve the pane never draws.  The gap is
# centred, so the mark reads as a dash pattern at swatch scale rather than as two
# unrelated ticks.
LEGEND_DASH_GAP = 4
# The mockup's box swatch is a band's fill, which is translucent by design; an
# opaque box would read as another colour rather than as the same hue.
LEGEND_BOX_ALPHA = 140

# 设计稿两块叠放的面板（反射率在上、残差在下）绘图区左边界都落在 viewBox 的 x=58，
# 也就是刻度文字和旋转的轴标题共用一条固定装订线。pyqtgraph 默认按当前刻度文字的宽度
# 自动伸缩轴宽：写着「10⁻⁶」的对数面板要 65px，只写「-2 0 2」的残差面板只要 38px，
# 叠起来就是两条错开近 30px 的左边界，网格线也对不上。四块面板钉同一个宽度，叠放的
# 绘图区才共用一条左边界；这个宽度足够放下原始视图的「60000」，没有标签会被丢掉。
PLOT_LEFT_GUTTER_PX = 58

# The mockup's `.sw` chip: square, so it does not read as either of the legend's
# marks, and small enough to sit inside a tree row's own line height.
STACK_SWATCH_PX = 14
STACK_SWATCH_RADIUS = 3

# 设计稿的 ``.stack`` 是 ``border-radius:8px`` 的一只盒子，靠 ``overflow:hidden`` 把里面首末
# 两行的方角裁掉。Qt 不裁子控件——那两只方角会顶在盒子的圆角外面露出尖来，所以首末行自己
# 也带圆角。行的半径比盒子小一像素：盒子的圆是画在 1px 边框的外缘上，行贴着的是框的内缘。
STACK_CONTAINER_RADIUS_PX = 8
STACK_ROW_RADIUS_PX = 7

# The colour roles a legend entry may name.  Entries spell a role rather than a
# hex string so a key mark and the curve it explains cannot drift apart: both
# resolve through the one series palette above.  ``accent`` is deliberately
# absent -- it follows the appearance, so it is resolved from the live palette.
LEGEND_ROLE_COLOURS = {
    "observed": DATA_OBSERVED,
    "candidate": DATA_CANDIDATE,
    "preview": DATA_PREVIEW,
    "range": DATA_RANGE,
    "neutral": DATA_NEUTRAL,
}


def _luminance(value: str) -> float:
    """WCAG relative luminance of a ``#RRGGBB`` string."""
    channels = []
    for index in (1, 3, 5):
        channel = int(value[index : index + 2], 16) / 255.0
        channels.append(channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def band_label_colour(fill: str) -> str:
    """The more readable of the two caption colours against `fill`.

    Computed rather than tabulated so a fill added to `DATA_SEQUENCE` later
    gets a correct label without a matching edit somewhere else.
    """
    if _contrast_ratio(BAND_LABEL_ON_DARK, fill) >= _contrast_ratio(BAND_LABEL_ON_LIGHT, fill):
        return BAND_LABEL_ON_DARK
    return BAND_LABEL_ON_LIGHT


# 拟合窗口那行说明画在只有 16% 不透明度的范围色带上，所以它读的底其实是画布本身。
# 设计稿给的是亮底那支暗琥珀（#8A6D00 对白 4.91:1）；同一支挪到深色画布只剩 3.38:1，
# 正文字号读不下来，于是暗底改用范围色本身（对 #1E1F22 是 7.36:1）。两端各有出处，
# 不是同一支颜色调亮调暗。
RANGE_LABEL_ON_LIGHT = "#8A6D00"
RANGE_LABEL_ON_DARK = DATA_RANGE


def range_label_colour(background: str) -> str:
    """The more readable of the two window-caption colours against `background`.

    Takes the canvas colour rather than the band's own fill for the reason above:
    at 16% the fill barely moves the luminance a reader's eye lands on.
    """
    if _contrast_ratio(RANGE_LABEL_ON_DARK, background) >= _contrast_ratio(RANGE_LABEL_ON_LIGHT, background):
        return RANGE_LABEL_ON_DARK
    return RANGE_LABEL_ON_LIGHT


@dataclass(frozen=True, slots=True)
class PlotPalette:
    """Structural colours for a Matplotlib figure, as RGBA 0..1 tuples.

    A ``Figure`` is not styled by the Qt stylesheet, so the diagnostic plots
    need the palette handed to them explicitly. Only structural colours live
    here; the data series keep their fixed Okabe-Ito hues, which stay legible
    against either background.
    """

    background: tuple[float, float, float, float]
    foreground: tuple[float, float, float, float]
    muted: tuple[float, float, float, float]
    grid: tuple[float, float, float, float]
    diverging_neg: tuple[float, float, float, float]
    diverging_pos: tuple[float, float, float, float]
    diverging_mid: tuple[float, float, float, float]


def _rgba(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    channels = tuple(int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    return (*channels, alpha)


def _tinted(value: str, alpha: float) -> str:
    """Render a token as a translucent Qt colour string.

    The status tokens are opaque hex per appearance, and a callout needs the
    same hue at a few percent. Deriving it here keeps the dark appearance from
    inheriting a light-theme tint, which is the bug the token table exists to
    prevent.
    """
    channels = ", ".join(str(int(value[index : index + 2], 16)) for index in (1, 3, 5))
    return f"rgba({channels}, {round(alpha * 255)})"


def token_colour(value: str) -> QColor:
    """Resolve a token to a painter-ready colour.

    半数 token 写成 ``rgba(r, g, b, a)``，因为它们的用途是进样式表。``QColor`` 只认
    十六进制与颜色名，收到函数记法时返回一个无效色，而无效色落笔是不透明黑——自绘的
    委托于是把次要文字画成黑色，深色外观下正好埋进背景。两种记法都从这里解开，调用处
    就不必先判断自己手上是哪一种。
    """
    text = value.strip()
    if not text.startswith("rgba"):
        return QColor(text)
    numbers = [part.strip() for part in text[text.index("(") + 1 : text.rindex(")")].split(",")]
    red, green, blue = (int(part) for part in numbers[:3])
    alpha = int(round(float(numbers[3]) if len(numbers) > 3 else 255))
    return QColor(red, green, blue, alpha)


# 设计稿 ``.hint`` 的底色与边框透明度。提示框要能从卡片底色里浮出来又不抢正文，所以
# 底色压到几个百分点、边框留到接近三成。
HINT_FILL_ALPHA = 0.06
HINT_BORDER_ALPHA = 0.28

# 设计稿 ``.badge`` 的胶囊几何与染色。半径大于任何可能的行高，所以两端始终是半圆
# 而非圆角矩形——这是 CSS ``border-radius:999px`` 在 Qt 里的等价写法。
BADGE_PAD_V_PX = 2
BADGE_PAD_H_PX = 8
BADGE_RADIUS_PX = 999
BADGE_FILL_ALPHA = 0.08
BADGE_BORDER_ALPHA = 0.40
# 胶囊那一档字：11px / 600。按 px 给是因为这枚胶囊在两条路上被画——样式表画带 ``badge``
# 属性的 ``QLabel``，结果检视器的候选行自绘同一只胶囊——两条路只有像素是同一个单位，混用点数
# 就得先猜当前 DPI 才知道两边是否等大（实测 ``FONT_PT_SM`` 的 9pt 落到 12px，比设计稿大一档）。
BADGE_FONT_PX = 11
BADGE_FONT_WEIGHT = QFont.Weight.DemiBold

# 设计稿 ``.jbanner`` 比 ``.hint`` 各重一档。提示有左侧竖条替它承载状态色，横幅没有，
# 整块底色是它唯一的可见性来源，压到 hint 的档位就会在卡片底色里化掉。
BANNER_FILL_ALPHA = 0.08
BANNER_BORDER_ALPHA = 0.30

# 设计稿 ``.confbadge``：判定框的内距、染色与字形字号。
# 底色按状态分档而不是折成一个常数——黄（多解）的亮度最高，同一个 alpha 铺出来比绿蓝红
# 都淡，所以设计稿单给它 ``.09``。边框统一 ``.35``，比 hint 的三成再实一点：这只框里装的
# 是整栏的结论，边界该清楚。
# 内距只在布局边距上落地，不写进样式表：这只框自己带布局，QSS 的 ``padding`` 会和布局边
# 距叠加一次，量出来是设计稿的两倍。
VERDICT_PAD_V_PX = 10
VERDICT_PAD_H_PX = SPACE_MD
VERDICT_FILL_ALPHAS: dict[str, float] = {"ok": 0.08, "info": 0.08, "warn": 0.09, "error": 0.08}
VERDICT_BORDER_ALPHA = 0.35
# 字形那 22px 折算成 16.5pt。字号在 Python 里落到控件上而不是写进 QSS：Qt 的样式表解析器
# 遇到带小数的 pt 会丢掉整条规则，字形会静默退回默认字号。
VERDICT_GLYPH_PT = 16.5

# 设计稿 ``.insp-sec .h`` 的字距（``letter-spacing:.6px``）。检视器那一栏的分节抬头是
# 11px 的小字，字距把它读成一行标签而不是一句被缩小的标题；正文与图卡抬头都不加。
SECTION_HEADING_TRACKING_PX = 0.6

# 设计稿 ``.nav-sec`` 的字距（``letter-spacing:.8px``）。左栏那两句抬头比检视器的松 .2px：
# 它们各自领着一整栏，而检视器那几句只领一节，所以两个值分开给，不合成一个。
RAIL_SECTION_TRACKING_PX = 0.8

# ``.nav-sec`` 与 ``.insp-sec .h`` 同为 ``font-size:11px``，所以分节抬头按像素给而不是折算
# 成 pt：11px 在 96dpi 下是 8.25pt，而这一档此前借用 ``FONT_PT_SM``（9pt = 12px），比设计稿
# 大一像素。抬头本就是最小的那号字，多一像素就够它跟正文平起平坐。
SECTION_HEADING_FONT_PX = 11

# 设计稿 ``.plotcard .ph .t`` 的 ``font-size:12.5px``——带框图卡那行抬头。取整到 13 而不是 12：
# 12 与 ``FONT_PT_SM`` 那档正文只差一像素，抬头就读不出是抬头了。同样按像素给：这一档要与
# 上面 11px 的 ``.insp-sec .h`` 直接比大小（右栏平抬头必须比图卡抬头小一档），两处混用 pt 与
# px 就得先猜当前 DPI 才知道谁大。``setPixelSize`` 只吃整数，所以 12.5 在这里没法原样落。
CARD_TITLE_FONT_PX = 13

# 设计稿 ``.plotcard .ph .sub`` 的 ``font-size:11px``——图卡抬头那行右端的提示语。跟着抬头一起
# 按像素给：抬头是 13px 的 ``CARD_TITLE_FONT_PX``，而提示语此前设的是 ``FONT_PT_SM``（9pt = 12px），
# 两处混用 pt 与 px 既没法直接比大小，实测也只差一像素——提示语等于跟抬头平起平坐，两级层次
# 只剩颜色在说。11px 让「先读抬头、再读提示」这个顺序由字号本身说出来。
CARD_HINT_FONT_PX = 11

# 设计稿 ``.wizbody .eyebrow2{font-size:12px;letter-spacing:.6px}``——引导页正文之上那行
# 「第 N 步 · 共 M 步」。按像素给的理由和上面几档一样：这一句要与同一张卡里 13.5px 的正文
# 直接比大小（它必须小一档，否则读作正文的第一句），两处混用 pt 与 px 就得先猜 DPI 才知道
# 谁大；字距本来也只能按像素落到字体对象上，Qt 的样式表没有 ``letter-spacing``。
STEP_EYEBROW_FONT_PX = 12
# 字距与 ``SECTION_HEADING_TRACKING_PX`` 恰好同值，但仍分开给：那一档是 11px 的分节抬头，
# 这一档是 12px 的强调色页眉，设计稿里是两条各自独立的规则。合成一个常量，改检视器抬头的
# 字距就会顺手挪走引导页的页眉。
STEP_EYEBROW_TRACKING_PX = 0.6

# 设计稿 ``.nav-sec .add`` 那枚 ``＋``：``font-size:16px;line-height:1``。它比抬头那行的字大
# 五像素，也是按像素给的——抬头与它是同一条 CSS 规则里的一对，折算成 pt 两个数会各差一点。
RAIL_ADD_GLYPH_FONT_PX = 16

# 设计稿 ``.field .inp{height:30px}``：选中层那张卡的四格比应用里其他输入框高一档，因为它们
# 是这张卡的主体而不是塞在一行工具里的附件。``CONTROL_MIN_H`` 不能跟着改——层堆叠拿它当行高
# 下限，流程步骤那一列也照它验高度；所以这一档按 objectName 只发给这张卡的输入框。
# 给的是 28 而不是 30：``min-height`` 不含框线，28 加上上下各 1px 的边正好是设计稿的 30。
SELECTED_LAYER_INPUT_H_PX = 28

# 设计稿 ``.lock{border:1.5px;border-radius:3px}``。Qt 的样式表只解析整数像素，1.5 会被整条
# 规则一起丢掉（连边都不画），所以取 2——方框只有 14px 见方，半像素的差别在这个尺寸上看不出，
# 而「有没有边」看得出。
SELECTED_LAYER_LOCK_BORDER_PX = 2
SELECTED_LAYER_LOCK_RADIUS_PX = 3


LIGHT_PLOT_PALETTE = PlotPalette(
    background=_rgba("#FFFFFF"),
    foreground=_rgba("#202020"),
    muted=_rgba("#6B6B70"),
    grid=_rgba("#202020", 0.16),
    diverging_neg=_rgba("#0072B2"),
    diverging_pos=_rgba("#D55E00"),
    diverging_mid=_rgba("#F2F2F4"),
)

DARK_PLOT_PALETTE = PlotPalette(
    background=_rgba("#1E1F22"),
    foreground=_rgba("#E8E8EA"),
    muted=_rgba("#A0A0A6"),
    grid=_rgba("#E8E8EA", 0.20),
    diverging_neg=_rgba("#0072B2"),
    diverging_pos=_rgba("#D55E00"),
    diverging_mid=_rgba("#2A2B2E"),
)


def plot_palette(tokens: ThemeTokens) -> PlotPalette:
    """Return the figure palette matching a resolved token set."""
    return DARK_PLOT_PALETTE if tokens is DARK_TOKENS else LIGHT_PLOT_PALETTE


# ``QColor.lightness()`` 的中点：到这个值以下算暗。
MID_LIGHTNESS = 128
# 底色淡到这个 alpha 以下，屏幕上呈现的主要是它下面那层，它自己就不再报得出外观的明暗。
OPAQUE_ENOUGH_ALPHA = 128


def palette_tokens(palette: QPalette) -> ThemeTokens:
    window = palette.color(QPalette.ColorRole.Window)
    if window.alpha() < OPAQUE_ENOUGH_ALPHA:
        # 样式表把控件底色写成 ``transparent``（``#candidateList`` 等处必须如此，否则那层底会
        # 压在委托画的行盒子下面透出来）时，Qt 把这个值反写进控件自己的调色板：屏幕上透出来
        # 的是父级那层底，可这里留下的是 ``rgba(0, 0, 0, 0)``——一个 alpha 为零的黑。
        # ``lightness()`` 不看 alpha，照它判浅色外观就成了深色。淡到这一档的底色不携带明暗
        # 信息，改问前景色：暗字必然配亮底。
        ink = palette.color(QPalette.ColorRole.WindowText)
        return DARK_TOKENS if ink.lightness() >= MID_LIGHTNESS else LIGHT_TOKENS
    return DARK_TOKENS if window.lightness() < MID_LIGHTNESS else LIGHT_TOKENS


def current_plot_palette() -> PlotPalette:
    """Resolve the plot palette from the running application's palette.

    Plots are drawn outside the stylesheet, so each draw reads the palette
    afresh; that is what lets a system appearance change reach both the
    matplotlib figures and the pyqtgraph panes on the next repaint without any
    switching logic.  This lives in ``theme`` rather than ``plots.diagnostics``
    so ``plots.live`` can share it without importing ``diagnostics`` (a cycle).
    """
    application = QApplication.instance()
    if application is None:
        return LIGHT_PLOT_PALETTE
    return plot_palette(palette_tokens(application.palette()))


def current_accent() -> str:
    """Resolve the accent colour from the running application's palette.

    ``PlotPalette`` deliberately carries no accent: for a data figure the
    accent is not a data colour.  A plot that draws *controls* -- the SLD
    profile's interface handles -- does need it, because blue is what tells the
    reader an element can be grabbed, and it has to be the same blue the rest
    of the window uses.  Resolved per draw for the same reason as
    :func:`current_plot_palette`.
    """
    application = QApplication.instance()
    if application is None:
        return LIGHT_TOKENS.accent
    return palette_tokens(application.palette()).accent


def _button_styles(tokens: ThemeTokens) -> str:
    return f"""
QPushButton {{
    border: 1px solid {tokens.border_strong};
    border-radius: 6px;
    padding: {SPACE_XS}px {SPACE_MD}px;
    min-height: {CONTROL_MIN_H}px;
    background: {tokens.surface};
}}
QPushButton:hover:enabled {{ border-color: {tokens.accent}; }}
QPushButton:pressed {{ background: {tokens.selection_bg}; }}
QPushButton:disabled {{ color: {tokens.muted_text}; }}
QPushButton[primary="true"] {{
    background: {tokens.accent};
    border-color: {tokens.accent};
    color: {tokens.accent_text};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover:enabled {{ background: {tokens.accent_hover}; }}
QPushButton[primary="true"]:disabled {{
    background: {tokens.surface};
    border-color: {tokens.border_strong};
    color: {tokens.muted_text};
    font-weight: 400;
}}
QPushButton[ghost="true"] {{
    border: none;
    background: transparent;
    color: {tokens.accent};
    font-weight: 600;
    padding: {SPACE_XS}px {SPACE_SM}px;
    min-height: {CONTROL_MIN_H}px;
}}
QPushButton[ghost="true"]:hover:enabled {{
    text-decoration: underline;
    color: {tokens.accent_hover};
}}
/* 设计稿 ``.btn.sm``：命令条里的按钮固定 26px 高，比它并排的 tab 矮一档。上下内边距因此
   归零并把高度钉死，否则字体一换按钮就长回去，把整条行头顶高。 */
QPushButton[commandBar="true"] {{
    padding: 0px {SPACE_SM}px;
    min-height: {COMMAND_BUTTON_H - 2}px;
    max-height: {COMMAND_BUTTON_H - 2}px;
}}
/* 设计稿 ``.btn.lg``：帧② 引导页那一行 CTA。高度钉死在两头，两枚按钮才会严格同高——
   ``.ctarow`` 垂直居中，差一像素就把其中一枚往下推一像素。``min/max-height`` 量的是内容
   矩形，上下各 1px 边框要自己扣掉；``ghost`` 那一档没有边框，所以它另算。 */
QPushButton[large="true"] {{
    padding: 0px {LARGE_BUTTON_PAD_PX - 1}px;
    min-height: {LARGE_BUTTON_H - 2}px;
    max-height: {LARGE_BUTTON_H - 2}px;
    font-size: {LARGE_BUTTON_FONT_PX}px;
    border-radius: {LARGE_BUTTON_RADIUS_PX}px;
}}
QPushButton[large="true"][ghost="true"] {{
    padding: 0px {LARGE_BUTTON_PAD_PX}px;
    min-height: {LARGE_BUTTON_H}px;
    max-height: {LARGE_BUTTON_H}px;
}}
QToolButton {{
    border: 1px solid transparent;
    border-radius: 5px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    min-height: {CONTROL_MIN_H}px;
}}
QToolButton:hover:enabled {{ border-color: {tokens.surface_border}; }}
QToolButton:checked {{
    background: {tokens.selection_bg};
    border-color: {tokens.accent};
}}
/* 设计稿的 ``.btn.primary`` 与那枚红字红边的 ⏹ 停止 都画在命令栏上，而命令栏里的按钮是
   QToolButton，此前只有 QPushButton 认这两个属性——于是「一键拟合」和「停止」在栏上一律
   画成普通灰边框，设置了属性也不生效。 */
QToolButton[primary="true"] {{
    background: {tokens.accent};
    border-color: {tokens.accent};
    color: {tokens.accent_text};
    font-weight: 600;
}}
QToolButton[primary="true"]:hover:enabled {{ background: {tokens.accent_hover}; }}
QToolButton[primary="true"]:disabled {{
    background: {tokens.surface};
    border-color: {tokens.surface_border};
    color: {tokens.muted_text};
    font-weight: 400;
}}
QToolButton[danger="true"] {{
    color: {tokens.error};
    border-color: {_tinted(tokens.error, BADGE_BORDER_ALPHA)};
}}
QToolButton[danger="true"]:hover:enabled {{ border-color: {tokens.error}; }}
QToolButton[danger="true"]:disabled {{
    color: {tokens.muted_text};
    border-color: {tokens.surface_border};
}}
/* 设计稿 ``.nav-sec .add``：一枚 accent 色的 16px ``＋``，行内 span，没有边框也没有内边距。
   ＋ 背后挂着一只 QMenu（三条导入命令住在里面），而 Qt 只要看见菜单就替按钮补画一枚
   ``::menu-indicator`` 下拉箭头——它把 12px 的字形撑成 22px 的墨迹，抬头于是读成
   「数据集 ＋⌄」。菜单本身留着（展开仍然选文件夹、换预设），只是箭头不画。 */
QToolButton#datasetAddButton {{
    border: none;
    background: transparent;
    padding: 0px;
    min-height: 0px;
    color: {tokens.accent};
    font-size: {RAIL_ADD_GLYPH_FONT_PX}px;
}}
QToolButton#datasetAddButton::menu-indicator {{ image: none; width: 0px; }}
QToolButton#datasetAddButton:hover:enabled {{ color: {tokens.accent_hover}; }}
QToolButton#datasetAddButton:disabled {{ color: {tokens.muted_text}; }}
/* The design's ``.seg``: one bordered pill whose chosen half is filled with the
   accent.  The halves lose their own borders and corner radius so the pair reads
   as a single control rather than as two adjacent buttons, and the filled half
   states the choice without relying on a subtle tint. */
QWidget[segmented="true"] {{
    border: 1px solid {tokens.border_strong};
    border-radius: 6px;
    background: {tokens.surface};
}}
QWidget[segmented="true"] QToolButton {{
    border: none;
    border-radius: 0px;
    padding: {SPACE_XS}px {SPACE_MD}px;
    min-height: {CONTROL_MIN_H}px;
    font-size: 13px;
    font-weight: 600;
    color: {tokens.muted_text};
    background: transparent;
}}
QWidget[segmented="true"] QToolButton:hover:enabled {{ background: {tokens.selection_bg}; }}
QWidget[segmented="true"] QToolButton:checked {{
    background: {tokens.accent};
    color: {tokens.accent_text};
}}
QWidget[segmented="true"] QToolButton:checked:hover:enabled {{ background: {tokens.accent_hover}; }}
"""


def _container_styles(tokens: ThemeTokens) -> str:
    return f"""
QFrame[sectionCard="true"] {{
    border: 1px solid {tokens.surface_border};
    border-radius: 8px;
    background: transparent;
}}
/* 设计稿 ``.insp-sec``：检视器那一栏是一栏被横线分节的正文，不是一列各带圆角框的卡片。
   五六段各画一圈框时读者看到的是一列卡片，段与段的关系要靠边距去猜；只留下边线，它们
   就读成同一栏的连续小节。末段之下没有下一节，那道线于是成了整栏的封边，要摘掉——设计稿
   把这个例外写成行内样式，这里挂在 ``lastSection`` 上由 ``mark_last_section`` 跟着可见性
   改。 */
QFrame[inspectorSection="true"] {{
    border: 0px;
    border-bottom: 1px solid {tokens.surface_border};
    border-radius: 0px;
    background: transparent;
}}
QFrame[inspectorSection="true"][lastSection="true"] {{ border-bottom: 0px; }}
/* 设计稿 ``.stack``（506-513 行）：引导页那一叠层是一只带外框的圆角盒子，里面每行用一道
   下边线分格，末行不画。分节的做法与上面那一族同源，区别是这一叠还有外框——外框说「这是
   一叠」，分隔线说「界面在这里」。四张各带边框圆角的卡片说的是「四件并列的东西」，而这
   四行是一叠：从空气到基底，相邻两行贴着的那个面就是一道界面。
   首末两行的外侧圆角替 CSS 的 ``overflow:hidden``。半无限那两行压一层淡底
   （``.lyr.semi``）：空气与基底不是这次拟合的对象，只是这一叠的上下界，颜色先把这件事
   说了，读者不必读完副行才明白第一行为什么没有厚度。 */
QWidget#structureStepStack {{
    border: 1px solid {tokens.border_strong};
    border-radius: {STACK_CONTAINER_RADIUS_PX}px;
    background: transparent;
}}
QFrame[stackRow="true"] {{
    border: 0px;
    border-bottom: 1px solid {tokens.surface_border};
    border-radius: 0px;
    background: transparent;
}}
QFrame[stackRow="true"][lastStackRow="true"] {{ border-bottom: 0px; }}
QFrame[stackRow="true"][semiInfinite="true"] {{ background: {tokens.surface}; }}
QFrame[stackRow="true"][firstStackRow="true"] {{
    border-top-left-radius: {STACK_ROW_RADIUS_PX}px;
    border-top-right-radius: {STACK_ROW_RADIUS_PX}px;
}}
QFrame[stackRow="true"][lastStackRow="true"] {{
    border-bottom-left-radius: {STACK_ROW_RADIUS_PX}px;
    border-bottom-right-radius: {STACK_ROW_RADIUS_PX}px;
}}
QWidget#plotInteractionToolbar {{
    background: {tokens.surface};
    border: 1px solid {tokens.surface_border};
    border-radius: 7px;
}}
/* The mode bar shares the plot's tab row, so its buttons must not carry the
   padding and min-height the general rule gives a button holding text: that
   turned an 18px glyph into a 39x43 slab and made the row tall enough to push
   the tabs out of line.  Square, glyph-sized cells keep the bar the narrow strip
   a charting mode bar is supposed to be. */
QWidget#plotInteractionToolbar QToolButton {{
    padding: 0px;
    min-width: {GLYPH_CELL_PX}px;
    max-width: {GLYPH_CELL_PX}px;
    min-height: {GLYPH_CELL_PX}px;
    max-height: {GLYPH_CELL_PX}px;
    border-radius: 4px;
}}
QLabel[sectionHeader="true"] {{
    font-weight: 700;
    padding: 2px 0px;
}}
/* 设计稿里的分区标题（.nav-sec 与 .insp-sec .h）自带精确内边距：.nav-sec 是
   12px/8px，量出来的行高就该是 12 + 一行字 + 8。上面那 2px 是给别处那些不按设计
   稿排的标题垫的，落到这一档会被加到 contentsMargins 上，让每个标题比设计稿高
   4px——六个标题就是 24px，正好够把机架里的一段内容挤出视口。这一档自己排版，所以
   把那点垫料收回去。 */
QLabel[sectionHeader="true"][faintText="true"] {{
    padding: 0px;
}}
QLabel[mutedText="true"] {{ color: {tokens.muted_text}; }}
QLabel[faintText="true"] {{ color: {tokens.faint_text}; }}
QLabel[emptyTitle="true"] {{ font-size: {FONT_PT_LG}pt; font-weight: 700; }}
QLabel[statusKind="ok"] {{ color: {tokens.ok}; }}
QLabel[statusKind="info"] {{ color: {tokens.info}; }}
QLabel[statusKind="warn"] {{ color: {tokens.warn}; }}
QLabel[statusKind="error"] {{ color: {tokens.error}; }}
/* 设计稿状态栏里帧③ 的「下一步：开始拟合」与帧④ 运行中的圆点都用强调色，它不是
   ok/info/warn/error 里的任何一个。 */
QLabel[statusKind="accent"] {{ color: {tokens.accent}; }}
/* ``.statusbar .dot{{width:8px;height:8px;border-radius:50%}}``：填色由语义决定，圆
   由半径给。控件是 8×8 的定尺，半径取一半就是整圆。 */
QLabel[statusDot="true"] {{
    border-radius: {STATUS_DOT_RADIUS_PX}px;
    background: {tokens.muted_text};
}}
QLabel[statusDot="true"][statusKind="ok"] {{ background: {tokens.ok}; }}
QLabel[statusDot="true"][statusKind="info"] {{ background: {tokens.info}; }}
QLabel[statusDot="true"][statusKind="warn"] {{ background: {tokens.warn}; }}
QLabel[statusDot="true"][statusKind="error"] {{ background: {tokens.error}; }}
QLabel[statusDot="true"][statusKind="accent"] {{ background: {tokens.accent}; }}
QLabel[hintBox="true"], QLabel[bannerBox="true"] {{
    border-radius: 8px;
    font-size: {FONT_PT_SM}pt;
}}
QLabel[hintBox="true"] {{ padding: 10px 12px; }}
QLabel[bannerBox="true"] {{ padding: 10px 14px; }}
QLabel[hintBox="true"][statusKind="info"] {{
    color: {tokens.muted_text};
    background: {_tinted(tokens.info, HINT_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.info, HINT_BORDER_ALPHA)};
    border-left: 3px solid {tokens.info};
}}
QLabel[hintBox="true"][statusKind="warn"] {{
    color: {tokens.muted_text};
    background: {_tinted(tokens.warn, HINT_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.warn, HINT_BORDER_ALPHA)};
    border-left: 3px solid {tokens.warn};
}}
QLabel[bannerBox="true"][statusKind="info"] {{
    color: {tokens.muted_text};
    background: {_tinted(tokens.info, BANNER_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.info, BANNER_BORDER_ALPHA)};
}}
QLabel[bannerBox="true"][statusKind="warn"] {{
    color: {tokens.muted_text};
    background: {_tinted(tokens.warn, BANNER_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.warn, BANNER_BORDER_ALPHA)};
}}
/* 设计稿 ``.badge``：胶囊形，底色与边框取自身状态色的淡染。判读结论只上文字颜色
   时和周围的正文一样是一行字，读者得靠颜色分辨哪几个词是结论;给它一个封闭的外形，
   一排结论才读成一排独立的标记。半径给足够大的定值，Qt 没有 999px 的等价写法,
   胶囊由高度决定而不是由半径决定。
   没有状态的徽章落在下面这条基础规则上，也就是设计稿的 ``.badge.mut``：中性底色配
   静音文字。这类结论（比如「未运行自助」）不能借用四种状态色里的任何一种，否则
   「这项没做」会读成「这项通过」或「这项有问题」。 */
QLabel[badge="true"] {{
    font-size: {BADGE_FONT_PX}px;
    font-weight: {BADGE_FONT_WEIGHT.value};
    padding: {BADGE_PAD_V_PX}px {BADGE_PAD_H_PX}px;
    border-radius: {BADGE_RADIUS_PX}px;
    color: {tokens.muted_text};
    background: {tokens.surface};
    border: 1px solid {tokens.surface_border};
}}
QLabel[badge="true"][statusKind="ok"] {{
    color: {tokens.ok};
    background: {_tinted(tokens.ok, BADGE_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.ok, BADGE_BORDER_ALPHA)};
}}
QLabel[badge="true"][statusKind="info"] {{
    color: {tokens.info};
    background: {_tinted(tokens.info, BADGE_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.info, BADGE_BORDER_ALPHA)};
}}
QLabel[badge="true"][statusKind="warn"] {{
    color: {tokens.warn};
    background: {_tinted(tokens.warn, BADGE_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.warn, BADGE_BORDER_ALPHA)};
}}
QLabel[badge="true"][statusKind="error"] {{
    color: {tokens.error};
    background: {_tinted(tokens.error, BADGE_FILL_ALPHA)};
    border: 1px solid {_tinted(tokens.error, BADGE_BORDER_ALPHA)};
}}
QLabel[mono="true"] {{ font-family: {MONO_FONT_STACK}; }}
QLabel[codeBlock="true"] {{
    font-family: {MONO_FONT_STACK};
    font-size: {FONT_PT_SM}pt;
    color: {tokens.muted_text};
    background: {tokens.surface};
    border: 1px solid {tokens.surface_border};
    border-radius: 6px;
    padding: {SPACE_SM}px {SPACE_MD}px;
}}
/* 设计稿三张「步骤地图」（``.pstep``、``.wizhead .wh``、``.stg``）只给当前那一步的名字上
   色：``.pstep.current .body .t``、``.wh.current .tt``、``.stg.current .nm`` 都是
   ``--accent``；done 与 pending 没有名字规则——完成靠圆点变绿说，未开始靠圆点的浅描边说。
   粗细留给各自的字体对象：``.pstep`` 的标题恒 600，梯子与向导的当前项是 700，一条规则给不
   出两个值。 */
QLabel[stepState="current"] {{ color: {tokens.accent}; }}
/* 拟合梯子（``.stg``）是三张地图里唯一整行降调的（``.stg.pending{{opacity:.5}}``），它的
   标记也还没做成 ``.stg .ic`` 那样的实心圈，所以这两档颜色只留给梯子自己。 */
QLabel[ladderStep="true"][stepState="done"] {{ color: {tokens.ok}; }}
QLabel[ladderStep="true"][stepState="pending"] {{ color: {tokens.muted_text}; opacity: 0.5; }}
/* ``.stg.current .nm{{font-weight:700}}``：粗细留在 QSS 里，这样每帧只换属性不换字体对象。
   同一条也压到标记上，而设计稿的 ``.stg .ic`` 本来就是 700。 */
QLabel[ladderStep="true"][stepState="current"] {{ font-weight: 700; background: {tokens.selection_bg}; }}
/* 设计稿 ``.stg .ic``：阶段标记是 18px 圆形，done 绿底白字，current 蓝底白字，pending 灰边白底。 */
QLabel[stageMarker="true"] {{ border-radius: 9px; border: 2px solid {tokens.border_strong}; background: {tokens.surface}; }}
QLabel[stageMarker="true"][stepState="done"] {{ background: {tokens.ok}; color: {tokens.accent_text}; border-color: {tokens.ok}; }}
QLabel[stageMarker="true"][stepState="current"] {{ background: {tokens.accent}; color: {tokens.accent_text}; border-color: {tokens.accent}; }}
QLabel[stageMarker="true"][stepState="pending"] {{ background: transparent; border-color: {tokens.muted_text}; }}
QLabel[stepDot="true"] {{
    border-radius: 10px;
    font-size: 9pt;
    font-weight: 600;
}}
/* 向导抬头那一格的记号大一档（``.wizhead .wh .n{{width:30px;height:30px}}``），圆角只能跟着
   直径走：上面那条 10px 是左栏管线 20px 圆点的一半，同一条压在 30px 的方块上画出来是个圆角
   方块。颜色与状态那四条仍由 ``stepDot`` 一处供，两张步骤地图共用同一套记号词汇。 */
QLabel[stepDot="true"][wizardStep="true"] {{ border-radius: {STEP_DOT_PX // 2}px; }}
QLabel[stepDot="true"][stepState="current"] {{
    background: {tokens.accent};
    color: {tokens.accent_text};
    border: 2px solid {tokens.accent_hover};
}}
QLabel[stepDot="true"][stepState="done"] {{
    background: {tokens.ok};
    color: {tokens.accent_text};
}}
QLabel[stepDot="true"][stepState="pending"] {{
    background: transparent;
    border: 1px solid {tokens.muted_text};
}}
/* The pipeline rail is a painted frame, so it needs a background of its own: a
   bare QFrame::VLine draws in the frame colour, which the dark palette renders
   near-invisible against the dock.  Keyed on a property rather than an
   objectName because every step builds its own rail. */
*[pipelineRail="true"] {{
    /* min-height 不能省：主窗口挂上 stylesheet 后 QStyleSheetStyle 接管裸 QWidget 的几何，
       纵向 ``QSizePolicy.Expanding`` 对没有内容的分隔条失效——它会缩到 26px、不再贯通整格。
       钉到与抬头行高同源的 STEP_DOT_PX + 2×SPACE_XS（圆点直径加上下内边距，正是四格行高），
       竖线才和格子一样高；末格不画（见 panel._StepHeader）。 */
    background: {tokens.surface_border};
    border: none;
    min-height: {STEP_DOT_PX + 2 * SPACE_XS}px;
}}
/* 设计稿 ``.confbadge``（HTML 181-188、421）：判定不是一行染了色的字，而是一只淡染的
   圆角框——字形一列，判定与理由叠成另一列。没有结果时落在下面这条基础规则上：中性边框
   配透明底，四种状态色一种都不借，否则「还没跑」会读成「通过」或「失败」。两个属性选择
   器压得住一个，所以四条状态规则不必再加权重。 */
QFrame[verdictBox="true"] {{
    border-radius: 8px;
    border: 1px solid {tokens.surface_border};
    background: transparent;
}}
QFrame[verdictBox="true"][statusKind="ok"] {{
    background: {_tinted(tokens.ok, VERDICT_FILL_ALPHAS["ok"])};
    border: 1px solid {_tinted(tokens.ok, VERDICT_BORDER_ALPHA)};
}}
QFrame[verdictBox="true"][statusKind="info"] {{
    background: {_tinted(tokens.info, VERDICT_FILL_ALPHAS["info"])};
    border: 1px solid {_tinted(tokens.info, VERDICT_BORDER_ALPHA)};
}}
QFrame[verdictBox="true"][statusKind="warn"] {{
    background: {_tinted(tokens.warn, VERDICT_FILL_ALPHAS["warn"])};
    border: 1px solid {_tinted(tokens.warn, VERDICT_BORDER_ALPHA)};
}}
QFrame[verdictBox="true"][statusKind="error"] {{
    background: {_tinted(tokens.error, VERDICT_FILL_ALPHAS["error"])};
    border: 1px solid {_tinted(tokens.error, VERDICT_BORDER_ALPHA)};
}}
/* ``.confbadge .txt .l`` / ``.r``：判定 700/15px，理由 11.5px 静音。两行同号时框里读起来
   是两句并列的话，小一档理由才读成判定的注脚。 */
QLabel#confidenceBadge {{ font-weight: 700; font-size: {FONT_PT_MD}pt; }}
QLabel#confidenceReason {{ font-size: {FONT_PT_SM}pt; }}
QTabBar::tab {{ padding: {SPACE_XS}px {SPACE_MD}px; }}
/* 帧①/⑤ 的画布列由 ``.canvas-top`` 那条线分上下两段，容器不再自己画框：两个
   ``QTabWidget`` 退成只放页面的壳（自己那条 tab 条由 ``canvas_tab_group`` 藏起来，露在
   外面的是行里那条扁 tab 条），``::pane`` 若还画一圈边框，就会紧贴着行的下边框再叠一道。 */
QTabWidget#reflectivityTabs::pane {{ border: 0px; }}
QTabWidget#analysisTabs::pane {{ border: 0px; }}
/* 设计稿 ``.canvas-top``（HTML 148-152）：画布列的行头，tab 贴着上沿，选中那个换成面板
   底色并戴一条 2px 强调色下划线。未选中的也留出同样厚的透明下边框，否则选中时文字会
   被那 2px 顶着上下跳一格。 */
QWidget[canvasTop="true"] {{ border-bottom: 1px solid {tokens.surface_border}; }}
QTabBar[canvasTab="true"]::tab {{
    padding: {SPACE_XS}px {SPACE_MD}px;
    margin: 0px 1px;
    color: {tokens.muted_text};
    font-weight: 600;
    background: transparent;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar[canvasTab="true"]::tab:hover {{ color: {tokens.accent}; }}
QTabBar[canvasTab="true"]::tab:selected {{
    color: {tokens.accent};
    border-color: {tokens.surface_border};
    border-bottom-color: {tokens.accent};
}}
QGroupBox {{
    border: 1px solid {tokens.border_strong};
    border-radius: 8px;
    margin-top: {SPACE_MD}px;
    padding-top: {SPACE_XS}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {SPACE_SM}px;
    padding: 0px {SPACE_XS}px;
    font-weight: 700;
}}
"""


def _chrome_styles(tokens: ThemeTokens) -> str:
    return f"""
QToolBar#mainToolbar {{
    padding: {SPACE_XS}px {SPACE_SM}px;
    spacing: {SPACE_SM}px;
    border-bottom: 1px solid {tokens.surface_border};
}}
/* ``.statusbar`` 是一条 30px 高、上边一道 1px 边线的行。段与段之间的 12px 由
   ``status_bar`` 的容器给，所以这里不再给标签补内边距——那点内边距是从前用来近似段间距
   的，留着会把说明文字与它的取值也撑开。底色跟机架同取 --panel-2，见 ``_shell_styles``。 */
QStatusBar#mainStatusBar {{ border-top: 1px solid {tokens.surface_border}; }}
QProgressBar {{
    border: 1px solid {tokens.surface_border};
    border-radius: 6px;
    text-align: center;
    min-height: 10px;
    max-height: 10px;
}}
QProgressBar::chunk {{ background: {tokens.accent}; border-radius: 6px; }}
QTreeWidget, QTableWidget, QListWidget {{
    border: 1px solid {tokens.surface_border};
    border-radius: 6px;
    selection-background-color: {tokens.selection_bg};
}}
/* 设计稿 ``.stack``：层堆叠是一只 8px 圆角的盒子，四行 ``.lyr`` 靠 1px 细线分隔，最后
   一行不画线（否则和盒子自己的下边框叠成两道）。行的内容由 ``structure/rows.py`` 的行
   控件画，这里只负责盒子、分隔线和那三档字号。 */
QTreeWidget#structureTree {{
    border-radius: 8px;
    outline: 0px;
}}
QWidget#structureLayerRow {{
    border-left: 3px solid transparent;
    border-bottom: 1px solid {tokens.surface_border};
}}
QWidget#structureLayerRow[lastRow="true"] {{ border-bottom: 0px; }}
QWidget#structureLayerRow[semiInfinite="true"] {{ background: {tokens.surface}; }}
/* ``.lyr.sel`` 的左缘 3px：树自己的高亮画在行控件底下，半无限那两行的底色会把它盖住，
   所以选中这件事得由行自己再说一遍。 */
QWidget#structureLayerRow[rowSelected="true"] {{ border-left-color: {tokens.accent}; }}
QLabel#structureLayerName {{ font-weight: 600; }}
QLabel#structureLayerDetail {{
    font-size: {FONT_PT_SM}pt;
    color: {tokens.muted_text};
}}
QLabel#structureLayerReading {{
    font-size: {FONT_PT_SM}pt;
    color: {tokens.muted_text};
}}
QLabel#structureLayerDrag {{ color: {tokens.muted_text}; }}
QHeaderView::section {{
    padding: 3px {SPACE_SM}px;
    border: 0px;
    border-bottom: 1px solid {tokens.surface_border};
    font-weight: 600;
}}
/* 设计稿 ``table.grid``：结果读数表不画外框，也不画竖线——横线由
   ``results.values.ResultValueDelegate`` 逐行画。外框留着就是双线的来源，这张表本来
   就装在检视器那一段里。表头跟着 ``th`` 走：小一号、700、次要色。 */
QTableWidget#resultValueTable {{
    border: 0px;
    border-radius: 0px;
    background: transparent;
    selection-background-color: transparent;
}}
QTableWidget#resultValueTable::item {{ padding: 3px {SPACE_SM}px; }}
QTableWidget#resultValueTable QHeaderView::section {{
    background: transparent;
    color: {tokens.muted_text};
    font-size: {FONT_PT_SM}pt;
    font-weight: 700;
    padding: {SPACE_XS}px {SPACE_SM}px;
}}
/* 设计稿的候选解一节：每一行自己是一只带边框的圆角盒子，盒子由
   ``results.candidate_row.CandidateRowDelegate`` 画。这张列表因此不能有自己的外框（双线）、
   自己的底色（压在行盒子下面透出来），也不能有 Qt 的选中蓝底——选中那行的主色框与
   ``--selection`` 底同样由委托画，两层叠起来会比设计稿重一档。
   通用规则给每张列表一道 1px 外框，``setFrameShape(NoFrame)`` 管不到它，只有按 id 覆盖
   才抹得掉；留着它，按内容算出来的列表高会被这道框吃掉两像素，末行被裁。 */
QListWidget#candidateList {{
    border: 0px;
    border-radius: 0px;
    background: transparent;
    selection-background-color: transparent;
}}
QListWidget#candidateList::item {{ padding: 0px; border: 0px; }}
QScrollArea#analysisScroll {{ border: 0px; background: transparent; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    border: 1px solid {tokens.border_strong};
    border-radius: 6px;
    padding: 2px {SPACE_SM}px;
    min-height: {CONTROL_MIN_H}px;
    background: transparent;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {tokens.accent};
    outline: none;
}}
QComboBox:hover:enabled, QSpinBox:hover:enabled, QDoubleSpinBox:hover:enabled, QLineEdit:hover:enabled {{
    border-color: {tokens.accent};
}}
QComboBox::drop-down {{
    border: 0px;
    padding-right: {SPACE_SM}px;
}}
/* 设计稿 ``.field .inp.sel``：材料那一格是一整只框，``▾`` 在框里靠右。边画在外面这只
   ``QFrame`` 上，里面的输入框和按钮各自去掉自己的边——两道边就成了两只挨着的框，读者会
   以为那枚 ``▾`` 是另一个控件，而设计稿画的是点哪儿都在编辑同一样东西的一整格。 */
QFrame#selectedLayerMaterialField {{
    border: 1px solid {tokens.surface_border};
    border-radius: 6px;
    background: transparent;
    min-height: {SELECTED_LAYER_INPUT_H_PX}px;
}}
QLineEdit#selectedLayerFormulaInput {{
    border: 0px;
    border-radius: 0px;
    padding: 0px;
    min-height: 0px;
    background: transparent;
}}
QToolButton#selectedLayerMaterialMenuButton {{
    border: 0px;
    padding: 0px;
    min-height: 0px;
    background: transparent;
    color: {tokens.faint_text};
}}
/* 带菜单的 ``QToolButton`` 自己还要画一枚下拉箭头，而这一枚 ``▾`` 是文本给的：不抹掉就是
   两个箭头并排。 */
QToolButton#selectedLayerMaterialMenuButton::menu-indicator {{ image: none; width: 0px; }}
/* 设计稿 ``.field .inp{{height:30px}}``：这三格比应用里其他输入框高一档，因为它们是这张卡的
   主体。按 id 只发给这张卡——``CONTROL_MIN_H`` 同时是层堆叠的行高下限，改它会连带把那一列
   撑开。 */
QDoubleSpinBox#selectedLayerThicknessInput,
QDoubleSpinBox#selectedLayerRoughnessInput,
QDoubleSpinBox#selectedLayerDensityInput {{
    padding: 0px {SPACE_SM}px;
    min-height: {SELECTED_LAYER_INPUT_H_PX}px;
}}
/* 设计稿的 ``.lock``：14px 见方的空框，锁住时填成主色并打一个白勾。设计稿那个勾是
   ``position:absolute`` 摆出去的 10px 字，故意越出方框；Qt 把文本居中画在内容区里（14 减去
   两道 2px 的边只剩 10px），同样字号会被裁掉一截，所以这里给 9px。 */
QLabel#selectedLayerDensityLock {{
    border: {SELECTED_LAYER_LOCK_BORDER_PX}px solid {tokens.muted_text};
    border-radius: {SELECTED_LAYER_LOCK_RADIUS_PX}px;
    background: transparent;
}}
QLabel#selectedLayerDensityLock[locked="true"] {{
    border-color: {tokens.accent};
    background: {tokens.accent};
    color: {tokens.accent_text};
    font-size: 9px;
}}
"""


def shell_washes(palette: QPalette) -> tuple[str, str]:
    """报回外壳的两块底色：中间画布那块，和两侧机架那块。

    设计稿 ``:root`` 把外壳分三层：``--app-bg`` 是窗底，``--panel`` 是内容面，``--panel-2`` 比内容面
    暗一档给两侧机架用。调色板里对得上的是 ``Base`` 与 ``AlternateBase``。深色调色板把
    ``AlternateBase`` 设成和 ``Base`` 同色，那时机架色改从 ``Window`` 取——否则三栏一样亮，设计稿
    「两侧是机架、中间才是内容」这层区分就没了。
    """
    base = palette.color(QPalette.ColorRole.Base)
    alternate = palette.color(QPalette.ColorRole.AlternateBase)
    rail = palette.color(QPalette.ColorRole.Window) if alternate == base else alternate
    return base.name(), rail.name()


def _shell_styles(tokens: ThemeTokens, canvas_wash: str, rail_wash: str) -> str:
    return f"""
/* 设计稿 .appbody 的三栏底色：canvas 取 --panel 当内容面，nav 与 inspector 同取 --panel-2
   退后一档当机架。三栏建出来是 QWidget，选择器写成 QFrame 一条也命中不了。
   三条规则分开写不并成一个选择器：并起来读样式表的人得先拆选择器才知道哪一栏是哪块底。 */
QWidget#navigationColumn {{
    background: {rail_wash};
}}
QWidget#canvasColumn {{
    background: {canvas_wash};
}}
QWidget#inspectorColumn {{
    background: {rail_wash};
}}
/* 设计稿的 .statusbar 也铺 --panel-2：它和两侧机架同属「不是内容面」的那一层。 */
QStatusBar#mainStatusBar {{
    background: {rail_wash};
}}
/* 设计稿 nav 的 border-right 是 1px solid var(--border)：栏与栏之间是一道线。
   线画在手柄上而不是画在栏上——手柄那几像素挪不走，边框画在栏上时手柄会留在线和画布之间，
   成为设计稿里没有的第二道缝。 */
QSplitter#workspaceSplitter::handle {{
    background: {tokens.surface_border};
}}
/* 机架色只有在上面那几层不铺底色时才看得见。QScrollArea 给视口和被滚动的那只控件都开了
   autoFillBackground，两层默认铺的是内容色，一整条机架会被涂成白的——设计稿里露在数据集行
   和管线步骤之间的是 --panel-2。三只滚动区连带各自的视口一起放平，栏色才透上来。 */
QScrollArea#dataPanelScroll, QScrollArea#pipelineNavScroll, QScrollArea#inspectorScroll {{
    border: 0px;
    background: transparent;
}}
QScrollArea#dataPanelScroll > QWidget#qt_scrollarea_viewport,
QScrollArea#pipelineNavScroll > QWidget#qt_scrollarea_viewport,
QScrollArea#inspectorScroll > QWidget#qt_scrollarea_viewport {{
    background: transparent;
}}
QWidget#dataPanel, QWidget#pipelineNav, QWidget#inspectorBody {{
    background: transparent;
}}
/* 中栏那只滚动区同理，只是它遮的不是机架色而是画布区。「总进度」卡自己带底色和圆角边框，
   滚动区在它周围再铺一层内容色时，圆角外会多出一圈白——看着像这张卡被垫在另一张卡上。 */
QScrollArea#fitProgressScroll {{
    border: 0px;
    background: transparent;
}}
QScrollArea#fitProgressScroll > QWidget#qt_scrollarea_viewport {{
    background: transparent;
}}
/* 设计稿的 .ds 行直接坐在机架上：这张列表不画外框，也不铺自己的底色。通用列表规则那只
   1px 圆角盒子在左栏是多出来的一层，盒里那片内容色还会把机架色整段盖掉——同一栏于是有
   两种白：列表里的和列表外的。
   白色留给 .ds.on：设计稿用整行的 --panel 说「此刻在看这一集」，--selection 那层淡染留给
   一掠而过的悬停。两者同色时这两句话在屏幕上就分不开了。选中写在悬停之后，掠过活动那一行
   时白底不会被淡染盖掉——设计稿的两条规则也是这个先后。 */
QTreeWidget#datasetTree {{
    border: 0px;
    border-radius: 0px;
    background: transparent;
    outline: 0px;
}}
QTreeWidget#datasetTree::item:hover {{
    background: {tokens.selection_bg};
}}
QTreeWidget#datasetTree::item:selected {{
    background: {canvas_wash};
}}
/* 设计稿 ``.wizhead``（HTML 487-493）：向导抬头是一条铺 ``--panel-2`` 的分格条，底下压一道
   1px 的线把它与内容面分开，当前那一格换成 ``--panel`` 的白。这三块底色只有这里拿得到，所以
   规则落在这一段而不是跟 ``stepDot`` 那几条放在一起。
   白底给的是「走到哪一步」：分格之后四格并列，光靠一枚圆点变色隔着一屏分不出哪格是活的，
   而整格换底是设计稿唯一用来报当前步的手段。选择器带上 ``stepHeader`` 与两个属性，压得住
   ``QPushButton[ghost="true"]`` 那条透明底，不必依赖两段样式表的拼接先后。 */
QWidget#stepHeader {{
    background: {rail_wash};
    border-bottom: 1px solid {tokens.surface_border};
}}
QWidget#stepHeader QPushButton[ghost="true"][stepState="current"] {{
    background: {canvas_wash};
}}
"""


def build_stylesheet(palette: QPalette) -> str:
    tokens = palette_tokens(palette)
    canvas_wash, rail_wash = shell_washes(palette)
    return "".join(
        (
            _button_styles(tokens),
            _container_styles(tokens),
            _chrome_styles(tokens),
            _shell_styles(tokens, canvas_wash, rail_wash),
        )
    )


# 设计稿命令栏最右边那枚键的两种面孔：亮色时给出「转深色」的月亮，深色时给出「转亮色」
# 的太阳。字形报的是按下去会去哪儿，不是当前在哪儿——一枚只显示现状的开关读者按之前
# 不知道会发生什么。
APPEARANCE_DARK_GLYPH = "☾"
APPEARANCE_LIGHT_GLYPH = "☀"

# 深色外观的三块底色。窗口比基底稍亮一档，让卡片浮在背景上而不是与之齐平。
DARK_WINDOW = "#1B1F24"
DARK_BASE = "#14181D"
DARK_TEXT = "#E6E9EE"

# 设计稿亮色的三层底色：``--app-bg`` 是窗口壳那层灰、``--panel`` 是内容面板的白、
# ``--panel-2`` 是机架上那层介于两者之间的洗色。三层必须分别落在 Window / Base /
# AlternateBase 上，``shell_washes`` 才能把「画布该白、机架该灰」从调色板里读出来：平台
# 默认的亮色板这三个角色全是白的，读出来机架色与画布色相等，屏幕上也就没有机架。
LIGHT_WINDOW = "#EDEFF3"
LIGHT_BASE = "#FFFFFF"
LIGHT_ALTERNATE = "#F7F8FA"
# 设计稿的 ``--ink``。它不只是「一种深灰」：浅色外观下每一个字的颜色都从这里来（次要与更淡
# 的两档是叠在它上面的 alpha 灰），所以这一位照设计稿写，而不是取个差不多的深灰。
LIGHT_TEXT = "#1B1F27"


def _appearance_palette(
    *,
    window: str,
    base: str,
    alternate: str,
    text: str,
    tokens: ThemeTokens,
) -> QPalette:
    """一份外观的调色板：三层底色、文字色，以及与词元对齐的选中色。"""
    palette = QPalette()
    window_color = QColor(window)
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button):
        palette.setColor(role, window_color)
    palette.setColor(QPalette.ColorRole.Base, QColor(base))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(alternate))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.ToolTipText,
    ):
        palette.setColor(role, QColor(text))
    palette.setColor(QPalette.ColorRole.ToolTipBase, window_color)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(tokens.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(tokens.accent_text))
    return palette


def dark_palette() -> QPalette:
    """Build the palette that ``palette_tokens`` resolves to ``DARK_TOKENS``."""
    return _appearance_palette(
        window=DARK_WINDOW,
        # 深色只有两层：面板与机架同色，``shell_washes`` 于是回退到窗口色当机架，而深色的
        # 窗口本来就比基底亮一档，机架该亮于画布这件事仍然成立。
        base=DARK_BASE,
        alternate=DARK_BASE,
        text=DARK_TEXT,
        tokens=DARK_TOKENS,
    )


def light_palette() -> QPalette:
    """Build the palette that ``shell_washes`` resolves to the design's three washes."""
    return _appearance_palette(
        window=LIGHT_WINDOW,
        base=LIGHT_BASE,
        alternate=LIGHT_ALTERNATE,
        text=LIGHT_TEXT,
        tokens=LIGHT_TOKENS,
    )


def _resolved_copy(source: QPalette) -> QPalette:
    """Copy a palette with every role written out explicitly.

    ``QApplication.setPalette`` merges the given palette over the current one and
    only takes the roles that palette has actually resolved.  A plain copy of the
    application palette usually resolves nothing, so handing it back would leave
    the dark colours in place.  Writing each role makes the return trip stick.
    """
    copy = QPalette()
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        for role in QPalette.ColorRole:
            if role == QPalette.ColorRole.NColorRoles:
                continue
            copy.setColor(group, role, source.color(group, role))
    return copy


# 第一次切深色前的那份调色板，留着做回程票。
LIGHT_PALETTE: QPalette | None = None


def set_dark_appearance(application: QApplication, dark: bool) -> str:
    """Switch the whole application between the light and dark appearances.

    The stylesheet is derived from the palette, so flipping the palette and
    re-applying is enough: every ``tokens.*`` colour, the plot palette read by
    ``current_plot_palette`` and the status kinds all follow from that one read.
    """
    global LIGHT_PALETTE
    if dark:
        # 记住来时的那一份而不是回头问 ``style().standardPalette()``：某些平台样式的
        # standardPalette 会跟着当前应用调色板走，换过去就再也换不回来了。
        if LIGHT_PALETTE is None:
            LIGHT_PALETTE = _resolved_copy(application.palette())
        application.setPalette(dark_palette())
    elif LIGHT_PALETTE is not None:
        application.setPalette(LIGHT_PALETTE)
    return apply_theme(application)


def apply_theme(application: QApplication) -> str:
    """Install the palette-matched stylesheet and return the applied sheet."""
    sheet = build_stylesheet(application.palette())
    if application.styleSheet() != sheet:
        application.setStyleSheet(sheet)
    return sheet


def repolish(widget: QWidget) -> None:
    """Re-run the style on `widget` so a just-changed QSS property takes effect.

    Qt resolves property selectors when it polishes a widget, not when the
    property changes, so a ``setProperty`` on a live widget is invisible until
    the style is torn down and re-applied.

    ``unpolish``/``polish`` 只重算画上去的东西；边框宽度还兼着控件的盒模型——``QFrame``
    把它缓存成 frameWidth 与内边距，只在收到 ``StyleChange`` 时才重算。少了这一步，一个
    改边框的属性会画对而量错：``contentsRect()`` 仍按旧边框缩进，靠它排版的邻居也跟着错。
    """
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    QCoreApplication.sendEvent(widget, QEvent(QEvent.Type.StyleChange))


def set_status_kind(widget: QWidget, kind: str) -> None:
    """Set the semantic status property and repolish so QSS reevaluates."""
    if kind not in ("", "ok", "info", "warn", "error", "accent"):
        raise ValueError(f"unsupported status kind: {kind}")
    if widget.property("statusKind") == kind:
        return
    widget.setProperty("statusKind", kind)
    repolish(widget)


def mark_last_section(sections: Iterable[QWidget]) -> None:
    """给这一串检视器分节里最后一个还露着的那段摘掉横线。

    ``.insp-sec`` 的下边线是「这一节到此为止」，末段之下没有下一节，那道线就成了整栏的
    封边——设计稿把这个例外写成行内样式（``style="border-bottom:none"``）。哪一段是末段
    随步骤变（按步骤收起的段落不同），所以这件事不能在构造期钉死，得跟着可见性一起改。

    判据是 ``isHidden()`` 而不是 ``isVisible()``：窗口还没 show 出来时后者对每一段都是
    False，末段就无从认定了；前者只读这一段自己被没被收起来。
    """
    ordered = list(sections)
    last = next((section for section in reversed(ordered) if not section.isHidden()), None)
    for section in ordered:
        wanted = section is last
        if bool(section.property("lastSection")) == wanted:
            continue
        section.setProperty("lastSection", wanted)
        repolish(section)


def mark_hint(widget: QWidget) -> None:
    """Turn an advisory line into a callout box.

    Advisories sit next to the controls they qualify, so plain coloured text
    reads as a verdict about those controls rather than as a note beside them.
    The box carries the status hue on its left edge instead, which both points at
    the neighbour being qualified and lets the body text stay muted.
    """
    _mark_callout(widget, "hintBox", "hint box")


def mark_banner(widget: QWidget) -> None:
    """Turn a run-wide precondition into a banner box.

    A banner states something that governs everything else on the card, so unlike
    a hint it has no neighbouring control to point at and drops the accent bar
    that would send the reader looking for one.  Losing the bar also loses the one
    solid edge of status colour, so the fill and border each come in a step
    stronger to keep the block visible against the card.
    """
    _mark_callout(widget, "bannerBox", "banner box")


def mark_code_block(widget: QWidget) -> None:
    """Frame a label as a quotation of file contents rather than as prose.

    Two separate things are wrong with showing a pretty-printed file in a plain
    label.  Proportional glyph widths pull the indentation out of column, and the
    indentation is the whole reason the payload was pretty-printed at all.  And
    without a frame the text reads as the dialog's own wording, when it is really
    an excerpt of what some other file is about to contain -- so the sunken fill
    carries the quoting that the surrounding prose cannot say for it.

    Unlike a callout this takes no status kind: an excerpt makes no claim about
    the run, it only shows what was written.
    """
    widget.setProperty("codeBlock", True)
    repolish(widget)


def _mark_callout(widget: QWidget, flag: str, described: str) -> None:
    """Flag a label for one of the callout rules and repolish it.

    Colour arrives through ``statusKind``: only ``info`` and ``warn`` are painted,
    because a callout is a note or a caution and never a success or a failure.
    Refusing the other kinds keeps a caller from getting padding and a radius with
    no border and no fill -- a box that silently lost its frame.
    """
    kind = widget.property("statusKind")
    if kind not in ("info", "warn"):
        raise ValueError(f"a {described} needs an info or warn status kind, got: {kind!r}")
    widget.setProperty(flag, True)
    repolish(widget)


def set_step_state(widget: QWidget, state: str) -> None:
    """Set the guided-step state property and repolish so QSS reevaluates.

    Progress is double-encoded: a done step also carries a ✓ glyph and a filled
    marker, so the state reads without relying on colour alone.  What this property
    paints is deliberately narrow -- the current step's accent, plus the fitting
    ladder's own done/pending shades -- because each of the three step maps in the
    design colours a different amount, and weight belongs to whichever map asks for
    it.  Colour still comes from the theme, which is why it follows a light/dark
    switch like every other status token.
    """
    if state not in ("", "done", "current", "pending"):
        raise ValueError(f"unsupported step state: {state}")
    if widget.property("stepState") == state:
        return
    widget.setProperty("stepState", state)
    repolish(widget)


def apply_section_heading(label: QLabel, *, tracking_px: float = SECTION_HEADING_TRACKING_PX) -> None:
    """把设计稿的分节抬头一档字排到一个已有的 ``QLabel`` 上。

    ``.nav-sec``（左栏的「数据集」「分析管线」）和 ``.insp-sec .h``（右栏各节抬头）在设计稿
    里是同一档字：11px、700、``--ink-faint``。所以这一档只有一处实现——分成两处，
    左右两栏的抬头就会各自漂移，而它们在同一屏上并列可见。

    字号与字距落在字体对象上，颜色走 ``faintText`` 规则：颜色得跟着浅/深外观切换，而 QSS
    才知道当前是哪套 token。粗细沿用 ``sectionHeader``，那条规则本就只给 ``font-weight``。

    字距是两栏唯一不同的那一项（``.nav-sec`` 是 .8px，``.insp-sec .h`` 是 .6px），所以它是
    参数而不是常量；其余三项在两栏里逐字相同，合成一处才不会各自漂移。
    """
    label.setProperty("sectionHeader", True)
    label.setProperty("faintText", True)
    font = label.font()
    font.setPixelSize(SECTION_HEADING_FONT_PX)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking_px)
    label.setFont(font)


def apply_step_eyebrow(label: QLabel) -> None:
    """把引导页正文之上那行「第 N 步 · 共 M 步」的一档字排到一个已有的 ``QLabel`` 上。

    设计稿 ``.wizbody .eyebrow2`` 是 12px、700、字距 .6px 的 ``--accent`` 小字。这一句唯一的
    作用是垫出位置感，所以它必须比同卡里 13.5px 的正文小一档、粗一档——同字号同字重时它读作
    正文的第一句，而不是页眉。

    颜色仍走 ``stepState`` 那条 QSS 通道，它得跟着浅/深外观换 token；其余三项落字体对象：
    Qt 的样式表没有 ``letter-spacing``，字号与字距同出设计稿的一条规则，字重按
    ``set_step_state`` 的分工归各自的字体对象（那条规则只给颜色）。
    """
    set_step_state(label, "current")
    font = label.font()
    font.setPixelSize(STEP_EYEBROW_FONT_PX)
    font.setWeight(QFont.Weight.Bold)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, STEP_EYEBROW_TRACKING_PX)
    label.setFont(font)


def titled_card(
    parent: QWidget | None,
    name: str,
    title: str,
    subtitle: str,
    *,
    flat: bool = False,
) -> tuple[QFrame, QVBoxLayout]:
    """A captioned section frame headed by a title and a muted subtitle.

    The card lives here rather than in one panel because several panels present
    the same unit and a bordered box on its own is ambiguous: the title says
    which section it is and the subtitle says how to drive it, so a proportional
    diagram is not mistaken for an ornament.  Returns the frame and the layout
    under the header so callers add content without counting header rows.

    Title and subtitle share one row, pushed to opposite ends the way the
    mockup's ``.insp-sec .h`` is a ``space-between`` flex row.  Stacking them
    instead cost two text rows per card, and letting the hint wrap made that
    cost grow with the copy, which is what pushed the results dock past the
    documented minimum window.

    The hint is typeset a step down from the title, as ``.plotcard .ph .sub`` is
    against ``.t``, and elides rather than demanding its whole string back from
    the column: it is the least load-bearing text in the card, so it is what
    gives way when the column is narrow.  Spelled in colour alone it carried the
    title's width while saying less, which put the stack card 45px past the
    264px column it lives in.

    The title is also the card's accessible name.  A section used to sit inside a
    QDockWidget, which announced itself; the fixed columns have no title bars, so
    the caption drawn here is the only label the section has left and a screen
    reader would otherwise enter an unnamed frame.

    ``flat`` 切到设计稿的 ``.insp-sec``：只有一道下边线、没有圆角、没有侧边框，抬头压成
    11px/700/字距 .6px 的 ``--ink-faint``。这一档必须由调用处点名而不是全局改掉——画布上
    那几张图卡是 ``.plotcard``，照旧带一圈框。两种形状挂在两个属性上（``sectionCard`` 与
    ``inspectorSection``）而不是共用一个：边框规则钉在前者上，共用就没法只改右栏。
    """
    card = QFrame(parent)
    card.setObjectName(name)
    card.setProperty("inspectorSection" if flat else "sectionCard", True)
    card.setAccessibleName(title)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
    # ``.insp-sec .h`` 的 ``margin-bottom:var(--sm)``：分节抬头与正文之间比图卡宽一档，
    # 因为这一栏没有边框替它分隔，留白就是分隔。
    layout.setSpacing(SPACE_SM if flat else SPACE_XS)
    heading = QLabel(title, card)
    heading.setObjectName(f"{name}Title")
    heading.setProperty("sectionHeader", True)
    if flat:
        apply_section_heading(heading)
    else:
        title_font = heading.font()
        title_font.setPixelSize(CARD_TITLE_FONT_PX)
        heading.setFont(title_font)
    caption = ElidingLabel(subtitle, card)
    caption.setObjectName(f"{name}Subtitle")
    caption.setProperty("mutedText", True)
    caption.setToolTip(subtitle)
    hint_font = caption.font()
    hint_font.setPixelSize(CARD_HINT_FONT_PX)
    caption.setFont(hint_font)
    header = QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(SPACE_SM)
    header.addWidget(heading)
    header.addStretch(1)
    header.addWidget(caption)
    layout.addLayout(header)
    return card, layout


def legend_swatch(shape: str, colour: str) -> QPixmap:
    """Draw one key mark, so shape and not only hue separates the entries."""
    pixmap = QPixmap(LEGEND_SWATCH_W, LEGEND_SWATCH_H)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    fill = QColor(colour)
    if shape == "box":
        fill.setAlpha(LEGEND_BOX_ALPHA)
        painter.setBrush(fill)
        painter.drawRoundedRect(0, 0, LEGEND_SWATCH_W, LEGEND_SWATCH_H, 2, 2)
    elif shape == "dot":
        painter.setBrush(fill)
        left = (LEGEND_SWATCH_W - LEGEND_DOT_D) // 2
        painter.drawEllipse(left, (LEGEND_SWATCH_H - LEGEND_DOT_D) // 2, LEGEND_DOT_D, LEGEND_DOT_D)
    elif shape == "dash":
        painter.setBrush(fill)
        top = (LEGEND_SWATCH_H - LEGEND_LINE_H) // 2
        run = (LEGEND_SWATCH_W - LEGEND_DASH_GAP) // 2
        painter.drawRect(0, top, run, LEGEND_LINE_H)
        painter.drawRect(LEGEND_SWATCH_W - run, top, run, LEGEND_LINE_H)
    else:
        painter.setBrush(fill)
        painter.drawRect(0, (LEGEND_SWATCH_H - LEGEND_LINE_H) // 2, LEGEND_SWATCH_W, LEGEND_LINE_H)
    painter.end()
    return pixmap


def stack_swatch(colour: str) -> QPixmap:
    """A solid chip standing for one band of a stack diagram.

    Kept apart from `legend_swatch`, which varies its mark by shape to separate
    key entries: here the rows are already named, and the chip's one job is to
    carry a hue across to the diagram.
    """
    pixmap = QPixmap(STACK_SWATCH_PX, STACK_SWATCH_PX)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(colour))
    painter.drawRoundedRect(0, 0, STACK_SWATCH_PX, STACK_SWATCH_PX, STACK_SWATCH_RADIUS, STACK_SWATCH_RADIUS)
    painter.end()
    return pixmap


def build_legend(parent: QWidget, name: str, entries: tuple[tuple[str, str, str], ...]) -> QWidget:
    """A caption row keying the marks drawn on one plot card.

    Each entry is ``(shape, colour role, caption)``.  Shape carries the same
    information as hue, so a key survives greyscale printing and colour vision
    deficiency; a ``plain`` shape draws no swatch, for an entry whose mark is a
    glyph already sitting in its caption.

    This lives beside ``titled_card`` because the two are one unit in the design:
    the mockup's ``.plotcard`` is a header, a key, then the axes.  A pyqtgraph
    in-plot legend cannot replace it -- it names curves only once they have been
    added, so an empty project would show an empty key, and it says nothing about
    the overlays (the fit window, the clipped-point glyph) that are not curves.
    """
    legend = QWidget(parent)
    legend.setObjectName(name)
    row = QHBoxLayout(legend)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(SPACE_MD)
    accent = palette_tokens(parent.palette()).accent
    for shape, role, caption in entries:
        if shape != "plain":
            swatch = QLabel(legend)
            swatch.setPixmap(legend_swatch(shape, accent if role == "accent" else LEGEND_ROLE_COLOURS[role]))
            row.addWidget(swatch)
        text = QLabel(caption, legend)
        text.setProperty("mutedText", True)
        row.addWidget(text)
    row.addStretch(1)
    return legend
