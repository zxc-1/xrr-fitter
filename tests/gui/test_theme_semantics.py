"""Semantic state colours resolve through theme tokens, not inline stylesheets.

The confidence badge used to carry four hardcoded hex values. They were the
light-theme colours, so on a dark desktop the badge kept emitting the light
green while every control around it had already switched. Worse, the colour
reached the screen through an inline ``setStyleSheet`` — the exact practice the
theme module exists to remove — while the ``semanticColor`` dynamic property it
also set was consumed by no stylesheet rule at all: computed on every refresh
and never painted.

The badge therefore joins the existing ``statusKind`` channel. Confidence has
four levels against three status colours, so an informational token covers the
"usable but correlated" case: it is a state, not a call to action, and must stay
distinguishable from the accent colour that marks primary buttons.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QFont, QFontInfo, QPalette
from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget

from xrr_fitter.gui import theme

STATUS_KINDS = ("ok", "info", "warn", "error")


def _palette(window: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(window))
    return palette


def test_every_status_kind_has_a_token_in_both_appearances() -> None:
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        colours = {getattr(tokens, kind) for kind in STATUS_KINDS}
        assert len(colours) == len(STATUS_KINDS)


def test_informational_token_tracks_the_appearance() -> None:
    """A state colour that ignores the palette is the bug being fixed."""
    assert theme.LIGHT_TOKENS.info != theme.DARK_TOKENS.info


def test_informational_token_stays_distinct_from_the_accent() -> None:
    """Reusing the accent would make a status badge read as a button."""
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        assert tokens.info != tokens.accent


@pytest.mark.parametrize("kind", STATUS_KINDS)
def test_stylesheet_paints_each_status_kind_from_the_resolved_token(kind: str) -> None:
    """A property no rule consumes is how the old badge lost its colour."""
    for window, tokens in (("#FFFFFF", theme.LIGHT_TOKENS), ("#1E1F22", theme.DARK_TOKENS)):
        sheet = theme.build_stylesheet(_palette(window))
        assert f'QLabel[statusKind="{kind}"]' in sheet
        assert f'QLabel[statusKind="{kind}"] {{ color: {getattr(tokens, kind)}; }}' in sheet


def test_confidence_badge_keeps_the_emphasis_the_inline_style_carried() -> None:
    """The inline rule set a weight as well as a colour; both had to survive.

    Moving the badge onto the token channel is only lossless if the stylesheet
    replaces every declaration the inline string carried, so the verdict still
    reads as the panel's headline rather than as one more label.

    设计稿 ``.confbadge .txt .l`` 是 ``font-weight:700;font-size:15px``，比行内样式那
    600 又重一档，还额外给了字号。判定框里两行字同号时读者分不出哪行是判定、哪行是理由，
    所以字号和字重都得写下来。
    """
    sheet = theme.build_stylesheet(_palette("#FFFFFF"))

    body = sheet.split("QLabel#confidenceBadge {", 1)[1].split("}", 1)[0]
    assert "font-weight: 700" in body
    assert f"font-size: {theme.FONT_PT_MD}pt" in body


def test_the_verdict_reason_is_typeset_as_the_smaller_second_line() -> None:
    """设计稿 ``.confbadge .txt .r``：11.5px 的静音字，比判定那行小一档。

    理由和判定同号时，框里读起来是两句并列的话；小一档它才读成判定的注脚。
    """
    sheet = theme.build_stylesheet(_palette("#FFFFFF"))

    body = sheet.split("QLabel#confidenceReason {", 1)[1].split("}", 1)[0]
    assert f"font-size: {theme.FONT_PT_SM}pt" in body


# 设计稿 HTML 185-188 的四条 ``.confbadge.<状态>`` 规则，逐位抄下来。
VERDICT_TINTS = (
    ("ok", "rgba(46, 125, 50, 20)", "rgba(46, 125, 50, 89)"),
    ("info", "rgba(21, 101, 192, 20)", "rgba(21, 101, 192, 89)"),
    ("warn", "rgba(154, 103, 0, 23)", "rgba(154, 103, 0, 89)"),
    ("error", "rgba(179, 38, 30, 20)", "rgba(179, 38, 30, 89)"),
)


@pytest.mark.parametrize(("kind", "fill", "border"), VERDICT_TINTS)
def test_the_verdict_box_is_tinted_exactly_as_the_design_specifies(kind: str, fill: str, border: str) -> None:
    """判定框的底色与边框，对齐设计稿那四条规则。

    这两个数不是随手挑的比例：底色压在 8% 上下、边框留到 35%，框才能从卡的底色上浮出来
    又不把判定那两个字压住。写成字面量是因为「差一点」在这里看得见——两成的底色会让判定
    框读成一块高亮，半成的读不出有框。
    """
    sheet = theme.build_stylesheet(_palette("#FFFFFF"))

    body = sheet.split(f'QFrame[verdictBox="true"][statusKind="{kind}"] {{', 1)[1].split("}", 1)[0]
    assert f"background: {fill}" in body
    assert f"border: 1px solid {border}" in body


def test_an_untinted_verdict_box_still_draws_its_frame() -> None:
    """未加后缀的 ``.confbadge``：中性边框 + 透明底，是「不可用」那一档落脚的地方。

    ``set_status_kind`` 的白名单收下空串，所以没有结果时框上那把键是空的，四条状态规则
    一条都不匹配。基础规则要是不画边框，判定就在这一档静默退回一行裸字。
    """
    sheet = theme.build_stylesheet(_palette("#FFFFFF"))

    body = sheet.split('QFrame[verdictBox="true"] {', 1)[1].split("}", 1)[0]
    assert f"border: 1px solid {theme.LIGHT_TOKENS.surface_border}" in body
    assert "background: transparent" in body
    assert "border-radius: 8px" in body


def test_the_verdict_tint_tracks_the_appearance() -> None:
    """判定框的底色也得跟着深浅外观走，理由同上面那条 ``info`` token。

    浅色的 ok 是 ``#2E7D32``，深色是另一支绿；把浅色那支的通道写死进深色样式表，深色
    外观下的判定框会是一块和周围都不搭的暗绿。
    """
    dark = theme.build_stylesheet(_palette("#1E1F22"))

    for kind, fill, border in VERDICT_TINTS:
        assert fill not in dark, kind
        assert border not in dark, kind


def test_the_amber_verdict_keeps_the_heavier_fill_the_design_gave_it() -> None:
    """设计稿只有 ``.confbadge.multiple`` 是 ``.09``，另外三档都是 ``.08``。

    黄的亮度最高，同一个 alpha 铺出来比绿蓝红都淡；折成一个常数会让「多解」这一档的底色
    在浅色卡上化掉，而它恰是最该被看见的两档之一。
    """
    alphas = theme.VERDICT_FILL_ALPHAS

    assert alphas["warn"] > alphas["ok"]
    assert alphas["ok"] == alphas["info"] == alphas["error"]


def test_confidence_glyphs_cover_every_class_with_distinct_shapes() -> None:
    """Double encoding: each confidence class gets a unique glyph, not just colour."""
    from xrr_fitter.gui.theme import CONFIDENCE_GLYPHS

    expected_classes = {"可信", "可用但相关", "多解", "不可信"}
    assert set(CONFIDENCE_GLYPHS.keys()) == expected_classes
    assert len(set(CONFIDENCE_GLYPHS.values())) == len(expected_classes)


def test_confidence_glyphs_are_the_documented_shapes() -> None:
    """The design spec pins specific Unicode glyphs for each verdict."""
    from xrr_fitter.gui.theme import CONFIDENCE_GLYPHS

    assert CONFIDENCE_GLYPHS["可信"] == "●"
    assert CONFIDENCE_GLYPHS["可用但相关"] == "◆"
    assert CONFIDENCE_GLYPHS["多解"] == "▲"
    assert CONFIDENCE_GLYPHS["不可信"] == "■"


def test_unclassified_verdict_has_a_glyph_authored_here_too() -> None:
    """The design's fifth state is a state, so its shape belongs with the other four.

    "不可用" is not a ``ConfidenceClass`` — it is what the panel shows before a
    dataset has any result — which is why it cannot become a fifth key without
    breaking the table above. It is still a rendered verdict, so leaving its
    glyph as a literal at the call site recreates the second glyph table this
    module exists to prevent.
    """
    from xrr_fitter.gui.theme import CONFIDENCE_FALLBACK_GLYPH, CONFIDENCE_GLYPHS

    assert CONFIDENCE_FALLBACK_GLYPH == "○"
    assert CONFIDENCE_FALLBACK_GLYPH not in set(CONFIDENCE_GLYPHS.values())


@pytest.mark.parametrize("kind", STATUS_KINDS)
def test_set_status_kind_accepts_every_painted_kind(qtbot, kind: str) -> None:
    label = QLabel()
    qtbot.addWidget(label)

    theme.set_status_kind(label, kind)

    assert label.property("statusKind") == kind


def test_set_status_kind_still_rejects_an_unpainted_kind(qtbot) -> None:
    """The guard is what keeps a typo from silently losing the colour."""
    label = QLabel()
    qtbot.addWidget(label)

    with pytest.raises(ValueError):
        theme.set_status_kind(label, "informational")


def _hint_probe(qtbot, *, boxed: bool, kind: str = "info", mark=None) -> tuple[QColor, QColor, QColor, QColor]:
    """Render one advisory line; report (left bar, fill, untouched, accent) colours.

    The fill is sampled below the text so a glyph cannot be mistaken for a
    background, and the reference is the window colour the label would show if
    no rule painted it.

    ``accent`` 由同一份调色板解出来，而且那份调色板要在贴 stylesheet 之前抓下来钉住：Qt 的
    ``QStyleSheetStyle`` 会在 stylesheet 生效后改写控件自己的 palette，所以 ``label.palette()``
    贴前贴后不是同一份。拿贴后那份去解状态色，比的就是另一套颜色——深浅两套主题的 warn 一个
    是 ``#9a6700`` 一个是 ``#e3b341``，本该抓到「竖条没取状态色」的断言会永远不成立，本该抓到
    「横幅多画了竖条」的断言会永远成立。
    """
    label = QLabel("厚度 d 与密度 ρ 可能相关。")
    qtbot.addWidget(label)
    theme.set_status_kind(label, kind)
    if boxed:
        (mark or theme.mark_hint)(label)
    palette = QPalette(label.palette())
    # 钉住：接下来量的每个像素都得出自这一份，包括不带 hint 时的窗口底色。
    label.setPalette(palette)
    label.setStyleSheet(theme.build_stylesheet(palette))
    label.resize(240, 60)
    label.show()
    qtbot.waitExposed(label)

    image = label.grab().toImage()
    return (
        QColor.fromRgba(image.pixel(1, image.height() // 2)),
        QColor.fromRgba(image.pixel(image.width() // 2, image.height() - 4)),
        palette.color(QPalette.ColorRole.Window),
        QColor(getattr(theme.palette_tokens(palette), kind)),
    )


@pytest.mark.parametrize("kind", ("info", "warn"))
def test_an_advisory_line_paints_a_callout_box_not_just_coloured_text(qtbot, kind: str) -> None:
    """设计稿的 ``.hint`` 是提示框：淡底色 + 细边 + 左侧竖条，正文用 muted。

    只把字染成状态色，一行提示就跟周围正文抢注意力却又不成块——它读起来像「这句话是
    结论」，而不是「这是一条旁注」。设计稿把三处提示（💡 建议加氧化层、🔎 d 与 ρ 相关、
    ⚠ 强负相关）画成同一种带底色的框，正是为了让它们一眼可辨、又不冒充结论。底色是渲染
    出来的，所以这里量像素而不是读 stylesheet 文本。
    """
    bar, fill, untouched, accent = _hint_probe(qtbot, boxed=True, kind=kind)

    assert fill.rgba() != untouched.rgba(), "提示框没有底色"
    assert bar.rgb() == accent.rgb(), "提示框左侧没有取状态色的强调竖条"
    assert bar.alpha() > fill.alpha(), "竖条没有比底色更实"


def test_a_plain_status_label_stays_unboxed(qtbot) -> None:
    """The box has to come from the hint property, not from every status label.

    Most ``statusKind`` labels are inline verdicts sitting inside a card that
    already draws its own frame; boxing all of them would nest a panel in a
    panel everywhere the theme paints a state colour.
    """
    _bar, fill, untouched, _accent = _hint_probe(qtbot, boxed=False)

    assert fill.rgba() == untouched.rgba(), "普通状态文字不该带底色"


def test_a_run_wide_banner_is_framed_but_carries_no_pointing_bar(qtbot) -> None:
    """设计稿 ``.jbanner``：同族的淡底框，但没有 ``.hint`` 的左侧竖条。

    竖条是个指向物——它把提示钉在紧邻的那个控件上（那条 🔎 就贴着结构卡）。横幅说的
    是整轮拟合的前提：这一轮是联合的，屏幕上每个共享参数都是几套数据折衷出来的。它笼罩
    整张卡，没有可指的邻居，留着竖条反而会让人去找"它在说哪一项"。所以框保留、竖条去掉。
    """
    bar, fill, untouched, accent = _hint_probe(qtbot, boxed=True, kind="warn", mark=theme.mark_banner)

    assert fill.rgba() != untouched.rgba(), "横幅没有底色"
    assert bar.rgb() != accent.rgb(), "横幅画了 hint 才该有的指向竖条"


def test_a_quoted_file_is_monospaced_and_sunken(qtbot) -> None:
    """设计稿 ``.mono`` 预览块：等宽字体 + 下沉底 + 细边。

    帧⑥ 的清单预览是一段 ``indent=2`` 的 JSON。比例字体会把那份缩进排歪，而缩进是它
    pretty-print 的全部理由——排歪了就不如压成一行。等宽是这块内容的正确性要求，不是装饰。

    下沉底解决的是另一件事：没有边框时，这段 JSON 读起来像对话框自己的说明文字，而它其实是
    「某个文件将写成什么样」的引用。框把这层引述关系画了出来。
    """
    label = QLabel('{\n  "app": "xrr-fitter"\n}')
    qtbot.addWidget(label)
    theme.mark_code_block(label)
    label.setStyleSheet(theme.build_stylesheet(label.palette()))
    label.resize(240, 80)
    label.show()
    qtbot.waitExposed(label)

    fill = QColor.fromRgba(label.grab().toImage().pixel(label.width() // 2, label.height() - 4))
    assert fill.rgba() != label.palette().color(QPalette.ColorRole.Window).rgba(), "引用块没有下沉底色"
    assert QFontInfo(label.font()).fixedPitch(), "引用块没有落到等宽字体，JSON 缩进会排歪"


def test_a_callout_refuses_a_status_kind_it_cannot_paint(qtbot) -> None:
    """A hint is advice or a caution, never a pass or a failure.

    Only ``info`` and ``warn`` carry box rules, so accepting the other kinds
    would hand back a label with padding and a radius but no border and no
    fill — a box that silently lost its frame.
    """
    label = QLabel()
    qtbot.addWidget(label)
    theme.set_status_kind(label, "error")

    with pytest.raises(ValueError):
        theme.mark_hint(label)


TRANSLUCENT_TOKENS = ("surface", "surface_border", "muted_text", "selection_bg")


def test_a_translucent_token_survives_the_trip_into_a_painter() -> None:
    """``QColor`` 读不懂 ``rgba(...)``：它返回无效色，落笔就是不透明黑。

    这几个 token 写成 CSS 的函数记法是为了进样式表，而自绘的委托拿同一个字符串塞进
    ``QColor`` 时静默退化成黑色——深色外观下等于把次要文字画进背景里。所以取色必须
    走这里，由主题自己把两种记法都解开。
    """
    rgba_tested = 0
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        for name in TRANSLUCENT_TOKENS:
            value = getattr(tokens, name)
            if QColor(value).isValid():
                continue  # now an opaque hex token, covered by the hex test

            colour = theme.token_colour(value)

            assert colour.isValid()
            assert colour.alpha() == int(value.rsplit(",", 1)[1].strip(" )"))
            assert (colour.red(), colour.green(), colour.blue()) != (0, 0, 0) or "0, 0, 0" in value
            rgba_tested += 1
    assert rgba_tested >= 4, f"只有 {rgba_tested} 个 rgba token，覆盖不足"


def test_an_opaque_token_reads_back_the_hex_it_was_written_as() -> None:
    """十六进制的 token 也走同一个入口，调用处才不必先判断记法。"""
    assert theme.token_colour(theme.LIGHT_TOKENS.accent) == QColor("#2F6BD8")
    assert theme.token_colour(theme.DARK_TOKENS.accent).alpha() == 255


def _section_probe(qtbot, *, flat: bool, last: bool = False) -> tuple[QColor, QColor, QColor, QColor]:
    """画一段检视器块，报回（下缘、上缘、左缘、内部）四个像素。

    宿主自己带一层底色是必需的：``surface_border`` 是 ``rgba(0, 0, 0, 34)``，把它叠在
    未上底色的抓图缓冲（黑）上，量到的是黑压黑——「有没有这道线」的断言会永远不成立。
    调色板在贴 stylesheet 之前抓下来钉住，理由同 ``_hint_probe``：``QStyleSheetStyle``
    会改写控件自己那份 palette。

    抓的是宿主而不是卡：卡的底色是 ``transparent``，单独抓它拿不到线所压的那层底。卡由
    零边距的布局撑满宿主，所以两者的坐标是同一套。
    """
    host = QWidget()
    qtbot.addWidget(host)
    palette = QPalette(host.palette())
    host.setPalette(palette)
    host.setAutoFillBackground(True)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    card, _ = theme.titled_card(host, "probeSection", "拟合判定", "aSi_ML_25C", flat=flat)
    layout.addWidget(card)
    if last:
        theme.mark_last_section((card,))
    host.setStyleSheet(theme.build_stylesheet(palette))
    host.resize(320, 96)
    host.show()
    qtbot.waitExposed(host)

    image = host.grab().toImage()
    width, height = image.width(), image.height()
    return (
        QColor.fromRgba(image.pixel(width // 2, height - 1)),
        QColor.fromRgba(image.pixel(width // 2, 0)),
        QColor.fromRgba(image.pixel(0, height // 2)),
        # 参照点取抬头以下、横线以上的空白：那里没有任何规则画东西。
        QColor.fromRgba(image.pixel(width // 2, height - 6)),
    )


def test_an_inspector_section_is_ruled_off_instead_of_boxed(qtbot) -> None:
    """设计稿 ``.insp-sec`` 只有一道下边线：段与段之间是界，不是各自一只盒子。

    右栏五六段各画一圈圆角边框时，读者看到的是一列卡片；设计稿画的是一栏连续的正文，
    被横线分节。
    """
    bottom, top, left, inside = _section_probe(qtbot, flat=True)

    assert bottom != inside, "下缘那道横线没画出来"
    assert top == inside, "上缘多了一道线"
    assert left == inside, "侧边多了一道线"


def test_a_bordered_card_keeps_every_one_of_its_four_edges(qtbot) -> None:
    """画布上那几张图卡照旧带一圈框（``.plotcard``）：扁平化是右栏的事，不是全局的。"""
    bottom, top, left, inside = _section_probe(qtbot, flat=False)

    assert bottom != inside
    assert top != inside
    assert left != inside


def test_the_last_section_of_a_column_is_not_ruled_off(qtbot) -> None:
    """末段之下没有下一段，那道线就成了整栏的封边。

    设计稿把这个例外写成行内样式（``<div class="insp-sec" style="border-bottom:none">``）。
    """
    bottom, _top, _left, inside = _section_probe(qtbot, flat=True, last=True)

    assert bottom == inside


def _sections(qtbot) -> tuple[QWidget, QLabel, QLabel]:
    # 宿主也一起交回去：只回那两个 QLabel 时，``host`` 这个局部一出函数就没人引用了，
    # 它带着两张卡和抬头一起被销毁，调用处读属性时拿到的是已析构的 C++ 对象。
    host = QWidget()
    qtbot.addWidget(host)
    flat, _ = theme.titled_card(host, "flatSection", "拟合判定", "aSi_ML_25C", flat=True)
    boxed, _ = theme.titled_card(host, "boxedSection", "反射率曲线", "归一化")
    return host, flat.findChild(QLabel, "flatSectionTitle"), boxed.findChild(QLabel, "boxedSectionTitle")


def test_the_flat_section_heading_is_typeset_smaller_and_fainter(qtbot) -> None:
    """``.insp-sec .h`` 是 11px/700/字距 .6px 的 ``--ink-faint``，比图卡那句 12.5px 抬头小一档。

    两者同字号同墨色时，右栏读起来像五个并列的标题；设计稿把分节抬头压成一行小字，正文
    才是主角。

    字号量的是像素：这一档此前借 ``FONT_PT_SM``（9pt = 12px）表达设计稿的 11px，量出来大
    一像素。字色量的是 ``faintText`` 而不是 ``mutedText``——顶替那一档等于把抬头和正文小字
    调成同色。
    """
    _host, flat, boxed = _sections(qtbot)

    assert flat.property("faintText") is True
    assert not boxed.property("faintText")
    assert not flat.property("mutedText")
    # ``QFontInfo`` 而不是 ``font().pointSize()``：像素给的字号读 ``pointSize()`` 会拿到 -1，
    # 于是「比图卡抬头小」这句会因为 -1 < 11 而空过，量的不再是字号。
    assert QFontInfo(flat.font()).pixelSize() == theme.SECTION_HEADING_FONT_PX
    assert QFontInfo(flat.font()).pixelSize() < QFontInfo(boxed.font()).pixelSize()
    assert flat.font().letterSpacingType() == QFont.SpacingType.AbsoluteSpacing
    # 容差是 1/64 px：Qt 把字距存成 1/64 像素的定点数，0.6px 落到最近的 38/64=0.59375，
    # 逐位相等是量不到的。
    assert flat.font().letterSpacing() == pytest.approx(theme.SECTION_HEADING_TRACKING_PX, abs=1 / 64)


def _shell_palette(window: str, base: str, alternate: str) -> QPalette:
    """三块外壳底色都写进调色板，断言才不必依赖跑测试那台机器的默认配色。"""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(window))
    palette.setColor(QPalette.ColorRole.Base, QColor(base))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(alternate))
    return palette


# 设计稿 ``:root`` 的三块外壳底色，原样喂进调色板：``--app-bg`` / ``--panel`` / ``--panel-2``。
DESIGN_SHELL_PALETTE = ("#EDEFF3", "#FFFFFF", "#F7F8FA")


def _column_rule(sheet: str, name: str) -> str:
    return sheet.split(f"QWidget#{name} {{", 1)[1].split("}", 1)[0]


def test_the_canvas_column_is_washed_with_the_panel_colour() -> None:
    """设计稿 ``.canvas{background:var(--panel)}``：画布是屏幕上最亮的一块。

    三栏一律留白时，画布与两侧栏同色，屏幕上就没有「内容区」这个概念了——读者靠什么
    判断中间那一栏是主角？设计稿用底色分的层，不是用边框。
    """
    sheet = theme.build_stylesheet(_shell_palette(*DESIGN_SHELL_PALETTE))

    assert "background: #ffffff" in _column_rule(sheet, "canvasColumn").lower()


def test_both_side_rails_are_washed_a_step_off_the_canvas() -> None:
    """设计稿 ``.nav`` 与 ``.inspector`` 同取 ``var(--panel-2)``：两侧是机架，不是内容。"""
    sheet = theme.build_stylesheet(_shell_palette(*DESIGN_SHELL_PALETTE))

    for name in ("navigationColumn", "inspectorColumn"):
        assert "background: #f7f8fa" in _column_rule(sheet, name).lower(), name


def test_the_rail_wash_falls_back_to_the_window_when_alternatebase_is_flat() -> None:
    """深色调色板把 AlternateBase 设成和 Base 同色，那时机架色得另找一处。

    ``dark_palette`` 的 Base 与 AlternateBase 都是 ``#14181D``；照搬 AlternateBase 会让
    三栏在深色外观下重新变成一整块平底色，正是上面两条测试要挡的那件事。
    """
    palette = theme.dark_palette()
    assert palette.color(QPalette.ColorRole.AlternateBase) == palette.color(QPalette.ColorRole.Base)
    sheet = theme.build_stylesheet(palette)

    window = palette.color(QPalette.ColorRole.Window).name()
    assert f"background: {window}" in _column_rule(sheet, "navigationColumn").lower()
    assert f"background: {window}" in _column_rule(sheet, "inspectorColumn").lower()
    assert window not in _column_rule(sheet, "canvasColumn").lower()


def _column_wash(qtbot, name: str, palette: QPalette) -> QColor:
    """画一栏裸控件，报回它自己铺出来的底色。

    抓裸控件而不是真窗口里的那一栏：栏里的列表和表格自带不透明底色，真窗口里几乎找不到
    一块只由栏自己上色的像素。「这条规则画没画出来」裸控件足以回答。
    """
    column = QWidget()
    qtbot.addWidget(column)
    column.setObjectName(name)
    column.setPalette(palette)
    column.setStyleSheet(theme.build_stylesheet(palette))
    column.resize(120, 60)
    column.show()
    qtbot.waitExposed(column)
    return QColor.fromRgba(column.grab().toImage().pixel(60, 30))


def test_the_column_washes_actually_reach_the_screen(qtbot) -> None:
    """选择器写对了、控件类型写错了，规则就一条也不命中——三栏是 ``QWidget``，不是 ``QFrame``。"""
    palette = _shell_palette(*DESIGN_SHELL_PALETTE)

    canvas = _column_wash(qtbot, "canvasColumn", palette)
    rail = _column_wash(qtbot, "navigationColumn", palette)

    assert canvas.name().lower() == "#ffffff"
    assert rail.name().lower() == "#f7f8fa"


def test_the_seam_between_two_columns_is_painted_as_a_border() -> None:
    """设计稿 ``.nav{border-right:1px solid var(--border)}``：栏与栏之间是一道细线。

    分栏器的手柄默认按样式给宽度（这里量到 4px），底色是窗口色——屏幕上那不是一道线，
    是一条比两侧都暗的空档。线由手柄自己画：把边框画在栏上，手柄那几像素会留在线和画布
    之间成为第二道缝，设计稿没有那道缝。
    """
    palette = _shell_palette(*DESIGN_SHELL_PALETTE)
    sheet = theme.build_stylesheet(palette)

    body = sheet.split("QSplitter#workspaceSplitter::handle {", 1)[1].split("}", 1)[0]
    assert f"background: {theme.palette_tokens(palette).surface_border}" in body


def _rule(sheet: str, selector: str) -> str:
    return sheet.split(f"{selector} {{", 1)[1].split("}", 1)[0]


def test_the_dataset_list_is_not_boxed_and_lets_the_rail_through() -> None:
    """设计稿的 ``.ds`` 行直接坐在机架上：这张列表既没有外框，也没有自己的底色。

    通用列表规则给每张列表一道 1px 圆角外框加一层内容色底。数据集列表照它画，左栏就多出
    一只设计稿里没有的盒子，盒里那片白还把 ``--panel-2`` 的机架色整段盖掉——同一栏于是有
    两种白：列表里的和列表外的。``setFrameShape(NoFrame)`` 管不到样式表画的框，只有按 id
    覆盖才抹得掉。
    """
    palette = _shell_palette(*DESIGN_SHELL_PALETTE)
    body = _rule(theme.build_stylesheet(palette), "QTreeWidget#datasetTree")

    assert "border: 0px" in body
    assert "background: transparent" in body


def test_only_the_active_dataset_card_takes_the_panel_white_while_hover_keeps_the_tint() -> None:
    """设计稿把两件事分开：``.ds.on`` 是 ``#fff``（即 ``--panel``），``.ds:hover`` 才是 ``--selection``。

    通用规则把选中底色设成那层主色淡染，于是「此刻在看哪一集」和「鼠标正掠过哪一集」在屏幕上
    长成一个样。活动集是这一栏最强的状态，它该拿到最亮的那块底，淡染留给一掠而过的悬停。
    """
    palette = _shell_palette(*DESIGN_SHELL_PALETTE)
    sheet = theme.build_stylesheet(palette)

    assert "background: #ffffff" in _rule(sheet, "QTreeWidget#datasetTree::item:selected").lower()
    selection = theme.palette_tokens(palette).selection_bg
    assert f"background: {selection}" in _rule(sheet, "QTreeWidget#datasetTree::item:hover")


def test_the_two_section_shapes_do_not_share_one_property(qtbot) -> None:
    """扁平那一档必须是另一个属性：``sectionCard`` 上挂着图卡的边框，改它会连图卡一起改。"""
    host = QWidget()
    qtbot.addWidget(host)
    flat, _ = theme.titled_card(host, "flatSection", "拟合判定", "aSi_ML_25C", flat=True)
    boxed, _ = theme.titled_card(host, "boxedSection", "反射率曲线", "归一化")

    assert flat.property("inspectorSection") is True
    assert not flat.property("sectionCard")
    assert boxed.property("sectionCard") is True
    assert not boxed.property("inspectorSection")


def _token_prominence(token: str) -> float:
    """越高越醒目：rgba 比 alpha，hex 比离白色的距离。"""
    c = theme.token_colour(token)
    if c.alpha() < 255:
        return float(c.alpha())
    return 255.0 - c.lightnessF() * 255.0


def test_the_faint_token_is_lighter_than_the_muted_one_in_both_appearances() -> None:
    """设计稿有三档字色，主题只有两档：``--ink-faint`` 没有对应的 token。

    ``--ink-muted:#5B6270`` 用在正文小字上，``--ink-faint:#8A909C`` 用在分区抬头上，抬头比
    小字更淡是它跟正文拉开层级的唯一手段。拿 ``muted_text`` 顶替，抬头就和小字同色，那一档
    层级在屏幕上消失。两套外观都要成立：这一档在浅色和深色下都比 ``muted_text`` 更透。
    """
    for tokens in (theme.LIGHT_TOKENS, theme.DARK_TOKENS):
        assert _token_prominence(tokens.faint_text) < _token_prominence(tokens.muted_text)


def test_the_stylesheet_paints_the_faint_ink_on_its_own_property() -> None:
    """``--ink-faint`` 得有自己的规则，不能挂回 ``sectionHeader`` 或 ``mutedText``。

    ``sectionHeader`` 是对话框标题也在用的属性，设计稿把那处画得很大；``mutedText`` 是正文
    小字那一档。抬头这一档要么另立一个属性，要么就得改掉别人的样子。
    """
    for window, tokens in (("#FFFFFF", theme.LIGHT_TOKENS), ("#1E1F22", theme.DARK_TOKENS)):
        sheet = theme.build_stylesheet(_palette(window))

        assert f"color: {tokens.faint_text}" in _rule(sheet, 'QLabel[faintText="true"]')
        assert "font-size" not in _rule(sheet, 'QLabel[sectionHeader="true"]')


def test_a_rail_heading_and_an_inspector_heading_share_one_type_scale(qtbot) -> None:
    """左右两栏的抬头是同一档字，只有字距按设计稿分 .8px / .6px。

    这两句抬头在同一屏上并列可见（左栏「数据集」「分析管线」，右栏各节抬头），分成两处实现
    就会各自漂移。所以只有 ``apply_section_heading`` 一处写这一档，字距是它的参数。
    """
    _host, flat, _boxed = _sections(qtbot)
    rail = QLabel("分析管线")
    qtbot.addWidget(rail)
    theme.apply_section_heading(rail, tracking_px=theme.RAIL_SECTION_TRACKING_PX)

    assert QFontInfo(rail.font()).pixelSize() == QFontInfo(flat.font()).pixelSize()
    assert rail.property("faintText") is flat.property("faintText") is True
    assert rail.property("sectionHeader") is flat.property("sectionHeader") is True
    assert rail.font().letterSpacing() == pytest.approx(theme.RAIL_SECTION_TRACKING_PX, abs=1 / 64)
    assert rail.font().letterSpacing() > flat.font().letterSpacing()


@pytest.mark.parametrize(
    ("ink", "expected"),
    ((theme.LIGHT_TEXT, "LIGHT_TOKENS"), (theme.DARK_TEXT, "DARK_TOKENS")),
)
def test_a_transparent_window_colour_does_not_decide_the_appearance(ink: str, expected: str) -> None:
    """透明的窗口色不是一种亮度，明暗得改问前景色。

    样式表把控件底色写成 ``transparent`` 时，Qt 把这个值反写进控件自己的调色板：屏幕上透出来
    的是父级那层白，可调色板里留下的是 ``rgba(0, 0, 0, 0)``——一个 alpha 为零的黑。
    ``lightness()`` 不看 alpha，于是浅色外观下的控件被判成深色，取到的次要字色是给深底备的
    浅灰，画到白底上等于没画。alpha 不足的底色不携带明暗信息，此时唯一还可信的线索是前景色：
    暗字必然配亮底。两个方向都要成立，否则这就只是把误判翻了个面。
    """
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0, 0))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(ink))

    assert theme.palette_tokens(palette) is getattr(theme, expected)


def test_an_opaque_window_colour_still_decides_the_appearance() -> None:
    """底色够实时判据不变：两套外观各自的调色板必须仍解析回自己那套 token。"""
    assert theme.palette_tokens(theme.light_palette()) is theme.LIGHT_TOKENS
    assert theme.palette_tokens(theme.dark_palette()) is theme.DARK_TOKENS


def test_the_light_appearance_carries_the_designs_own_four_values() -> None:
    """设计稿 ``:root`` 的三层洗色与墨色（HTML 18 与 20 行）逐位落在这四个常量上。

    ``--app-bg`` / ``--panel`` / ``--panel-2`` / ``--ink`` 是整套亮色外观的全部来源：
    ``light_palette`` 把它们分别放到 Window / Base / AlternateBase / Text 上，其余每一档灰都
    是叠在这四层上的 alpha 值，会跟着它们一起走。所以这四个数是设计稿与实现之间那道对照的
    锚点，而墨色此前写的是 ``#1B1F24``——与设计稿差 ΔB=3，肉眼分不出，可它是浅色外观下每一
    个字的唯一来源；没有一处断言，它已经漂了这件事就无从发现。
    """
    palette = theme.light_palette()

    assert (theme.LIGHT_WINDOW, theme.LIGHT_BASE, theme.LIGHT_ALTERNATE, theme.LIGHT_TEXT) == (
        "#EDEFF3",
        "#FFFFFF",
        "#F7F8FA",
        "#1B1F27",
    )
    # 常量对上了还不够：这四层得真落到那四个角色上，``shell_washes`` 才读得出机架与画布之分。
    roles = (
        QPalette.ColorRole.Window,
        QPalette.ColorRole.Base,
        QPalette.ColorRole.AlternateBase,
        QPalette.ColorRole.Text,
    )
    assert tuple(palette.color(role).name().upper() for role in roles) == (
        theme.LIGHT_WINDOW,
        theme.LIGHT_BASE,
        theme.LIGHT_ALTERNATE,
        theme.LIGHT_TEXT,
    )


def test_a_list_the_stylesheet_paints_transparent_keeps_the_window_appearance(qtbot) -> None:
    """候选行的字色由委托按 ``option.palette`` 取，而那张列表的底色被样式表抹成了透明。

    ``QListWidget#candidateList`` 必须是透明底——设计稿的行盒子由委托画，列表再铺一层底就
    压在行盒子下面透出来。代价是它的调色板从此不再报窗口的白，委托于是在浅色外观下按深色
    取字色：``muted_text`` 那档次要文字（目标值、排序值）在白底上对比度掉到 1.0 附近，屏幕上
    整列读不出来。这一条钉的是委托拿到的那份调色板仍要解析回窗口的外观。
    """
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    listing = QListWidget(host)
    listing.setObjectName("candidateList")
    layout.addWidget(listing)
    host.setStyleSheet(theme.build_stylesheet(host.palette()))
    host.show()
    qtbot.waitExposed(host)

    assert theme.palette_tokens(listing.palette()) is theme.palette_tokens(host.palette())
