"""帧⑤ 相关矩阵那两条轴上写谁的名字，以及那些名字放不放得进画布。

设计稿的刻度写 ``d·aSi``——符号加归属，两三个字符；实现写的是 ``component.0.thickness_a``
这样的机器路径。差别不止「长一点」：17 个参数的真实规模下实测 16 个 ytick 越出画布左界
（最长 155px），matplotlib 于是放弃 ``constrained_layout``（``axes sizes collapsed to zero``），
矩阵格子和它的名字一起读不到。这几条把两半都钉住——写的是设计稿那套短名，且短到能落在
帧⑤ 真实给出的那块画布里。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
from tests.support.model_cases import final_fit_result

import xrr_fitter.api as api

# 帧⑤ 的证据对话框实际给相关矩阵的画布：400×202 px @ dpi 100。名字读不读得全取决于这个
# 真实预算，而不是测试里随手给的一块大画布——够大的画布装得下机器路径，屏上那块装不下。
CANVAS_INCHES = (4.0, 2.02)

# 同一页在对话框被拉高之后实得的画布：370×540 px @ dpi 100（实测——「不确定度分析…」那个
# 对话框拖到 320×1400 时卡里这只 pages 长到 374×563）。宽度拖不下去（卡有最小宽），高度却
# 随对话框一路长，于是这一档高过宽——扁框那个「短边是高」的前提在这里整个翻过来了，而竖色标
# 是按分配框的整**高**竖立的。
TALL_CANVAS_INCHES = (3.70, 5.40)

# 一次真实拟合摊在这张矩阵上的 17 个参数：机器路径、项目既有的 display_name、以及矩阵轴上
# 该出现的短名。层参数取「符号·归属」（``d·ox``）；instrument 十项里六项的 display_name 末尾
# 本来就写着符号（``相对分辨率 σq/q``），轴上只留那一段；``scale`` 与 ``Δθ`` 分别是设计稿和
# ``docs/algorithm.md`` 逐字写过的记号；``常数背景``/``线性背景`` 没有出处，保持中文原样。
PARAMETERS = (
    ("component.0.thickness_a", "ox 厚度", "layer", "d·ox"),
    ("component.0.density_scale", "ox 相对密度", "layer", "ρ·ox"),
    ("component.0.roughness_a", "ox 入射侧粗糙度", "layer", "σ·ox"),
    ("component.1.thickness_a", "aSi 厚度", "layer", "d·aSi"),
    ("component.1.density_scale", "aSi 相对密度", "layer", "ρ·aSi"),
    ("component.1.roughness_a", "aSi 入射侧粗糙度", "layer", "σ·aSi"),
    ("backing.roughness_a", "基底连接界面粗糙度", "layer", "σ·基底"),
    ("instrument.angle_offset_deg", "入射角零点偏移", "instrument", "Δθ"),
    ("instrument.scale", "尺度", "instrument", "scale"),
    ("instrument.background", "常数背景", "instrument", "常数背景"),
    ("instrument.linear_background_per_a_inv", "线性背景", "instrument", "线性背景"),
    ("instrument.powerlaw_background_amplitude", "幂律背景幅值 B₂", "instrument", "B₂"),
    ("instrument.powerlaw_background_exponent", "幂律背景指数 p", "instrument", "p"),
    ("instrument.relative_sigma", "相对分辨率 σq/q", "instrument", "σq/q"),
    ("instrument.footprint_spill_angle_deg", "足迹满斑角 θ_fp", "instrument", "θ_fp"),
    ("instrument.absolute_sigma_a_inv", "绝对分辨率 σq,0", "instrument", "σq,0"),
    ("instrument.sigma_theta_deg", "角域分辨率 σθ", "instrument", "σθ"),
)

# 一格至少要有这么多像素。刻度不越界只是「名字读得到」；格子读不读得到取决于名字吃掉多少
# 画布——热图是方阵（``aspect='equal'``），旋转 45° 的 xtick 有多高，绘图区的边长就少多少，
# 而边长一被压缩，17 列一起变窄。402×202 的画布上把整个高度都让给格子也只有约 11px 一格，
# 这条要求其中一半：名字最多只能吃掉理论上限的一半。
MINIMUM_CELL_PIXELS = 6.0

# 帧⑤ 中栏那张双图在 1272×1060 主窗口里实得的画布：666×898 px @ dpi 100（实测）。矩阵是方阵，
# 两半各只有 174px 宽——剖面能用的横向空间不到整张画布的三分之一，区间那几行就是在这里被裁的。
PANE_INCHES = (6.66, 8.98)

# 同一张双图在 1400×900 主窗口里实得的画布：794×738 px @ dpi 100（实测）。矮一档、宽一档——
# 读数栏那几段就是在这一档溢出的（见
# ``test_the_summary_column_keeps_every_reading_inside_its_own_column``）。两档都留着：缺陷对
# 画布高度敏感，只测一档等于把「换个窗口大小就散架」放过去。
SHORT_PANE_INCHES = (7.94, 7.38)

# 同一张双图在中栏被压扁时那一档：794×400 px @ dpi 100。读数栏在这里只有 87px 高，而它是
# ``draw_uncertainty`` 换算行距的依据——窗口随后拉大到 SHORT_PANE_INCHES，栏高变成 240px，行距
# 却还按 87px 那套摊开。这一档单独留着不是为了「测个小窗口」，是为了拿到那次落差。
SQUEEZED_PANE_INCHES = (7.94, 4.0)


def _definition(name: str, display_name: str, category: str) -> api.ParameterDefinition:
    return api.ParameterDefinition(
        name=name,
        display_name=display_name,
        unit="",
        category=category,
        initial=1.0,
        lower=0.0,
        upper=2.0,
        transform="linear",
        locked=False,
    )


def _result() -> api.FitResult:
    """一份带齐 17 个参数声明的结果——短名的素材就在 ``parameter_definitions`` 上。"""
    names = tuple(name for name, *_ in PARAMETERS)
    matrix = np.eye(len(names))
    matrix[0, 3] = matrix[3, 0] = 0.93
    report = api.UncertaintyReport(
        correlation_names=names,
        correlation_matrix=matrix,
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=((names[0], names[3], 0.93),),
        systematic_residual=False,
        diagnostics=(),
        candidate_id="candidate-a",
    )
    return replace(
        final_fit_result(),
        parameter_definitions=tuple(_definition(name, display, category) for name, display, category, _ in PARAMETERS),
        uncertainty=report,
    )


def _correlation_page(qtbot):
    """帧⑤ 那只相关矩阵页，已按屏上真实的画布尺寸排完版。

    返回的是整只 ``UncertaintyPages``：``qtbot`` 只留弱引用，调用方不抓住这个父控件的话，
    Qt 会连它下面的 canvas 一起删掉，读 artist 时只剩 ``Internal C++ object already deleted``。
    """
    from xrr_fitter.gui.plots.posterior import UncertaintyPages

    pages = UncertaintyPages()
    qtbot.addWidget(pages)
    pages.set_result(_result(), "candidate-a")
    pages.correlation.figure.set_size_inches(*CANVAS_INCHES)
    return pages


def _tall_correlation_page(qtbot):
    """同一页，画布换成对话框被拉高之后那一档。"""
    pages = _correlation_page(qtbot)
    pages.correlation.figure.set_size_inches(*TALL_CANVAS_INCHES)
    return pages


def _colour_key(view):
    """这一页那条竖色标。"""
    return next(axes for axes in view.figure.axes if axes.get_label() == "<colorbar>")


def test_the_upright_colour_key_spans_exactly_the_square_it_explains(qtbot) -> None:
    """竖色标的上下两端各自对齐方阵的上下两端——它解释的是这个方阵的颜色，不是整张画布的。

    这是横躺那条（``test_correlation_matrix_readings`` 里的
    ``test_the_colour_key_spans_exactly_the_square_it_explains``）把宽高对调过来的同一件事，
    而它只在这一档才现形。方阵是 ``aspect='equal'``，边长由分配框的**短边**定死；
    ``figure.colorbar`` 竖立的那条则按分配框的整**高**。扁框里短边就是高，于是两者天然等长
    ——实测 ``CANVAS_INCHES`` 那档方阵 115.7px、色标 115.7px，一分不差，屏上看不出这里有条
    不变式。高过宽的画布上短边换成了宽，色标却仍按整高竖立：实测这一档方阵 215.7px、色标
    517.8px，高出 302.0px，上下各伸出 151.0px——一条比矩阵本身长一倍半的色带，两端悬在方阵
    上下那片空无一物的边距里。

    这一档不是假想的：卡里那只 ``pages`` 宽度有下限拖不动，高度却随对话框一路长（
    ``ResultsPanel.open_uncertainty_dialog`` 用 ``addWidget(self.uncertainty, 1)`` 带 stretch），
    所以「把对话框拉高」就到这里。断言写成两端的差值而不是 ``approx``，是为了让红的时候直接
    报出溢出多少像素。
    """
    pages = _tall_correlation_page(qtbot)
    view = pages.correlation
    view.canvas.draw()
    renderer = view.canvas.get_renderer()
    matrix = view.figure.axes[0]
    # 方阵实框要 ``apply_aspect`` 之后才是方的，否则量到的还是布局分配的那个长方框。
    matrix.apply_aspect()

    square = matrix.get_window_extent(renderer)
    assert square.height < square.width + 1.0, f"这一档的方阵没被高度压住：{square.width:.1f}×{square.height:.1f}"
    bar = _colour_key(view).get_window_extent(renderer)
    misfit = {"下端": round(bar.y0 - square.y0, 1), "上端": round(bar.y1 - square.y1, 1)}
    assert max(abs(value) for value in misfit.values()) <= 1.0, f"色标与方阵两端错开 {misfit} px"


def test_the_matrix_axes_carry_the_designed_short_names(qtbot) -> None:
    """两条轴上写设计稿那套短名，而不是参数在代码里的那条路径。

    ``component.0.thickness_a`` 与 ``d·ox`` 指的是同一个数，但矩阵是拿来「扫一眼看哪两格发
    亮」的：读者要在 17×17 的格子间反复来回对照行列，路径式的名字每次都得读到第三段才知道
    是谁。设计稿把它压成「符号·归属」正是为这个来回。

    纵轴逐条量全套：名字的完整一份在这条轴上，任何一格空掉都是少了一个参数。横轴只量「写出来
    的那些写对了没有」——45° 斜排下相邻两条挤不挤取决于画布，实现会按当下的间距隔几列写一个（见
    ``test_the_matrix_ticks_stay_far_enough_apart_to_be_read``），所以这里钉的是「写的是短名、
    且写在自己那一列上」，不是「每一列都写」。
    """
    pages = _correlation_page(qtbot)
    axes = pages.correlation.figure.axes[0]
    expected = tuple(row[-1] for row in PARAMETERS)

    assert tuple(text.get_text() for text in axes.get_yticklabels()) == expected
    printed = {index: text.get_text() for index, text in enumerate(axes.get_xticklabels()) if text.get_text()}
    assert printed, "横轴一个名字都没写"
    assert printed == {index: expected[index] for index in printed}, printed


def test_every_matrix_label_lands_inside_the_canvas(qtbot) -> None:
    """每个刻度都要落在画布里——越出左界的那半截在屏上根本不存在。

    这不是「挤一点」：17 个参数的实测里 16 个 ytick 的左端是负坐标，最长的
    ``instrument.powerlaw_background_amplitude`` 越界 155px，比矩阵自己的宽度还多。
    """
    pages = _correlation_page(qtbot)
    view = pages.correlation
    view.canvas.draw()
    renderer = view.canvas.get_renderer()

    overflow = {
        text.get_text(): round(-text.get_window_extent(renderer).x0, 1)
        for text in view.figure.axes[0].get_yticklabels()
        if text.get_window_extent(renderer).x0 < 0.0
    }
    assert overflow == {}, f"越出画布左界的刻度：{overflow}"


def test_the_names_leave_the_matrix_enough_room_to_be_read(qtbot) -> None:
    """名字排得进画布之后，还得给格子留下地方。

    上一条只管刻度不越界，它有个绿得很轻松的走法：名字长到把绘图区挤成一条窄带，刻度自然
    全在画布里。实测 17 个机器路径下矩阵只剩 76px 边长——4.5px 一格，一眼扫过去连主对角线
    都分不出来，而那时前两条测试是可以同时为绿的。
    """
    pages = _correlation_page(qtbot)
    view = pages.correlation
    view.canvas.draw()
    renderer = view.canvas.get_renderer()

    extent = view.figure.axes[0].get_window_extent(renderer)
    cell = extent.width / len(PARAMETERS)
    assert cell >= MINIMUM_CELL_PIXELS, (
        f"矩阵边长 {extent.width:.0f}px，{len(PARAMETERS)} 列摊下来一格只有 {cell:.1f}px"
    )


def test_the_matrix_still_gets_laid_out_at_the_shipped_canvas_size(qtbot) -> None:
    """``constrained_layout`` 必须真的排上，否则连边距都不再是它算出来的那一套。

    名字长到挤没了绘图区时，matplotlib 不报错，它放弃排版并留下一句 warning——之后每一条
    边距、colorbar 位置和格子大小都退回默认值，页面看着「只是丑」，其实是没排版。
    """
    pages = _correlation_page(qtbot)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pages.correlation.canvas.draw()

    collapsed = [str(item.message) for item in caught if "constrained_layout" in str(item.message)]
    assert collapsed == [], f"排版被放弃：{collapsed}"


def _profiled_result() -> api.FitResult:
    """同一份 17 参数结果，另挂三条剖面与三段自助区间。

    三条是帧⑤ 画的规模：剖面扫描按强相关挑几个参数跑，不会给 17 个都跑一遍。短名的素材还是
    ``parameter_definitions``——和矩阵同一处。
    """
    result = _result()
    names = tuple(name for name, *_ in PARAMETERS)[:3]
    profiles = tuple(
        api.ParameterProfile(
            name,
            np.linspace(0.9, 1.1, 5),
            np.array([1.4, 0.6, 0.2, 0.6, 1.4]),
            True,
            True,
            objective_threshold=1.2,
        )
        for name in names
    )
    report = replace(
        result.uncertainty,
        profiles=profiles,
        bootstrap_intervals=tuple((name, 0.92, 1.08) for name in names),
    )
    return replace(result, uncertainty=report)


def _fully_profiled_result() -> api.FitResult:
    """同一份 17 参数结果，17 条剖面全跑了——屏上真实那一档的规模。

    ``_profiled_result`` 只挂三条，那是「按强相关挑几个跑」的规模；但产品真跑一遍
    ``analysis/profiles.py`` 时逐个自由参数都扫（实测切到不确定度页，``report.profiles`` 是
    17 条），于是图例项 18 行——17 条曲线加一条阈值线。这个差别不是「多几行字」：三条时图例
    205×79.5px 落在绘图区里绰绰有余，17 条时它 258×345.7px，比整格还高，于是同一段代码在两个
    规模上是两种排布结果。缺陷只在真实规模上现形，替身也就得按真实规模造。
    """
    result = _result()
    names = tuple(name for name, *_ in PARAMETERS)
    profiles = tuple(
        api.ParameterProfile(
            name,
            np.linspace(0.9, 1.1, 5),
            np.array([1.4, 0.6, 0.2, 0.6, 1.4]),
            True,
            True,
            objective_threshold=1.2,
        )
        for name in names
    )
    report = replace(
        result.uncertainty,
        profiles=profiles,
        bootstrap_intervals=tuple((name, 0.92, 1.08) for name in names),
    )
    return replace(result, uncertainty=report)


def _profile_page(qtbot):
    """帧⑤ 那只 Profile 似然页，画布同样收到屏上真实的那块预算。

    与 ``_correlation_page`` 共用 ``_page_view()``，所以两页拿到的是同一块画布——名字放不放
    得下这件事在两页上是同一道题。
    """
    from xrr_fitter.gui.plots.posterior import UncertaintyPages

    pages = UncertaintyPages()
    qtbot.addWidget(pages)
    pages.set_result(_profiled_result(), "candidate-a")
    pages.profile.figure.set_size_inches(*CANVAS_INCHES)
    return pages


def _uncertainty_pane(qtbot, inches: tuple[float, float] = PANE_INCHES, result: api.FitResult | None = None):
    """帧⑤ 中栏那张双图，画布收到主窗口里实测的那块预算。

    对话框里剖面独占一页（``_profile_page``），中栏这张是矩阵和剖面上下叠的两格——剖面拿到整个
    栏宽，但只有半栏高，名字放不放得下这件事在这里最紧。

    ``result`` 留着换素材：剖面条数改变的不只是图例有几行，还有这两格各分到多少高度（图例参与
    ``constrained_layout``），所以真实规模那一档要另外传一份（``_fully_profiled_result``）。
    """
    from matplotlib.figure import Figure

    from xrr_fitter.gui.plots import diagnostics
    from xrr_fitter.gui.plots.diagnostics import DiagnosticCanvas, DiagnosticView
    from xrr_fitter.gui.plots.sld import draw_uncertainty

    figure = Figure(figsize=inches, layout="constrained")
    # 形状取自实现自己那一处声明，不在这里手抄一遍：抄一遍就会和实现分叉，而分叉的那一次
    # 恰好就是 ``_reset_uncertainty_axes`` 的重建分支从没被测到的原因。
    view = DiagnosticView(figure, DiagnosticCanvas(figure), diagnostics._axes(figure, "uncertainty"))
    qtbot.addWidget(view.canvas)
    draw_uncertainty(view, _profiled_result() if result is None else result, "candidate-a")
    return view


def test_the_profile_curves_carry_the_designed_short_names(qtbot) -> None:
    """剖面图的图例和矩阵的刻度写同一套名字。

    这两张图在帧⑤ 是上下叠的同一屏：矩阵轴上写 ``d·aSi``，剖面图例里同一个参数写
    ``component.1.thickness_a``，读者要在一屏之内自己完成一次翻译才敢说「亮的那格就是这条
    曲线」。翻译发生在两张图之间，而它们本来是为了互相印证才叠在一起的。

    图例项不只有短名：每项前面还有中文量名、后面还有一句形状判读（那两件事归
    ``tests/gui/test_profile_likelihood_readings.py`` 钉）。这里钉的是「短名照原样出现在项里」
    ——它是读者拿去矩阵上找那一格的钥匙，措辞一分叉，那次翻译就又回来了。
    """
    pages = _profile_page(qtbot)
    legend = pages.profile.figure.axes[0].get_legend()

    labels = tuple(text.get_text() for text in legend.get_texts())
    assert len(labels) == 4, labels
    for short, entry in zip(("d·ox", "ρ·ox", "σ·ox"), labels, strict=False):
        assert short in entry, (short, entry)
    assert "区间闭合阈值" in labels[3], labels


def test_the_profile_half_keeps_every_label_inside_its_axes(qtbot) -> None:
    """剖面那一格里的每一笔字都要整行落在格子里——越出右界的半截在屏上不存在。

    这条原先钉的是自助区间那几行：实测帧⑤ 上前两行读作
    ``component.0.thickness_a: [32.1, 36.4|`` 和 ``component.1.thickness_a: [481.2, 492|``，
    区间上界正好落在被裁掉的那一段里，于是这行字唯一有信息量的部分没了。那几行现在不画在图上
    （见 ``_draw_profiles``），判据就不能只数 ``axes.texts``——那样它会变成一条永远为真的空断言。
    改成把这一格里所有会上屏的文字都量一遍：图例条目、阈值那条的标签、占位文案都算在内。
    """
    view = _uncertainty_pane(qtbot)
    view.canvas.draw()
    renderer = view.canvas.get_renderer()
    # colorbar 排在 ``figure.axes`` 末尾，剖面是第二个。
    axes = view.figure.axes[1]
    right = axes.get_window_extent(renderer).x1
    legend = axes.get_legend()
    labels = (*axes.texts, *(legend.get_texts() if legend is not None else ()))
    assert labels, "剖面那一格上一个字都没有，这条测试量不到东西"

    overflow = {
        text.get_text(): round(text.get_window_extent(renderer).x1 - right, 1)
        for text in labels
        if text.get_window_extent(renderer).x1 > right
    }
    assert overflow == {}, f"越出子图右界的标注：{overflow}"


def _assert_summary_clipping(panel, frame) -> None:
    unclipped = [text.get_text().replace("\n", " ") for text in panel.texts if not text.get_clip_on()]
    assert unclipped == [], f"没开 clip、放不下就会画到别的格子上的段：{unclipped}"

    # 矩形 clip path 会折成 clipbox；只能接受当前读数栏自己的框。
    astray = {
        text.get_text().replace("\n", " "): text.get_clip_box()
        for text in panel.texts
        if text.get_clip_box() is None or not np.allclose(text.get_clip_box().extents, frame.extents)
    }
    assert astray == {}, f"clip 框不是这一栏的框：{astray}"


def test_the_summary_column_keeps_every_reading_inside_its_own_column(qtbot) -> None:
    """矩阵旁那栏读数必须整段留在这一栏里，不能溢到别的格子上。

    这一栏自上而下码五段（最强相关、系数、次强、判定门、一句判读）。行距一旦算错，实现不会
    报错也不会截断：``clip_on`` 对 ``Text`` 默认是关的，越出下界的那几段照样画在画布上，落进
    下方 Profile 那一格里。屏上读到的是「次强」孤零零悬在 Profile 的抬头旁边，而它下面那半句
    和整句判读干脆没了。

    此前实测（1400×900 主窗口，切到不确定度那一页）：栏框高 240px，而实现给每段留了 202px 的
    行距——第三段起全部落到栏外，末段的基线掉到 y=−56，连画布都出去了。

    行距之所以荒谬，是因为它按绘制那一刻的 ``panel.get_window_extent()`` 换算成 axes 分数。画布
    一变尺寸，这个分数对应的像素间距跟着变，栏高也跟着变——但两者的比例被钉死在了旧尺寸上。所以
    这条测试要走屏上那条时序：先在中栏被压扁的那一档（``SQUEEZED_PANE_INCHES``，栏高 87px）画一次，
    再把窗口拉到 ``SHORT_PANE_INCHES``（栏高 240px）重绘。重绘不重算数据，正如拖窗口不会重跑拟合。

    判据分三条。头两条是「放不下时怎么收场」的兜底——更小的画布档（对话框压到最扁那一档）仍然
    码不下，而折行必须在构图时定下来（往文本里插的是真的换行符），栏那时还没排版：
    ① 每段都开着 clip 且 clip 框就是栏框：放不下的部分被栏底切掉，而不是画到 Profile 那一格上；
    ② clip 框只能是栏框：换成画布或矩阵的框，「切掉」就又变成「画到别处」。
    第三条是这一档该有的结果：
    ③ 四段全部整段在栏内。这一条原先写的是「除末段外」——那时栏里码着五段（多一句判定门），
    240px 的栏要 268px，末尾那句判读注定被栏底切掉。判定门那句现在归卡抬头右端（见
    ``tests/gui/test_correlation_matrix_readings.py::test_the_matrix_states_the_threshold_in_its_heading_not_its_readings``），
    少掉的那段约 54px 正是这栏一直差的高度，于是「被牺牲的只能是末段」这个让步不再需要。
    原始缺陷下三条全红：clip 是关的、第三段起整段在栏外、末段的基线掉出画布。
    """
    from xrr_fitter.gui.plots.correlation import SUMMARY_AXES_LABEL

    view = _uncertainty_pane(qtbot, SQUEEZED_PANE_INCHES)
    view.canvas.draw()
    view.figure.set_size_inches(*SHORT_PANE_INCHES)
    view.canvas.draw()
    renderer = view.canvas.get_renderer()
    matrix = view.figure.axes[0]
    panel = next(axes for axes in matrix.child_axes if axes.get_label() == SUMMARY_AXES_LABEL)
    frame = panel.get_window_extent(renderer)
    assert len(panel.texts) == 4, [text.get_text() for text in panel.texts]

    _assert_summary_clipping(panel, frame)

    overflow = {
        text.get_text().replace("\n", " "): (
            round(frame.y0 - text.get_window_extent(renderer).y0, 1),
            round(text.get_window_extent(renderer).y1 - frame.y1, 1),
        )
        for text in panel.texts
        if text.get_window_extent(renderer).y0 < frame.y0 or text.get_window_extent(renderer).y1 > frame.y1
    }
    assert overflow == {}, f"越出读数栏的段（下溢, 上溢 px）：{overflow}"


def test_the_legend_stays_inside_the_profile_half_at_the_real_profile_count(qtbot) -> None:
    """Profile 图例在真实规模（17 条曲线）下仍整个落在那一格里——不溢出底边、不盖住绘图区。

    ``_profiled_result`` 只挂三条 profile，那是「按强相关挑几个跑」的规模；图例 3+1 项（三条
    曲线加一条阈值线）205×79.5px，落在绘图区里绰绰有余。但产品真跑一遍时逐个自由参数都扫
    （实测 17 条），于是图例 18 项单列、258×345.7px——比半栏高的绘图区（211.8px）还高 163%。

    图例参与 ``constrained_layout``：它撑大 Profile 那格的 tightbbox，layout 为它让出高度，
    **反过来压小矩阵**——而读数栏整块挂在矩阵的 axes 坐标上（``SUMMARY_PANEL_BOUNDS``），矩阵
    一小，读数栏跟着矮，四段读数于是被栏底裁掉。所以图例溢出与读数栏溢出是同一个根因，这条
    RED 之后改图例排布、两处一起解。

    判据只说「整个在格子里」，不规定用什么手段：多列、移出绘图区、或限项数都可以——只要图例
    不再撑大那格的 tightbbox 去压矩阵，读数栏就能拿回高度。
    """

    view = _uncertainty_pane(qtbot, SHORT_PANE_INCHES, result=_fully_profiled_result())
    view.canvas.draw()
    renderer = view.canvas.get_renderer()
    profile = view.figure.axes[1]
    legend = profile.get_legend()
    assert legend is not None, "Profile 图例缺失，这条测试量不到东西"

    axes_box = profile.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    overflow = {
        "底部溢出": round(axes_box.y0 - legend_box.y0, 1) if legend_box.y0 < axes_box.y0 else 0.0,
        "顶部溢出": round(legend_box.y1 - axes_box.y1, 1) if legend_box.y1 > axes_box.y1 else 0.0,
        "左侧溢出": round(axes_box.x0 - legend_box.x0, 1) if legend_box.x0 < axes_box.x0 else 0.0,
        "右侧溢出": round(legend_box.x1 - axes_box.x1, 1) if legend_box.x1 > axes_box.x1 else 0.0,
    }
    assert all(v == 0.0 for v in overflow.values()), f"图例越出 Profile 绘图区：{overflow}"


def test_the_readings_survive_a_full_profile_run_at_the_designed_scale(qtbot) -> None:
    """真实规模（17 条 profile）下读数栏的四段仍整段留在栏内——不被栏底裁掉。

    ``test_the_summary_column_keeps_every_reading_inside_its_own_column`` 只用三条 profile
    的替身（``_profiled_result``），那一档图例不溢出、矩阵不被压、读数栏 240px 高够装四段。
    但真实规模是 17 条：18 项图例撑大 Profile 那格的 tightbbox，``constrained_layout`` 为它让出
    高度，反过来把矩阵从 379.9 压到 211.8px——读数栏挂在矩阵坐标上，跟着从 235.5 矮到 131.3px，
    而四段总跨 175.0px，溢出 43.7px，末段「需结合 Profile 似然判读」被 ``clip_on=True`` 从
    栏底裁掉。这条 RED 坐实这件事，之后改图例排布两处一起绿。
    """
    from xrr_fitter.gui.plots.correlation import SUMMARY_AXES_LABEL

    view = _uncertainty_pane(qtbot, SHORT_PANE_INCHES, result=_fully_profiled_result())
    view.canvas.draw()
    renderer = view.canvas.get_renderer()
    matrix = view.figure.axes[0]
    panel = next(axes for axes in matrix.child_axes if axes.get_label() == SUMMARY_AXES_LABEL)
    frame = panel.get_window_extent(renderer)
    assert len(panel.texts) == 4, [text.get_text() for text in panel.texts]

    overflow = {
        text.get_text().replace("\n", " "): (
            round(frame.y0 - text.get_window_extent(renderer).y0, 1),
            round(text.get_window_extent(renderer).y1 - frame.y1, 1),
        )
        for text in panel.texts
        if text.get_window_extent(renderer).y0 < frame.y0 or text.get_window_extent(renderer).y1 > frame.y1
    }
    assert overflow == {}, f"越出读数栏的段（下溢, 上溢 px）：{overflow}"


def _tick_clearance(axes, which: str) -> tuple[float, float]:
    """那条轴上会上屏的刻度，相邻两条之间的净距，以及一行字有多高（都是像素）。

    45° 旋转下包围盒必然彼此交叠（斜排的东西包围盒就是这样），拿 bbox 相交当判据会把「排得
    很开」也判成挤。真正决定读不读得出来的是**相邻两条文字基线之间的垂直距离**：标签沿 45°
    伸出，相邻两条是两条平行的文本行，刻度间距 d 对应的垂距是 ``d·sin45° = d/√2``——与标签
    多长无关（长度决定的是刻度带有多高，不是两行挤不挤）。横排刻度（rotation=0）的净距就是
    d 本身。

    间距从刻度自己的数据坐标换算，不按刻度条数摊 axes 宽度：实现若改成隔位标注，相邻可见标签
    的间距就是两格，按条数摊会把那条路误判成挤。``apply_aspect`` 得先跑，否则 ``transData``
    还是布局分配的那个长方框（见 ``_cell_side_pixels`` 同一处理由）。

    字高只能从字号推，不能量 ``get_window_extent().height``：45° 下那个高度是文本长度的对角
    投影，几个字符就比一行字高得多。
    """
    labels = axes.get_xticklabels() if which == "x" else axes.get_yticklabels()
    locations = axes.get_xticks() if which == "x" else axes.get_yticks()
    axes.apply_aspect()
    index = 0 if which == "x" else 1
    kept = [
        float(axes.transData.transform((float(location), float(location)))[index])
        for location, label in zip(locations, labels, strict=False)
        if label.get_visible() and label.get_text()
    ]
    if len(kept) < 2:
        return (0.0, 0.0)
    pitch = min(abs(second - first) for first, second in zip(kept, kept[1:], strict=False))
    rotation = math.radians(float(labels[0].get_rotation()) % 180.0)
    clearance = pitch * math.sin(rotation) if rotation else pitch
    # 一行汉字的行高按字号本身算（dpi/72），与 ``offset_copy(units="points")`` 同一口径。
    height = float(labels[0].get_fontsize()) * float(axes.figure.dpi) / 72.0
    return (clearance, height)


def test_the_matrix_ticks_stay_far_enough_apart_to_be_read(qtbot) -> None:
    """矩阵刻度上屏了就得读得出来——相邻两条的净距不能小于一行字的高度。

    前面几条只管「名字排得进画布」「格子还剩多少像素」，两者都可以在刻度已经糊成一团时为绿：
    17 个 45° 标签全在画布里、格子也有 14px，而相邻两条标签的基线只隔 10.0px、字却有 13.9px
    高——后一条压在前一条的上半截里，屏上是一片斜向的墨。实测（1400×900 主窗口那一档）就是
    这个数：净距 10.0px 对字高 13.9px，互压 3.9px。

    两档画布都量，且横纵两条轴都量。横轴那排是 45°（净距 = 间距/√2，只有间距的七成），纵轴
    是横排（净距就是间距）——所以先失守的总是横轴，但把纵轴一起钉住才拦得下「把 x 轴的问题挪
    到 y 轴上」这种走法。

    判据只说「上屏的刻度要读得出」，不规定用什么手段达到：缩字号、隔位标注、或者干脆不写这一轴
    的刻度（对称矩阵的名字从另一轴读，见
    ``test_the_matrix_says_how_to_read_its_columns_when_it_drops_their_ticks``）都算合格。
    """
    for inches in (PANE_INCHES, SHORT_PANE_INCHES):
        view = _uncertainty_pane(qtbot, inches)
        view.canvas.draw()
        axes = view.figure.axes[0]

        crowded = {}
        for which in ("x", "y"):
            clearance, height = _tick_clearance(axes, which)
            if clearance and clearance < height:
                crowded[which] = (round(clearance, 2), round(height, 2))
        assert crowded == {}, f"画布 {inches}：刻度互相压（轴: (净距, 字高) px）{crowded}"


def test_the_matrix_says_how_to_read_its_columns_when_it_drops_their_ticks(qtbot) -> None:
    """真的撤掉横轴那排名字时，得留一句话交代列是谁——不能让读者对着无名的列发呆。

    17 个参数下横轴那排 45° 标签是挤得最凶的一处，把它整个撤掉能把约 36px 的刻度带高度还给
    方阵（实测 240.5→260.5px），是收益最大的一条路；相关矩阵又必然对称，行名与列名本就是同
    一套，撤掉的只是重复的那一份。但「重复」这件事是读者不知道的：他看到的是纵轴有名字、横轴
    光秃秃，第 12 列是谁只能靠猜。所以撤掉的前提是把读法写出来——一句轴标签，点明列与行同序、
    名字在纵轴上。

    落地实现没走这条路：它按当下的刻度间距隔几列写一个，且把最宽的那个名字钉在可见集合里（带高
    由最宽的可见标签定，跳掉它会让带高、方阵边长、间距连锁变化，一次 draw 一个结论地来回翻）——
    所以横轴永远至少写着一个名字，这条测试当下走 early return 空过。留着它是为了换手段的那一天：
    真撤光刻度时它会立刻要求补上读法，而
    ``test_the_matrix_ticks_stay_far_enough_apart_to_be_read`` 对「一条都不写」是宽裕到底的。
    """
    view = _uncertainty_pane(qtbot, SHORT_PANE_INCHES)
    view.canvas.draw()
    axes = view.figure.axes[0]
    if [text for text in axes.get_xticklabels() if text.get_visible() and text.get_text()]:
        return

    label = axes.get_xlabel()
    assert label, "横轴既没有刻度也没有轴标签，17 列全是无名的"
    assert "纵轴" in label or "行" in label, label


def test_the_strong_correlation_line_names_the_pair_the_matrix_way() -> None:
    """「强相关」那一行和矩阵共用一套名字。

    这两处讲的是同一件事：读者拿着这行里的一对参数，去矩阵上找那一格。一边写
    ``component.0.thickness_a``、另一边写 ``d·ox``，中间就多了一次翻译——而这一步翻译恰恰
    发生在读者最需要相信「说的是同一对」的时候。
    """
    from xrr_fitter.gui.results.uncertainty import _evidence_lines

    lines = _evidence_lines(_result(), "candidate-a")
    strong = next(line for line in lines if line.startswith("强相关"))

    assert "d·ox/d·aSi" in strong, strong
    assert "component." not in strong, strong
