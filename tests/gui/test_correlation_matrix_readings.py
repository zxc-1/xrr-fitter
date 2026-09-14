"""帧⑤ 那张相关矩阵上读得出数，而不只读得出颜色。

设计稿的矩阵每格印着系数本身（``-0.72``、``+0.55``，主对角线一个 ``1``），色标横躺在矩阵
下方并写明「Pearson 相关系数 ρ」，矩阵右边还有三行读数点名最强的那一对是谁、外加一句
「这对强相关意味着什么」的判读。实现此前只有 ``imshow`` 加一条纵置色标，标题写「固定范围
[-1, 1]」——颜色深浅要靠眼睛去比，「哪两格最亮」得读者自己扫完 n² 格再回头对刻度。

这个文件钉住那几件读数。规模按设计稿给的 6 个参数，因为格内数值这件事在 6×6 与 17×17 上
是两个结论：17 参数时一格只有几个像素，印上去就是一团糊字，所以「印」这件事必须自己看
格子够不够宽（见 ``test_a_seventeen_parameter_matrix_leaves_its_cells_bare``）。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import final_fit_result

import xrr_fitter.api as api
from xrr_fitter.model.inference import CovarianceEvidence

# 设计稿帧⑤ 画的那 6 个参数，以及矩阵轴上该出现的短名。挑的是一层氧化加一层 aSi 的厚度、
# 密度、粗糙度，再加仪器标度——也就是「厚度与密度纠缠」这件事看得见的最小规模。
PARAMETERS = (
    ("component.0.thickness_a", "ox 厚度", "layer", "d·ox"),
    ("component.0.density_scale", "ox 相对密度", "layer", "ρ·ox"),
    ("component.1.thickness_a", "aSi 厚度", "layer", "d·aSi"),
    ("component.1.density_scale", "aSi 相对密度", "layer", "ρ·aSi"),
    ("component.1.roughness_a", "aSi 入射侧粗糙度", "layer", "σ·aSi"),
    ("instrument.scale", "尺度", "instrument", "scale"),
)

# 设计稿那张热图逐格写着的系数（SVG 里一格一个 ``<text>``）。最强的一对是 d·aSi 与 ρ·aSi
# 的 −0.72，次强是 d·ox 与 ρ·ox 的 +0.55——后者 |ρ| 不到 0.6，所以它进不了
# ``strong_correlations``，「次强」这一行只能从矩阵本身排出来。
COEFFICIENTS = (
    (1.00, 0.55, 0.10, -0.08, 0.05, 0.03),
    (0.55, 1.00, -0.06, -0.12, 0.09, 0.04),
    (0.10, -0.06, 1.00, -0.72, 0.18, -0.20),
    (-0.08, -0.12, -0.72, 1.00, -0.15, 0.31),
    (0.05, 0.09, 0.18, -0.15, 1.00, -0.10),
    (0.03, 0.04, -0.20, 0.31, -0.10, 1.00),
)

# 帧⑤ 中栏那张双图在 1272×1060 主窗口里实得的画布，与 ``test_correlation_labels`` 同一处实测。
PANE_INCHES = (6.66, 8.98)

# 同一张双图在 1400×900 主窗口里实得的画布：794×738 px @ dpi 100（实测，与
# ``test_correlation_labels.SHORT_PANE_INCHES`` 同一处）。矮一档、宽一档——读数栏那几段就是在这一
# 档溢出的（见 ``test_the_readings_all_fit_the_column_at_the_designed_scale``）。
SHORT_PANE_INCHES = (7.94, 7.38)

# 中栏被压扁到读数栏只有 62.8px 宽那一档（实测）：短名对 ``d·aSi ↔ ρ·aSi`` 在 9pt 下就要 78px，
# 这个宽度上一行都排不下。
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


def _report(names: tuple[str, ...], matrix: np.ndarray) -> api.UncertaintyReport:
    """一份只带相关证据的报告。

    ``strong_correlations`` 按产品默认的 ``ConfidenceThresholds.strong_correlation``（0.95）挑，
    于是这份替身里它是空的——0.72 够不上那条判定门。这是故意的：图上「最强 / 次强」两行和
    逐格加粗认的是读图那条界（|ρ|≥0.6），素材只能是矩阵自己。读 ``strong_correlations`` 的实现
    在这份替身上会画出一栏空白，也就是说这个字段替不了排序。
    """
    strong = tuple(
        (names[row], names[column], float(matrix[row, column]))
        for row in range(len(names))
        for column in range(row + 1, len(names))
        if abs(float(matrix[row, column])) >= 0.95
    )
    return api.UncertaintyReport(
        correlation_names=names,
        correlation_matrix=matrix,
        covariance_evidence=CovarianceEvidence(names, matrix, "gaussian_known_sigma", len(names)),
        parameter_sigma=np.ones(len(names)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=strong,
        systematic_residual=False,
        diagnostics=(),
        candidate_id="candidate-a",
    )


def _result() -> api.FitResult:
    """设计稿规模的一份结果：6 个参数、一对强相关、一对够不上强相关的次强。"""
    names = tuple(name for name, *_ in PARAMETERS)
    matrix = np.array(COEFFICIENTS, dtype=float)
    return replace(
        final_fit_result(),
        parameter_definitions=tuple(_definition(name, display, category) for name, display, category, _ in PARAMETERS),
        uncertainty=_report(names, matrix),
    )


def _wide_result() -> api.FitResult:
    """17 个参数摊在同一块画布上——一格只剩几像素的那个规模。"""
    names = tuple(f"component.{index}.thickness_a" for index in range(17))
    matrix = np.eye(17)
    matrix[0, 3] = matrix[3, 0] = 0.93
    definitions = tuple(_definition(name, f"层{index} 厚度", "layer") for index, name in enumerate(names))
    return replace(final_fit_result(), parameter_definitions=definitions, uncertainty=_report(names, matrix))


def _pane(qtbot, result: api.FitResult, inches: tuple[float, float] = PANE_INCHES):
    """帧⑤ 中栏那张双图，画布收到屏上实测的那块预算。"""
    from matplotlib.figure import Figure

    from xrr_fitter.gui.plots import diagnostics
    from xrr_fitter.gui.plots.diagnostics import DiagnosticCanvas, DiagnosticView
    from xrr_fitter.gui.plots.sld import draw_uncertainty

    figure = Figure(figsize=inches, layout="constrained")
    # 形状取自实现自己那一处声明，与 ``test_correlation_labels`` 同一条理由：手抄一遍就会分叉。
    view = DiagnosticView(figure, DiagnosticCanvas(figure), diagnostics._axes(figure, "uncertainty"))
    qtbot.addWidget(view.canvas)
    draw_uncertainty(view, result, "candidate-a")
    view.canvas.draw()
    return view


def _matrix_axes(view):
    """矩阵那一格：按标题找，而不是按 ``figure.axes`` 的下标。

    这张 figure 上的 axes 数目会随着色标和读数栏变动，下标式的定位每加一格就错一次。
    抬头在哪一端由 ``test_the_matrix_is_titled_the_way_the_design_names_the_card`` 单独钉；这里
    左中两端都认，免得那一条契约的红牵连到别的十来条身上——那样报出来的全是 ``StopIteration``，
    看不出真正断的是哪件事。
    """
    from xrr_fitter.gui.plots.correlation import CORRELATION_TITLE

    return next(
        axes
        for axes in view.figure.axes
        if CORRELATION_TITLE in (axes.get_title(loc="left"), axes.get_title(loc="center"))
    )


def _title_text(axes, content: str):
    """抬头那一行里写着 ``content`` 的那个 ``Text``。

    matplotlib 的三个 title 只有居中那个有公开属性（``axes.title``），左右两个挂在私有名下。
    这里按文本内容从 ``get_children()`` 里认——公开取法，且顺带确认这段字真的挂在这个 axes 上。
    """
    from matplotlib.text import Text

    return next(child for child in axes.get_children() if isinstance(child, Text) and child.get_text() == content)


def _cell_texts(axes) -> dict[tuple[int, int], object]:
    """屏上真读得到的那些格内读数，按 (行, 列) 取。

    两道过滤。一是坐标系：只认画在数据坐标里的，抬头旁那句阈值注写在 axes 坐标上，不该被数成
    一格读数。二是可见性：格子够不够宽是画布尺寸的函数，而画布会被拖动，所以实现把这个判断放在
    每次 draw 里（建 artist 那一刻的结论会随 resize 过期）。于是「不印」在对象上表现为 artist
    仍在、``get_visible()`` 为假——读者眼里没有这个数，这里也就不该数它。
    """
    cells: dict[tuple[int, int], object] = {}
    for text in axes.texts:
        if text.get_transform() is not axes.transData or not text.get_visible():
            continue
        column, row = text.get_position()
        cells[(round(row), round(column))] = text
    return cells


def test_every_cell_prints_the_coefficient_it_colours(qtbot) -> None:
    """36 格各印着自己那个系数，主对角线写 ``1``。

    热图的颜色只给「大概多强、什么符号」；读者真正要带走的是数字本身——「d·aSi 与 ρ·aSi
    是 −0.72」这句话要能从矩阵上直接抄下来，而不是先目测色深、再回头去比色标。设计稿正是
    逐格印上了数，主对角线只写一个 ``1``（它恒等于 1，印 ``+1.00`` 是拿三个字符换零信息）。
    """
    view = _pane(qtbot, _result())
    cells = _cell_texts(_matrix_axes(view))

    assert len(cells) == len(PARAMETERS) ** 2, sorted(cells)
    expected = {
        (row, column): "1" if row == column else f"{COEFFICIENTS[row][column]:+.2f}"
        for row in range(len(PARAMETERS))
        for column in range(len(PARAMETERS))
    }
    assert {key: text.get_text() for key, text in cells.items()} == expected


def test_the_strongest_cells_are_the_ones_set_in_bold(qtbot) -> None:
    """|ρ| 过了强相关阈值的格子加粗，其余不加。

    逐格印数之后 36 个数字权重相同，扫一眼仍分不出该看哪格——设计稿把过阈值的那几格连字
    一起加粗，也就是把「哪对纠缠」这个判读直接画在矩阵上。主对角线一并加粗：它是每行的
    参照点，读者顺着它找自己在第几行。
    """
    view = _pane(qtbot, _result())
    cells = _cell_texts(_matrix_axes(view))

    bold = {key for key, text in cells.items() if text.get_fontweight() in ("bold", 700)}
    expected = {
        (row, column)
        for row in range(len(PARAMETERS))
        for column in range(len(PARAMETERS))
        if row == column or abs(COEFFICIENTS[row][column]) >= 0.6
    }
    assert bold == expected


def test_a_seventeen_parameter_matrix_leaves_its_cells_bare(qtbot) -> None:
    """格子窄到装不下一个数时就不印——17 参数下印上去是一团互相压掉的糊字。

    ``test_correlation_labels`` 已经钉住 17 参数时一格只有 6px 上下。``+0.55`` 五个字符在
    那个宽度里必然彼此重叠，读到的不是数而是墨。所以印数这件事得自己看格子够不够宽：够
    就印，不够就退回纯色块，读者去点开单页视图（那里矩阵独占整张画布）。
    """
    view = _pane(qtbot, _wide_result())

    assert _cell_texts(_matrix_axes(view)) == {}


def test_the_designed_six_columns_all_keep_their_names(qtbot) -> None:
    """设计稿画的这 6 列，横轴上一个名字都不许省。

    17 参数下横轴那排 45° 标签会挤成一片斜向的墨，实现于是按当下的刻度间距隔几列写一个（见
    ``xrr_fitter.gui.plots.correlation._ColumnLabels`` 与
    ``tests/gui/test_correlation_labels.py::test_the_matrix_ticks_stay_far_enough_apart_to_be_read``）。
    那是给参数多到读不出的那一档准备的救火，而设计稿画的就是这 6 列：一格 53.9px、相邻两条标签
    的基线垂距 38.1px 对 9pt 字高 12.5px——宽裕三倍，没有任何一列有理由让位。

    这条钉的就是「救火手段不许蔓延到设计稿那一档」：隔位的门槛若被写松（比如无条件隔一列写一个，
    或者把步长按参数个数而不是按当下间距算），6 列里会立刻少掉三个名字，而
    ``test_the_matrix_ticks_stay_far_enough_apart_to_be_read`` 对此照样为绿——省得越多，那条越宽裕。
    """
    view = _pane(qtbot, _result())

    axes = _matrix_axes(view)
    expected = tuple(row[-1] for row in PARAMETERS)
    assert tuple(text.get_text() for text in axes.get_xticklabels()) == expected
    assert tuple(text.get_text() for text in axes.get_yticklabels()) == expected


def test_the_column_names_ride_above_the_matrix(qtbot) -> None:
    """45° 的列名写在矩阵**上方**，把底下那条边整条让给色标。

    设计稿把两者分在矩阵的两条边上（列名基线 y=50 而首行格子从 y=59 起，色带在 y=356、
    最后一行格子止于 y=333），这不是排版口味：色标已经收到方阵实宽（见
    ``test_the_colour_key_spans_exactly_the_square_it_explains``），横躺在下方时它和斜标签带
    争的是同一条边——斜标签在 17 参数那档高 36px 上下，色标被整条推远，读者要跨过一片斜向的字
    才找到那把尺。列名挪到上方之后两者各占一边，中间只隔着矩阵自己。

    旋转角仍是实现的 45°（设计稿画的是 32°）：那个角度在 17 参数下有实测理由——相邻标签的基线
    垂距是 ``pitch × sin θ``，压到 32° 会让
    ``tests/gui/test_correlation_labels.py::test_the_matrix_ticks_stay_far_enough_apart_to_be_read``
    那条判据在密档上失守。这条只钉「在哪条边上」。
    """
    view = _pane(qtbot, _result())
    axes = _matrix_axes(view)
    # 方阵实框要 ``apply_aspect`` 之后才是方的，否则量到的还是布局分配的那个长方框。
    axes.apply_aspect()
    renderer = view.canvas.get_renderer()

    square = axes.get_window_extent(renderer)
    printed = [text for text in axes.get_xticklabels() if text.get_text()]
    assert printed, "横轴一个名字都没写"
    below = {
        text.get_text(): round(square.y1 - text.get_window_extent(renderer).y0, 1)
        for text in printed
        if text.get_window_extent(renderer).y0 < square.y1 - 1.0
    }
    assert below == {}, f"这些列名还压在矩阵下方（低于方阵上边缘多少 px）：{below}"


def test_the_colour_key_lies_below_the_matrix_and_names_pearson_rho(qtbot) -> None:
    """色标横躺在矩阵下方，标题写「Pearson 相关系数 ρ」。

    两件事各自的理由。一是方位：矩阵是方阵（``aspect='equal'``），在半栏高的格子里它的宽度
    由高度定死，右侧本就空着一大片；色标竖在右边等于把那片空白再切一刀，而矩阵自己一点没变
    宽。躺到下面之后横向整条都归它，刻度 −1 / 0 / +1 也读得开。二是标题：「固定范围 [-1, 1]」
    说的是这条色标的量程，可读者站在这里要问的是「这颜色代表什么量」——答案是 Pearson ρ，
    量程 [-1, 1] 由两端刻度自己说。
    """
    view = _pane(qtbot, _result())
    from xrr_fitter.gui.plots.correlation import COLOUR_KEY_LABEL

    matrix = _matrix_axes(view)
    keys = [
        axes
        for axes in view.figure.axes
        if axes.get_label() == "<colorbar>" and axes.get_position().y1 <= matrix.get_position().y1
    ]
    assert len(keys) == 1, [axes.get_label() for axes in view.figure.axes]
    key = keys[0]
    box = key.get_position()
    assert box.width > box.height, f"色标 {box.width:.3f}×{box.height:.3f}（归一化），仍是竖的"
    assert box.y1 <= matrix.get_position().y0, "色标没落在矩阵下方"
    assert COLOUR_KEY_LABEL == "Pearson 相关系数 ρ"
    assert key.get_xlabel() == COLOUR_KEY_LABEL, key.get_xlabel()


def _colour_key(view):
    """那条横色标。"""
    return next(axes for axes in view.figure.axes if axes.get_label() == "<colorbar>")


def test_the_colour_key_spans_exactly_the_square_it_explains(qtbot) -> None:
    """色标的两端各自对齐方阵的两端——它解释的是这个方阵的颜色，不是整张画布的。

    矩阵是方阵（``aspect='equal'``）又靠左（``set_anchor("W")``），实宽由分配框的**高度**定死；
    而 ``figure.colorbar(ax=axes)`` 是按**分配框的整宽**横躺的，于是两者差着好几倍。实测
    1400×900 那一档：方阵 246.8px 宽，色标 723.5px——宽出 476.7px，左端更是伸到 x=53.3，落在
    方阵左边那片空无一物的边距上（方阵左端在 290.6）。屏上于是有一条比矩阵本身还醒目的长条，
    而读者没有任何线索把它和上方那个小方块对上；设计稿画的正是「色标与矩阵同宽」。

    收窄不能写成静态比例（``shrink=0.34``）：方阵占分配框多少是排版结果，1272×1060 那档是
    0.55、1400×900 只有 0.34，比例随画布变。所以这条只钉结果——两端对齐，怎么做到不管。
    """
    view = _pane(qtbot, _result())
    matrix = _matrix_axes(view)
    # 方阵实框要 ``apply_aspect`` 之后才是方的，否则量到的还是布局分配的那个长方框。
    matrix.apply_aspect()
    renderer = view.canvas.get_renderer()

    square = matrix.get_window_extent(renderer)
    bar = _colour_key(view).get_window_extent(renderer)
    assert bar.x0 == pytest.approx(square.x0, abs=1.0), (bar.x0, square.x0)
    assert bar.x1 == pytest.approx(square.x1, abs=1.0), (bar.x1, square.x1)


def test_the_colour_key_marks_only_its_ends_and_the_neutral_point(qtbot) -> None:
    """色标上只写 −1 / 0 / +1 三个刻度。

    收窄之后这不是取舍而是算术：``-1.00`` 这样的五字符标签一个约 30px，matplotlib 默认给出的
    九档（±1 / ±0.75 / ±0.5 / ±0.25 / 0）摊下来要 270px，而收到方阵实宽后色标只有 250px 上下
    ——九档必然互相压掉。

    三档也够了。色标要回答的是「这颜色是正还是负、大概多强」；具体是多少不用在这里读，每一格
    里都印着系数本身（见 ``test_every_cell_prints_the_coefficient_it_colours``）。正号照 ``+1``
    写出来，与格内 ``+0.55`` 那套带符号的写法同一口径：这张图上符号本身就是要读的东西。
    """
    view = _pane(qtbot, _result())

    labels = tuple(text.get_text() for text in _colour_key(view).get_xticklabels())
    assert labels == ("−1", "0", "+1"), labels


def test_the_colour_key_names_its_quantity_above_the_bar(qtbot) -> None:
    """「Pearson 相关系数 ρ」写在色带**上方**，刻度留在下方。

    设计稿把两者分在色带两侧（标题基线 y=352 而色带自 y=356 起、刻度在 y=380）。理由是读者
    走到这里的顺序：先要知道「这条尺量的是什么」，才轮到「这一头是多少」。标题压在刻度下面时
    顺序整个倒过来——眼睛先撞上三个数字，问「什么的 −1」还得往下再看一行。

    还有一件排版上的事：色标已经贴着矩阵下沿（``test_the_colour_key_lies_below_the_matrix_...``），
    上方那 4px 的空隙本来就闲着，标题挪上去不占新的高度；留在下方则要在刻度之后再排一行，整张
    卡跟着长高，而这张图和 Profile 那格是共享一栏高度的。
    """
    from xrr_fitter.gui.plots.correlation import COLOUR_KEY_LABEL

    view = _pane(qtbot, _result())
    key = _colour_key(view)
    renderer = view.canvas.get_renderer()

    assert key.get_xlabel() == COLOUR_KEY_LABEL, key.get_xlabel()
    bar = key.get_window_extent(renderer)
    label = key.xaxis.label.get_window_extent(renderer)
    assert label.y0 >= bar.y1 - 1.0, f"标题压在色带下方 {bar.y1 - label.y0:.1f}px"
    ticks = [text.get_window_extent(renderer) for text in key.get_xticklabels() if text.get_text()]
    assert ticks, "色标一个刻度都没写"
    assert max(box.y1 for box in ticks) <= bar.y0 + 1.0, "刻度跑到色带上方去了"


def _summary_panel(view):
    """矩阵旁那栏读数的 axes。

    它是矩阵的 ``inset_axes``——跟着方阵收缩后的那个框走，所以站得住右侧那片空白；代价是它不在
    ``figure.axes`` 里，而挂在父 axes 的 ``child_axes`` 上。
    """
    from xrr_fitter.gui.plots.correlation import SUMMARY_AXES_LABEL

    return next(axes for axes in _matrix_axes(view).child_axes if axes.get_label() == SUMMARY_AXES_LABEL)


def _summary_lines(view) -> list[str]:
    """矩阵旁那栏读数，自上而下，折行还原成整句。

    两件事解释一下取法。一是 ``\\n``：栏宽只有一百五十来像素，长句必须折行，而折在哪儿是排版
    细节、不是读数契约——实现每次绘制都按当帧的栏实宽重折（``_SummaryStack.draw``），同一句话在
    不同窗口尺寸下断在不同位置。所以这里把换行去掉再断言：折行一个字符也不增不减，去掉换行
    就是原句。

    二是顺序：这里按 ``panel.texts`` 的加入顺序取，而不按 y 坐标排。实现让每段都站在栏的左上角
    再用 ``offset_copy`` 往下推像素（行距不能按栏高的比例给，见
    ``tests/gui/test_correlation_labels.py`` 那条溢出测试），于是四段的 ``get_position()`` 全是
    ``(0.0, 1.0)``——按它排序的话 key 全相等，顺序只是碰巧靠 ``sorted`` 的稳定性保住了加入顺序。
    """
    return [text.get_text().replace("\n", "") for text in _summary_panel(view).texts]


def _summary_line(lines: list[str], caption: str) -> str:
    return next(line for line in lines if caption in line)


def test_the_panel_beside_the_matrix_names_the_strongest_pair(qtbot) -> None:
    """矩阵右边那三行直接报出最强的一对、它的系数、以及次强。

    n² 个数印在格子里解决了「这格是多少」，没解决「该看哪格」——读者仍要把 36 个数扫一遍
    再排个序。设计稿把排序的结果写成三行读数放在矩阵旁：最强那对是谁、系数多少、次强是谁。
    名字用矩阵刻度上的同一套短名，读者拿着这行回矩阵上就能找到那一格，不必再翻译一次。

    次强是 ``d·ox ↔ ρ·ox`` 的 +0.55，|ρ| 不到 0.6，所以它并不在 ``strong_correlations``
    里——这两行只能从矩阵自己按 |ρ| 排出来，读 ``strong_correlations`` 会漏掉次强。
    """
    view = _pane(qtbot, _result())
    lines = _summary_lines(view)

    assert "最强相关" in lines[0], lines
    assert "d·aSi ↔ ρ·aSi" in lines[0], lines[0]
    coefficient = _summary_line(lines, "系数 ρ")
    assert "-0.72" in coefficient, coefficient
    runner_up = _summary_line(lines, "次强")
    assert "d·ox ↔ ρ·ox" in runner_up, runner_up
    assert "+0.55" in runner_up, runner_up
    assert not any("component." in line for line in lines), lines


def test_the_panel_says_what_a_strong_pair_costs_the_reader(qtbot) -> None:
    """还有一句判读：这对强相关意味着 ±1σ 会低估真实不确定度。

    「−0.72」本身不告诉读者该怎么办。两个参数强相关时协方差派生的 ±1σ 是沿各自坐标轴切的，
    真实的联合不确定区是斜的椭圆，于是逐参数的 ±1σ 系统性偏窄——这正是同屏下半那张 Profile
    似然存在的理由。设计稿把这句因果写在矩阵旁，读者才知道下一步该看哪里。
    """
    view = _pane(qtbot, _result())
    lines = _summary_lines(view)
    note = next((line for line in lines if "±1σ" in line), "")

    assert "低估" in note, lines
    assert "参数剖面" in note, note


def test_the_readings_never_split_a_number_or_a_latin_word_across_lines(qtbot) -> None:
    """折行只能落在词与词之间——从数字或西文词中间切开会把读数读成另一个数。

    实测（1400×900 那一档，渲染核对图上直接读得到）：次强那段折成 ``次强 / d·ox ↔ ρ·ox +0.5 / 5``，
    判读句折成 ``…需结合 Pr / ofile 似然判读。``。前者不是「丑」——扫过去读到的是 +0.5，而真值是
    +0.55，末位那个 5 孤零零挂在下一行开头，读者没有理由把它接回上一行。后者把这句话指向的那张图
    的名字（``Profile`` 似然，同屏下半那一格的抬头）拆成两截，读者再也认不出它说的是哪张图。

    病根是 ``_wrap_to_width`` 逐字符累宽：中文长句整句没有空白，逐字符折是对的，但同一套规则套到
    西文和数字上就成了在词中间下刀。

    判据按段逐词核：某段的原文里有这个词，这段折出来的某一行就得完整含着它。不去检查「换行处两侧
    是什么字符」——读数里本来就有构图时插的真换行（``系数 ρ`` 与 ``-0.72`` 分两行是设计要的），那种
    换行两侧同样是窄字符，按字符判会把它一起判红。词表显式列出来也更好审计：这几个词正是别处那两条
    内容测试断言过的那些（短名、系数、判读句指向的图名）。

    这条不规定用什么手段做到，也不禁止硬切：某个词本身宽过一整行时除了硬切没有别的走法，只是这四段
    里没有那样的词。
    """
    # 这四段里不能被折断的词：矩阵刻度上的短名、两个系数、以及判读句指向的那张图的名字。
    unbreakable = ("d·ox", "ρ·ox", "d·aSi", "ρ·aSi", "-0.72", "+0.55", "±1σ", "Profile")

    view = _pane(qtbot, _result(), SHORT_PANE_INCHES)
    panel = _summary_panel(view)
    assert len(panel.texts) == 4, [text.get_text() for text in panel.texts]

    split = []
    for text in panel.texts:
        rows = text.get_text().split("\n")
        whole = "".join(rows)
        split += [word for word in unbreakable if word in whole and not any(word in row for row in rows)]
    assert split == [], f"被折行切断的词：{split}，实际排布 {[t.get_text() for t in panel.texts]}"


def test_the_readings_all_fit_the_column_at_the_designed_scale(qtbot) -> None:
    """1400×900 那一档，四段读数整段都在栏内——末尾那句判读不该被栏底裁掉。

    ``tests/gui/test_correlation_labels.py::test_the_summary_column_keeps_every_reading_inside_its_own_column``
    量的是 17 参数那份替身，判读句只有五行；设计稿画的这 6 个参数里最强的一对是厚度与密度，
    ``_correlation_hint`` 于是多写一句「薄层的电子密度与厚度难以同时唯一确定」，同一段折成七行、
    高 126.9px。四段内容合起来 253.8px，栏高 261.5px——装得下，可实测跨了 286.4px 溢出 25.1px，
    多出来的 32.6px 全是三个段间隙（``SUMMARY_BLOCK_GAP`` 按行高的固定倍数给）。被栏底裁掉的正是
    「需结合 参数剖面判读」这半句：读者要带走的下一步动作没了。

    间隙是这里唯一可让的东西——内容不能删（那半句物理解释是这对参数为什么纠缠的原因），字号不能
    再小（9pt 已是全图最小档）。所以判据只说「四段都在栏内」，怎么腾出这 25px 不管。
    """
    view = _pane(qtbot, _result(), SHORT_PANE_INCHES)
    panel = _summary_panel(view)
    renderer = view.canvas.get_renderer()
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


def test_the_column_stands_down_when_it_gets_too_narrow_to_read(qtbot) -> None:
    """栏窄到一行短名对都排不下时整栏让位，宽回来了再补上。

    ``MINIMUM_SUMMARY_PANEL_PIXELS`` 声明的就是这条：``d·aSi ↔ ρ·aSi`` 在 9pt 下约 78px，栏比
    这个下限还窄时读到的不是读数而是每段最左边一两个字（折行按下限折的，其余被栏框裁掉）。

    但那道门此前是死的：``_draw_correlation_summary`` 在构图时取 ``panel.get_window_extent()``，
    读到的是上一次排版留下的框，中栏压扁那一档栏实宽 62.8px 却照样判「够宽」。这和读数溢出、
    与色标宽度是同一个病根——「画多宽、画不画」都要等 ``constrained_layout`` 排完版才有依据。

    往返两档一起量：压扁时收掉，拉回 1400×900 那档要重新出现。只判「不画」不判「补画」的话，
    把门改成一次性的 ``remove()`` 也能绿，而那样拖一次窗口这一栏就永久消失了。
    """
    from xrr_fitter.gui.plots.correlation import MINIMUM_SUMMARY_PANEL_PIXELS

    view = _pane(qtbot, _result(), SQUEEZED_PANE_INCHES)
    panel = _summary_panel(view)
    renderer = view.canvas.get_renderer()
    narrow = panel.get_window_extent(renderer).width
    assert narrow < MINIMUM_SUMMARY_PANEL_PIXELS, f"这一档栏宽 {narrow:.1f}px，量不到「太窄」这件事"
    showing = [text.get_text().replace("\n", " ") for text in panel.texts if text.get_visible()]
    assert showing == [], f"栏只有 {narrow:.1f}px 宽却仍画着这些段：{showing}"

    view.figure.set_size_inches(*SHORT_PANE_INCHES)
    view.canvas.draw()
    renderer = view.canvas.get_renderer()
    wide = panel.get_window_extent(renderer).width
    assert wide >= MINIMUM_SUMMARY_PANEL_PIXELS, wide
    hidden = [text.get_text().replace("\n", " ") for text in panel.texts if not text.get_visible()]
    assert hidden == [], f"栏宽回到 {wide:.1f}px 之后这些段没补画：{hidden}"


def test_the_matrix_states_the_threshold_in_its_heading_not_its_readings(qtbot) -> None:
    """「|ρ|≥0.6 视为强相关」写在卡抬头上，而不是占掉读数栏的一段。

    不写出来的话，加粗的格子和那几行读数都是没有出处的断言：读者看得见「这几格粗」，不知道
    粗在哪一条线之后开始。但它也不该排在读数之间——它讲的是**整张图怎么读**，与「最强那对是
    d·aSi ↔ ρ·aSi」这类逐条读数不是一个层级；夹在中间时读者要在两种句子之间来回切换语境。
    设计稿把它放在卡抬头右端当副题（``.ph`` 是 ``justify-content:space-between``，主标题左、
    副题右、同一行），也就是判读规则与被判读的图同屏、又不与读数抢位置。

    搬走之后读数栏正好剩四段，和设计稿那栏一样（最强相关 / 系数 / 次强 / 一句判读）。这不只是
    数字对上了：``tests/gui/test_correlation_labels.py::test_the_summary_column_keeps_every_reading_inside_its_own_column``
    钉着「240px 的栏码不下五段（要 268px）、被牺牲的只能是末尾那句判读」——少一段就少约 54px，
    正是那栏一直缺的那点高度。
    """
    from xrr_fitter.gui.plots.correlation import CORRELATION_TITLE, SUMMARY_PANEL_BOUNDS

    view = _pane(qtbot, _result())
    axes = _matrix_axes(view)
    axes.apply_aspect()
    renderer = view.canvas.get_renderer()

    subtitle = axes.get_title(loc="right")
    assert "Pearson" in subtitle and "0.6" in subtitle, subtitle
    lines = _summary_lines(view)
    assert len(lines) == 4, lines
    assert not any("视为强相关" in line for line in lines), lines

    # 抬头那一行横跨方阵**和**读数栏——设计稿的 ``.ph`` 是整张卡的宽度，而这张图里「卡」就是
    # 这两块。副题只对齐方阵右端的话，窄档（方阵 174px）上它和主标题必然压在一起：两段 9pt
    # 与标题字号的文字合起来要 245px 上下。
    left = _title_text(axes, CORRELATION_TITLE).get_window_extent(renderer)
    right = _title_text(axes, subtitle).get_window_extent(renderer)
    assert left.x1 <= right.x0 + 1.0, f"主标题与副题重叠 {left.x1 - right.x0:.1f}px"
    column_end = axes.transAxes.transform((SUMMARY_PANEL_BOUNDS[0] + SUMMARY_PANEL_BOUNDS[2], 0.0))[0]
    assert right.x1 == pytest.approx(column_end, abs=2.0), (right.x1, column_end)


def test_the_matrix_is_titled_the_way_the_design_names_the_card(qtbot) -> None:
    """抬头左端写「参数相关矩阵」。

    「相关矩阵」没说清是谁之间的相关——同屏还有残差自相关、SLD 可信带这些同样带「相关」
    字样的证据。设计稿的卡抬头点明是参数之间。

    靠左而不居中：抬头那一行右端归阈值副题（见
    ``test_the_matrix_states_the_threshold_in_its_heading_not_its_readings``），主标题居中会
    把两者推到一起。设计稿的 ``.ph`` 也正是这个分法。
    """
    from xrr_fitter.gui.plots.correlation import CORRELATION_TITLE

    view = _pane(qtbot, _result())

    assert CORRELATION_TITLE == "参数相关矩阵"
    assert _matrix_axes(view).get_title(loc="left") == CORRELATION_TITLE
