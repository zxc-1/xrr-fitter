"""帧⑤ 那张 参数剖面要能读出「哪个参数被约束住了」，而不只读出三条上扬的曲线。

设计稿的 Profile 卡：纵轴从 0 起算、写明「相对最优」，横轴以 σ 为单位、三条曲线共享同一个
0 点，于是「对称 / 偏斜 / 平坦」一眼分得出；图例每项除短名外还写着中文量名与一句形状判读
（``厚度 d·aSi（对称）``）。实现此前把纵轴画成目标函数的绝对值、横轴按各参数自己的量程
独立归一化到 [0, 1]——三条曲线于是各有一个 0 点、各有一个 1 点，最低点落在横轴哪儿取决于
扫描点怎么排，对称性读不出来；图例只有 ``d·aSi``、``ρ·aSi``、``σ·aSi`` 三个符号。

有意偏离设计稿一处，写在这里免得后人当成漏改：设计稿把纵轴写作 ``Δχ²（相对最优）``、参考线
写作 ``Δχ²=1 → ±1σ``。本项目的目标函数是 ``robust_log_cost``，它不是 χ²，所以这个 Δ 没有
χ² 的单位，「Δ=1 对应 ±1σ」这个推论在这里不成立。纵轴因此只写「相对最优的目标增量」，参考线
只标本次扫描自己用来判定闭合的那个阈值——``analysis/profiles.py`` 的 ``_closure_threshold``
明说它「既是判定 flags 的比较值、也是发布在 profile 上的数」，图上标的必须是同一个数。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import final_fit_result

import xrr_fitter.api as api

# 设计稿帧⑤ 图例里的那三条曲线：同一层的厚度、密度、粗糙度。短名与量名都从 ``display_name``
# 派生，所以这里连 display_name 一起给——它同时是矩阵刻度上的 ``d·aSi`` 的出处。
PARAMETERS = (
    ("component.1.thickness_a", "aSi 厚度"),
    ("component.1.density_scale", "aSi 相对密度"),
    ("component.1.roughness_a", "aSi 入射侧粗糙度"),
)

# 三个参数的 ±1σ。``parameter_sigma`` 与 ``correlation_names`` 同序同长，这是模型侧的校验
# 条件（``model/analysis.py`` 的 ``_parameter_sigma``），横轴换算成 σ 单位靠的就是这条对齐。
SIGMA = (2.0, 0.05, 1.5)

# 扫描中心的参数值。数值本身无关紧要，重要的是三条曲线的中心落在各自量程的不同位置——横轴
# 若还按各自量程归一化，三个 0 点就会落在 0.5 附近但并不重合。
CENTRES = (30.0, 1.0, 8.0)

# 扫描中心的目标值，以及判定一侧闭合所需的增量。阈值发布在 profile 上时是绝对值
# ``center_objective + delta``，读图要的是相对量，于是图上应当出现 ``DELTA`` 而不是 2.0。
CENTRE_OBJECTIVE = 1.0
DELTA = 1.0

# 采样点：−3σ 到 +3σ，与设计稿横轴的六个刻度同一个跨度。
OFFSETS = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

# 帧⑤ 中栏那张双图在 1272×1060 主窗口里实得的画布，与 ``test_correlation_matrix_readings``
# 同一处实测。
PANE_INCHES = (6.66, 8.98)

# 三种曲线形状各自的目标增量，按 σ 单位的偏移算。
# 对称：Δ=o²，两侧都在 ±1σ 处跨过 DELTA。
# 偏斜：左支陡（Δ=2o²，0.71σ 处跨过）右支缓（Δ=0.5o²，1.41σ 处跨过），半宽比 1:2。
# 平坦：Δ=0.04o²，扫到 ±3σ 峰值也只有 0.36，两侧都没跨过 DELTA——这就是「弱约束」。
INCREMENTS = (
    OFFSETS**2,
    np.where(OFFSETS < 0.0, 2.0, 0.5) * OFFSETS**2,
    0.04 * OFFSETS**2,
)
CLOSED = (True, True, False)


def _definition(name: str, display_name: str) -> api.ParameterDefinition:
    return api.ParameterDefinition(
        name=name,
        display_name=display_name,
        unit="",
        category="layer",
        initial=1.0,
        lower=0.0,
        upper=2.0,
        transform="linear",
        locked=False,
    )


def _profiles() -> tuple[api.ParameterProfile, ...]:
    """设计稿那三条曲线：对称、偏斜、平坦各一条，中心与阈值三条共用。"""
    return tuple(
        api.ParameterProfile(
            name=name,
            values=CENTRES[index] + OFFSETS * SIGMA[index],
            objectives=CENTRE_OBJECTIVE + INCREMENTS[index],
            lower_closed=CLOSED[index],
            upper_closed=CLOSED[index],
            objective_threshold=CENTRE_OBJECTIVE + DELTA,
        )
        for index, (name, _) in enumerate(PARAMETERS)
    )


def _report(*, with_sigma: bool) -> api.UncertaintyReport:
    """一份只带剖面证据的报告；``with_sigma`` 控制横轴能不能换算成 σ。"""
    names = tuple(name for name, _ in PARAMETERS)
    return api.UncertaintyReport(
        correlation_names=names,
        correlation_matrix=np.eye(len(names)),
        profiles=_profiles(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id="candidate-a",
        parameter_sigma=np.array(SIGMA) if with_sigma else None,
    )


def _result(*, with_sigma: bool = True) -> api.FitResult:
    return replace(
        final_fit_result(),
        parameter_definitions=tuple(_definition(name, display) for name, display in PARAMETERS),
        uncertainty=_report(with_sigma=with_sigma),
    )


def _pane(qtbot, result: api.FitResult):
    """帧⑤ 中栏那张双图，画布收到屏上实测的那块预算。"""
    from matplotlib.figure import Figure

    from xrr_fitter.gui.plots import diagnostics
    from xrr_fitter.gui.plots.diagnostics import DiagnosticCanvas, DiagnosticView
    from xrr_fitter.gui.plots.sld import draw_uncertainty

    figure = Figure(figsize=PANE_INCHES, layout="constrained")
    # 形状取自实现自己那一处声明，与 ``test_correlation_matrix_readings`` 同一条理由：手抄一遍
    # 就会分叉。
    view = DiagnosticView(figure, DiagnosticCanvas(figure), diagnostics._axes(figure, "uncertainty"))
    qtbot.addWidget(view.canvas)
    draw_uncertainty(view, result, "candidate-a")
    view.canvas.draw()
    return view


def _profile_axes(view):
    """剖面那一格：按标题找，而不是按 ``figure.axes`` 的下标。

    这张 figure 上的 axes 数目会随色标与读数栏变动，下标式的定位每加一格就错一次。
    """
    from xrr_fitter.gui.plots.sld import PROFILE_TITLE

    return next(axes for axes in view.figure.axes if axes.get_title() == PROFILE_TITLE)


def _curves(axes) -> dict[str, object]:
    """按图例项取那三条扫描曲线。

    按点数认，而不是按 label 认：阈值线也是一条 ``Line2D``、也带 label，而它只有两个点。
    label 的措辞是这些测试要断言的内容之一，拿它当筛选条件就成了自证。
    """
    return {line.get_label(): line for line in axes.lines if len(line.get_xdata()) == OFFSETS.size}


def test_the_curves_all_start_from_zero_at_their_own_best(qtbot) -> None:
    """纵轴是相对最优的增量：三条曲线的最低点都落在 0，而不是各自的目标绝对值。

    绝对目标值读不出「代价涨了多少」——``robust_log_cost`` 的量级由数据点数和权重定，1.0 与
    1.36 之间那 0.36 是大还是小，光看纵轴无从判断。减掉各自的最优之后纵轴上的数就是「离开
    最优点付出的代价」，而这正是判定区间的那把尺子：阈值本身就是以 ``center_objective`` 为
    起点加一个增量定义的。
    """
    view = _pane(qtbot, _result())
    axes = _profile_axes(view)
    curves = _curves(axes)

    assert len(curves) == len(PARAMETERS), sorted(curves)
    for label, line in curves.items():
        assert float(np.min(line.get_ydata())) == 0.0, label
    assert float(axes.get_ylim()[0]) <= 0.0


def test_the_vertical_axis_says_relative_to_best_and_does_not_claim_chi_square(qtbot) -> None:
    """纵轴写「相对最优」，且不含 ``χ²``。

    前半句是读数的前提：不写出基准，0 这个数就没有意义——读者不知道是「目标值等于 0」还是
    「比最优高 0」。后半句是本文件头写的那处有意偏离：设计稿写 ``Δχ²（相对最优）``，而本项目
    的目标函数不是 χ²，标上去会让读者用 χ² 的分位数去读这条纵轴（Δ=1 → 68%），得出的区间
    没有出处。
    """
    view = _pane(qtbot, _result())
    label = _profile_axes(view).get_ylabel()

    assert "相对最优" in label, label
    assert "χ" not in label and "chi" not in label.lower(), label


def test_the_horizontal_axis_counts_sigma_so_the_curves_share_one_zero(qtbot) -> None:
    """横轴以 σ 为单位：三条曲线的最低点都落在 0，跨度都是 ±3σ。

    独立归一化到 [0, 1] 时三条曲线各有各的 0 点与 1 点，最低点落在横轴哪儿取决于扫描点怎么
    排；而设计稿这张图要回答的正是「哪条窄、哪条歪、哪条平」——那是三条曲线摆在同一把尺子上
    才有的比较。σ 是现成的那把尺子：它逐参数带着自己的量纲，除完之后 ``d·aSi`` 的 2 Å 与
    ``ρ·aSi`` 的 0.05 都读作「一个 σ」。
    """
    view = _pane(qtbot, _result())
    axes = _profile_axes(view)

    for label, line in _curves(axes).items():
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        # 0 点用逐位相等：中心那一点是同一个数相减，浮点上就该是精确的 0，而三条曲线共享这一个
        # 0 点正是这张图能横向比较的前提。两端只到显示精度——``1.0 - 3 * 0.05`` 再除 0.05 得
        # −3.0000000000000004，那是替身里 σ 的十进制表示，不是横轴换算的偏差。
        assert float(x[int(np.argmin(y))]) == 0.0, label
        assert float(np.min(x)) == pytest.approx(-3.0) and float(np.max(x)) == pytest.approx(3.0), (label, x)
    assert "σ" in axes.get_xlabel(), axes.get_xlabel()


def test_a_report_without_sigma_keeps_the_normalized_coordinate(qtbot) -> None:
    """没有 ``parameter_sigma`` 就退回各自归一化的横轴，并照实写在轴标签上。

    σ 出自协方差，而协方差不是每条路径都算得出来（只跑了剖面扫描、或者 Hessian 没有正定的
    那一支）。缺了它就没有共同的尺子，此时按各自量程归一化至少让三条曲线并排画得下——但轴
    标签必须跟着换，否则读者会拿 σ 的读法去读一个不是 σ 的横轴。
    """
    view = _pane(qtbot, _result(with_sigma=False))
    axes = _profile_axes(view)

    for label, line in _curves(axes).items():
        x = np.asarray(line.get_xdata(), dtype=float)
        assert float(np.min(x)) == 0.0 and float(np.max(x)) == 1.0, (label, x)
    assert "独立归一化" in axes.get_xlabel(), axes.get_xlabel()
    assert "σ" not in axes.get_xlabel(), axes.get_xlabel()


def test_the_legend_names_the_quantity_and_the_shape_of_each_curve(qtbot) -> None:
    """图例每项是「量名 短名（形状）」，三条各得一句判读。

    两件事各自的理由。一是量名：``ρ·aSi`` 的 ``ρ`` 只对读过矩阵刻度的人成立，而图例常常是
    这条曲线第一次出现的地方；补上中文量名之后这一行自己读得懂。量名取参数表里那个
    ``display_name`` 的尾段（「aSi 相对密度」→「相对密度」），不另造一套——设计稿写的是
    「密度」，但同一个参数在表里叫「相对密度」，图上换个叫法读者就得先学一遍对照关系。

    二是形状：设计稿在括号里写着「对称 / 偏斜 / 平坦→弱约束」，这是这张图真正要读者带走的
    东西——曲线本身只是三条上扬的线，谁窄谁歪要靠比较。判读出自扫描自己发布的读数（两侧是否
    闭合、闭合处离中心多远），不是目测。
    """
    view = _pane(qtbot, _result())
    legend = _profile_axes(view).get_legend()

    entries = [text.get_text() for text in legend.get_texts()]
    assert entries[: len(PARAMETERS)] == [
        "厚度 d·aSi（对称）",
        "相对密度 ρ·aSi（偏斜）",
        "入射侧粗糙度 σ·aSi（平坦→弱约束）",
    ], entries


def test_a_curve_with_one_open_side_is_not_called_symmetric(qtbot) -> None:
    """一侧没闭合的曲线，图例点明是哪一侧没闭合。

    「平坦」说的是曲线抬不起来，「单侧未闭合」说的是扫描范围不够或者边界挡住了——两者都让
    ±1σ 不可信，但下一步不一样：前者要加约束或换参数化，后者放宽扫描范围就够了。把二者都
    写成「平坦」会把读者引到错的那一步；写成「对称」更糟，那一侧的区间根本没有右端。
    """
    result = _result()
    report = result.uncertainty
    profiles = (replace(report.profiles[0], upper_closed=False), *report.profiles[1:])
    view = _pane(qtbot, replace(result, uncertainty=replace(report, profiles=profiles)))

    entries = [text.get_text() for text in _profile_axes(view).get_legend().get_texts()]
    assert entries[0] == "厚度 d·aSi（上侧未闭合）", entries


def test_the_threshold_line_marks_the_increment_that_closes_a_side(qtbot) -> None:
    """参考线画在 Δ=1.00 上，也就是扫描判定闭合时用的那个增量。

    profile 上发布的阈值是绝对目标值（``center_objective + delta``）；纵轴一旦改成相对量，
    照搬那个绝对值会把线画到画面外。减掉同一个基准之后线落在 Δ=delta 上，而这仍是扫描做过的
    那次比较——``_closure_threshold`` 的 docstring 要的就是「图上标的数与定 flags 的数是同
    一个」。线上标着数值而不只是一句「阈值」：读者要判「这条曲线离闭合还差多少」。
    """
    view = _pane(qtbot, _result())
    axes = _profile_axes(view)

    lines = [line for line in axes.lines if len(line.get_xdata()) == 2]
    assert len(lines) == 1, [line.get_label() for line in axes.lines]
    assert float(np.asarray(lines[0].get_ydata(), dtype=float)[0]) == DELTA
    assert lines[0].get_label() == f"区间闭合阈值 Δ={DELTA:.2f}", lines[0].get_label()


def test_the_vertical_axis_frames_the_threshold_instead_of_the_far_tail(qtbot) -> None:
    """纵轴按阈值取景，不被曲线远端撑开。

    这不是「挤一点」：扫描范围本身按 σ 的倍数定，对称抛物线扫到 ±3σ 时端点的增量就是 9 倍
    delta，偏斜那条的陡侧是 18 倍。按全部数据取景的话阈值线落在画面底部约 1/19 处，而这张图
    要读的三件事——哪条窄、哪条歪、哪条离闭合还差多少——全发生在 0 到 Δ 那一段里。设计稿的
    纵轴因此只画到 0/1/2/3/4 五格（阈值的四倍），三条曲线超出顶边的部分直接截掉。

    裁的是取景而不是数据：远端那些点仍在线上（悬停读得到、后续导出用得着），只是不出现在
    这一屏里。
    """
    view = _pane(qtbot, _result())
    axes = _profile_axes(view)
    lower, upper = (float(value) for value in axes.get_ylim())
    tail = max(float(np.max(line.get_ydata())) for line in _curves(axes).values())

    assert tail == pytest.approx(18.0), tail
    assert upper < tail / 2.0, (upper, tail)
    # 0..Δ 那一段至少占住画面纵向的五分之一，否则三条曲线在阈值附近仍然叠成一团。
    assert (DELTA - lower) / (upper - lower) >= 0.2, (lower, upper)


def test_a_scan_that_never_reaches_the_threshold_still_shows_its_whole_curve(qtbot) -> None:
    """三条都抬不到阈值时，纵轴贴着曲线收回来，而不是空着上面四分之三。

    上一条的取景基准是阈值，照字面执行会让「全是弱约束」这一屏变成三条压在底部的平线加一大
    片空白——而这一屏恰恰是要看清「到底抬起来多少」的时候。阈值是上限而非固定值：曲线不够高
    就按曲线取景，但阈值线本身必须留在视野里，否则读者失去判断「差多远」的那把尺子。
    """
    result = _result()
    report = result.uncertainty
    flat = tuple(
        replace(profile, objectives=CENTRE_OBJECTIVE + INCREMENTS[2], lower_closed=False, upper_closed=False)
        for profile in report.profiles
    )
    view = _pane(qtbot, replace(result, uncertainty=replace(report, profiles=flat)))
    axes = _profile_axes(view)
    lower, upper = (float(value) for value in axes.get_ylim())

    assert upper > DELTA, (upper, DELTA)
    assert upper < 2.0 * DELTA, (upper, DELTA)
    assert lower <= 0.0, lower


def test_the_card_is_titled_the_way_the_design_names_it(qtbot) -> None:
    """抬头写「参数剖面」，且空态与有数据时是同一句。

    抬头此前在实现里写了两遍（有数据一处、``_unavailable`` 一处），两处拼写分叉时读者会以为
    换了张图。这条断言把它锁在一个常量上。
    """
    from xrr_fitter.gui.plots.sld import PROFILE_TITLE

    view = _pane(qtbot, _result())

    assert PROFILE_TITLE == "参数剖面 · 逐参数扫描"
    assert _profile_axes(view).get_title() == PROFILE_TITLE
