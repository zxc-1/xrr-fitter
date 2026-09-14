"""帧⑤ 中栏那两张图排成什么形状，以及重建它们时碰不碰那只冻结的 view。

设计稿帧⑤（HTML 870-983）的中栏是两张上下叠的 ``plotcard``：相关矩阵一张、参数剖面
一张，各占满整个栏宽。实现把它们排成 1×2 并肩——矩阵是方阵，并肩之后两半各只剩 174px 宽，
刻度、图例和区间全挤在这个宽度里。形状是这一屏所有「放不下」的根因，所以单独钉住。

另一条钉的是重建路径：``DiagnosticView`` 是 ``frozen=True``，而重建分支里写着
``view.axes = primary[0]``。这一支在现有流程里从不进入（每次重置都把 figure 收回两个
axes），1364 条 GUI 测试因此全绿——它是个潜伏的 ``FrozenInstanceError``，不是死代码。
"""

from __future__ import annotations

from tests.gui.test_correlation_labels import PANE_INCHES, _profiled_result


def _view(qtbot, axes_shape: str = "tab"):
    """帧⑤ 那张双图，形状取自实现自己那一处声明。

    ``axes_shape="tab"`` 走 ``diagnostics._axes("uncertainty")``——tab 建 view 时用的正是它，
    所以形状永远跟着实现走。测试自己抄一遍 ``subplots`` 就会和实现分叉，而分叉的那一次恰好
    就是重建分支测不到的原因。``"single"`` 给一只单 axes 的 figure，逼实现走重建那一支。
    """
    from matplotlib.figure import Figure

    from xrr_fitter.gui.plots import diagnostics
    from xrr_fitter.gui.plots.diagnostics import DiagnosticCanvas, DiagnosticView

    figure = Figure(figsize=PANE_INCHES, layout="constrained")
    axes = diagnostics._axes(figure, "uncertainty") if axes_shape == "tab" else figure.subplots()
    view = DiagnosticView(figure, DiagnosticCanvas(figure), axes)
    qtbot.addWidget(view.canvas)
    return view


def _drawn(qtbot, axes_shape: str = "tab"):
    from xrr_fitter.gui.plots.sld import draw_uncertainty

    view = _view(qtbot, axes_shape)
    draw_uncertainty(view, _profiled_result(), "candidate-a")
    view.canvas.draw()
    return view


def _plot_boxes(view):
    """两张图各自的绘图区，按 figure 归一化坐标；colorbar 排在末尾所以取前两只。"""
    matrix, profile = view.figure.axes[0], view.figure.axes[1]
    return matrix.get_position(), profile.get_position()


def test_the_uncertainty_pane_stacks_its_two_plots_top_to_bottom(qtbot):
    """设计稿把矩阵和 Profile 排成两张各占满栏宽的卡，不是并肩的两个半栏。"""
    view = _drawn(qtbot)

    geometry = view.figure.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert geometry == (2, 1), f"两张图仍排成 {geometry[0]}×{geometry[1]}，设计稿是上下 2×1"


def test_the_stacked_plots_do_not_share_a_row(qtbot):
    """形状之外再钉实际落点：矩阵整个在上，两图横向占同一段宽度。

    只断言 gridspec 会漏掉「声明 2×1 却被别处改回并排」的情形，所以另外量一次真实
    ``get_position()``。矩阵是 ``aspect="equal"`` 且带 colorbar，横向不会和 Profile 严格
    等宽，因此只要求重叠够多——并排时两者横向重叠为 0，区分得开。
    """
    view = _drawn(qtbot)
    matrix, profile = _plot_boxes(view)

    assert matrix.y0 >= profile.y1, f"矩阵底 {matrix.y0:.3f} 压在 Profile 顶 {profile.y1:.3f} 之下"
    overlap = min(matrix.x1, profile.x1) - max(matrix.x0, profile.x0)
    assert overlap > 0.3, f"两图横向只重叠 {overlap:.3f}，说明还是左右并排"


def test_the_pane_rebuilds_its_halves_without_touching_the_frozen_view(qtbot):
    """从单 axes 的 figure 起手，逼实现走重建那一支。

    ``DiagnosticView`` 冻结，重建时不能替换 ``view.axes``——只能把它自己挪进新格子，
    这也是同文件 ``_reset_page`` 已经在用的路子。
    """
    view = _drawn(qtbot, axes_shape="single")

    assert view.axes in view.figure.axes, "重建把主 axes 丢在了 figure 之外"
    assert view.figure.axes[0] is view.axes, "重建后主 axes 不再是上面那张矩阵"
    assert view.axes.get_subplotspec().get_gridspec().get_geometry() == (2, 1)


def test_the_profile_plot_leaves_the_bootstrap_intervals_to_the_evidence_text(qtbot):
    """图内不再逐条写 bootstrap 区间——那些行归证据文本框。

    区间是「每个自由参数一条」：这份样品 17 个自由参数就是 17 行，叠在绘图区左上角必然越出
    右边界，也必然和图例抢同一块地方。设计稿卡二（HTML 940-983）图内只有三条曲线、一条
    ``Δχ²=1`` 参考线和坐标轴，区间不在图上。所以这里钉两件事：图上没有，且证据文本里有。
    """
    from xrr_fitter.gui.results.uncertainty import _report_lines

    view = _drawn(qtbot)
    profile = view.figure.axes[1]
    bracketed = [text.get_text() for text in profile.texts if "[" in text.get_text()]
    assert not bracketed, f"绘图区里还写着区间：{bracketed}"

    report = _profiled_result().uncertainty
    line = next(item for item in _report_lines(report) if item.startswith("bootstrap 区间："))
    assert not line.endswith("不可用"), "证据文本没接上区间，移出图内就等于把信息丢了"
    for name, *_bounds in report.bootstrap_intervals:
        assert name in line, f"证据文本漏了 {name} 的区间"
