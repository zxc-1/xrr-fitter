"""Candidate-owned SLD and uncertainty diagnostic rendering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, NullLocator

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.correlation import CORRELATION_TITLE, _draw_correlation
from xrr_fitter.gui.plots.diagnostics import (
    DiagnosticView,
    apply_figure_palette,
    candidate_is_inspection_only,
    draw_empty,
    finish_view,
)
from xrr_fitter.gui.plots.parameter_labels import quantity_names, short_labels
from xrr_fitter.gui.structure import naming, stack

# A layer thinner than this reads as roughness, not a film; dragging an interface
# past its predecessor is clamped here so a gesture can never invert the stack.
MIN_LAYER_THICKNESS_A = 2.0

# ``LayerSpec`` rejects a density scale that is not finite and positive, so a
# level dragged down to or past the axis floor is clamped to a vanishing film
# rather than an invalid one the commit would refuse.
MIN_DENSITY_SCALE = 1e-6

# X 射线的 SLD 落在 10⁻⁶ Å⁻² 这个量级，按 Å⁻² 画出来刻度就成了 1.9e-05，读者得数六位
# 小数才知道自己在看什么。整张图统一乘这个数并把单位写进纵轴标注，刻度才是文献里那套
# 「Si 约 20、SiO2 约 19」的读法。
#
# 拖拽不受影响：手柄的高度都是从画好的数组里取的，落回结构时走的是 to/from 的比值
# （见 ``sld_drag`` 的 level 提交），整条线同乘一个数，比值不变。
SLD_DISPLAY_SCALE = 1e6

# 两条轴的标注。深度必须说清从哪头算起——同一张图既可以从表面往基底走也可以反过来，
# 而「先氧化层、后薄膜、最后基底」这个读法就建立在方向上。
DEPTH_AXIS_LABEL = "深度 z（nm，从表面向基底）"
SLD_AXIS_LABEL = "散射长度密度 SLD（10⁻⁶ Å⁻²）"

# 图例里那条当前剖面的名字。「实部」是把复折射率的记法搬到了图例上；读者要认的是
# 「这条就是我这套结构现在的剖面」，虚部另有一行说自己是虚部。
CANDIDATE_REAL_LABEL = "当前 SLD 剖面"

# 拟合结果之外再叠一条结构标称时，那是第二条曲线（拿手上这套声明去比对拟合出来的剖面），
# 所以留自己的名字；还没有候选时它就是当前剖面本身，名字随之换成上面那个。
NOMINAL_OVERLAY_LABEL = "结构标称 实部"

INNER_BAND_LABEL = "16–84% 不确定带"

# 手柄按 ``_interface_{i}`` 这类下划线开头的名字画，既躲开图例又给拖拽控制器一个稳定的
# 抓取名（``SldDragMixin._sld_handle`` 按这个名字精确匹配），所以它在图例里只能由别人代说。
INTERFACE_KEY_LABEL = "可拖拽界面"

# 卡片抬头下面那排色标（``sld_state.SLD_LEGEND_ENTRIES``）已经说了这三样。轴内再画一遍
# 就是同一张卡里两份图例，而设计稿帧③ 的图区里根本没有图例框——那几条 pyqtgraph 面板
# （log/raw/qz4/residual）同样只有卡片色标、没有内嵌图例。轴内那份因此只负责色标说不到
# 的记号：虚部、对比候选、以及叠在拟合结果上的结构标称。
CARD_KEY_LABELS = (CANDIDATE_REAL_LABEL, INNER_BAND_LABEL, INTERFACE_KEY_LABEL)

# 深度轴按介质分段着色：半无限的空气与基底取中性色，真正的层取自己那一号色。层色与层
# 列表那一行的色块同源（``stack.component_fill``），两处一对上，读者才不必回头数层序。
# 两档 alpha 是设计稿 rect 上的 opacity：色带只是背景，压不住上面那条曲线。
REGION_MEDIUM_ALPHA = 0.05
REGION_LAYER_ALPHA = 0.10

# 半无限的两段没有边界，但 ``axvspan`` 要一个数。取总厚度的这个比例（且不小于 10nm）
# 铺到视野之外，轴自己会把多出来的裁掉，看上去正是设计稿那两条顶到画框边的 rect；写
# 成有限值是因为 ``axvspan`` 会请求一次 x 自动缩放，无穷大会把整条轴带走。
REGION_OVERHANG_FRACTION = 0.25
REGION_OVERHANG_FLOOR_NM = 10.0

# 分区标注压在图区顶边下面一点。设计稿把这行字的基线放在 y=29，而图区是 y=16…196，
# 换成轴坐标就是顶边下方约 2.6%。
REGION_LABEL_Y = 0.974

# 半无限那两段的说法。空气与层列表抬头（``editor.py`` 的 fronting 行）同一个词；基底那
# 行在自己的名字后面补一个角色名，写出来是「c-Si 基底」。
FRONTING_REGION_LABEL = "空气"
BACKING_REGION_ROLE = "基底"

# 界面手柄：一根从曲线落到轴底的虚线，线头顶一个白心圆环。蓝是这套设计里「可以动」的
# 意思，所以颜色跟着 ``theme.current_accent()`` 走，而不是数据色里的任何一支。alpha 单独
# 给线，不给整个 artist——``Artist.set_alpha`` 会把圆点一起调淡，而设计稿的点是实心的。
HANDLE_ALPHA = 0.6
HANDLE_DASH_PATTERN = (3, 3)
HANDLE_LINEWIDTH = 1.0
HANDLE_DOT_SIZE = 7.2
HANDLE_DOT_FACE = "#FFFFFF"
HANDLE_DOT_EDGE_WIDTH = 2.0

# 表面（深度 0）也画这一笔，但它不是层的边界、不能拖。标签必须落在 ``sld_drag`` 认的
# ``_interface_{i}`` 之外，否则按下它会被当成某个界面而改错层厚。
SURFACE_HANDLE_LABEL = "_surface"

# 深度视野：按总厚度留边，而不是拿采样网格再放 5%。标称网格的尾巴只跟粗糙度有关，照它
# 取景基底那条带会只剩一丝。这两个比例是设计稿的构图（左边距占画框 8.8%、右边距 14.5%）。
DEPTH_HEAD_FRACTION = 0.115
DEPTH_TAIL_FRACTION = 1.19

# 纵轴顶到峰值的这个倍数，刻度取五格。设计稿顶在 22.0 = 1.10×20.0，横线正好五条，每级
# 台阶各自贴着一条；默认的 AutoLocator 在同一段上每 2.5 一格，会画出十来条。
SLD_HEADROOM = 1.10
SLD_TICK_BINS = 5
SLD_TICK_STEPS = (1, 2, 2.5, 5, 10)

# 只留横向网格，线宽跟着设计稿那五条走。
GRID_LINEWIDTH = 0.6


def _plain_layers(structure: object | None) -> tuple[object, ...]:
    """Return the stack's components when every one is a plain, drag-editable layer.

    A periodic block or a gradient layer has neither a single thickness nor a
    single SLD a handle could stand for, so a stack holding one offers no handles
    at all rather than a partial set the reader would have to reason about.
    """
    if structure is None:
        return ()
    components = tuple(getattr(structure, "components", ()))
    if not components or not all(isinstance(component, api.LayerSpec) for component in components):
        return ()
    return components


def draggable_interfaces(structure: object | None) -> tuple[tuple[int, float], ...]:
    """Return ``(layer_index, backing_depth_nm)`` for each drag-editable interface.

    Each layer contributes one handle at its backing edge, whose depth is the
    running sum of every thickness up to and including it.
    """
    interfaces: list[tuple[int, float]] = []
    cumulative_a = 0.0
    for index, component in enumerate(_plain_layers(structure)):
        cumulative_a += float(component.thickness_a)
        interfaces.append((index, cumulative_a / 10.0))
    return tuple(interfaces)


def draggable_layers(structure: object | None) -> tuple[tuple[int, float, float], ...]:
    """Return ``(layer_index, front_depth_nm, backing_depth_nm)`` per editable layer.

    A level handle spans the whole layer it stands for rather than marking one
    point of it, so a grab anywhere across the film raises or lowers that film
    and not a neighbour.
    """
    spans: list[tuple[int, float, float]] = []
    cumulative_a = 0.0
    for index, component in enumerate(_plain_layers(structure)):
        front_a = cumulative_a
        cumulative_a += float(component.thickness_a)
        spans.append((index, front_a / 10.0, cumulative_a / 10.0))
    return tuple(spans)


def draggable_roughness(structure: object | None) -> tuple[tuple[int, float, float], ...]:
    """Return ``(layer_index, interface_nm, sigma_nm)`` per drag-editable interface.

    Each layer's backing edge carries the roughness of the interface just below
    it: an interior layer ``k`` is smeared by ``components[k+1].roughness_a`` and
    the deepest layer by the stack's ``backing_roughness_a``.  The air/film
    interface (``components[0].roughness_a``) sits at depth zero with no backing
    edge of its own, so no handle stands for it.

    A layer whose backing neighbour draws its width from an interface transition
    offers no whisker at all: a transition and a plain roughness cannot both set
    that width, so committing a roughness there is refused, and a handle that
    could only ever snap back would mislead more than it helps.
    """
    components = _plain_layers(structure)
    if not components:
        return ()
    whiskers: list[tuple[int, float, float]] = []
    cumulative_a = 0.0
    last = len(components) - 1
    for index, component in enumerate(components):
        cumulative_a += float(component.thickness_a)
        if index == last:
            sigma_a = float(getattr(structure, "backing_roughness_a", 0.0))
        else:
            backing = components[index + 1]
            if getattr(backing, "transition", None) is not None:
                continue
            sigma_a = float(backing.roughness_a)
        whiskers.append((index, cumulative_a / 10.0, sigma_a / 10.0))
    return tuple(whiskers)


def _nominal_level_at(depth_nm: np.ndarray, profile_real: np.ndarray, target_nm: float) -> float:
    """Sample the nominal real SLD nearest ``target_nm`` off an already-drawn grid.

    Nearest grid point rather than an interpolation: the handle then sits on a
    value the pane actually drew, so what the reader grabs is what the reader
    sees.
    """
    if depth_nm.size == 0:
        return float("nan")
    return float(profile_real[int(np.argmin(np.abs(depth_nm - float(target_nm))))])


def _stack_total_nm(structure: object) -> float | None:
    """The depth of the last interface in nm, or ``None`` for a layerless stack."""
    layers = draggable_layers(structure)
    return layers[-1][2] if layers else None


def depth_regions(structure: object) -> tuple[tuple[str, int, float, float], ...]:
    """每段介质在深度轴上的位置：``(kind, index, start_nm, end_nm)``。

    ``kind`` 是 ``"fronting"`` / ``"layer"`` / ``"backing"`` 三者之一，``index`` 只对层
    有意义（层在结构里的序号，用来取它那一号色和名字）。半无限的两段没有边界，这里给
    的是一段按总厚度算的有限外延，轴会把视野之外的那截裁掉。
    """
    layers = draggable_layers(structure)
    if not layers:
        return ()
    total_nm = layers[-1][2]
    overhang = max(REGION_OVERHANG_FRACTION * total_nm, REGION_OVERHANG_FLOOR_NM)
    regions: list[tuple[str, int, float, float]] = [("fronting", -1, -overhang, 0.0)]
    regions.extend(("layer", index, front_nm, backing_nm) for index, front_nm, backing_nm in layers)
    regions.append(("backing", -1, total_nm, total_nm + overhang))
    return tuple(regions)


def _draw_region_bands(axes: object, structure: object) -> None:
    """把深度轴按介质分成竖色带。

    色带交代的是「这一段是什么」——没有它，读者只能从曲线的台阶反推哪一段是氧化层。
    ``zorder=0`` 让它待在网格和曲线底下：它是背景，不是第三套线。
    """
    for kind, index, start_nm, end_nm in depth_regions(structure):
        if kind == "layer":
            colour, alpha = stack.component_fill(index), REGION_LAYER_ALPHA
        else:
            colour, alpha = theme.DATA_NEUTRAL, REGION_MEDIUM_ALPHA
        axes.axvspan(start_nm, end_nm, facecolor=colour, alpha=alpha, linewidth=0.0, zorder=0)


def _region_name(name: object) -> str:
    """介质名里读者要认的那半段，与层列表那一行同源。

    ``naming.expert_name`` 把生成的占位名写成「SiO₂ · 表面氧化层」，树里那一行显示的就是
    它；色带上地方只够写一个词，所以取分隔符前面那半段，两处仍然对得上。
    """
    text = str(name or "").strip()
    if not text:
        return ""
    return naming.expert_name(text).partition(naming.SEPARATOR)[0].strip()


def _layer_region_label(component: object, thickness_nm: float) -> str:
    """一层的分区标注：名字加厚度，例如「SiO₂ 3.4nm」。"""
    thickness = f"{thickness_nm:.1f}nm"
    name = _region_name(getattr(component, "name", ""))
    return f"{name} {thickness}" if name else thickness


def _backing_region_label(structure: object) -> str:
    """基底那段的分区标注，例如「c-Si 基底」。"""
    name = _region_name(getattr(getattr(structure, "backing", None), "name", ""))
    return f"{name} {BACKING_REGION_ROLE}" if name else BACKING_REGION_ROLE


def _label_regions(axes: object, structure: object) -> None:
    """在每条色带顶上写一行「这段是什么」。

    必须在定好视野之后调用：半无限的两段没有中点，标注要落在画框内那一截的中间，而那
    一截有多长只有 ``get_xlim`` 知道。字号与字重是设计稿的读数（10.5px / 600），颜色让每
    行字跟自己那条带同色系——``apply_figure_palette`` 只改标题和轴标注，不会动这几行。
    """
    left, right = (float(value) for value in axes.get_xlim())
    palette = theme.current_plot_palette()
    layers = _plain_layers(structure)
    for kind, index, start_nm, end_nm in depth_regions(structure):
        if kind == "fronting":
            text, colour = FRONTING_REGION_LABEL, theme.DATA_NEUTRAL
            centre = 0.5 * (max(start_nm, left) + end_nm)
        elif kind == "backing":
            text, colour = _backing_region_label(structure), theme.DATA_NEUTRAL
            centre = 0.5 * (start_nm + min(end_nm, right))
        else:
            text, colour = _layer_region_label(layers[index], end_nm - start_nm), palette.muted
            centre = 0.5 * (start_nm + end_nm)
        axes.text(
            centre,
            REGION_LABEL_Y,
            text,
            transform=axes.get_xaxis_transform(),
            horizontalalignment="center",
            verticalalignment="top",
            color=colour,
            fontsize=theme.FONT_PT_SM,
            fontweight="semibold",
        )


def _draw_interface_handle(axes: object, depth_nm: float, level: float, label: str) -> None:
    """界面手柄：一根从曲线落到轴底的蓝虚线，线头顶一个白心蓝环的圆点。

    两笔画成一条 2 点线（``markevery=[1]`` 只给末端那一点上记号），因为 ``sld_drag`` 抓的
    就是这条线，而它读的是 ``get_xdata()`` 的两个端点。alpha 只压在线色上，圆点另给自己的
    面色与环色，这样点是实心的、线是淡的，跟设计稿一致。
    """
    accent = theme.current_accent()
    axes.plot(
        [depth_nm, depth_nm],
        [0.0, level],
        color=to_rgba(accent, HANDLE_ALPHA),
        dashes=HANDLE_DASH_PATTERN,
        linewidth=HANDLE_LINEWIDTH,
        marker="o",
        markevery=[1],
        markersize=HANDLE_DOT_SIZE,
        markerfacecolor=HANDLE_DOT_FACE,
        markeredgecolor=accent,
        markeredgewidth=HANDLE_DOT_EDGE_WIDTH,
        label=label,
        zorder=3,
    )


def _draw_nominal_structure(
    axes: object,
    structure: object,
    wavelength_a: float,
    *,
    label: str = NOMINAL_OVERLAY_LABEL,
    color: str = theme.DATA_NEUTRAL,
    linestyle: str = ":",
    linewidth: float = 1.2,
    regions: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay the structure's nominal real SLD and its draggable handles.

    How the curve is named and drawn is the caller's to say: beside a fitted
    candidate it is a second opinion and reads as a faint dotted overlay, while
    with no candidate at all it *is* the profile the reader is editing and takes
    the candidate's own name and hue. ``regions`` 加上设计稿帧③ 那四条介质色带，只有
    「这条线就是读者在编辑的剖面」那一路才要——旁边压着拟合结果时，色带说的是标称结构，
    会与候选曲线抢读者的注意。

    Returns the nominal depth grid in nm and the displayed real SLD, so a
    nominal-only pane can scale both axes to what it just drew.
    """
    depth_a, profile = api.sld_nominal_profile(structure, wavelength_a=float(wavelength_a))
    depth_nm = np.asarray(depth_a, dtype=float) / 10.0
    profile = np.asarray(profile, dtype=complex)
    # 手柄的高度从这条已经换算过的曲线上取（见 ``_nominal_level_at``），所以在这里换算
    # 一次，曲线和手柄就不会一个按 Å⁻² 一个按 10⁻⁶ Å⁻²。
    real_display = profile.real * SLD_DISPLAY_SCALE
    if regions:
        _draw_region_bands(axes, structure)
    axes.plot(
        depth_nm,
        real_display,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    # 表面也画同一笔，但它不是层的边界；标签落在可拖的那套名字之外（见
    # ``SURFACE_HANDLE_LABEL``），按下它不会被当成某个界面。
    _draw_interface_handle(axes, 0.0, _nominal_level_at(depth_nm, real_display, 0.0), SURFACE_HANDLE_LABEL)
    for index, interface_nm in draggable_interfaces(structure):
        # An underscore-prefixed label keeps the handle out of the key while still
        # giving the drag controller a stable name to grab and move it by.
        level = _nominal_level_at(depth_nm, real_display, interface_nm)
        _draw_interface_handle(axes, interface_nm, level, f"_interface_{index}")
    for index, front_nm, backing_nm in draggable_layers(structure):
        # 层的高度就是曲线自己那一段平台，所以这条把手与曲线同色同粗——它不是压在曲线上
        # 的第二条线，它就是曲线的那一段。
        level = _nominal_level_at(depth_nm, real_display, 0.5 * (front_nm + backing_nm))
        axes.plot(
            [front_nm, backing_nm],
            [level, level],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"_level_{index}",
        )
    for index, interface_nm, sigma_nm in draggable_roughness(structure):
        # The whisker runs from the interface outward by its current roughness, so
        # its visible length is the width the reader is editing.  末端不再单独画点：这套
        # 读数下 σ 折成 1.6…2.4px，那个点整个压在界面那个点底下，看上去就是重影；界面那个
        # 点本身仍然落在 ``_roughness_tip_hit`` 的容差里，按下它照样抓到粗糙度。
        level = _nominal_level_at(depth_nm, real_display, interface_nm)
        axes.plot(
            [interface_nm, interface_nm + sigma_nm],
            [level, level],
            color=to_rgba(theme.current_accent(), HANDLE_ALPHA),
            linestyle="-",
            linewidth=HANDLE_LINEWIDTH,
            label=f"_roughness_{index}",
            zorder=3,
        )
    return depth_nm, real_display


def _draw_nominal_only(
    view: DiagnosticView,
    structure: object,
    wavelength_a: float,
    bands: object | None = None,
) -> None:
    """Draw the structure's nominal profile as the profile under edit.

    A user can shape the stack by hand before the first fit, so the companion
    pane shows the current declaration and its handles rather than an empty
    placeholder. 结构与参数这两步走的也是这条路（``panel._sld_follows_edits``）：那时候
    选可能已经有了，但读者改的是结构，跟着改的就是这条线。
    """
    axes = view.axes
    axes.clear()
    try:
        # 还没有候选时，这条标称就是读者正在编辑的那条剖面，所以按设计稿帧③ 用候选橙实
        # 线画，名字也就是「当前 SLD 剖面」——卡片色标那一行说的正是它。
        depth_nm, real_display = _draw_nominal_structure(
            axes,
            structure,
            wavelength_a,
            label=CANDIDATE_REAL_LABEL,
            color=theme.DATA_CANDIDATE,
            linestyle="-",
            linewidth=2.2,
            regions=True,
        )
    except np.linalg.LinAlgError:
        finish_view(view)
        return
    # 抬头交给卡片（``sldCard`` 写着「SLD 深度剖面」），轴里不再写第二遍——设计稿帧③ 的图区
    # 没有标题，而这一段只有 ~120px 高，一行标题就是一成高度。可信带那句说明是抬头说不到的，
    # 所以只有它留在轴上。
    axes.set(xlabel=DEPTH_AXIS_LABEL, ylabel=SLD_AXIS_LABEL)
    if bands is not None:
        _draw_bands(axes, bands)
        axes.set_title(bands.caption(), fontsize=theme.FONT_PT_SM)
    _limit_depth_axis(axes, depth_nm, total_nm=_stack_total_nm(structure))
    if bands is None:
        # 可信带的下界会落到 0 以下，把纵轴钉在 0 会把它裁掉，所以只有单纯那一路收紧纵轴。
        _scale_sld_axis(axes, real_display)
    # 分区标注要落在「看得见的那一段」的正中，所以排在定好视野之后；又必须排在
    # ``finish_view`` 之前，那一步才会把中文字体套到这几行字上。
    _label_regions(axes, structure)
    _draw_legend(axes, [])
    finish_view(view)
    _frame_axes(view)


def draw_sld(
    view: DiagnosticView,
    candidate: object | None,
    others: tuple[object, ...] = (),
    bands: object | None = None,
    *,
    structure: object | None = None,
    wavelength_a: float | None = None,
) -> None:
    """Draw the selected candidate's SLD with comparison overlays from others.

    The reflectivity comparison tab already overlays all candidates so users can
    judge which model best matches the data. This extends that comparison to the
    SLD depth profiles: the selected candidate's real and imaginary parts stay
    prominent at full opacity, while other valid candidates' real parts appear as
    faint overlays behind them. This lets users compare structural interpretations
    side-by-side without switching between rows.

    When a structure is supplied its nominal real profile and draggable interface
    handles are overlaid too, so the depth axis doubles as a hand-editing surface.
    With no fitted candidate the nominal profile stands in for the empty pane.
    """
    has_nominal = structure is not None and wavelength_a is not None
    if candidate is None:
        if has_nominal:
            _draw_nominal_only(view, structure, wavelength_a, bands)
            return
        draw_empty(view, "SLD 深度剖面", "暂无当前候选")
        return
    depth_nm = np.asarray(candidate.sld_depth_a, dtype=float) / 10.0
    profile = np.asarray(candidate.sld_profile_a2, dtype=complex)
    axes = view.axes
    axes.clear()
    # Draw comparison overlays first so the selected candidate's curves sit on top
    selected_id = candidate.candidate_id
    overlays: list[object] = []
    for other in others:
        if other.candidate_id == selected_id or candidate_is_inspection_only(other):
            continue
        other_depth = np.asarray(other.sld_depth_a, dtype=float) / 10.0
        other_profile = np.asarray(other.sld_profile_a2, dtype=complex)
        overlays.extend(
            axes.plot(
                other_depth,
                other_profile.real * SLD_DISPLAY_SCALE,
                linewidth=0.8,
                alpha=0.35,
                label=f"{other.candidate_id} 实部",
            )
        )
    # Draw the selected candidate's real/imag at full opacity on top. These stay
    # the first two entries in axes.lines; the nominal overlay is appended after
    # so index-addressed assertions keep reading the candidate curves.
    axes.plot(depth_nm, profile.real * SLD_DISPLAY_SCALE, label=CANDIDATE_REAL_LABEL)
    axes.plot(depth_nm, profile.imag * SLD_DISPLAY_SCALE, "--", label="SLD 虚部")
    axes.set(title="SLD 深度剖面", xlabel=DEPTH_AXIS_LABEL, ylabel=SLD_AXIS_LABEL)
    if has_nominal:
        try:
            _draw_nominal_structure(axes, structure, wavelength_a)
        except np.linalg.LinAlgError:
            pass
    if bands is not None:
        _draw_bands(axes, bands)
        axes.set_title(f"SLD 深度剖面 — {bands.caption()}", fontsize=theme.FONT_PT_SM)
    _limit_depth_axis(axes, depth_nm)
    _draw_legend(axes, overlays)
    finish_view(view)
    _frame_axes(view)


def _limit_depth_axis(axes: object, depth_nm: np.ndarray, *, total_nm: float | None = None) -> None:
    """Scale the depth axis to the selected candidate, not to the overlays.

    A run that keeps a badly fitted candidate can carry a stack hundreds of nm
    thick.  Letting the shared autoscale see that overlay stretched the axis to
    its depth, and the selected profile - the one the user is reading - collapsed
    into a sliver at the left edge.  The overlays stay drawn for comparison; they
    just no longer decide the scale, so the part of them that falls outside the
    selected range is simply out of frame.

    ``total_nm`` 给的是层堆叠的总厚度：有它就按设计稿的构图留边（两侧各占画框 8.8% 与
    14.5%），而不是拿采样网格再放 5%。标称网格的尾巴只跟粗糙度有关，照它取景基底那条色带
    会只剩一丝；用比例而不是常数，换一套层厚时构图仍然成立，再与网格取 min/max，曲线
    永远不会被裁掉。
    """
    finite = depth_nm[np.isfinite(depth_nm)]
    if finite.size == 0:
        return
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if total_nm is not None and total_nm > 0.0:
        axes.set_xlim(
            min(-DEPTH_HEAD_FRACTION * total_nm, lower),
            max(DEPTH_TAIL_FRACTION * total_nm, upper),
        )
        return
    span = upper - lower
    if span <= 0.0:
        return
    margin = 0.05 * span
    axes.set_xlim(lower - margin, upper + margin)


def _scale_sld_axis(axes: object, real_display: np.ndarray) -> None:
    """纵轴从 0 顶到峰值上方一点，刻度取五格。

    默认的 AutoLocator 在这段上每 2.5 一格，画出十来条横线，把一条只有三级台阶的曲线埋
    在网格里。五格是设计稿的读数密度（0/5/10/15/20），也正好让每级台阶各自贴着一条线。
    必须排在 ``finish_view`` 之前：新刻度标签是在那一步才套上字体的。
    """
    finite = real_display[np.isfinite(real_display)]
    if finite.size == 0:
        return
    peak = float(np.max(finite))
    if peak <= 0.0:
        return
    axes.set_ylim(0.0, SLD_HEADROOM * peak)
    axes.yaxis.set_major_locator(MaxNLocator(nbins=SLD_TICK_BINS, steps=list(SLD_TICK_STEPS)))


def _frame_axes(view: DiagnosticView) -> None:
    """只留横向网格和左、下两条边框。

    竖向的分界已经由色带和界面虚线交代过了，再铺一层竖网格就是三套线在同一处重复；次刻度
    也一并去掉——``apply_axes_grid`` 会装上 ``AutoMinorLocator``，留着就多出一批没画完的短线。
    必须排在 ``finish_view`` 之后：那一步会把两向网格都重新打开。只是隐藏线条，不会产生新的
    刻度标签，所以字体那一遍仍然作数。
    """
    axes = view.axes
    palette = theme.current_plot_palette()
    axes.xaxis.set_minor_locator(NullLocator())
    axes.yaxis.set_minor_locator(NullLocator())
    axes.grid(False, which="both")
    axes.grid(True, axis="y", which="major", color=palette.grid, linewidth=GRID_LINEWIDTH)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    view.canvas.draw_idle()


def _draw_legend(axes: object, overlays: list[object]) -> None:
    """Key whatever the card's own colour strip above the axes cannot name.

    A run that keeps several candidates gave the key one row per comparison
    curve, and the key then took most of the pane's width and squeezed the
    profiles into a strip.  Every curve keeps its own label so selection and
    inspection still name it; only the key collapses them into a counted entry.

    设计稿帧③ 的图区里没有图例框，卡片抬头下面那排色标就是这张图的图例；四条 pyqtgraph
    面板同样只有色标、不画内嵌图例。所以色标已经说过的记号在这里一律不重复，剩下的什么
    都不剩时就干脆不画框——那正是帧③ 的样子。
    """
    handles, labels = axes.get_legend_handles_labels()
    entries = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=True)
        if handle not in overlays and label not in CARD_KEY_LABELS
    ]
    if overlays:
        entries.append((overlays[0], f"其他候选 实部 ×{len(overlays)}"))
    existing = axes.get_legend()
    if not entries:
        if existing is not None:
            existing.remove()
        return
    axes.legend(
        [handle for handle, _ in entries],
        [label for _, label in entries],
        fontsize=theme.FONT_PT_SM,
        loc="best",
        framealpha=0.75,
    )


# Credible bands mirror the exported PNG: (quantile pair, fill alpha, legend
# label). The inner 16-84% band is more opaque than the outer 2.5-97.5% band, so
# nesting reads correctly. Depth is converted Å→nm to match the profile curves.
#
# 标注写全「不确定带」：这张图上同时还有分位数着色的对比候选，光写一个区间，读者得先猜
# 它说的是哪个量的百分位。
BAND_PAIRS = (
    ((0.16, 0.84), 0.28, INNER_BAND_LABEL),
    ((0.025, 0.975), 0.14, "2.5–97.5% 不确定带"),
)


def _band_index(quantiles: tuple[float, ...], level: float) -> int | None:
    return next((i for i, value in enumerate(quantiles) if value == level), None)


def _draw_band_pair(
    axes: object,
    bands: object,
    pair: tuple[float, float],
    alpha: float,
    label: str,
) -> None:
    lower = _band_index(bands.quantiles, pair[0])
    upper = _band_index(bands.quantiles, pair[1])
    if lower is None or upper is None:
        return
    depth_nm = np.asarray(bands.depth_a, dtype=float) / 10.0
    # 帧③把可信带定为单一候选橙阴影，仅以透明度区分内外带。不传 color 会让填充
    # 跟随 axes 轮转色（且随前面绘制的对比候选数量漂移），四个填充会落到互不相干
    # 的颜色；固定为 DATA_CANDIDATE 让内外带、实部虚部同色，与设计稿对齐。
    axes.fill_between(
        depth_nm,
        bands.real[lower] * SLD_DISPLAY_SCALE,
        bands.real[upper] * SLD_DISPLAY_SCALE,
        color=theme.DATA_CANDIDATE,
        alpha=alpha,
        label=label,
    )
    axes.fill_between(
        depth_nm,
        bands.imaginary[lower] * SLD_DISPLAY_SCALE,
        bands.imaginary[upper] * SLD_DISPLAY_SCALE,
        color=theme.DATA_CANDIDATE,
        alpha=alpha,
    )


def _draw_bands(axes: object, bands: object) -> None:
    for pair, alpha, label in BAND_PAIRS:
        _draw_band_pair(axes, bands, pair, alpha, label)


def _owned_report(result: object | None, candidate_id: str | None) -> tuple[object | None, str]:
    """The uncertainty report, but only when the inspected candidate owns it.

    Returns ``(None, reason)`` whenever the evidence must not be drawn, so every
    caller states the same refusal wording.  Ownership is checked rather than
    assumed because a report survives a candidate switch: drawing candidate-b's
    matrix while candidate-a is inspected would relabel whose evidence it is.
    """
    report = None if result is None else result.uncertainty
    if report is None:
        return None, "不确定性报告不可用"
    owner = report.candidate_id
    if candidate_id is None or owner != candidate_id:
        return (
            None,
            f"不确定性证据属于 {owner or '未标识候选'}，当前查看 {candidate_id or '未选择候选'}",
        )
    return report, ""


def _reset_page(view: DiagnosticView) -> object:
    """Clear a single-axes page back to its primary axes, colour key included.

    A colour key is an axes of the figure, so a page that drew one last time would
    otherwise accumulate a second, third and fourth key beside a matrix that has
    only one scale.  ``DiagnosticView`` is frozen, so the primary axes is the one
    thing that cannot be replaced -- it is kept and cleared.
    """
    for axes in tuple(view.figure.axes):
        if axes is not view.axes:
            axes.remove()
    view.axes.clear()
    return view.axes


def _page_placeholder(view: DiagnosticView, message: str) -> None:
    """State why a page is empty, in the muted colour and without tick marks.

    Default 0–1 ticks on an empty pane read as a real plot whose curve failed to
    draw, which is the opposite of what an unavailable-evidence page has to say.
    """
    axes = _reset_page(view)
    axes.set_xticks(())
    axes.set_yticks(())
    palette = apply_figure_palette(view.figure)
    axes.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axes.transAxes,
        color=palette.muted,
    )
    finish_view(view)


def _definitions(result: object | None) -> tuple[object, ...]:
    """结果自己带的参数声明——短名的素材，缺席时轴上退回机器路径。"""
    return tuple(getattr(result, "parameter_definitions", ()) or ())


# Profile 卡的抬头。设计稿的卡抬头是「Profile 似然」、副题「逐参数扫描 · Δχ²=1 对应 ±1σ」；
# 前半段并进抬头，后半段不采用——本项目的目标函数是 ``robust_log_cost``，它不是 χ²，所以
# 「Δ=1 对应 ±1σ」这个推论在这里不成立（见 ``_draw_profiles``）。写成常量是因为空态那一支
# 也要写同一句：两处拼写分叉时读者会以为换了张图。
PROFILE_TITLE = "参数剖面 · 逐参数扫描"

# 两侧半宽比过了这个数才叫「对称」。0.75 即「窄的一侧不短于宽的一侧的四分之三」——再接近一些
# 的差别在半栏高的图上本来就看不出来，而低于它时曲线的歪已经肉眼可辨，此时逐参数的 ±1σ
# （左右等宽的那一对）就不该再单独读。
PROFILE_SYMMETRY_RATIO = 0.75

# 纵轴最多画到阈值的这么多倍，超出的部分裁掉——取自设计稿那张卡的刻度（0/1/2/3/4，阈值在 1）。
# 扫描跨度按 σ 的倍数定，对称抛物线扫到 ±3σ 时端点就是 9 倍 delta；按全部数据取景的话读者要
# 读的 0..Δ 那一段只剩画面的十几分之一。四倍留得下阈值线上方的那截「还差多少」，又不至于让远端
# 决定取景。
PROFILE_HEADROOM = 4.0


def draw_correlation_page(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    """Fill one whole page with the correlation matrix.

    The same matrix also rides half of the ``uncertainty`` diagnostic tab.  Here
    it gets the full figure, because reading a 17×17 matrix in half a pane means
    the parameter names elide down to nothing.

    这一页按扁框排：色标竖在右边，方阵居中。画布是 400×202px，方阵的边长由高度定死，而横躺的
    色标正是从高度里再切一刀——实测 17 个参数下一格从 6px 掉到 3.4px，连主对角线都分不出来（见
    ``test_correlation_labels``）。中栏那一屏的格子高 449px，躺下去之后矩阵仍有 300px 可用，
    右侧还能整块让给读数栏，所以在那里划算；在这里不划算。
    """
    report, message = _owned_report(result, candidate_id)
    if report is None:
        _page_placeholder(view, message)
        return
    axes = _reset_page(view)
    _draw_correlation(view.figure, axes, report, _definitions(result), wide_layout=False)
    finish_view(view)


def draw_profile_page(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    """Fill one whole page with the profile-likelihood curves and intervals."""
    report, message = _owned_report(result, candidate_id)
    if report is None:
        _page_placeholder(view, message)
        return
    axes = _reset_page(view)
    _draw_profiles(axes, report, _definitions(result))
    finish_view(view)


def draw_band_page(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    """Fill one whole page with the SLD credible bands and what produced them.

    The caption comes off the bands themselves rather than being written here, so
    the on-screen provenance and the exported provenance are the same sentence.
    """
    report, message = _owned_report(result, candidate_id)
    if report is None:
        _page_placeholder(view, message)
        return
    bands = report.sld_bands
    if bands is None:
        _page_placeholder(view, "SLD 可信带不可用（未运行自助抽样或采样）")
        return
    axes = _reset_page(view)
    _draw_bands(axes, bands)
    axes.set(title="SLD 剖面可信带", ylabel=SLD_AXIS_LABEL)
    # 出处跟着横轴走：带宽读起来只是「宽」还是「窄」，抽样数和对齐面才说得清它凭什么这么宽。
    axes.set_xlabel(f"{DEPTH_AXIS_LABEL}\n{bands.caption()}", fontsize=theme.FONT_PT_SM)
    axes.legend(fontsize=theme.FONT_PT_SM)
    finish_view(view)


def _reset_uncertainty_axes(view: DiagnosticView) -> tuple[object, object]:
    """把这一页收回上下两张图，主 axes 留在上面那格。

    重建这一支平时不走——每次重置都把 figure 交还两个 axes，所以它只在 view 是单 axes 起手
    时进入。``DiagnosticView`` 冻结，主 axes 换不掉：给 figure 铺一张 2×1 的网格，把主 axes
    挪进上格，再补一个下格。同文件 ``_reset_page`` 收拾单图页用的是同一条路子。
    """
    primary = tuple(view.figure.axes[:2])
    if len(primary) != 2:
        for axes in tuple(view.figure.axes):
            if axes is not view.axes:
                axes.remove()
        grid = GridSpec(2, 1, figure=view.figure)
        view.axes.set_subplotspec(grid[0])
        primary = (view.axes, view.figure.add_subplot(grid[1]))
    for axes in tuple(view.figure.axes[2:]):
        axes.remove()
    for axes in primary:
        axes.clear()
    return primary


def _unavailable(
    view: DiagnosticView,
    message: str,
) -> None:
    # Both halves get their ticks blanked and their placeholder drawn in the
    # muted colour.  Leaving the default 0-1 ticks in place made an empty pane
    # read as a real plot whose curve had failed to draw.
    correlation, profile = _reset_uncertainty_axes(view)
    for axes in (correlation, profile):
        axes.set_xticks(())
        axes.set_yticks(())
    correlation.set_title(CORRELATION_TITLE)
    profile.set_title(PROFILE_TITLE)
    palette = apply_figure_palette(view.figure)
    for axes, text in ((correlation, message), (profile, "区间证据不可用")):
        axes.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=axes.transAxes,
            color=palette.muted,
        )
    finish_view(view)


def _sigma_by_name(report: object) -> dict[str, float]:
    """报告发布的 ±1σ，按参数名取；一个都拿不到时是空映射。

    ``parameter_sigma`` 与 ``correlation_names`` 同序同长（模型侧的 ``_parameter_sigma`` 校验
    过这一条），所以按位置对齐就够。非有限值和 0 一并丢掉：横轴要拿它做除数，除完得不出
    「一个 σ」这个刻度。
    """
    sigma = getattr(report, "parameter_sigma", None)
    if sigma is None:
        return {}
    names = tuple(report.correlation_names)
    values = np.asarray(sigma, dtype=float)
    if values.shape != (len(names),):
        return {}
    return {
        name: float(value)
        for name, value in zip(names, values, strict=True)
        if np.isfinite(value) and float(value) > 0.0
    }


def _profile_offsets(values: np.ndarray, objectives: np.ndarray, sigma: float | None) -> np.ndarray:
    """横轴坐标：给了 σ 就以 σ 为单位、0 落在最优点；否则退回按各自量程归一化。

    归一化那一支里三条曲线各有一个 0 点和一个 1 点，最低点落在横轴哪儿取决于扫描点怎么排，
    「哪条窄、哪条歪」于是比不出来。σ 是现成的公共尺子：它逐参数带着自己的量纲，除完之后
    2 Å 的厚度与 0.05 的相对密度都读作「一个 σ」。σ 出自协方差，而协方差不是每条路径都算得
    出来，所以归一化那一支留着——但轴标签跟着换，免得读者拿 σ 的读法去读一个不是 σ 的横轴。
    """
    if sigma is not None:
        centre = float(values[int(np.argmin(objectives))])
        return (values - centre) / float(sigma)
    span = float(np.ptp(values))
    return np.zeros_like(values) if span == 0.0 else (values - float(np.min(values))) / span


def _half_width(offsets: np.ndarray, increments: np.ndarray, threshold: float, *, upper: bool) -> float | None:
    """中心到这一侧闭合点的距离，按横轴自己的单位算；这一侧没跨过阈值时是 ``None``。

    闭合点在相邻两个扫描点之间线性插值。扫描点是离散的，取「第一个超过阈值的那个点」会把半宽
    系统性地读大一格——而这个距离正是用来判左右是否对称的，两侧各偏一格就足以把对称读成偏斜。
    """
    centre = int(np.argmin(increments))
    step = 1 if upper else -1
    for index in range(centre, offsets.size - 1 if upper else 0, step):
        near, far = index, index + step
        if float(increments[far]) < threshold:
            continue
        span = float(increments[far]) - float(increments[near])
        fraction = 1.0 if span <= 0.0 else (threshold - float(increments[near])) / span
        crossing = float(offsets[near]) + fraction * (float(offsets[far]) - float(offsets[near]))
        return abs(crossing - float(offsets[centre]))
    return None


def _profile_shape(profile: object, offsets: np.ndarray, increments: np.ndarray, threshold: float | None) -> str:
    """图例括号里那句形状判读；判不出来时是空串，图例就只写量名与短名。

    判读出自扫描自己发布的读数，不是目测：两侧是否闭合是 ``lower_closed`` / ``upper_closed``，
    左右半宽出自曲线与本次扫描那条阈值的交点。四种措辞各自对应一个不同的下一步，所以不能混：
    「平坦→弱约束」是两侧都没闭合——曲线抬不起来，得加约束或换参数化；「单侧未闭合」是那一侧
    扫描范围不够或被边界挡住，放宽范围就够了，而它的区间根本没有那一端，写成「对称」是错的；
    「偏斜」是两侧都闭合但一宽一窄，此时逐参数那对左右等宽的 ±1σ 不该再单独读；「对称」才是
    ±1σ 读得下去的那种。
    """
    lower_closed = bool(profile.lower_closed)
    upper_closed = bool(profile.upper_closed)
    if not lower_closed and not upper_closed:
        return "平坦→弱约束"
    if not lower_closed:
        return "下侧未闭合"
    if not upper_closed:
        return "上侧未闭合"
    if threshold is None:
        return ""
    lower = _half_width(offsets, increments, threshold, upper=False)
    upper = _half_width(offsets, increments, threshold, upper=True)
    if lower is None or upper is None or max(lower, upper) <= 0.0:
        return ""
    return "对称" if min(lower, upper) / max(lower, upper) >= PROFILE_SYMMETRY_RATIO else "偏斜"


def _legend_columns(axes: object, entry_count: int) -> int:
    """Profile 图例该排几列——按绘图区高度与项数自适应，让图例收进绘图区里。

    单列图例每项约 19.2px（9pt 字 + 行距），18 项 345.7px，而半栏高绘图区只有 211.8px——
    图例溢出撑大 Profile 那格的 tightbbox，``constrained_layout`` 为它让出高度，反过来压小
    矩阵（矩阵从 379.9 缩到 211.8px），读数栏挂在矩阵坐标上跟着从 235.5 矮到 131.3px，四段
    读数 175.0px 装不下，末段被栏底裁掉。

    按可用高度算列数时须把每行的 ``labelspacing`` 和图例内外边距一起扣掉。只算字高会把
    8 项误判为单列可容纳，实际图例高 159px；它随后又把 constrained_layout 的绘图区压到
    99px。构图时尚未排版，预留这份真实排版开销，避免图例反过来撑坏整个上下布局。
    """
    box = axes.get_position()
    height_inches = box.height * float(axes.figure.get_figheight())
    height_pixels = height_inches * float(axes.figure.dpi)
    font_pixels = theme.FONT_PT_SM * float(axes.figure.dpi) / 72.0
    line_height = font_pixels * (1.2 + float(rcParams["legend.labelspacing"]))
    padding = 2.0 * font_pixels * (float(rcParams["legend.borderpad"]) + float(rcParams["legend.borderaxespad"]))
    usable = height_pixels * 0.9 - padding
    max_rows = max(int(usable / line_height), 1)
    return max((entry_count + max_rows - 1) // max_rows, 1)


def _draw_profile_curve(
    axes: object, profile: object, label: str, quantity: str, sigma: float | None
) -> tuple[float, float | None]:
    """一条扫描及其形状标注；返回同一目标增量标度上的峰值与闭合阈值。"""
    values = np.asarray(profile.values, dtype=float)
    objectives = np.asarray(profile.objectives, dtype=float)
    best = float(np.min(objectives))
    increments = objectives - best
    offsets = _profile_offsets(values, objectives, sigma)
    published = profile.objective_threshold
    threshold = None if published is None else float(published) - best
    shape = _profile_shape(profile, offsets, increments, threshold)
    entry = " ".join(part for part in (quantity, label) if part)
    axes.plot(offsets, increments, "-o", label=f"{entry}（{shape}）" if shape else entry)
    return float(np.max(increments)), threshold


def _draw_profile_thresholds(axes: object, levels: dict[str, float], palette: theme.PlotPalette) -> None:
    """每个不同增量一条阈值线，按数值排序并保留本次扫描的读数。"""
    for text, level in sorted(levels.items(), key=lambda item: item[1]):
        axes.axhline(
            level,
            color=palette.muted,
            linestyle="--",
            linewidth=1.0,
            label=f"区间闭合阈值 Δ={text}",
        )


def _draw_profiles(axes: object, report: object, definitions: Iterable[object] = ()) -> None:
    """Parameter profiles and the published threshold that closes their intervals.

    刻度和图例写的是短名（``d·ox``）。这张图在帧⑤ 与相关矩阵上下叠，读者拿图例里的曲线去
    矩阵上找那一格；一边写 ``d·ox``、另一边写 ``component.0.thickness_a``，两张互证的图之间
    就多了一次翻译。图例除短名外还写着中文量名与一句形状判读：``ρ`` 这个符号只对读过矩阵刻度
    的人成立，而图例常常是这条曲线第一次出现的地方；判读那一句见 ``_profile_shape``。

    纵轴画的是相对各自最优的增量，不是目标函数的绝对值。``robust_log_cost`` 的量级由数据点数
    和权重定，「1.0 与 1.36 之间那 0.36 是大还是小」光看绝对值无从判断；减掉最优之后纵轴上的数
    就是「离开最优点付出的代价」，而这正是判定区间的那把尺子——阈值本身就是以扫描中心的目标值
    为起点加一个增量定义的。同一个减法也落在阈值线上，否则线会画到画面外。

    纵轴与那条线都不标成 ``Δχ²``，这是对设计稿的一处有意偏离：设计稿写 ``Δχ²（相对最优）`` 与
    ``Δχ²=1 → ±1σ``，而本项目的目标函数不是 χ²，这个 Δ 没有 χ² 的单位，「Δ=1 对应 ±1σ」这个
    推论在这里不成立。标上去读者会拿 χ² 的分位数去读这条纵轴，得出的区间没有出处。线上因此只
    标本次扫描自己用来判定闭合的那个增量——``analysis/profiles.py`` 的 ``_closure_threshold``
    明说它「既是判定 flags 的比较值、也是发布在 profile 上的数」，图上标的必须是同一个数。

    自助区间不画在这里。它是「每个自由参数一条」，这份样品 17 个自由参数就是 17 行——叠在
    绘图区左上角既越出右边界、又和图例抢同一块地方，而绘图区再宽也只能容下前几行。那些行归
    ``results/uncertainty.py`` 的证据文本框，那里是可滚动的纯文本，17 行放得下。图上留下的是
    曲线和阈值线：阈值线标出每一侧在哪里闭合，也就是 profile 区间的两端。
    """
    profiles = tuple(report.profiles)
    palette = theme.current_plot_palette()
    names = tuple(profile.name for profile in profiles)
    labels = short_labels(names, definitions)
    quantities = quantity_names(names, definitions)
    sigma = _sigma_by_name(report)
    # 整张图一起换或一起不换：一条曲线以 σ 为单位、另一条留在自己的量程里时，横轴上的同一个数
    # 在两条曲线上不是同一件事，而这张图要读的正是三条摆在同一把尺子上的比较。
    by_sigma = bool(names) and all(name in sigma for name in names)
    # 阈值按印出来的精度聚合：几条扫描的增量常常只差在小数第三位，各画一条在屏上是一条粗线加
    # 几句重复的图例。画的是组内第一个真值，标的是它印出来的那个数。
    levels: dict[str, float] = {}
    tallest = 0.0
    for profile, label, quantity in zip(profiles, labels, quantities, strict=True):
        height, threshold = _draw_profile_curve(
            axes, profile, label, quantity, sigma[profile.name] if by_sigma else None
        )
        tallest = max(tallest, height)
        if threshold is not None:
            levels.setdefault(f"{threshold:.2f}", threshold)
    # 每个不同的增量一条线，各带自己的读数：扫描就是拿目标值与这个数比较来判一侧是否闭合，
    # 线不画出来，曲线只显示「涨了」而藏起了区间在哪儿结束。线上标数值而不只标「阈值」，读者要
    # 判的是「这条曲线离闭合还差多少」。
    _draw_profile_thresholds(axes, levels, palette)
    if profiles:
        # 图例项数 = len(profiles) + len(levels)，真实规模 17 条 profile 加一条阈值线就是 18 项。
        # 单列时高约 19.2px/项，18 项 345.7px，而半栏高绘图区只有 211.8px——图例撑大 Profile 那格
        # 的 tightbbox，constrained_layout 为它让出高度，反过来压小矩阵，读数栏跟着矮，末段被裁。
        # 按可用高度自适应列数：每列最多 n 项，n 行高 ≤ 绘图区高 × 0.9（留 10% 呼吸）。
        axes.legend(fontsize=theme.FONT_PT_SM, ncols=_legend_columns(axes, len(profiles) + len(levels)))
    else:
        axes.text(0.5, 0.5, "剖面与区间证据未执行/不可用", ha="center", va="center", transform=axes.transAxes)
    # 取景按阈值而不是按曲线远端，见 ``PROFILE_HEADROOM``。裁的是视野不是数据：远端那些点仍留在
    # 线上（悬停读得到、导出用得着），只是不出现在这一屏里。阈值是上限而非固定值——三条都抬不到
    # 阈值时（全是弱约束）按曲线收回来，否则那一屏会变成三条压在底部的平线加一大片空白；但阈值线
    # 本身始终留在视野里，读者判「还差多少」用的就是它。
    if levels:
        ceiling = max(levels.values())
        upper = max(min(tallest, PROFILE_HEADROOM * ceiling), ceiling)
        margin = 0.05 * upper
        axes.set_ylim(-margin, upper + margin)
    axes.set(
        title=PROFILE_TITLE,
        xlabel="参数偏移（以 σ 为单位）" if by_sigma else "参数坐标（各参数独立归一化至 [0, 1]）",
        ylabel="目标增量 Δ（相对最优）",
    )


def draw_uncertainty(
    view: DiagnosticView,
    result: object | None,
    candidate_id: str | None,
) -> None:
    report = None if result is None else result.uncertainty
    if report is None:
        _unavailable(view, "不确定性报告不可用")
        return
    owner = report.candidate_id
    if candidate_id is None or owner != candidate_id:
        _unavailable(
            view,
            f"不确定性证据属于 {owner or '未标识候选'}，当前查看 {candidate_id or '未选择候选'}",
        )
        return
    correlation, profile = _reset_uncertainty_axes(view)
    # The matrix and its colour key are drawn by the same helper the single-page
    # correlation view uses, so the tab and the page cannot drift apart on the
    # fixed [-1, 1] scale -- the one thing that makes a cell readable as a number.
    # The key is rebuilt each draw because the reset above removes it along with
    # every other companion axes.
    _draw_correlation(view.figure, correlation, report, _definitions(result))
    # The fixed scale rides on the colour key, which is the thing it describes.
    # In the title it made that title 222px wide inside an 835px pane and it ran
    # into 「参数剖面似然与区间」; as a secondary line above the axes -- the way
    # the residual heatmap states its own range -- it overran the narrow matrix
    # subplot sideways and printed over 「相关矩阵」 instead.
    _draw_profiles(profile, report, _definitions(result))
    finish_view(view)
