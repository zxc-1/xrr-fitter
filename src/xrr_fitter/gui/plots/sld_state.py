"""SLD companion pane controls and view-only band replay state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.plots import sld
from xrr_fitter.gui.plots.diagnostics import COMPANION_SPEC, DiagnosticView

# The selector labels are Chinese UI text; these are the matching algorithm keys
# in the same order the combo box items are added.
ALIGN_KEYS = ("backing", "surface")
BatchTrends = tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]

SLD_CARD_SUBTITLE = "实时预览 · 界面手柄可拖拽"

# (shape, colour role, caption), keyed by ``theme.build_legend``.  The captions
# come from the drawing module so this row and the marks it explains cannot drift
# apart; ``sld._draw_legend`` reads the same names back to keep the in-axes key
# from repeating any of them.  Shapes differ per entry, so the key survives
# greyscale and colour blindness.
SLD_LEGEND_ENTRIES = (
    ("line", "candidate", sld.CANDIDATE_REAL_LABEL),
    ("box", "candidate", sld.INNER_BAND_LABEL),
    ("dot", "accent", sld.INTERFACE_KEY_LABEL),
)

# The shortest SLD plot still worth drawing: a depth axis, its labels, and enough
# vertical room to tell a step profile from a slope.  The pane's floor is this
# plus whatever the card chrome asks for, so adding a header row pushes the floor
# up instead of eating into the axes.
MIN_SLD_PLOT_H = 120


@dataclass(frozen=True, slots=True)
class Projection:
    data: api.PreparedData | None
    mask: np.ndarray | None
    result: object | None
    candidate_id: str | None
    trends: BatchTrends | None
    visible_range: tuple[float, float] | None
    dataset_id: str | None = None
    structure: object | None = None
    # 当前数据集读不出来时画布上那句说明；跟着投影走，回滚时说明也一起回到上一次的样子。
    source_issue: str | None = None


@dataclass(frozen=True, slots=True)
class SldBandReplay:
    dataset_id: str | None
    report: object
    alignment: str
    bands: object


@dataclass(frozen=True, slots=True)
class SldViewState:
    cache: SldBandReplay | None
    alignment_index: int
    toggle_enabled: bool
    toggle_checked: bool
    toggle_tooltip: str
    selector_enabled: bool


def build_sld_legend(parent: QWidget) -> QWidget:
    """The key for the SLD pane, as a caption row above the plot.

    Named ``sldCardLegend`` because it is one of the plot cards' keys: the marks
    and the geometry are ``theme``'s, only the entries are this pane's.
    """
    return theme.build_legend(parent, "sldCardLegend", SLD_LEGEND_ENTRIES)


def build_sld_companion_pane(
    parent: QWidget,
    views: dict[str, DiagnosticView],
    on_bands_toggled: Callable[..., None],
    on_align_changed: Callable[..., None],
) -> tuple[QWidget, QCheckBox, QComboBox]:
    key, title, description = COMPANION_SPEC
    pane = QWidget(parent)
    pane.setObjectName("sldPane")
    pane.setAccessibleName(title)
    pane.setAccessibleDescription(description)
    # The card's own title replaces the former bare header label: two headings
    # for one pane read as two sections.
    card, body = theme.titled_card(pane, "sldCard", title, SLD_CARD_SUBTITLE)
    toggle, selector, controls = build_sld_band_controls(card, on_bands_toggled, on_align_changed)
    body.addWidget(build_sld_legend(card))
    body.addLayout(controls)
    body.addWidget(views[key].canvas, 1)
    layout = QVBoxLayout(pane)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_XS)
    layout.addWidget(card, 1)
    # The floor is derived, not counted: ask the assembled pane what its chrome
    # needs and add a plot worth drawing.  A hand-written number goes stale the
    # moment a header row is added, and the pane then honours the number by
    # collapsing the axes instead of by growing.
    canvas_floor = views[key].canvas.minimumSizeHint().height()
    chrome = max(pane.minimumSizeHint().height() - canvas_floor, 0)
    pane.setMinimumHeight(chrome + MIN_SLD_PLOT_H)
    return pane, toggle, selector


def build_sld_band_controls(
    pane: QWidget,
    on_bands_toggled: Callable[..., None],
    on_align_changed: Callable[..., None],
) -> tuple[QCheckBox, QComboBox, QHBoxLayout]:
    controls = QHBoxLayout()
    controls.setContentsMargins(0, 0, 0, 0)
    controls.setSpacing(theme.SPACE_SM)
    toggle = QCheckBox("显示不确定度带", pane)
    toggle.setObjectName("sldBandsToggle")
    toggle.setEnabled(False)
    toggle.setToolTip("需要先运行 MCMC")
    toggle.toggled.connect(on_bands_toggled)
    selector = QComboBox(pane)
    selector.setObjectName("sldAlignSelector")
    selector.addItems(("基底界面", "表面界面"))
    selector.setEnabled(False)
    selector.currentIndexChanged.connect(on_align_changed)
    controls.addWidget(toggle)
    controls.addWidget(selector)
    controls.addStretch(1)
    return toggle, selector, controls


def projection_bands(result: object | None) -> object | None:
    uncertainty = None if result is None else result.uncertainty
    return None if uncertainty is None else uncertainty.sld_bands


def projection_mcmc(result: object | None) -> object | None:
    uncertainty = None if result is None else result.uncertainty
    return None if uncertainty is None else uncertainty.mcmc


def candidate_for_result(result: object | None, candidate_id: str | None) -> object | None:
    if result is None or candidate_id is None:
        return None
    matches = tuple(candidate for candidate in result.candidates if candidate.candidate_id == candidate_id)
    if len(matches) != 1:
        raise KeyError(f"unknown candidate: {candidate_id}")
    return matches[0]


def current_projection(
    datasets: Mapping[str, api.PreparedData],
    masks: Mapping[str, np.ndarray],
    dataset_id: str | None,
    result: object | None,
    candidate_id: str | None,
    trends: BatchTrends | None,
    visible_range: tuple[float, float] | None,
    structure: object | None,
    changes: Mapping[str, object] | None = None,
    source_issue: str | None = None,
) -> Projection:
    values: dict[str, object] = {
        "data": datasets.get(dataset_id) if dataset_id is not None else None,
        "mask": masks.get(dataset_id) if dataset_id is not None else None,
        "result": result,
        "candidate_id": candidate_id,
        "trends": trends,
        "visible_range": visible_range,
        "dataset_id": dataset_id,
        "structure": structure,
        "source_issue": source_issue,
    }
    if changes is not None:
        values.update(changes)
    return Projection(**values)


def committed_projection(
    datasets: Mapping[str, api.PreparedData],
    masks: Mapping[str, np.ndarray],
    dataset_id: str | None,
    result: object | None,
    candidate_id: str | None,
    trends: BatchTrends | None,
    visible_range: tuple[float, float] | None,
    structure: object | None,
    source_issue: str | None = None,
) -> Projection:
    data = None
    mask = None
    if dataset_id is not None and dataset_id in datasets:
        data = datasets[dataset_id]
        mask = masks[dataset_id]
    return Projection(data, mask, result, candidate_id, trends, visible_range, dataset_id, structure, source_issue)


def comparison_candidates(result: object | None, candidate_id: str | None) -> tuple[object, ...]:
    if result is None or candidate_id is None:
        return ()
    return tuple(candidate for candidate in result.candidates if candidate.candidate_id != candidate_id)


def alignment_index_from_bands(bands: object | None, surface_label: str) -> int:
    label = None if bands is None else getattr(bands, "align_label", None)
    return 1 if label == surface_label else 0


def alignment_index_from_cache(
    cache: SldBandReplay | None,
    persisted: object | None,
    surface_label: str,
) -> int:
    if cache is not None:
        return ALIGN_KEYS.index(cache.alignment)
    return alignment_index_from_bands(persisted, surface_label)


def reset_band_view(
    persisted: object | None,
    selector: QComboBox,
) -> None:
    set_alignment_index(selector, alignment_index_from_bands(persisted, selector.itemText(1)))


def check_new_bands(toggle: QCheckBox, previous: object | None, incoming: object | None) -> None:
    """一组新的带到来时勾上开关——只在这一刻勾，此后勾不勾由读者决定。

    按带对象本身判「新」：同一份结果反复投影（换步骤、换视图都会）带是同一个对象，读者在结果页
    取消的勾选就留着；没有 MCMC 报告、只有自助抽样带的结果也算到来。
    """
    if incoming is None or incoming is previous:
        return
    toggle.blockSignals(True)
    toggle.setChecked(True)
    toggle.blockSignals(False)


def set_alignment_index(selector: QComboBox, index: int) -> None:
    if selector.currentIndex() == index:
        return
    selector.blockSignals(True)
    selector.setCurrentIndex(index)
    selector.blockSignals(False)


def capture_sld_view_state(
    cache: SldBandReplay | None,
    toggle: QCheckBox,
    selector: QComboBox,
) -> SldViewState:
    return SldViewState(
        cache,
        selector.currentIndex(),
        toggle.isEnabled(),
        toggle.isChecked(),
        toggle.toolTip(),
        selector.isEnabled(),
    )


def restore_sld_view_state(
    state: SldViewState,
    toggle: QCheckBox,
    selector: QComboBox,
) -> SldBandReplay | None:
    set_alignment_index(selector, state.alignment_index)
    toggle.blockSignals(True)
    toggle.setEnabled(state.toggle_enabled)
    toggle.setChecked(state.toggle_checked)
    toggle.setToolTip(state.toggle_tooltip)
    toggle.blockSignals(False)
    selector.setEnabled(state.selector_enabled)
    return state.cache


def cache_matches(
    cache: SldBandReplay | None,
    dataset_id: str | None,
    report: object | None,
    alignment: str,
) -> bool:
    return (
        cache is not None and cache.dataset_id == dataset_id and cache.report is report and cache.alignment == alignment
    )


def visible_bands(
    *,
    checked: bool,
    cache: SldBandReplay | None,
    persisted: object | None,
    dataset_id: str | None,
    report: object | None,
    alignment: str,
    surface_label: str,
) -> object | None:
    if not checked:
        return None
    if cache_matches(cache, dataset_id, report, alignment):
        return cache.bands
    if ALIGN_KEYS[alignment_index_from_bands(persisted, surface_label)] == alignment:
        return persisted
    return None


# 带是围着拟合结果算的；结构与参数两步画的是读者正在编辑的标称剖面，把它叠上去就成了「我刚改的
# 这套结构有多确定」——那不是这条带回答的问题。开关在那两步灰掉，并说出为什么。
BANDS_NEED_MCMC_TIP = "需要先运行 MCMC"
BANDS_FOLLOW_FIT_TIP = "不确定度带以拟合结果为中心，结构编辑时画的是标称剖面，回到结果页查看"


def sync_band_controls(
    toggle: QCheckBox,
    selector: QComboBox,
    *,
    bands: object | None,
    has_structure: bool,
    drawable: bool = True,
) -> None:
    """只管能不能点；勾不勾由读者决定，新报告到来时由面板勾上一次（``reset_band_view``）。

    此前每次从灰变亮都替读者勾一次，于是在结果页取消勾选、去结构页再回来，带又自己回来了。
    """
    enabled = bands is not None and drawable
    toggle.setEnabled(enabled)
    toggle.setToolTip("" if enabled else BANDS_FOLLOW_FIT_TIP if bands is not None else BANDS_NEED_MCMC_TIP)
    selector.setEnabled(enabled and has_structure)


def project_structure(project: api.XrrProject, dataset_id: str | None) -> object | None:
    for dataset in project.datasets:
        if dataset.dataset_id == dataset_id:
            return dataset.structure
    return None
