"""SLD companion pane controls and view-only band replay state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.diagnostics import COMPANION_SPEC, DiagnosticView

# The selector labels are Chinese UI text; these are the matching algorithm keys
# in the same order the combo box items are added.
ALIGN_KEYS = ("backing", "surface")
BatchTrends = tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]


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
    heading = QLabel(title, pane)
    heading.setObjectName("sldPaneHeader")
    heading.setProperty("sectionHeader", True)
    toggle, selector, controls = build_sld_band_controls(pane, on_bands_toggled, on_align_changed)
    layout = QVBoxLayout(pane)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_XS)
    layout.addWidget(heading)
    layout.addLayout(controls)
    layout.addWidget(views[key].canvas, 1)
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
) -> Projection:
    values: dict[str, object] = {
        "data": None if dataset_id is None else datasets[dataset_id],
        "mask": None if dataset_id is None else masks[dataset_id],
        "result": result,
        "candidate_id": candidate_id,
        "trends": trends,
        "visible_range": visible_range,
        "dataset_id": dataset_id,
        "structure": structure,
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
) -> Projection:
    data = None
    mask = None
    if dataset_id is not None and dataset_id in datasets:
        data = datasets[dataset_id]
        mask = masks[dataset_id]
    return Projection(data, mask, result, candidate_id, trends, visible_range, dataset_id, structure)


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


def sync_band_controls(
    toggle: QCheckBox,
    selector: QComboBox,
    *,
    bands: object | None,
    has_structure: bool,
) -> None:
    enabled = bands is not None
    was_enabled = toggle.isEnabled()
    toggle.setEnabled(enabled)
    toggle.setToolTip("" if enabled else "需要先运行 MCMC")
    selector.setEnabled(enabled and has_structure)
    if enabled and not was_enabled:
        toggle.blockSignals(True)
        toggle.setChecked(True)
        toggle.blockSignals(False)


def project_structure(project: api.XrrProject, dataset_id: str | None) -> object | None:
    for dataset in project.datasets:
        if dataset.dataset_id == dataset_id:
            return dataset.structure
    return None
