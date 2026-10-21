"""View-only SLD band controls of the plot panel.

带的显隐与对齐界面只改画面、不写项目：重算失败时退回原来的勾选与对齐，已提交的投影不动。选中组件在剖面上
的高亮（``highlight_component``）同样只是画面状态。这一块从
:mod:`xrr_fitter.gui.plots.panel` 分出来，面板模块只留投影与绘制事务，复杂度门禁才能各自衡量。

:class:`SldBandControlsMixin` 混入 ``PlotPanel``：运行时 ``self`` 就是面板，所以方法直接读它的
``_sld_band_cache``、带开关与对齐下拉框。
"""

from __future__ import annotations

from dataclasses import replace

import xrr_fitter.api as api
from xrr_fitter.gui.plots.sld import component_span_nm
from xrr_fitter.gui.plots.sld_state import (
    ALIGN_KEYS,
    Projection,
    SldBandReplay,
    alignment_index_from_cache,
    cache_matches,
    capture_sld_view_state,
    projection_bands,
    projection_mcmc,
    reset_band_view,
    restore_sld_view_state,
)


class SldBandControlsMixin:
    """Toggle and realign the persisted MCMC bands, and emphasise a picked component, as view-only state."""

    #
    # Reset sld band view
    #
    def _reset_sld_band_view(self, projection: Projection) -> None:
        """Drop a view-only replay when its dataset or MCMC owner changes."""
        self._sld_band_cache = None
        reset_band_view(projection_bands(projection.result), self.sld_align_selector)

    #
    # On bands toggled
    #
    def _on_bands_toggled(self) -> None:
        if self._released or self._dataset_id is None:
            return
        if self.sld_bands_toggle.isChecked():
            self._on_align_changed(self.sld_align_selector.currentIndex())
            return
        self._transact(self._current_projection())

    #
    # On align changed
    #
    def _on_align_changed(self, index: int) -> None:
        """Recompute the bands for the picked alignment as a view-only overlay."""
        if self._released or self._dataset_id is None or self._structure is None or not 0 <= index < len(ALIGN_KEYS):
            return
        projection = self._current_projection()
        report = projection_mcmc(projection.result)
        if report is None or not self.sld_bands_toggle.isChecked():
            return
        alignment = ALIGN_KEYS[index]
        if cache_matches(self._sld_band_cache, self._dataset_id, report, alignment):
            self._transact(projection)
            return
        previous_index = alignment_index_from_cache(
            self._sld_band_cache,
            projection_bands(projection.result),
            self.sld_align_selector.itemText(1),
        )
        previous_sld_state = capture_sld_view_state(
            self._sld_band_cache,
            self.sld_bands_toggle,
            self.sld_align_selector,
        )
        rollback_sld_state = replace(previous_sld_state, alignment_index=previous_index)
        try:
            bands = api.sld_uncertainty_bands(
                self._structure,
                report,
                wavelength_a=self._active_data().beam.effective_wavelength_a,
                align=alignment,
            )
            self._sld_band_cache = SldBandReplay(self._dataset_id, report, alignment, bands)
            self._transact(
                projection,
                rollback_projection=self._committed_projection(),
                rollback_sld_state=rollback_sld_state,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError):
            self._sld_band_cache = restore_sld_view_state(
                rollback_sld_state,
                self.sld_bands_toggle,
                self.sld_align_selector,
            )

    #
    # Highlight component
    #
    def highlight_component(self, index: int | None) -> None:
        """在 SLD 剖面上强调结构里第 ``index`` 个组件那一段；``None`` 撤掉。

        ``index`` 必须指着此刻投影的结构里的一个组件，否则抛 ``IndexError``。只是画面状态：不进投影事务、
        不写项目。高亮认的是点下时的数据集与结构，换数据集、结构一变就撤掉（``_drop_stale_highlight``）。
        """
        if index is not None and type(index) is not int:
            raise TypeError("component index must be int or None")
        if index is not None and component_span_nm(self._structure, index) is None:
            raise IndexError(f"component index {index} is outside the projected structure")
        self._highlight = None if index is None else (index, self._dataset_id, self._structure)
        if not self._released and self._dataset_id is not None:
            self._redraw_sld()

    def _highlight_for(self, projection: Projection) -> int | None:
        """这次投影该强调哪一段：高亮只认它被点下时的那个数据集与结构。"""
        if self._highlight is None:
            return None
        index, dataset_id, structure = self._highlight
        return index if (dataset_id, structure) == (projection.dataset_id, projection.structure) else None

    def _drop_stale_highlight(self) -> None:
        """数据集或结构变了，高亮就作废——撤销回原结构时它也不该自己回来。"""
        if self._highlight is not None and self._highlight[1:] != (self._dataset_id, self._structure):
            self._highlight = None

    def _redraw_sld(self) -> None:
        """只重画剖面那一张；复位基线也只动这一张的，别的图上读者缩放过的视野不跟着被改成「复位点」。"""
        self._draw_or_defer("sld", self._committed_projection())
        self._interactions.navigators()["sld"].push_baseline()
