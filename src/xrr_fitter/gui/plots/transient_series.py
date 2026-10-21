"""Transient series of the plot panel: the live search preview and the dataset overlay.

两样都画在已提交的投影之上、却不属于它：搜索中的模型一秒刷新很多次，叠加对比画的是别的数据集。它们不走
投影事务，每次提交之后由面板重新铺一遍，并共用卡片上那行动态色标（``series_key``）。这一块从
:mod:`xrr_fitter.gui.plots.panel` 分出来，面板模块只留投影与绘制事务，复杂度门禁才能各自衡量。

:class:`TransientSeriesMixin` 混入 ``PlotPanel``：运行时 ``self`` 就是面板。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.reflectivity import preview_display_values
from xrr_fitter.gui.plots.series_key import SERIES_KEY_CARDS, show_series_key

# 搜索中的模型画在这两页上：对数页按显示下限截底，原始页乘回归一化系数，与那一页的拟合模型同一把尺子。
PREVIEW_PAGES = ("log", "raw")
# 色标里那条搜索中的模型：颜色与线型照 ``LiveReflectivityPlot.set_preview`` 那一笔。
PREVIEW_KEY = ("搜索中模型", theme.DATA_PREVIEW, "solid")


class TransientSeriesMixin:
    """Draw the search preview and the dataset overlay above the committed projection."""

    #
    # Set preview curve
    #
    def set_preview_curve(self, qz_a_inv: object, model_normalized: object) -> bool:
        """Overlay the searching model on both reflectivity pages without a projection.

        A live preview updates many times per search, so it pushes into one owned
        pyqtgraph curve per page instead of running the transactional redraw. It
        carries no committed evidence and is discarded whenever a real projection
        lands.
        """
        if not self.active_data_available() or self._released:
            return False
        data = self._active_data()
        angles, values = preview_display_values(data, qz_a_inv, model_normalized)
        raw_values = np.asarray(model_normalized, dtype=float) * data.normalization
        shown = self._views["log"].set_preview(angles, values) and self._views["raw"].set_preview(angles, raw_values)
        self._sync_series_keys()
        return shown

    #
    # Clear preview curve
    #
    def clear_preview_curve(self) -> None:
        """Drop the live overlay from both pages so committed evidence renders on its own."""
        if self._released:
            return
        for key in PREVIEW_PAGES:
            self._views[key].clear_preview()
        self._sync_series_keys()

    #
    # On overlay toggled
    #
    def _on_overlay_toggled(self, _checked: bool) -> None:
        if not self._released:
            self._sync_overlay()

    #
    # Sync overlay
    #
    def _sync_overlay(self) -> None:
        """按此刻已提交的数据集重铺叠加对比：画的是当前这条之外的每一个数据集。

        此前只在勾选那一刻铺一次，换了当前数据集仍画着旧视角下的「其他曲线」，当前那条也混在里面。每条按
        它自己的显示下限截底，和它当上当前数据集时画出来的一样。
        """
        others: tuple[tuple[str, object, object], ...] = ()
        if self.toolbar.overlay_action.isChecked():
            others = tuple(
                (
                    self._dataset_names.get(dataset_id, dataset_id),
                    data.two_theta_deg,
                    np.maximum(np.asarray(data.intensity_normalized, dtype=float), data.r_floor),
                )
                for dataset_id, data in self._datasets.items()
                if dataset_id != self._dataset_id
            )
        self._views["log"].set_overlay_datasets(others)
        self._sync_series_keys()

    #
    # Sync series keys
    #
    def _sync_series_keys(self) -> None:
        """把叠加曲线与搜索中的模型写进两张反射率卡的动态色标；条目没变就不重建那一行。"""
        for key in SERIES_KEY_CARDS:
            view = self._views[key]
            entries = view.overlay_keys() + ((PREVIEW_KEY,) if view.preview_item is not None else ())
            if entries == self._series_keys.get(key):
                continue
            self._series_keys[key] = entries
            show_series_key(self.findChild(QWidget, f"plotCard:{key}SeriesKey"), entries)
