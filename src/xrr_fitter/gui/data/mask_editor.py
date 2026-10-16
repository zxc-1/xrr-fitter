"""API-backed fit-range and point-mask transactions."""

from __future__ import annotations

from math import isfinite

from PySide6.QtCore import QObject, Signal

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument


def _persisted_mask(dataset: api.DatasetProject, data: api.PreparedData) -> object:
    """项目里存的 fit_mask，铺进一份与源行对齐、可写的 bool 向量。

    ``set_fit_mask`` 只收 bool ndarray，而本层不直接依赖 numpy：借预处理数据里同长度的
    ``validation_mask`` 作底板，长度对不上时赋值本身就会报错，不会悄悄截断。
    """
    mask = data.validation_mask.copy()
    mask[:] = dataset.fit_mask
    return mask


def _disabled_in_saved_range(dataset: api.DatasetProject, data: api.PreparedData) -> object:
    """已存范围里有效、却被读者亲手排除的点。

    这份记忆只从项目推：撤销回滚的是项目，所以推出来的永远是撤销之后的样子。编辑器自己记一份
    会话缓存的话，撤销掉的范围和点会在下一次编辑时被原样写回去。
    """
    lower, upper = dataset.fit_range_two_theta_deg
    inside = data.validation_mask & (data.two_theta_deg >= lower) & (data.two_theta_deg <= upper)
    return inside & ~_persisted_mask(dataset, data)


class MaskEditor(QObject):
    """Compose range and point choices, then publish one immutable API result."""

    mask_changed = Signal(str, tuple)

    def __init__(self, document: ProjectDocument, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._document = document

    def set_fit_range(self, dataset_id: str, lower: float, upper: float) -> None:
        if not isfinite(lower) or not isfinite(upper) or lower > upper:
            raise ValueError("fit range requires finite lower <= upper")
        dataset = self._dataset(dataset_id)
        data = self._prepared(dataset_id, dataset)
        mask = data.validation_mask & (data.two_theta_deg >= lower)
        mask &= data.two_theta_deg <= upper
        mask &= ~_disabled_in_saved_range(dataset, data)
        if not mask.any():
            raise ValueError("fit range must retain at least one valid point")
        self._commit(dataset_id, mask)

    def set_point_enabled(self, dataset_id: str, index: int, enabled: bool) -> None:
        dataset = self._dataset(dataset_id)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(dataset.fit_mask):
            raise IndexError("point index out of range")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        # 只改这一个点：项目的 fit_mask 就是全部状态，范围由服务按留下的点重新推出。
        mask = _persisted_mask(dataset, self._prepared(dataset_id, dataset))
        mask[index] = enabled
        self._commit(dataset_id, mask)

    def _commit(self, dataset_id: str, mask: object) -> None:
        updated = api.set_fit_mask(self._document.project, dataset_id, mask)
        self._document.replace_project(updated)
        self._emit_mask(updated, dataset_id)

    def _prepared(self, dataset_id: str, dataset: api.DatasetProject) -> api.PreparedData:
        source = self._document.source_record(dataset_id).source_identity
        if source is None:
            raise ValueError(f"source is unavailable for dataset {dataset_id}")
        return api.import_data(
            source,
            dataset.beam,
            dataset.import_angle_offset_deg,
            dataset.column_mapping,
            angle_convention=dataset.angle_convention,
            noise_model=self._document.project.fit_config.noise_model,
        )

    def _dataset(self, dataset_id: str) -> api.DatasetProject:
        matches = tuple(dataset for dataset in self._document.project.datasets if dataset.dataset_id == dataset_id)
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return matches[0]

    def _emit_mask(self, project: api.XrrProject, dataset_id: str) -> None:
        dataset = next(item for item in project.datasets if item.dataset_id == dataset_id)
        self.mask_changed.emit(dataset_id, dataset.fit_mask)
