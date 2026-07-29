"""API-backed fit-range and point-mask transactions."""

from __future__ import annotations

from math import isfinite

from PySide6.QtCore import QObject, Signal

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument


class MaskEditor(QObject):
    """Compose range and point choices, then publish one immutable API result."""

    mask_changed = Signal(str, tuple)

    def __init__(self, document: ProjectDocument, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._document = document
        self._ranges: dict[str, tuple[float, float]] = {}
        self._disabled_points: dict[str, set[int]] = {}

    def set_fit_range(self, dataset_id: str, lower: float, upper: float) -> None:
        if not isfinite(lower) or not isfinite(upper) or lower > upper:
            raise ValueError("fit range requires finite lower <= upper")
        dataset = self._dataset(dataset_id)
        data = self._prepared(dataset_id, dataset)
        mask = data.validation_mask & (data.two_theta_deg >= lower)
        mask &= data.two_theta_deg <= upper
        for index in self._disabled_points.get(dataset_id, set()):
            mask[index] = False
        if not mask.any():
            raise ValueError("fit range must retain at least one valid point")
        updated = api.set_fit_mask(self._document.project, dataset_id, mask)
        self._document.replace_project(updated)
        self._ranges[dataset_id] = (lower, upper)
        self._emit_mask(updated, dataset_id)

    def set_point_enabled(self, dataset_id: str, index: int, enabled: bool) -> None:
        dataset = self._dataset(dataset_id)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(dataset.fit_mask):
            raise IndexError("point index out of range")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        disabled = set(self._disabled_points.get(dataset_id, set()))
        disabled.discard(index) if enabled else disabled.add(index)
        data = self._prepared(dataset_id, dataset)
        lower, upper = self._ranges.get(dataset_id, dataset.fit_range_two_theta_deg)
        mask = data.validation_mask & (data.two_theta_deg >= lower)
        mask &= data.two_theta_deg <= upper
        for point in disabled:
            mask[point] = False
        updated = api.set_fit_mask(self._document.project, dataset_id, mask)
        self._document.replace_project(updated)
        self._disabled_points[dataset_id] = disabled
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
        )

    def _dataset(self, dataset_id: str) -> api.DatasetProject:
        matches = tuple(
            dataset
            for dataset in self._document.project.datasets
            if dataset.dataset_id == dataset_id
        )
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return matches[0]

    def _emit_mask(self, project: api.XrrProject, dataset_id: str) -> None:
        dataset = next(item for item in project.datasets if item.dataset_id == dataset_id)
        self.mask_changed.emit(dataset_id, dataset.fit_mask)
