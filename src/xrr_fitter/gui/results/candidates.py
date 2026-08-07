"""Candidate audit rows and stable-ID selection projection."""

from __future__ import annotations

from math import isfinite

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget


def candidate_is_selectable(candidate: object) -> bool:
    """Return whether a candidate may be recommended or sampled."""
    ranking = getattr(candidate, "ranking_objective", None)
    ranking_is_finite = ranking is None or isfinite(ranking)
    return bool(
        getattr(candidate, "valid", False)
        and getattr(candidate, "stop_reason", "") != "early_eliminated"
        and isfinite(getattr(candidate, "objective", float("inf")))
        and ranking_is_finite
    )


def candidate_is_archived(candidate: object) -> bool:
    ranking = getattr(candidate, "ranking_objective", None)
    return bool(
        getattr(candidate, "valid", False)
        and getattr(candidate, "stop_reason", "") == "early_eliminated"
        and isfinite(getattr(candidate, "objective", float("inf")))
        and (ranking is None or isfinite(ranking))
    )


def candidate_is_mcmc_ready(candidate: object | None, result: object | None) -> bool:
    if candidate is None or result is None or result.uncertainty is None:
        return False
    return bool(
        candidate_is_selectable(candidate)
        and len(candidate.unit_vector) > 0
        and result.uncertainty.candidate_id == candidate.candidate_id
    )


def _objective_text(candidate: object) -> str:
    local = f"局部目标值 J={candidate.objective:.12g}"
    ranking = getattr(candidate, "ranking_objective", None)
    if ranking is None:
        return local
    return f"{local} · 全局排序目标值 J={ranking:.12g}"


def _audit_state(candidate: object) -> str:
    if candidate_is_selectable(candidate):
        return "有效"
    if getattr(candidate, "stop_reason", "") == "early_eliminated":
        return "仅供检查 · 早期淘汰 · 不参与收敛统计"
    reason = getattr(candidate, "stop_reason", "unknown")
    return f"仅供检查 · 无效 · {reason}"


def candidate_line(candidate: object, *, selected: bool, recommended: bool) -> str:
    parts = [str(candidate.candidate_id), _objective_text(candidate), _audit_state(candidate)]
    if recommended and candidate_is_selectable(candidate):
        parts.append("推荐")
    if selected:
        parts.append("查看中")
    return " · ".join(parts)


def active_dataset(project: object) -> object | None:
    dataset_id = project.ui_state.active_dataset_id
    return next(
        (
            dataset
            for dataset in project.datasets
            if dataset.dataset_id == dataset_id
        ),
        None,
    )


def persisted_candidate_id(project: object, dataset_id: str) -> str | None:
    return next(
        (
            candidate_id
            for owner, candidate_id in project.ui_state.selected_candidate_ids
            if owner == dataset_id
        ),
        None,
    )


class CandidateList(QListWidget):
    """Render every retained candidate without hiding archived evidence."""

    candidate_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset_id: str | None = None
        self._result: object | None = None
        self._recommended_id: str | None = None
        self._inspection_result: object | None = None
        self._inspection_candidate_id: str | None = None
        self.setObjectName("candidateList")
        self.setAccessibleName("拟合候选解")
        self.setAccessibleDescription("用方向键在候选解之间切换，证据面板会随当前行更新")
        self.setToolTip("用方向键或点击切换候选解；证据面板随当前行更新")
        self.setWordWrap(True)
        self.currentItemChanged.connect(self._current_item_changed)

    @property
    def result(self) -> object | None:
        return self._result

    @property
    def recommended_id(self) -> str | None:
        return self._recommended_id

    def project_result(
        self,
        dataset_id: str,
        result: object,
        persisted_id: str | None,
    ) -> str | None:
        best = result.best_candidate
        self._recommended_id = None if best is None else best.candidate_id
        inspection_is_current = (
            self._dataset_id == dataset_id
            and self._inspection_result is result
            and self._inspection_candidate_id is not None
        )
        if not inspection_is_current:
            self.clear_inspection()
        self._dataset_id = dataset_id
        self._result = result
        visible_id = self._inspection_candidate_id or persisted_id or self._recommended_id
        self.load(
            result.candidates,
            selected_id=visible_id,
            recommended_id=self._recommended_id,
        )
        return visible_id

    def inspect(self, candidate_id: str) -> object:
        candidate = self.candidate(candidate_id)
        if candidate is None or self._result is None:
            raise ValueError("fit result is no longer current")
        self._inspection_result = self._result
        self._inspection_candidate_id = candidate_id
        self.load(
            self._result.candidates,
            selected_id=candidate_id,
            recommended_id=self._recommended_id,
        )
        return candidate

    def candidate(self, candidate_id: str | None) -> object | None:
        if candidate_id is None or self._result is None:
            return None
        return next(
            (
                candidate
                for candidate in self._result.candidates
                if candidate.candidate_id == candidate_id
            ),
            None,
        )

    def clear_inspection(self) -> None:
        self._inspection_result = None
        self._inspection_candidate_id = None

    def clear_projection(self) -> None:
        self._dataset_id = None
        self._result = None
        self._recommended_id = None
        self.clear_inspection()
        self.clear_candidates()

    def load(
        self,
        candidates: tuple[object, ...],
        *,
        selected_id: str | None,
        recommended_id: str | None,
    ) -> None:
        blocker = QSignalBlocker(self)
        self.clear()
        selected_row = -1
        for row, candidate in enumerate(candidates):
            item = QListWidgetItem(
                candidate_line(
                    candidate,
                    selected=candidate.candidate_id == selected_id,
                    recommended=candidate.candidate_id == recommended_id,
                )
            )
            item.setData(Qt.ItemDataRole.UserRole, candidate.candidate_id)
            self.addItem(item)
            if candidate.candidate_id == selected_id:
                selected_row = row
        self.setCurrentRow(selected_row)
        del blocker

    def clear_candidates(self) -> None:
        blocker = QSignalBlocker(self)
        self.clear()
        del blocker

    def candidate_count(self) -> int:
        return self.count()

    def candidate_text(self, row: int) -> str:
        item = self.item(row)
        if item is None:
            raise IndexError(f"candidate row out of range: {row}")
        return item.text()

    def selected_candidate_id(self) -> str | None:
        item = self.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

    def _current_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is not None:
            self.candidate_requested.emit(
                str(current.data(Qt.ItemDataRole.UserRole))
            )
