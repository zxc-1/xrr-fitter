"""Compact visible projection of immutable fit-progress values."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

import xrr_fitter.api as api


PROGRESS_RESOLUTION = 1000

# Fractional share of one search that each stage occupies, measured from real
# single-dataset runs. The bar advances monotonically across stage changes
# because every stage maps into its own slice of one fixed global range.
STAGE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("A", 0.06),
    ("B", 0.34),
    ("C", 0.16),
    ("D", 0.16),
    ("E", 0.16),
    ("basin-recovery", 0.02),
    ("bootstrap", 0.05),
    ("profile", 0.04),
    ("finalizing", 0.01),
)

STAGE_LABELS = {
    "A": "初筛候选",
    "B": "全局搜索",
    "C": "密度精修",
    "D": "粗糙度与仪器精修",
    "E": "最终种子搜索",
    "basin-recovery": "基态复核",
    "bootstrap": "自助抽样",
    "profile": "置信区间",
    "finalizing": "汇总",
}


def _stage_offsets() -> dict[str, tuple[float, float]]:
    offsets: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for name, weight in STAGE_WEIGHTS:
        offsets[name] = (cursor, weight)
        cursor += weight
    return offsets


STAGE_OFFSETS = _stage_offsets()


def stage_position(stage: str, completed: int, total: int) -> int | None:
    """Map one stage-local count into the fixed global progress range."""
    placement = STAGE_OFFSETS.get(stage)
    if placement is None:
        return None
    start, weight = placement
    fraction = 0.0 if total <= 0 else min(1.0, max(0.0, completed / total))
    return round((start + weight * fraction) * PROGRESS_RESOLUTION)


def stage_text(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


class ProgressView(QWidget):
    """Show stage identity, completion, objective, and worker message."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fitProgressView")
        self._floor = 0
        self.stage_label = QLabel("等待开始")
        self.stage_label.setObjectName("fitProgressStage")
        self.bar = QProgressBar()
        self.bar.setObjectName("fitProgressBar")
        self.bar.setRange(0, PROGRESS_RESOLUTION)
        self.bar.setValue(0)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("fitProgressDetail")
        self.detail_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.detail_label)

    def set_progress(self, progress: api.FitProgress) -> None:
        owner = progress.dataset_id or "联合拟合"
        stage_index, stage_count = self._stage_ordinal(progress.stage)
        self.stage_label.setText(
            f"{owner} · 阶段 {stage_index}/{stage_count} · {stage_text(progress.stage)}"
        )
        self._advance(progress)
        self.detail_label.setText(
            f"{progress.stage} {progress.completed}/{progress.total}"
            f" · {progress.message} · best={progress.best_objective:.12g}"
        )

    def _stage_ordinal(self, stage: str) -> tuple[int, int]:
        names = tuple(name for name, _weight in STAGE_WEIGHTS)
        count = len(names)
        if stage not in names:
            return count, count
        return names.index(stage) + 1, count

    def _advance(self, progress: api.FitProgress) -> None:
        position = stage_position(progress.stage, progress.completed, progress.total)
        if position is None:
            return
        # An unknown or out-of-order stage must never move the bar backwards.
        self._floor = max(self._floor, position)
        self.bar.setValue(self._floor)

    def reset(self) -> None:
        self._floor = 0
        self.stage_label.setText("等待开始")
        self.bar.setRange(0, PROGRESS_RESOLUTION)
        self.bar.setValue(0)
        self.detail_label.clear()
