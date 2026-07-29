"""Compact visible projection of immutable fit-progress values."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

import xrr_fitter.api as api


class ProgressView(QWidget):
    """Show stage identity, completion, objective, and worker message."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fitProgressView")
        self.stage_label = QLabel("等待开始")
        self.stage_label.setObjectName("fitProgressStage")
        self.bar = QProgressBar()
        self.bar.setObjectName("fitProgressBar")
        self.bar.setRange(0, 1)
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
        self.stage_label.setText(f"{owner} · {progress.stage}")
        self.bar.setRange(0, progress.total)
        self.bar.setValue(progress.completed)
        self.detail_label.setText(
            f"{progress.message} · best={progress.best_objective:.12g}"
        )

    def reset(self) -> None:
        self.stage_label.setText("等待开始")
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.detail_label.clear()
