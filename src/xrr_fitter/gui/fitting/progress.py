"""Compact visible projection of immutable fit-progress values."""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

import xrr_fitter.api as api


PROGRESS_RESOLUTION = 1000

# Below this completed fraction the elapsed time is too short a lever to
# extrapolate an honest remaining estimate, so the view shows a placeholder
# instead of a wild guess during the fast opening stage.
ETA_MIN_FRACTION = 0.02

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


def joint_layout_text(layout: object) -> str:
    """Name the coupled datasets and shared parameters of a joint run.

    A joint fit optimises shared parameters across several datasets at once, so
    its progress frames carry no single owning dataset. Stating the participants
    and the shared-parameter groups up front turns an anonymous "联合拟合" into a
    run the user can actually attribute.
    """
    datasets = "、".join(layout.dataset_ids)
    if layout.shared_parameters:
        keys = "、".join(rule.sharing_key for rule in layout.shared_parameters)
        shared = f"共享参数：{keys}"
    else:
        shared = "无显式共享参数"
    return f"联合拟合 · 数据集：{datasets} · {shared}"


def _stage_percent_text(completed: int, total: int) -> str:
    """Render the stage-local count with its own percentage for a live feel.

    The global bar deliberately cannot rewind across stages, so a slow stage
    with many seeds can look stalled even while it advances. Showing the
    stage-local percent next to the raw count gives that motion a visible home.
    """
    if total <= 0:
        return f"{completed}/{total}"
    percent = min(100, max(0, round(100 * completed / total)))
    return f"{completed}/{total} ({percent}%)"


def _format_duration(seconds: float) -> str:
    """Render a non-negative duration as MM:SS, or H:MM:SS past an hour."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# Refresh cadence of the elapsed/remaining clock while the worker is silent. A
# long fit spends most stages emitting no progress frames, so an idle screen
# reads as frozen; ticking the clock once a second proves the run is alive.
HEARTBEAT_INTERVAL_MS = 1000


class _RemainingEstimator:
    """Project the remaining time from the recent completion rate.

    The cumulative average rate lags badly when stages carry uneven real cost:
    the fast opening stage makes a slow later stage look nearly done, so a naive
    ``elapsed * (1 - frac) / frac`` estimate lurches at every stage boundary.
    Smoothing the instantaneous rate with an exponential moving average tracks
    the stage actually running, and freezing the projected finish in elapsed
    terms lets the estimate count down honestly during the silent gaps between
    frames instead of standing still.
    """

    def __init__(self, *, smoothing: float = 0.35) -> None:
        self._smoothing = smoothing
        self._rate: float | None = None
        self._previous: tuple[float, float] | None = None
        self._finish_elapsed: float | None = None

    def reset(self) -> None:
        self._rate = None
        self._previous = None
        self._finish_elapsed = None

    def update(self, elapsed: float, fraction: float) -> None:
        previous = self._previous
        self._previous = (elapsed, fraction)
        if previous is None:
            return
        delta_elapsed = elapsed - previous[0]
        delta_fraction = fraction - previous[1]
        if delta_elapsed <= 0.0 or delta_fraction <= 0.0:
            return
        rate = delta_fraction / delta_elapsed
        if self._rate is None:
            self._rate = rate
        else:
            self._rate += self._smoothing * (rate - self._rate)
        self._finish_elapsed = elapsed + (1.0 - fraction) / self._rate

    def remaining(self, elapsed: float) -> float | None:
        if self._finish_elapsed is None:
            return None
        return max(0.0, self._finish_elapsed - elapsed)


class ProgressView(QWidget):
    """Show stage identity, completion, objective, and worker message."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        heartbeat_interval_ms: int = HEARTBEAT_INTERVAL_MS,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("fitProgressView")
        self._clock = clock
        self._floor = 0
        self._started: float | None = None
        self._cancelling = False
        self._estimator = _RemainingEstimator()
        self._heartbeat = QTimer(self)
        self._heartbeat.setObjectName("fitProgressHeartbeat")
        self._heartbeat.setInterval(heartbeat_interval_ms)
        self._heartbeat.timeout.connect(self._tick)
        self.joint_label = QLabel("")
        self.joint_label.setObjectName("fitProgressJointLayout")
        self.joint_label.setWordWrap(True)
        self.joint_label.hide()
        self.stage_label = QLabel("等待开始")
        self.stage_label.setObjectName("fitProgressStage")
        self.bar = QProgressBar()
        self.bar.setObjectName("fitProgressBar")
        self.bar.setRange(0, PROGRESS_RESOLUTION)
        self.bar.setValue(0)
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("fitProgressMeta")
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("fitProgressDetail")
        self.detail_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.joint_label)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.meta_label)
        layout.addWidget(self.detail_label)

    def set_joint_layout(self, layout: object | None) -> None:
        """Show a joint run's datasets and shared parameters, or hide the banner.

        Passing ``None`` (an independent fit) clears the banner so stale member
        names never linger from a previous joint run.
        """
        if layout is None:
            self.joint_label.clear()
            self.joint_label.hide()
            return
        self.joint_label.setText(joint_layout_text(layout))
        self.joint_label.show()

    def set_progress(self, progress: api.FitProgress) -> None:
        now = self._clock()
        if self._started is None:
            self._started = now
        elapsed = now - self._started
        owner = progress.dataset_id or "联合拟合"
        stage_index, stage_count = self._stage_ordinal(progress.stage)
        self.stage_label.setText(
            f"{owner} · 阶段 {stage_index}/{stage_count} · {stage_text(progress.stage)}"
        )
        self._advance(progress)
        self._estimator.update(elapsed, self._floor / PROGRESS_RESOLUTION)
        self._render_meta(elapsed)
        self.detail_label.setText(
            f"{progress.stage} {_stage_percent_text(progress.completed, progress.total)}"
            f" · {progress.message} · best={progress.best_objective:.12g}"
        )
        if not self._heartbeat.isActive():
            self._heartbeat.start()

    def mark_cancelling(self) -> None:
        """Acknowledge a cancel request while the worker winds a seed down."""
        self._cancelling = True
        self.meta_label.setText("正在取消，等待当前种子结束…")

    def freeze(self) -> None:
        """Stop the live clock without clearing the last rendered values."""
        self._heartbeat.stop()

    def _tick(self) -> None:
        """Advance the elapsed/remaining clock between silent progress frames."""
        if self._started is None:
            return
        self._render_meta(self._clock() - self._started)

    def _render_meta(self, elapsed: float) -> None:
        if self._cancelling:
            self.meta_label.setText("正在取消，等待当前种子结束…")
            return
        elapsed_text = f"已用 {_format_duration(elapsed)}"
        fraction = self._floor / PROGRESS_RESOLUTION
        remaining = self._estimator.remaining(elapsed)
        if fraction >= ETA_MIN_FRACTION and remaining is not None:
            self.meta_label.setText(
                f"{elapsed_text} · 预计剩余 {_format_duration(remaining)}"
            )
            return
        self.meta_label.setText(f"{elapsed_text} · 预计剩余 --:--")

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
        self._heartbeat.stop()
        self._estimator.reset()
        self._floor = 0
        self._started = None
        self._cancelling = False
        self.stage_label.setText("等待开始")
        self.bar.setRange(0, PROGRESS_RESOLUTION)
        self.bar.setValue(0)
        self.detail_label.clear()
        self.meta_label.clear()
