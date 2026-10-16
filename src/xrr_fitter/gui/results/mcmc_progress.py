"""Live MCMC telemetry; a finished sampling step is not a published result."""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class McmcProgressView(QWidget):
    """Keep one operation's identity, measured steps and elapsed time visible."""

    # 对话框关掉之后状态栏是唯一能看见采样的地方：每次读数更新都把「步数 · 接受率 · 预计剩余」
    # 这一句发出去，停了发空串。与拟合卡 ``summary_changed`` 同一个用意，状态栏不必自己再估一遍。
    summary_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(parent)
        self.setObjectName("mcmcProgress")
        self.setAccessibleName("MCMC 采样进度")
        self._clock = clock
        self._started: float | None = None
        self._active = False
        self._stopping = False
        self._dataset_id: str | None = None
        self._sample: tuple[float, int, int] | None = None
        self._finish_time: float | None = None
        self._sampling_done = False
        self.identity = QLabel()
        self.identity.setObjectName("mcmcProgressIdentity")
        self.identity.setWordWrap(True)
        self.details = QLabel()
        self.details.setObjectName("mcmcProgressDetails")
        self.details.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setObjectName("mcmcProgressBar")
        self.bar.setAccessibleName("MCMC 已完成采样步数")
        self.timing = QLabel()
        self.timing.setObjectName("mcmcProgressTiming")
        self.timing.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        for widget in (self.identity, self.details, self.bar, self.timing):
            layout.addWidget(widget)
        self._heartbeat = QTimer(self)
        self._heartbeat.setObjectName("mcmcProgressHeartbeat")
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._tick)
        self.hide()

    def begin(self, dataset_id: str, candidate_id: str, config: api.McmcConfig) -> None:
        self._dataset_id = dataset_id
        self._started = self._clock()
        self._active = True
        self._stopping = False
        self._sample = None
        self._finish_time = None
        self._sampling_done = False
        self.identity.setText(f"{dataset_id} · 候选 {candidate_id} · {config.walkers} walkers")
        self.details.setText(f"准备采样 · 0 / {config.burn_in + config.production_steps} 步 · 接受率 —")
        self.bar.setRange(0, 0)
        self._tick()
        self._heartbeat.start()
        self.show()

    def set_progress(self, progress: api.FitProgress) -> None:
        if not self._active or progress.stage != "MCMC" or progress.dataset_id != self._dataset_id:
            return
        now = self._clock()
        self._update_estimate(now, progress.completed, progress.total)
        self._sampling_done = progress.total > 0 and progress.completed >= progress.total
        self.bar.setRange(0, max(1, progress.total))
        self.bar.setValue(min(progress.completed, self.bar.maximum()))
        rate = "—" if progress.acceptance_rate is None else f"{progress.acceptance_rate:.1%}"
        self.details.setText(f"采样 {progress.completed} / {progress.total} 步 · 接受率 {rate}")
        self._render_time(now)

    def _update_estimate(self, now: float, completed: int, total: int) -> None:
        previous = self._sample
        if previous is None or previous[2] != total or completed < previous[1]:
            self._sample = (now, completed, total)
            self._finish_time = None
            return
        elapsed, steps = now - previous[0], completed - previous[1]
        if elapsed > 0 and steps > 0:
            self._finish_time = now + max(0, total - completed) * elapsed / steps
            self._sample = (now, completed, total)

    def _remaining_text(self, now: float) -> str:
        if self._stopping:
            return "等待停止确认"
        if self._sampling_done:
            return "等待结果"
        if self._finish_time is None:
            return "--:--"
        # A stalled/long step makes an old estimate expire, not a finished job.
        if self._finish_time <= now:
            return "估算中"
        return _duration(self._finish_time - now)

    def _render_time(self, now: float) -> None:
        if self._started is not None:
            remaining = self._remaining_text(now)
            self.timing.setText(f"已用 {_duration(now - self._started)} · 预计剩余 {remaining}")
            self.summary_changed.emit(f"{self.details.text()} · 预计剩余 {remaining}")

    def _tick(self) -> None:
        if self._active:
            self._render_time(self._clock())

    def elapsed_seconds(self) -> float:
        """How long the live chain has run; 0 when none is running."""
        if not self._active or self._started is None:
            return 0.0
        return self._clock() - self._started

    def mark_stopping(self) -> None:
        self._stopping = True
        self._tick()

    def freeze(self) -> None:
        if self._active and self._started is not None:
            self.timing.setText(f"已用 {_duration(self._clock() - self._started)} · 已结束")
            self.summary_changed.emit("")
        self._active = False
        self._heartbeat.stop()
