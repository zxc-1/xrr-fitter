"""Qt timer projection for the public nonblocking operation-job protocol."""

from __future__ import annotations

import traceback
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal

import xrr_fitter.api as api


class FitController(QObject):
    """Poll one API-owned job without recreating its process protocol."""

    running_changed = Signal(bool)
    event_received = Signal(object)
    progress_changed = Signal(object)
    checkpoint_ready = Signal(object)
    fit_finished = Signal(object)
    mcmc_finished = Signal(object)
    cancelled = Signal(str)
    failed = Signal(object)
    stopped = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        poll_interval_ms: int = 25,
        idle_poll_interval_ms: int = 200,
        backoff_after_empty_polls: int = 8,
    ) -> None:
        super().__init__(parent)
        self._job: object | None = None
        self._base_interval_ms = poll_interval_ms
        self._idle_interval_ms = idle_poll_interval_ms
        self._backoff_after = backoff_after_empty_polls
        self._empty_polls = 0
        self._timer = QTimer(self)
        self._timer.setObjectName("operationPollTimer")
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self.poll_now)

    @property
    def timer(self) -> QTimer:
        return self._timer

    @property
    def is_running(self) -> bool:
        return self._job is not None

    def start_fit(
        self,
        project: api.XrrProject,
        checkpoint_path=None,
    ) -> bool:
        return self._begin(
            lambda: api.start_fit_job(project, checkpoint_path),
        )

    def start_automatic_fit(
        self,
        project: api.XrrProject,
        import_batch_id: str | None = None,
        checkpoint_path=None,
    ) -> bool:
        return self._begin(
            lambda: api.start_automatic_fit_job(
                project,
                import_batch_id,
                checkpoint_path,
            ),
        )

    def start_mcmc(
        self,
        project: api.XrrProject,
        dataset_id: str,
        candidate_id: str,
        config: api.McmcConfig,
    ) -> bool:
        return self._begin(
            lambda: api.start_mcmc_job(project, dataset_id, candidate_id, config),
        )

    def _begin(self, start: Callable[[], object]) -> bool:
        if self.is_running:
            raise RuntimeError("an operation is already running")
        try:
            job = start()
        except Exception as error:
            self.failed.emit(
                api.OperationError(
                    type(error).__name__,
                    str(error) or type(error).__name__,
                    traceback.format_exc(),
                )
            )
            return False
        self._job = job
        self._empty_polls = 0
        self._timer.setInterval(self._base_interval_ms)
        self._timer.start()
        self.running_changed.emit(True)
        return True

    def poll_now(self) -> tuple[api.OperationEvent, ...]:
        job = self._job
        if job is None:
            return ()
        events = tuple(job.poll())
        self._dispatch_batch(events)
        self._adjust_poll_interval(bool(events))
        return events

    def _adjust_poll_interval(self, had_events: bool) -> None:
        """Back off toward the idle interval during silence; snap back on events.

        A long fit spends most of its wall-clock producing no events, so a fixed
        fast interval burns tens of thousands of empty polls. Widening the timer
        after sustained silence cuts that waste, while any single event restores
        the responsive interval immediately so progress never lags. The timer is
        only ever reprogrammed while a job is running.
        """
        if self._job is None:
            return
        if had_events:
            self._empty_polls = 0
            if self._timer.interval() != self._base_interval_ms:
                self._timer.setInterval(self._base_interval_ms)
            return
        self._empty_polls += 1
        if self._empty_polls >= self._backoff_after and self._timer.interval() != self._idle_interval_ms:
            self._timer.setInterval(self._idle_interval_ms)

    def _dispatch_batch(self, events: tuple[api.OperationEvent, ...]) -> None:
        """Project one frame per stage while preserving durable event order.

        A single poll may carry hundreds of progress values, so consecutive
        frames of the same stage coalesce into their latest value. A stage
        change still publishes the stage it leaves behind, which keeps every
        stage visible instead of collapsing the whole batch into one frame.
        """
        pending_progress: api.FitProgress | None = None
        for event in events:
            self.event_received.emit(event)
            if event.kind == "progress":
                if pending_progress is not None and event.progress.stage != pending_progress.stage:
                    self._flush_progress(pending_progress)
                pending_progress = event.progress
                continue
            self._flush_progress(pending_progress)
            pending_progress = None
            self._dispatch_durable(event)
        self._flush_progress(pending_progress)

    def _flush_progress(self, progress: api.FitProgress | None) -> None:
        if progress is not None:
            self.progress_changed.emit(progress)

    def _dispatch_durable(self, event: api.OperationEvent) -> None:
        if event.kind == "checkpoint":
            self.checkpoint_ready.emit(event.checkpoint)
        elif event.kind == "fit_result":
            self.fit_finished.emit(event.fit_result)
        elif event.kind == "mcmc_result":
            self.mcmc_finished.emit(event.mcmc_result)
        elif event.kind == "cancelled":
            self.cancelled.emit(event.cancellation)
        elif event.kind == "error":
            self.failed.emit(event.error)
        elif event.kind == "stopped":
            self._finish()

    def _finish(self) -> None:
        job = self._job
        if job is None:
            return
        self._timer.stop()
        job.close()
        self._job = None
        self.running_changed.emit(False)
        self.stopped.emit()

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()

    def force_stop(self) -> None:
        if self._job is not None:
            self._job.force_stop()
