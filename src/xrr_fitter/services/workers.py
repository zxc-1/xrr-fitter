"""Spawn process, queue protocol, cancellation, and worker lifecycle owner."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from queue import Empty
from time import monotonic
import traceback as traceback_module

from xrr_fitter.model.analysis import McmcConfig
from xrr_fitter.model.operations import OperationError, OperationEvent
from xrr_fitter.model.project import XrrProject
from xrr_fitter.services.fitting import fit_worker_handler, mcmc_worker_handler
from xrr_fitter.services.projects import save_project


TERMINAL_KINDS = frozenset({"fit_result", "mcmc_result", "cancelled", "error"})
PAYLOAD_KINDS = frozenset({"progress", "checkpoint", *TERMINAL_KINDS, "stopped"})
FORCE_KILL_AFTER_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class _FitJobRequest:
    project: XrrProject
    checkpoint_path: str | None


@dataclass(frozen=True, slots=True)
class _McmcJobRequest:
    project: XrrProject
    dataset_id: str
    candidate_id: str
    config: McmcConfig


def _operation_error(error: BaseException) -> OperationError:
    return OperationError(
        type(error).__name__,
        str(error) or type(error).__name__,
        traceback_module.format_exc(),
    )


def _put(queue, kind: str, payload) -> None:
    queue.put((kind, payload))


def _run_fit_worker(request: _FitJobRequest, queue, cancellation) -> None:
    try:
        def progress(value) -> None:
            _put(queue, "progress", value)

        def checkpoint(value) -> None:
            if request.checkpoint_path is not None:
                save_project(value, request.checkpoint_path)
            _put(queue, "checkpoint", value)

        result = fit_worker_handler(
            request.project,
            progress,
            checkpoint,
            cancellation.is_set,
        )
        if result.cancelled:
            _put(queue, "cancelled", "requested")
        else:
            _put(queue, "fit_result", result)
    except BaseException as error:
        _put(queue, "error", _operation_error(error))
    finally:
        _put(queue, "stopped", None)


def _run_mcmc_worker(request: _McmcJobRequest, queue, cancellation) -> None:
    try:
        result = mcmc_worker_handler(
            request.project,
            request.dataset_id,
            request.candidate_id,
            request.config,
            lambda value: _put(queue, "progress", value),
            cancellation.is_set,
        )
        _put(queue, "mcmc_result", result)
    except BaseException as error:
        if type(error).__name__ in {"SearchCancelled", "InterruptedError"}:
            _put(queue, "cancelled", "requested")
        else:
            _put(queue, "error", _operation_error(error))
    finally:
        _put(queue, "stopped", None)


def _protocol_error(message: str) -> OperationError:
    return OperationError(
        "WorkerProtocolError",
        message,
        "worker protocol validation failed",
    )


def _event(sequence: int, kind: str, payload) -> OperationEvent:
    if kind == "stopped":
        return OperationEvent(sequence, kind)
    return OperationEvent(sequence, kind, **{kind if kind != "cancelled" else "cancellation": payload})


class OperationJob:
    """One concrete process job with a validated event stream."""

    def __init__(self, process, queue, cancellation) -> None:
        self._process = process
        self._queue = queue
        self._cancellation = cancellation
        self._sequence = 0
        self._pending_terminal: tuple[str, object] | None = None
        self._protocol_failed = False
        self._stop_received = False
        self._stopped = False
        self._force_started: float | None = None
        self._closed = False

    @property
    def pid(self) -> int:
        value = self._process.pid
        if not isinstance(value, int):
            raise RuntimeError("worker process has no pid")
        return value

    @property
    def is_running(self) -> bool:
        return not self._stopped

    def _append(self, events: list[OperationEvent], kind: str, payload=None) -> None:
        events.append(_event(self._sequence, kind, payload))
        self._sequence += 1

    def _fail_protocol(self, events: list[OperationEvent], message: str) -> None:
        del events
        if not self._protocol_failed:
            self._pending_terminal = ("error", _protocol_error(message))
            self._protocol_failed = True
        if self._process.is_alive():
            if self._force_started is None:
                self._force_started = monotonic()
            self._process.terminate()

    def _accept_message(self, events: list[OperationEvent], message) -> None:
        if self._protocol_failed:
            return
        if not isinstance(message, tuple) or len(message) != 2:
            self._fail_protocol(events, f"malformed worker message: {message!r}")
            return
        kind, payload = message
        if kind not in PAYLOAD_KINDS:
            self._fail_protocol(events, f"unexpected worker event kind: {kind!r}")
            return
        if kind == "stopped":
            self._accept_stopped(events, payload)
            return
        self._accept_payload(events, kind, payload)

    def _accept_stopped(self, events: list[OperationEvent], payload) -> None:
        if payload is not None or self._pending_terminal is None:
            self._fail_protocol(events, "stopped must follow one terminal event")
            return
        if self._stop_received:
            self._fail_protocol(events, "stopped event must occur exactly once")
            return
        self._stop_received = True

    def _accept_payload(self, events: list[OperationEvent], kind: str, payload) -> None:
        if self._pending_terminal is not None:
            self._fail_protocol(events, f"event after terminal: {kind}")
            return
        try:
            if kind in TERMINAL_KINDS:
                _event(self._sequence, kind, payload)
                self._pending_terminal = (kind, payload)
            else:
                self._append(events, kind, payload)
        except (TypeError, ValueError) as error:
            self._fail_protocol(events, f"invalid {kind} payload: {error}")

    def _drain(self, events: list[OperationEvent]) -> None:
        while True:
            try:
                message = self._queue.get_nowait()
            except Empty:
                return
            except (EOFError, OSError) as error:
                self._fail_protocol(events, f"worker queue failed: {error}")
                return
            self._accept_message(events, message)

    def _advance_force_stop(self) -> None:
        if self._force_started is None or not self._process.is_alive():
            return
        if monotonic() - self._force_started >= FORCE_KILL_AFTER_SECONDS:
            self._process.kill()

    def _finish_exited(self, events: list[OperationEvent]) -> None:
        if self._process.is_alive():
            return
        self._process.join(timeout=0)
        self._drain(events)
        terminal = self._terminal_after_exit()
        self._append(events, terminal[0], terminal[1])
        if not self._stopped:
            self._append(events, "stopped")
            self._stopped = True

    def _terminal_after_exit(self) -> tuple[str, object]:
        if self._protocol_failed:
            assert self._pending_terminal is not None
            return self._pending_terminal
        if self._force_started is not None and self._pending_terminal is None:
            return "cancelled", "force_stop"
        if self._pending_terminal is None:
            return (
                "error",
                _protocol_error(
                    f"worker exited without terminal event: {self._process.exitcode}"
                ),
            )
        if not self._stop_received:
            return (
                "error",
                _protocol_error(
                    f"worker exited without stopped event: {self._process.exitcode}"
                ),
            )
        if self._process.exitcode != 0:
            return (
                "error",
                _protocol_error(
                    f"worker exited with unexpected exit status {self._process.exitcode}"
                ),
            )
        return self._pending_terminal

    def poll(self) -> tuple[OperationEvent, ...]:
        """Return all currently available events without blocking."""
        if self._closed:
            raise RuntimeError("operation job is closed")
        if self._stopped:
            return ()
        events: list[OperationEvent] = []
        self._drain(events)
        self._advance_force_stop()
        self._finish_exited(events)
        return tuple(events)

    def cancel(self) -> None:
        if self.is_running:
            self._cancellation.set()

    def force_stop(self) -> None:
        if not self.is_running or self._force_started is not None:
            return
        self._cancellation.set()
        self._force_started = monotonic()
        if self._process.is_alive():
            self._process.terminate()

    def close(self) -> None:
        if self.is_running:
            raise RuntimeError("cannot close a running operation job")
        if self._closed:
            return
        self._queue.close()
        self._queue.join_thread()
        self._process.close()
        self._closed = True


def _spawn_context():
    return multiprocessing.get_context("spawn")


def _start(target, request) -> OperationJob:
    context = _spawn_context()
    queue = context.Queue()
    cancellation = context.Event()
    process = context.Process(target=target, args=(request, queue, cancellation))
    try:
        process.start()
    except BaseException:
        queue.close()
        queue.join_thread()
        try:
            process.close()
        except ValueError:
            pass
        raise
    return OperationJob(process, queue, cancellation)


def start_fit_job(
    project: XrrProject,
    checkpoint_path: str | Path | None = None,
) -> OperationJob:
    path = None if checkpoint_path is None else str(checkpoint_path)
    return _start(_run_fit_worker, _FitJobRequest(project, path))


def start_mcmc_job(
    project: XrrProject,
    dataset_id: str,
    candidate_id: str,
    config: McmcConfig,
) -> OperationJob:
    return _start(
        _run_mcmc_worker,
        _McmcJobRequest(project, dataset_id, candidate_id, config),
    )
