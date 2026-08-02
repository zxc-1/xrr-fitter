"""Reusable ordered thread scheduling for independent fitting tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from threading import Lock
from typing import TypeVar


T = TypeVar("T")


class OrderedTaskRunner:
    """Run independent callables concurrently and observe them in input order."""

    def __init__(self, max_workers: int) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self._executor = (
            None
            if max_workers == 1
            else ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="xrr-fit-local",
            )
        )
        self._lock = Lock()
        self._active: tuple[Future[object], ...] = ()
        self._running = False
        self._cancel_requested = False
        self._closed = False

    def __enter__(self) -> OrderedTaskRunner:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _begin(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("ordered task runner is closed")
            if self._running:
                raise RuntimeError("ordered task runner is already running")
            self._running = True
            self._cancel_requested = False

    def _finish(self) -> None:
        with self._lock:
            self._active = ()
            self._running = False
            self._cancel_requested = False

    def _register(self, futures: list[Future[T]], future: Future[T]) -> None:
        futures.append(future)
        with self._lock:
            self._active = tuple(futures)
            cancel_requested = self._cancel_requested
        if cancel_requested:
            future.cancel()

    @staticmethod
    def _cancel_after(futures: tuple[Future[T], ...], index: int) -> None:
        for future in futures[index + 1 :]:
            future.cancel()

    def _cancel_on_failure(
        self,
        futures: tuple[Future[T], ...],
        index: int,
        completed: Future[T],
    ) -> None:
        try:
            failed = completed.exception() is not None
        except CancelledError:
            failed = False
        if failed:
            self._cancel_after(futures, index)

    def run(self, tasks: Iterable[Callable[[], T]]) -> tuple[T, ...]:
        """Execute one batch, selecting results and failures by input index."""
        values = tuple(tasks)
        if any(not callable(task) for task in values):
            raise TypeError("ordered tasks must be callable")
        self._begin()
        try:
            if self._executor is None:
                return tuple(task() for task in values)
            futures: list[Future[T]] = []
            try:
                for task in values:
                    self._register(futures, self._executor.submit(task))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            submitted = tuple(futures)
            for index, future in enumerate(submitted):
                future.add_done_callback(
                    lambda completed, index=index: self._cancel_on_failure(
                        submitted,
                        index,
                        completed,
                    )
                )
            return tuple(future.result() for future in submitted)
        finally:
            self._finish()

    def cancel(self) -> None:
        """Cancel every task in the current batch that has not started."""
        with self._lock:
            if not self._running:
                return
            self._cancel_requested = True
            futures = self._active
        for future in futures:
            future.cancel()

    def close(self) -> None:
        """Release thread resources after cancelling queued work."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = self._active
            executor = self._executor
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
