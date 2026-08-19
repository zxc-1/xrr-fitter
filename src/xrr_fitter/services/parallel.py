"""Reusable ordered thread scheduling for independent fitting tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from queue import SimpleQueue
from threading import Lock
from typing import TypeVar, cast

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
    def _validated_tasks(
        tasks: Iterable[Callable[[], T]],
        completed: Callable[[int, T], None] | None,
    ) -> tuple[Callable[[], T], ...]:
        values = tuple(tasks)
        if any(not callable(task) for task in values):
            raise TypeError("ordered tasks must be callable")
        if completed is not None and not callable(completed):
            raise TypeError("completed callback must be callable")
        return values

    def _run_inline(
        self,
        tasks: tuple[Callable[[], T], ...],
        completed: Callable[[int, T], None] | None,
    ) -> tuple[T, ...]:
        results = []
        for index, task in enumerate(tasks):
            with self._lock:
                cancel_requested = self._cancel_requested
            if cancel_requested:
                raise CancelledError()
            value = task()
            results.append(value)
            if completed is not None:
                completed(index, value)
        return tuple(results)

    @staticmethod
    def _cancel_pending(
        futures: Iterable[Future[object]],
        completed: Future[object] | None = None,
    ) -> None:
        for future in futures:
            if future is not completed and not future.done():
                future.cancel()

    def _submit_tasks(
        self,
        tasks: tuple[Callable[[], T], ...],
        completed_positions: SimpleQueue[int],
    ) -> tuple[Future[T], ...]:
        if self._executor is None:
            raise RuntimeError("threaded execution requires an executor")
        futures: list[Future[T]] = []
        try:
            for index, task in enumerate(tasks):

                def observed_task(index=index, task=task):
                    try:
                        return task()
                    finally:
                        completed_positions.put(index)

                future = self._executor.submit(observed_task)
                future.add_done_callback(
                    lambda value, index=index: completed_positions.put(index) if value.cancelled() else None
                )
                self._register(futures, future)
        except BaseException:
            self._cancel_pending(futures)
            raise
        return tuple(futures)

    def _run_threaded(
        self,
        tasks: tuple[Callable[[], T], ...],
        completed: Callable[[int, T], None] | None,
    ) -> tuple[T, ...]:
        completed_positions: SimpleQueue[int] = SimpleQueue()
        submitted = self._submit_tasks(tasks, completed_positions)
        results: list[T | None] = [None] * len(submitted)
        for _ in submitted:
            index = completed_positions.get()
            future = submitted[index]
            try:
                value = future.result()
            except BaseException:
                self._cancel_pending(submitted, future)
                raise
            results[index] = value
            if completed is not None:
                try:
                    completed(index, value)
                except BaseException:
                    self._cancel_pending(submitted, future)
                    raise
        return tuple(cast(T, value) for value in results)

    def run(
        self,
        tasks: Iterable[Callable[[], T]],
        completed: Callable[[int, T], None] | None = None,
    ) -> tuple[T, ...]:
        """Execute a batch and publish successful results in finish order."""
        values = self._validated_tasks(tasks, completed)
        self._begin()
        try:
            if self._executor is None:
                return self._run_inline(values, completed)
            return self._run_threaded(values, completed)
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
