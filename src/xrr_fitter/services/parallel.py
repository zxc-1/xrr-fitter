"""Reusable ordered thread scheduling for independent fitting tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
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

    def run(
        self,
        tasks: Iterable[Callable[[], T]],
        completed: Callable[[int, T], None] | None = None,
    ) -> tuple[T, ...]:
        """Execute a batch and publish successful results in finish order."""
        values = tuple(tasks)
        if any(not callable(task) for task in values):
            raise TypeError("ordered tasks must be callable")
        if completed is not None and not callable(completed):
            raise TypeError("completed callback must be callable")
        self._begin()
        try:
            if self._executor is None:
                results: list[T] = []
                for index, task in enumerate(values):
                    value = task()
                    results.append(value)
                    if completed is not None:
                        completed(index, value)
                return tuple(results)
            futures: list[Future[T]] = []
            try:
                for task in values:
                    self._register(futures, self._executor.submit(task))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            submitted = tuple(futures)
            completed_queue: Queue[tuple[int, Future[T]]] = Queue()
            positions = {future: index for index, future in enumerate(submitted)}
            for future in submitted:
                future.add_done_callback(
                    lambda completed_future, queue=completed_queue: queue.put(
                        (positions[completed_future], completed_future)
                    )
                )
            results: list[T | None] = [None] * len(submitted)
            errors: dict[int, BaseException] = {}
            for _ in submitted:
                index, future = completed_queue.get()
                try:
                    value = future.result()
                except BaseException as error:
                    errors[index] = error
                    for pending in submitted:
                        if pending is not future and not pending.done():
                            pending.cancel()
                    continue
                results[index] = value
                if completed is not None:
                    completed(index, value)
            if errors:
                raise errors[min(errors)]
            return tuple(cast(T, value) for value in results)
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
