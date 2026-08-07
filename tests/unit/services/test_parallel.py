from __future__ import annotations

from concurrent.futures import CancelledError, Future
from threading import Event, Thread, get_ident

import pytest


def test_single_worker_runs_serially_without_constructing_executor(monkeypatch) -> None:
    from xrr_fitter.services import parallel

    monkeypatch.setattr(
        parallel,
        "ThreadPoolExecutor",
        lambda **_kwargs: pytest.fail("single-worker runner constructed an executor"),
    )
    calls: list[int] = []

    with parallel.OrderedTaskRunner(1) as runner:
        result = runner.run(
            tuple(lambda index=index: calls.append(index) or index for index in range(3))
        )

    assert result == (0, 1, 2)
    assert calls == [0, 1, 2]


def test_parallel_runner_returns_results_in_input_order() -> None:
    from xrr_fitter.services.parallel import OrderedTaskRunner

    started = tuple(Event() for _ in range(3))
    releases = tuple(Event() for _ in range(3))
    completed: list[int] = []

    def task(index: int) -> int:
        started[index].set()
        assert releases[index].wait(timeout=2.0)
        completed.append(index)
        return index * 10

    def release_in_reverse() -> None:
        assert all(event.wait(timeout=2.0) for event in started)
        for index in reversed(range(3)):
            releases[index].set()
            while index not in completed:
                pass

    controller = Thread(target=release_in_reverse)
    controller.start()
    with OrderedTaskRunner(3) as runner:
        result = runner.run(tuple(lambda index=index: task(index) for index in range(3)))
    controller.join(timeout=2.0)

    assert controller.is_alive() is False
    assert completed == [2, 1, 0]
    assert result == (0, 10, 20)


def test_completed_callback_observes_finish_order_but_return_stays_input_order() -> None:
    from xrr_fitter.services.parallel import OrderedTaskRunner

    release_slow = Event()
    completed: list[tuple[int, str]] = []
    callback_threads: list[int] = []
    caller_thread = get_ident()

    def slow() -> str:
        assert release_slow.wait(timeout=2.0)
        return "slow"

    def fast() -> str:
        return "fast"

    def observe(index: int, value: str) -> None:
        callback_threads.append(get_ident())
        completed.append((index, value))
        if index == 1:
            release_slow.set()

    with OrderedTaskRunner(2) as runner:
        values = runner.run(
            (slow, fast),
            completed=observe,
        )

    assert values == ("slow", "fast")
    assert completed == [(1, "fast"), (0, "slow")]
    assert callback_threads == [caller_thread, caller_thread]


def test_completed_callback_preserves_tasks_finished_during_submission(
    monkeypatch,
) -> None:
    from xrr_fitter.services import parallel

    hashes = (1, 2, 4, 8, 16, 32, 64, 128)

    class CompletedFuture(Future[int]):
        def __init__(self, value: int, hash_value: int) -> None:
            super().__init__()
            self._hash_value = hash_value
            self.set_result(value)

        def __hash__(self) -> int:
            return self._hash_value

    class CompletingExecutor:
        def __init__(self, **_kwargs) -> None:
            self._position = 0

        def submit(self, task) -> Future[int]:
            value = task()
            future = CompletedFuture(value, hashes[self._position])
            self._position += 1
            return future

        def shutdown(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", CompletingExecutor)
    finished: list[int] = []
    observed: list[int] = []

    def task(index: int) -> int:
        finished.append(index)
        return index

    with parallel.OrderedTaskRunner(len(hashes)) as runner:
        values = runner.run(
            tuple(lambda index=index: task(index) for index in range(len(hashes))),
            completed=lambda _index, value: observed.append(value),
        )

    assert finished == list(range(len(hashes)))
    assert observed == finished
    assert values == tuple(finished)


def test_parallel_runner_rethrows_first_exception_without_late_callbacks() -> None:
    from xrr_fitter.services.parallel import OrderedTaskRunner

    started = Event()
    rendezvous = Event()
    failed = Event()
    completed: list[tuple[int, str]] = []

    def slow_success() -> str:
        started.set()
        assert rendezvous.wait(timeout=2.0)
        return "late"

    def fast_failure() -> str:
        assert started.wait(timeout=2.0)
        failed.set()
        raise RuntimeError("failure-fast")

    def release_slow() -> None:
        assert failed.wait(timeout=2.0)
        rendezvous.set()

    controller = Thread(target=release_slow)
    controller.start()
    with OrderedTaskRunner(2) as runner:
        with pytest.raises(RuntimeError, match="failure-fast"):
            runner.run(
                (slow_success, fast_failure),
                completed=lambda index, value: completed.append((index, value)),
            )
    controller.join(timeout=2.0)

    assert controller.is_alive() is False
    assert completed == []


def test_parallel_runner_cancel_prevents_queued_task_from_starting() -> None:
    from xrr_fitter.services.parallel import OrderedTaskRunner

    started = (Event(), Event())
    release = Event()
    queued_started = Event()
    outcome: list[BaseException] = []

    def blocking(index: int) -> int:
        started[index].set()
        assert release.wait(timeout=2.0)
        return index

    def queued() -> int:
        queued_started.set()
        return 2

    runner = OrderedTaskRunner(2)

    def invoke() -> None:
        try:
            runner.run((lambda: blocking(0), lambda: blocking(1), queued))
        except BaseException as error:
            outcome.append(error)

    worker = Thread(target=invoke)
    worker.start()
    assert all(event.wait(timeout=2.0) for event in started)
    runner.cancel()
    release.set()
    worker.join(timeout=2.0)
    runner.close()

    assert worker.is_alive() is False
    assert queued_started.is_set() is False
    assert len(outcome) == 1
    assert isinstance(outcome[0], CancelledError)


def test_single_worker_cancel_prevents_next_task_from_starting() -> None:
    from xrr_fitter.services.parallel import OrderedTaskRunner

    first_started = Event()
    release_first = Event()
    second_started = Event()
    outcome: list[BaseException] = []

    def first() -> int:
        first_started.set()
        assert release_first.wait(timeout=2.0)
        return 1

    def second() -> int:
        second_started.set()
        return 2

    runner = OrderedTaskRunner(1)

    def invoke() -> None:
        try:
            runner.run((first, second))
        except BaseException as error:
            outcome.append(error)

    worker = Thread(target=invoke)
    worker.start()
    assert first_started.wait(timeout=2.0)
    runner.cancel()
    release_first.set()
    worker.join(timeout=2.0)
    runner.close()

    assert worker.is_alive() is False
    assert second_started.is_set() is False
    assert len(outcome) == 1
    assert isinstance(outcome[0], CancelledError)
