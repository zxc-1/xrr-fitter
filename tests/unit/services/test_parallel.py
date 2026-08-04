from __future__ import annotations

from concurrent.futures import CancelledError
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


def test_parallel_runner_propagates_lowest_index_exception() -> None:
    from xrr_fitter.services.parallel import OrderedTaskRunner

    releases = (Event(), Event())
    failed = (Event(), Event())

    def task(index: int) -> None:
        assert releases[index].wait(timeout=2.0)
        failed[index].set()
        raise RuntimeError(f"failure-{index}")

    def fail_higher_index_first() -> None:
        releases[1].set()
        assert failed[1].wait(timeout=2.0)
        releases[0].set()

    controller = Thread(target=fail_higher_index_first)
    controller.start()
    with OrderedTaskRunner(2) as runner:
        with pytest.raises(RuntimeError, match="failure-0"):
            runner.run((lambda: task(0), lambda: task(1)))
    controller.join(timeout=2.0)

    assert controller.is_alive() is False


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
