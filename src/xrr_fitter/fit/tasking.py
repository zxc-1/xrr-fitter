"""Small deterministic task-runner boundary shared by fit orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar


T = TypeVar("T")


class TaskRunner(Protocol):
    """Execute a declared batch and return one result for every task."""

    def __call__(
        self,
        tasks: tuple[Callable[[], T], ...],
        /,
    ) -> tuple[T, ...]: ...


def run_tasks(
    tasks: tuple[Callable[[], T], ...],
    task_runner: TaskRunner | None,
) -> tuple[T, ...]:
    """Run tasks locally or through an injected ordered runner."""
    results = tuple(task() for task in tasks) if task_runner is None else tuple(
        task_runner(tasks)
    )
    if len(results) != len(tasks):
        raise RuntimeError("task runner returned an unexpected result count")
    return results
