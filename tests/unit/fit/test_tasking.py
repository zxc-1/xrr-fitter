from __future__ import annotations

import pytest

from xrr_fitter.fit.tasking import run_tasks


def test_run_tasks_preserves_declared_order_without_runner() -> None:
    observed: list[int] = []

    result = run_tasks(
        (
            lambda: observed.append(1) or "first",
            lambda: observed.append(2) or "second",
        ),
        None,
    )

    assert observed == [1, 2]
    assert result == ("first", "second")


def test_run_tasks_rejects_a_runner_result_with_wrong_cardinality() -> None:
    with pytest.raises(RuntimeError, match="unexpected result count"):
        run_tasks((lambda: 1,), lambda _tasks: ())
