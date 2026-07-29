from __future__ import annotations

from time import monotonic, sleep

import xrr_fitter.api as api


def _poll_until_stopped(
    job: api.OperationJob,
    deadline: float,
    events: list[api.OperationEvent],
) -> bool:
    while monotonic() < deadline:
        current = job.poll()
        events.extend(current)
        if any(event.kind == "stopped" for event in current):
            return True
        sleep(0.01)
    return False


def _drain_after_force_stop(
    job: api.OperationJob,
    deadline: float,
    events: list[api.OperationEvent],
) -> None:
    while job.is_running and monotonic() < deadline:
        events.extend(job.poll())
        sleep(0.01)


def collect_events(
    job: api.OperationJob,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[api.OperationEvent, ...]:
    deadline = monotonic() + timeout_seconds
    events: list[api.OperationEvent] = []
    if _poll_until_stopped(job, deadline, events):
        return tuple(events)
    job.force_stop()
    _drain_after_force_stop(job, deadline + 5.0, events)
    raise TimeoutError(f"worker did not stop; events={[event.kind for event in events]}")
