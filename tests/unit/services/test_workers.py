from __future__ import annotations

from queue import Empty
from types import SimpleNamespace

import pytest
from tests.support.model_cases import project

from xrr_fitter.services import workers


class FakeQueue:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.closed = False

    def get_nowait(self):
        if not self.messages:
            raise Empty
        return self.messages.pop(0)

    def close(self):
        self.closed = True

    def join_thread(self):
        pass


class FakeEvent:
    def __init__(self):
        self.is_set = False

    def set(self):
        self.is_set = True


class FakeProcess:
    pid = 7319

    def __init__(
        self,
        *,
        alive: bool,
        terminate_stops: bool = True,
        exitcode: int = 0,
    ):
        self.alive = alive
        self.terminate_stops = terminate_stops
        self.exitcode = exitcode
        self.joined = False
        self.terminated = False
        self.killed = False
        self.closed = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=0):
        self.joined = True

    def terminate(self):
        self.terminated = True
        if self.terminate_stops:
            self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False

    def close(self):
        self.closed = True


def test_operation_job_publishes_one_terminal_then_stopped_and_reaps() -> None:
    process = FakeProcess(alive=False)
    queue = FakeQueue((("cancelled", "requested"), ("stopped", None)))
    job = workers.OperationJob(process, queue, FakeEvent())

    events = job.poll()

    assert [(event.sequence, event.kind) for event in events] == [
        (0, "cancelled"),
        (1, "stopped"),
    ]
    assert events[0].cancellation == "requested"
    assert process.joined is True
    assert job.is_running is False
    job.close()
    assert process.closed is True
    assert queue.closed is True


def test_operation_job_reports_malformed_protocol_as_error_then_stopped() -> None:
    process = FakeProcess(alive=False)
    job = workers.OperationJob(
        process,
        FakeQueue((("unexpected", object()),)),
        FakeEvent(),
    )

    events = job.poll()

    assert [event.kind for event in events] == ["error", "stopped"]
    assert events[0].error.exception_type == "WorkerProtocolError"
    assert "unexpected" in events[0].error.message
    assert job.is_running is False


@pytest.mark.parametrize(
    "messages",
    (
        (("cancelled", "requested"),),
        (("cancelled", "requested"), ("progress", object()), ("stopped", None)),
        (("cancelled", "requested"), ("cancelled", "again"), ("stopped", None)),
    ),
)
def test_operation_job_never_publishes_success_for_an_invalid_terminal_stream(
    messages,
) -> None:
    process = FakeProcess(alive=False)
    job = workers.OperationJob(process, FakeQueue(messages), FakeEvent())

    events = job.poll()

    assert [event.kind for event in events] == ["error", "stopped"]
    assert events[0].error.exception_type == "WorkerProtocolError"
    assert job.is_running is False


def test_operation_job_rejects_a_success_terminal_from_an_abnormal_exit() -> None:
    process = FakeProcess(alive=False, exitcode=9)
    messages = (("cancelled", "requested"), ("stopped", None))
    job = workers.OperationJob(process, FakeQueue(messages), FakeEvent())

    events = job.poll()

    assert [event.kind for event in events] == ["error", "stopped"]
    assert "exit status 9" in events[0].error.message


def test_cancel_is_cooperative_and_poll_is_nonblocking_while_running() -> None:
    process = FakeProcess(alive=True)
    cancellation = FakeEvent()
    job = workers.OperationJob(process, FakeQueue(), cancellation)

    assert job.poll() == ()
    assert job.is_running is True
    job.cancel()
    assert cancellation.is_set is True
    with pytest.raises(RuntimeError, match="running"):
        job.close()


def test_force_stop_owns_process_termination_and_finishes_on_poll() -> None:
    process = FakeProcess(alive=True)
    job = workers.OperationJob(process, FakeQueue(), FakeEvent())

    job.force_stop()
    events = job.poll()

    assert process.terminated is True
    assert [event.kind for event in events] == ["cancelled", "stopped"]
    assert events[0].cancellation == "force_stop"
    assert job.is_running is False


def test_force_stop_escalates_to_kill_after_the_bounded_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(workers, "monotonic", lambda: now[0])
    process = FakeProcess(alive=True, terminate_stops=False)
    job = workers.OperationJob(process, FakeQueue(), FakeEvent())

    job.force_stop()
    assert job.poll() == ()
    now[0] += workers.FORCE_KILL_AFTER_SECONDS
    events = job.poll()

    assert process.terminated is True
    assert process.killed is True
    assert [event.kind for event in events] == ["cancelled", "stopped"]
    assert events[0].cancellation == "force_stop"


def test_spawn_failure_is_raised_and_closes_owned_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue()
    process = FakeProcess(alive=False)

    def fail_start():
        raise RuntimeError("spawn failed")

    process.start = fail_start

    def queue_factory():
        return queue

    def event_factory():
        return FakeEvent()

    def process_factory(**_kwargs):
        return process

    context = SimpleNamespace(
        Queue=queue_factory,
        Event=event_factory,
        Process=process_factory,
    )
    monkeypatch.setattr(workers, "_spawn_context", lambda: context)

    with pytest.raises(RuntimeError, match="spawn failed"):
        workers.start_fit_job(project())

    assert queue.closed is True
    assert process.closed is True


def test_automatic_job_request_keeps_import_batch_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[object, object]] = []
    sentinel = object()
    monkeypatch.setattr(
        workers,
        "_start",
        lambda target, request: (captured.append((target, request)), sentinel)[1],
    )
    value = project()

    job = workers.start_automatic_fit_job(
        value,
        import_batch_id="batch-gui",
        checkpoint_path="automatic-checkpoint.json",
    )

    assert job is sentinel
    target, request = captured[0]
    assert target is workers._run_automatic_fit_worker
    assert request.project is value
    assert request.import_batch_id == "batch-gui"
    assert request.checkpoint_path == "automatic-checkpoint.json"
