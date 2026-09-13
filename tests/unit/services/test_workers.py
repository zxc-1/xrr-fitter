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


class FakeToggle:
    """A ``multiprocessing.Event`` stand-in that can be cleared as well as set."""

    def __init__(self, initial: bool = False):
        self._set = initial

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def is_set(self):
        return self._set


def test_the_pause_probe_holds_the_worker_until_it_is_resumed() -> None:
    """暂停不是新增停车点，而是让既有的取消探针在那儿等。

    求解器每到一个阶段边界就问一次「要停了吗」。暂停复用同一个问句：探针在暂停期间不
    返回，等到恢复才放行——所以不必往 ``fit``/``services`` 里加新的等待点，也就不会碰
    到那些按阶段指纹冻结的测试。
    """
    cancellation = FakeToggle()
    pause = FakeToggle(initial=True)
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        pause.clear()

    probe = workers._pause_aware_probe(cancellation, pause, sleep=sleep)

    assert probe() is False
    assert waits  # 真的等过，不是直接放行


def test_the_pause_probe_still_answers_yes_to_a_cancel_arriving_while_paused() -> None:
    """暂停中按停止，探针必须马上说「要停」——否则停止键在暂停时是死的。"""
    cancellation = FakeToggle()
    pause = FakeToggle(initial=True)

    def sleep(_seconds: float) -> None:
        cancellation.set()

    probe = workers._pause_aware_probe(cancellation, pause, sleep=sleep)

    assert probe() is True


def test_the_job_pauses_and_resumes_through_its_own_event() -> None:
    """``OperationJob`` 把暂停开关暴露出来，GUI 因此不必自己拿到那个 Event。"""
    process = FakeProcess(alive=True)
    pause = FakeToggle()
    job = workers.OperationJob(process, FakeQueue(), FakeEvent(), pause=pause)

    job.pause()
    assert job.is_paused is True

    job.resume()
    assert job.is_paused is False


def test_the_skip_probe_raises_stage_skipped_once_and_then_lets_the_run_continue() -> None:
    """⏭ 只作废当前这一个阶段，所以开关是一次性的。

    探针抛 ``StageSkipped`` 之后必须马上把开关放下：留着它，下一个阶段一开始就又被
    跳掉，读者按一次会连着丢好几层加工。
    """
    from xrr_fitter.fit.local_search import StageSkipped

    cancellation = FakeToggle()
    skip = FakeToggle(initial=True)
    probe = workers._pause_aware_probe(cancellation, None, skip)

    with pytest.raises(StageSkipped):
        probe()

    assert skip.is_set() is False
    assert probe() is False


def test_a_cancel_outranks_a_pending_skip() -> None:
    """同时按下停止与跳过时，停止说了算——跳过之后还有的跑，停止之后没有。"""
    cancellation = FakeToggle(initial=True)
    skip = FakeToggle(initial=True)
    probe = workers._probe(cancellation, None, skip)

    assert probe() is True


def test_the_job_hands_the_skip_switch_to_the_gui() -> None:
    job = workers.OperationJob(
        FakeProcess(alive=True),
        FakeQueue(),
        FakeEvent(),
        pause=FakeToggle(),
        skip=FakeToggle(),
    )

    job.skip_stage()

    assert job._skip.is_set() is True
