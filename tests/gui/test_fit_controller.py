from __future__ import annotations

from PySide6.QtCore import QTimer

import xrr_fitter.api as api


class _FakeJob:
    def __init__(self, *batches) -> None:
        self.batches = list(batches)
        self.cancel_calls = 0
        self.force_stop_calls = 0
        self.close_calls = 0
        self.is_running = True

    def poll(self):
        events = self.batches.pop(0) if self.batches else ()
        if any(event.kind == "stopped" for event in events):
            self.is_running = False
        return events

    def cancel(self) -> None:
        self.cancel_calls += 1

    def force_stop(self) -> None:
        self.force_stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _progress(sequence: int, completed: int) -> api.OperationEvent:
    return api.OperationEvent(
        sequence,
        "progress",
        progress=api.FitProgress(
            "curve",
            "stage-a",
            completed,
            10,
            1.0 / (completed + 1),
            f"step {completed}",
        ),
    )


def test_fit_controller_runs_job_through_qtimer_and_public_start_api(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    controller = FitController()
    job = _FakeJob()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "start_fit_job",
        lambda project, checkpoint_path=None: (
            calls.append((project, checkpoint_path)),
            job,
        )[1],
    )
    project = api.new_project()
    running: list[bool] = []
    controller.running_changed.connect(running.append)

    started = controller.start_fit(project, checkpoint_path="checkpoint.json")

    assert started is True
    assert calls == [(project, "checkpoint.json")]
    assert running == [True]
    assert controller.is_running is True
    assert isinstance(controller.timer, QTimer)
    assert controller.timer.isActive() is True


def test_fit_controller_coalesces_progress_without_reordering_durable_records(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    project = api.new_project()
    checkpoint = api.set_expert_mode(project, True)
    job = _FakeJob(
        (
            _progress(0, 1),
            _progress(1, 2),
            api.OperationEvent(2, "checkpoint", checkpoint=checkpoint),
            _progress(3, 3),
            api.OperationEvent(4, "cancelled", cancellation="requested"),
            api.OperationEvent(5, "stopped"),
        )
    )
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    controller = FitController()
    projected: list[tuple[str, object]] = []
    controller.progress_changed.connect(
        lambda value: projected.append(("progress", value.completed))
    )
    controller.checkpoint_ready.connect(
        lambda value: projected.append(("checkpoint", value))
    )
    controller.cancelled.connect(lambda value: projected.append(("cancelled", value)))

    controller.start_fit(project)
    controller.poll_now()

    assert projected == [
        ("progress", 2),
        ("checkpoint", checkpoint),
        ("progress", 3),
        ("cancelled", "requested"),
    ]
    assert job.close_calls == 1
    assert controller.is_running is False


def test_fit_controller_cancel_sets_job_event_without_terminating_worker(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    job = _FakeJob()
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    controller = FitController()
    controller.start_fit(api.new_project())

    controller.cancel()

    assert job.cancel_calls == 1
    assert job.force_stop_calls == 0
    assert controller.is_running is True


def test_fit_controller_force_stop_is_nonblocking_and_keeps_polling(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    job = _FakeJob()
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    controller = FitController()
    controller.start_fit(api.new_project())

    controller.force_stop()

    assert job.force_stop_calls == 1
    assert controller.timer.isActive() is True
    assert controller.is_running is True


def test_fit_controller_process_start_failure_does_not_emit_running_true(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    monkeypatch.setattr(
        api,
        "start_fit_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("spawn failed")),
    )
    controller = FitController()
    running: list[bool] = []
    failures: list[api.OperationError] = []
    controller.running_changed.connect(running.append)
    controller.failed.connect(failures.append)

    started = controller.start_fit(api.new_project())

    assert started is False
    assert running == []
    assert failures[0].exception_type == "RuntimeError"
    assert failures[0].message == "spawn failed"
    assert "RuntimeError: spawn failed" in failures[0].traceback


def test_fit_controller_worker_error_has_no_finish_signal(qtbot, monkeypatch) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    error = api.OperationError("ValueError", "bad fit", "traceback text")
    job = _FakeJob(
        (
            api.OperationEvent(0, "error", error=error),
            api.OperationEvent(1, "stopped"),
        )
    )
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    controller = FitController()
    failures: list[object] = []
    finished: list[object] = []
    controller.failed.connect(failures.append)
    controller.fit_finished.connect(finished.append)

    controller.start_fit(api.new_project())
    controller.poll_now()

    assert failures == [error]
    assert finished == []


def test_fit_controller_routes_mcmc_start_and_result_through_same_timer(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    project = api.new_project()
    updated = api.set_expert_mode(project, True)
    job = _FakeJob(
        (
            api.OperationEvent(0, "mcmc_result", mcmc_result=updated),
            api.OperationEvent(1, "stopped"),
        )
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "start_mcmc_job",
        lambda *args: (calls.append(args), job)[1],
    )
    controller = FitController()
    results: list[object] = []
    controller.mcmc_finished.connect(results.append)
    config = api.McmcConfig(walkers=8, burn_in=5, production_steps=20, thin=1)

    controller.start_mcmc(project, "curve", "candidate-0", config)
    controller.poll_now()

    assert calls == [(project, "curve", "candidate-0", config)]
    assert results == [updated]


def test_fit_controller_routes_automatic_start_with_batch_identity(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.fitting.controller import FitController

    project = api.new_project()
    job = _FakeJob()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "start_automatic_fit_job",
        lambda *args: (calls.append(args), job)[1],
        raising=False,
    )
    controller = FitController()

    assert controller.start_automatic_fit(project, "batch-9", "checkpoint.json") is True
    assert calls == [(project, "batch-9", "checkpoint.json")]


def _stage_progress(sequence: int, stage: str, completed: int, total: int) -> api.OperationEvent:
    return api.OperationEvent(
        sequence,
        "progress",
        progress=api.FitProgress(
            "curve",
            stage,
            completed,
            total,
            1.0 / (completed + 1),
            f"{stage} step {completed}",
        ),
    )


def test_fit_controller_keeps_one_progress_frame_per_stage_in_a_batch(
    qtbot,
    monkeypatch,
) -> None:
    """A large batch must still show every stage it covered.

    One poll can carry hundreds of Stage-A frames followed by later stages.
    Collapsing the whole batch to a single frame hides Stage A entirely, so the
    projection keeps the last frame of each stage while preserving stage order.
    """
    from xrr_fitter.gui.fitting.controller import FitController

    batch = tuple(
        _stage_progress(index, "A", index + 1, 400) for index in range(400)
    ) + (
        _stage_progress(400, "B", 1, 2),
        _stage_progress(401, "B", 2, 2),
        _stage_progress(402, "C", 1, 3),
    )
    job = _FakeJob(batch)
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    controller = FitController()
    seen: list[tuple[str, int]] = []
    controller.progress_changed.connect(
        lambda value: seen.append((value.stage, value.completed))
    )

    controller.start_fit(api.new_project())
    controller.poll_now()

    assert seen == [("A", 400), ("B", 2), ("C", 1)]


def test_fit_controller_progress_frames_survive_across_durable_events(
    qtbot,
    monkeypatch,
) -> None:
    """Stage frames before a checkpoint must not be swallowed by it."""
    from xrr_fitter.gui.fitting.controller import FitController

    project = api.new_project()
    checkpoint = api.set_expert_mode(project, True)
    job = _FakeJob(
        (
            _stage_progress(0, "A", 1, 2),
            _stage_progress(1, "A", 2, 2),
            api.OperationEvent(2, "checkpoint", checkpoint=checkpoint),
            _stage_progress(3, "B", 1, 2),
            _stage_progress(4, "B", 2, 2),
        )
    )
    monkeypatch.setattr(api, "start_fit_job", lambda *_args, **_kwargs: job)
    controller = FitController()
    ordered: list[object] = []
    controller.progress_changed.connect(
        lambda value: ordered.append((value.stage, value.completed))
    )
    controller.checkpoint_ready.connect(lambda _value: ordered.append("checkpoint"))

    controller.start_fit(project)
    controller.poll_now()

    assert ordered == [("A", 2), "checkpoint", ("B", 2)]
