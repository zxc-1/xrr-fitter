"""Worker skip requests are scoped to searches, not analysis or MCMC."""

from __future__ import annotations

from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest
from tests.support.model_cases import dataset_project, project
from tests.unit.fit.test_resume import _problem

from xrr_fitter.fit.local_search import StageSkipped
from xrr_fitter.model.analysis import ConfidenceClass, FitResult, McmcConfig
from xrr_fitter.services import fitting, workers
from xrr_fitter.services.fitting_phases.common import PreparedDatasetFit


def test_paused_skip_is_consumed_once_without_resuming_next_stage() -> None:
    cancellation, pause, skip = Event(), Event(), Event()
    pause.set()
    skip.set()
    waits = []

    def resume(seconds):
        waits.append(seconds)
        pause.clear()

    probe = fitting.pause_aware_probe(cancellation, pause, skip, sleep=resume)
    with pytest.raises(StageSkipped):
        probe()
    assert not skip.is_set()
    assert pause.is_set()
    assert waits == []
    assert probe() is False
    assert waits == [fitting.PAUSE_POLL_SECONDS]


def test_worker_probe_discards_skip_outside_search() -> None:
    cancellation, skip = Event(), Event()
    skip.set()
    probe = workers._probe(cancellation, None, skip)
    assert probe() is False
    assert not skip.is_set()
    cancellation.set()
    assert probe() is True


@pytest.mark.parametrize("automatic", [False, True])
def test_service_search_consumes_skip_but_analysis_and_recovery_do_not(monkeypatch, automatic) -> None:
    problem = _problem(seed=811)
    prepared = PreparedDatasetFit("curve", 0, dataset_project(), problem)
    cancellation, skip = Event(), Event()
    probe = workers._probe(cancellation, None, skip)
    phases = []

    def checkpoint(value):
        if value.stage == "B":
            skip.set()

    def recovery(_problem, _candidate, *, cancelled):
        skip.set()
        assert cancelled() is False
        phases.append("recovery")
        return None

    def analysis(request, *, cancelled, **_kwargs):
        skip.set()
        assert cancelled() is False
        phases.append("analysis")
        return FitResult.from_search(request.search_result, confidence=ConfidenceClass.UNTRUSTED, uncertainty=None)

    monkeypatch.setattr(fitting, "recover_profile_basin", recovery)
    monkeypatch.setattr(fitting, "run_analysis", analysis)
    monkeypatch.setattr(
        fitting,
        "assess_automatic_quality",
        lambda *_args: SimpleNamespace(
            passed=True, reasons=(), search_upgrade=False, absorption_names=(), profile_names=()
        ),
    )
    run = fitting.fit_automatic_prepared_dataset if automatic else fitting.fit_prepared_dataset
    value = run(prepared, cancelled=probe, checkpoint=checkpoint)
    result = value.fit_result if automatic else value
    assert result.skipped_stages == ("C",)
    assert phases == (["analysis", "analysis"] if automatic else ["recovery", "analysis"])
    assert not skip.is_set()


def test_stale_skip_does_not_leak_into_next_dataset_search(monkeypatch) -> None:
    problem = _problem(seed=811)
    prepared = PreparedDatasetFit("curve", 0, dataset_project(), problem)
    cancellation, skip = Event(), Event()
    probe = workers._probe(cancellation, None, skip)
    skip.set()

    def analysis(request, **_kwargs):
        skip.set()  # A late request after the search has already ended.
        return FitResult.from_search(request.search_result, confidence=ConfidenceClass.UNTRUSTED, uncertainty=None)

    monkeypatch.setattr(fitting, "recover_profile_basin", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fitting, "run_analysis", analysis)
    for _ in range(2):
        result = fitting.fit_prepared_dataset(prepared, cancelled=probe)
        assert result.skipped_stages == ()


def test_mcmc_worker_ignores_search_skip_request(monkeypatch) -> None:
    cancellation, skip = Event(), Event()
    skip.set()
    value = project()
    queue = Queue()

    def handler(_project, _dataset, _candidate, _config, _progress, cancelled):
        assert cancelled() is False
        return value

    monkeypatch.setattr(workers, "mcmc_worker_handler", handler)
    request = workers._McmcJobRequest(value, "curve", "E-0", McmcConfig(16, 10, 20))
    workers._run_mcmc_worker(request, queue, cancellation, skip=skip)
    events = list(queue.queue)
    assert [kind for kind, _value in events] == ["mcmc_result", "stopped"]
