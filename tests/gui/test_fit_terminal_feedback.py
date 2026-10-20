"""Worker snapshots and terminal evidence, not callback names, own GUI claims."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
from tests.gui.fit_feedback_cases import manual_case, terminal_case, with_checkpoint
from tests.gui.test_fit_progress import _FakeJob, _project

import xrr_fitter.api as api
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.panel import FitPanel
from xrr_fitter.gui.fitting.progress import CANCELLING_TEXT, PAUSE_REQUESTED_TEXT


@contextmanager
def _running(qtbot, monkeypatch, project, *, automatic=True, import_batch_id=None):
    # Isolate the worker/preflight boundary, not the real Qt publication path.
    monkeypatch.setattr(api, "preflight_fit", lambda *_a: api.FitReadiness(True, "ready"))
    monkeypatch.setattr(api, "preflight_automatic_fit", lambda *_a: api.FitReadiness(True, "ready"))
    panel = FitPanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    job = _FakeJob()
    job.pause = lambda: None
    job.resume = lambda: None
    monkeypatch.setattr(api, "start_automatic_fit_job", lambda *_a, **_k: job)
    monkeypatch.setattr(api, "start_fit_job", lambda *_a, **_k: job)
    try:
        started = panel.start_automatic_fit(import_batch_id) if automatic else panel.start_fit()
        assert started
        yield panel, job
    finally:
        panel.controller._finish()


def _publish(panel, job, result):
    job.events = (
        api.OperationEvent(10, "fit_result", fit_result=result),
        api.OperationEvent(11, "stopped"),
    )
    panel.controller.poll_now()
    assert panel.document.project is result.updated_project
    assert not panel.is_running


def _assert_completed_progress(view):
    summary = view.status_summary()
    assert view.overall_percent() == 100
    assert summary.position == "进度 1000 / 1000"
    assert summary.remaining == "00:00"
    assert not view._cancelling
    assert not view._pause_requested
    assert not view._heartbeat.isActive()


def _assert_failed_feedback(panel, reason):
    assert panel.status_label.property("statusKind") == "error"
    assert reason in panel.status_text()
    assert panel.progress_view.overall_percent() < 100


@pytest.mark.parametrize("intermediate", (False, True))
def test_checkpoint_rollback_does_not_claim_a_resumable_checkpoint(qtbot, tmp_path, monkeypatch, intermediate):
    project = _project(tmp_path)
    with _running(qtbot, monkeypatch, project) as (panel, job):
        snapshots = (with_checkpoint(project), project) if intermediate else (project,)
        seen = []
        panel.checkpoint_published.connect(seen.append)
        events = [api.OperationEvent(index, "checkpoint", checkpoint=value) for index, value in enumerate(snapshots)]
        job.events = (
            *events,
            api.OperationEvent(len(events), "cancelled", cancellation="requested"),
            api.OperationEvent(len(events) + 1, "stopped"),
        )

        panel.controller.poll_now()

        assert panel.document.project is project
        assert len(seen) == len(snapshots)
        assert all(actual is expected for actual, expected in zip(seen, snapshots, strict=True))
        assert not panel._checkpoint_saved
        assert "未产生检查点" in panel.status_text()
        assert "可从中恢复" not in panel.status_text()


def test_real_checkpoint_snapshot_keeps_positive_resume_feedback(qtbot, tmp_path, monkeypatch):
    project = _project(tmp_path)
    with _running(qtbot, monkeypatch, project) as (panel, job):
        snapshot = with_checkpoint(project)
        job.events = (
            api.OperationEvent(0, "checkpoint", checkpoint=snapshot),
            api.OperationEvent(1, "cancelled", cancellation="requested"),
            api.OperationEvent(2, "stopped"),
        )

        panel.controller.poll_now()

        assert panel.document.project is snapshot
        assert panel._checkpoint_saved
        assert "已保存检查点，可从中恢复" in panel.status_text()


@pytest.mark.parametrize(
    "states,kind,title",
    (
        ((api.AutomaticStatus.FAILED, api.AutomaticStatus.FAILED), "error", "拟合失败"),
        ((api.AutomaticStatus.PASSED, api.AutomaticStatus.FAILED), "warn", "部分"),
        ((api.AutomaticStatus.REVIEW,), "warn", "需复核"),
    ),
)
def test_terminal_feedback_preserves_failure_and_review_reasons(qtbot, tmp_path, monkeypatch, states, kind, title):
    project, result = terminal_case(tmp_path, states)
    with _running(qtbot, monkeypatch, project) as (panel, job):
        observed = []
        panel.result_published.connect(observed.append)

        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") == kind
        assert title in panel.status_text()
        for dataset in result.updated_project.datasets:
            if dataset.automation.reason:
                assert dataset.automation.reason in panel.status_text()
        assert len(observed) == 1 and observed[0] is result
        if api.AutomaticStatus.FAILED in states:
            assert panel.progress_view.overall_percent() < 100
        else:
            _assert_completed_progress(panel.progress_view)


def test_trusted_result_can_still_require_automatic_review(qtbot, tmp_path, monkeypatch):
    project, result = terminal_case(tmp_path, (api.AutomaticStatus.REVIEW,))
    with _running(qtbot, monkeypatch, project) as (panel, job):
        assert result.datasets[0].fit_result.confidence is api.ConfidenceClass.TRUSTED
        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") == "warn"
        assert panel.document.project.datasets[0].last_valid_result is result.datasets[0].fit_result
        _assert_completed_progress(panel.progress_view)


@pytest.mark.parametrize(
    "confidence,evidence",
    (
        (api.ConfidenceClass.CORRELATED, {"classification_evidence": ("strong parameter correlation",)}),
        (api.ConfidenceClass.MULTIPLE, {"warnings": ("equivalent candidate basins",)}),
        (api.ConfidenceClass.UNTRUSTED, {"classification_evidence": ("uncertainty incomplete",)}),
        (api.ConfidenceClass.UNTRUSTED, {}),
    ),
)
def test_manual_nontrusted_results_remain_published_with_review_feedback(
    qtbot, tmp_path, monkeypatch, confidence, evidence
):
    project, result = manual_case(tmp_path, confidence, **evidence)
    with _running(qtbot, monkeypatch, project, automatic=False) as (panel, job):
        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") == "warn"
        assert "需复核" in panel.status_text()
        reasons = (*result.datasets[0].fit_result.classification_evidence, *result.datasets[0].fit_result.warnings)
        for reason in reasons or (confidence.value,):
            assert reason in panel.status_text()
        assert panel.document.project.datasets[0].last_valid_result is result.datasets[0].fit_result
        _assert_completed_progress(panel.progress_view)


def test_winner_without_publication_does_not_claim_success(qtbot, tmp_path, monkeypatch):
    project, valid = manual_case(tmp_path, api.ConfidenceClass.TRUSTED)
    reason = "publication rejected: inconsistent member projection"
    result = replace(valid, updated_project=project, warnings=(reason,))
    with _running(qtbot, monkeypatch, project, automatic=False) as (panel, job):
        _publish(panel, job, result)

        assert result.datasets[0].fit_result.best_candidate is not None
        _assert_failed_feedback(panel, reason)


def test_failed_manual_outcome_is_not_hidden_by_a_previous_result(qtbot, tmp_path, monkeypatch):
    _initial, valid = manual_case(tmp_path, api.ConfidenceClass.TRUSTED)
    project = valid.updated_project
    reason = "no eligible candidate in this run"
    failed = api.FitResult.failed(reason, project.datasets[0].last_valid_result.region_labels.size)
    result = replace(valid, datasets=(replace(valid.datasets[0], fit_result=failed),))
    with _running(qtbot, monkeypatch, project, automatic=False) as (panel, job):
        _publish(panel, job, result)

        _assert_failed_feedback(panel, reason)


def test_failed_automatic_status_cannot_be_overridden_by_a_winner(qtbot, tmp_path, monkeypatch):
    project, valid = terminal_case(tmp_path, (api.AutomaticStatus.PASSED,))
    row = valid.updated_project.datasets[0]
    reason = "automatic publication qualification rejected"
    rejected = replace(
        row,
        automation=replace(
            row.automation,
            status=api.AutomaticStatus.FAILED,
            statistics_member=False,
            reason=reason,
        ),
    )
    result = replace(valid, updated_project=replace(valid.updated_project, datasets=(rejected,)))
    with _running(qtbot, monkeypatch, project) as (panel, job):
        _publish(panel, job, result)

        assert result.datasets[0].fit_result.best_candidate is not None
        _assert_failed_feedback(panel, reason)


def test_empty_result_event_does_not_claim_success(qtbot, tmp_path, monkeypatch):
    project, valid = terminal_case(tmp_path, (api.AutomaticStatus.PASSED,))
    result = replace(valid, datasets=(), updated_project=project)
    with _running(qtbot, monkeypatch, project) as (panel, job):
        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") != "ok"
        assert "未发布" in panel.status_text()
        assert panel.progress_view.overall_percent() < 100


def test_manual_success_does_not_reuse_historical_automatic_failure(qtbot, tmp_path, monkeypatch):
    project, whole = terminal_case(tmp_path, (api.AutomaticStatus.FAILED,))
    _initial, successful = manual_case(tmp_path / "manual", api.ConfidenceClass.TRUSTED)
    row = replace(whole.updated_project.datasets[0], last_valid_result=successful.datasets[0].fit_result)
    updated = replace(whole.updated_project, datasets=(row,))
    result = replace(successful, updated_project=updated)
    with _running(qtbot, monkeypatch, project, automatic=False) as (panel, job):
        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") == "ok"
        assert panel.status_text() == "拟合完成"
        _assert_completed_progress(panel.progress_view)


def test_published_result_warnings_remain_visible_without_becoming_failure(qtbot, tmp_path, monkeypatch):
    project, valid = manual_case(tmp_path, api.ConfidenceClass.TRUSTED)
    reason = "diagnostic resampling notice"
    result = replace(valid, warnings=(reason,))
    with _running(qtbot, monkeypatch, project, automatic=False) as (panel, job):
        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") == "warn"
        assert "拟合完成" in panel.status_text()
        assert reason in panel.status_text()
        _assert_completed_progress(panel.progress_view)


def test_unrelated_historical_failure_does_not_poison_the_selected_batch(qtbot, tmp_path, monkeypatch):
    project, whole = terminal_case(tmp_path, (api.AutomaticStatus.PASSED, api.AutomaticStatus.FAILED))
    failed = whole.updated_project.datasets[1]
    historical = replace(failed, automation=replace(failed.automation, import_batch_id="old-batch"))
    project = replace(project, datasets=(project.datasets[0], historical))
    updated = replace(whole.updated_project, datasets=(whole.updated_project.datasets[0], historical))
    result = replace(whole, datasets=(whole.datasets[0],), updated_project=updated)
    with _running(qtbot, monkeypatch, project, import_batch_id="feedback-batch") as (panel, job):
        assert tuple(panel.progress_view._member_positions) == (project.datasets[0].dataset_id,)
        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") == "ok"
        assert panel.status_text() == "拟合完成"
        assert historical.automation.reason not in panel.status_text()
        _assert_completed_progress(panel.progress_view)


def test_success_reaches_100_only_after_durable_result_publication(qtbot, tmp_path, monkeypatch):
    project, result = terminal_case(tmp_path, (api.AutomaticStatus.PASSED,))
    with _running(qtbot, monkeypatch, project) as (panel, job):
        job.events = (
            api.OperationEvent(
                0,
                "progress",
                progress=api.FitProgress(
                    "curve",
                    "finalizing",
                    1,
                    1,
                    1.0,
                    "prefit analysis completed",
                ),
            ),
        )
        panel.controller.poll_now()
        assert panel.is_running
        assert panel.progress_view.overall_percent() < 100

        _publish(panel, job, result)

        assert panel.status_label.property("statusKind") == "ok"
        _assert_completed_progress(panel.progress_view)


@pytest.mark.parametrize("pending_action", (None, "pause", "cancel"))
def test_published_success_overrides_pending_requests_and_needs_no_progress_frame(
    qtbot, tmp_path, monkeypatch, pending_action
):
    project, result = terminal_case(tmp_path, (api.AutomaticStatus.PASSED,))
    with _running(qtbot, monkeypatch, project) as (panel, job):
        if pending_action is not None:
            panel.controller.progress_changed.emit(api.FitProgress("curve", "B", 1, 10, 1.0, "search"))
            assert panel.progress_view._heartbeat.isActive()
            if pending_action == "pause":
                panel._toggle_pause()
                expected = PAUSE_REQUESTED_TEXT
            else:
                panel._request_cancel()
                expected = CANCELLING_TEXT
            assert panel.progress_view.status_summary().position == expected
        observed = []
        panel.progress_view.summary_changed.connect(observed.append)

        _publish(panel, job, result)

        summary = panel.progress_view.status_summary()
        _assert_completed_progress(panel.progress_view)
        assert observed[-1] == summary
        assert panel.progress_view.stage_label.text() != "等待开始"
