from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import monotonic

import pytest
from tests.support.automatic_recovery import (
    build_direct_sld_project,
    build_two_point_joint_project,
)
from tests.support.processes.run_analysis_worker import start as start_analysis_worker
from tests.support.processes.run_fit_worker import collect_events

import xrr_fitter.api as api

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_KINDS = {"fit_result", "mcmc_result", "cancelled", "error"}


def _fast_project() -> api.XrrProject:
    value = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    budget = replace(
        value.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=5,
        local_nfev_per_parameter=1,
        bootstrap_samples=1,
    )
    value = replace(
        value,
        fit_config=replace(
            api.FitConfig.fast(value.master_seed),
            budget=budget,
            local_workers=1,
            scale_prior_enabled=False,
        ),
    )
    dataset_id = value.datasets[0].dataset_id
    free_name = "component.0.thickness_a"
    settings = tuple(
        api.ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name == free_name else definition.initial,
            definition.upper if definition.name == free_name else definition.initial,
            locked=definition.name != free_name,
        )
        for definition in api.describe_parameters(value, dataset_id)
    )
    return api.set_parameter_settings(value, dataset_id, settings)


def _protocol_snapshot(events: tuple[api.OperationEvent, ...]) -> tuple[object, ...]:
    kinds = tuple(event.kind for event in events)
    return (
        tuple(kind for kind in kinds if kind in TERMINAL_KINDS),
        kinds[-2:],
        tuple(event.sequence for event in events),
    )


def _assert_terminal_then_stopped(events: tuple[api.OperationEvent, ...], terminal: str) -> None:
    assert _protocol_snapshot(events) == (
        (terminal,),
        (terminal, "stopped"),
        tuple(range(len(events))),
    )


def _event(events: tuple[api.OperationEvent, ...], kind: str) -> api.OperationEvent:
    return next(event for event in events if event.kind == kind)


def _checkpoints_before_terminal(
    events: tuple[api.OperationEvent, ...],
    kind: str,
) -> tuple[object, ...]:
    terminal_index = next(index for index, event in enumerate(events) if event.kind == kind)
    return tuple(event.checkpoint for event in events[:terminal_index] if event.kind == "checkpoint")


def _contains_dataset_checkpoint(snapshots) -> bool:
    return any(dataset.checkpoint is not None for snapshot in snapshots for dataset in snapshot.datasets)


def _contains_joint_checkpoint(snapshots) -> bool:
    return any(
        len(checkpoints) >= 2
        and len({checkpoint.joint_layout_fingerprint for checkpoint in checkpoints}) == 1
        and checkpoints[0].joint_layout_fingerprint
        for snapshot in snapshots
        if (checkpoints := tuple(dataset.checkpoint for dataset in snapshot.datasets if dataset.checkpoint is not None))
    )


@pytest.mark.spawn
def test_real_spawn_fit_and_mcmc_workers_publish_ordered_terminal_protocol(
    tmp_path: Path,
) -> None:
    project = _fast_project()
    checkpoint_path = tmp_path / "worker-checkpoint.xrrproj.json"
    fit_job = api.start_fit_job(project, checkpoint_path)

    started = monotonic()
    initial = fit_job.poll()
    nonblocking = monotonic() - started < 0.5
    fit_events = (*initial, *collect_events(fit_job))

    _assert_terminal_then_stopped(fit_events, "fit_result")
    assert (
        nonblocking,
        any(event.kind == "progress" for event in fit_events),
        any(event.kind == "checkpoint" for event in fit_events),
        checkpoint_path.is_file(),
        fit_job.is_running,
    ) == (True, True, True, True, False)
    fit_job.close()

    fit_result = _event(fit_events, "fit_result").fit_result
    fitted = fit_result.updated_project
    dataset = fitted.datasets[0]
    candidate = dataset.last_valid_result.best_candidate
    assert candidate is not None
    mcmc_job = start_analysis_worker(fitted, dataset.dataset_id, candidate.candidate_id)
    mcmc_events = collect_events(mcmc_job)

    _assert_terminal_then_stopped(mcmc_events, "mcmc_result")
    updated = _event(mcmc_events, "mcmc_result").mcmc_result
    assert updated.datasets[0].last_valid_result.uncertainty.mcmc is not None
    mcmc_job.close()


@pytest.mark.spawn
def test_real_spawn_automatic_worker_publishes_partial_checkpoint_before_terminal(
    tmp_path: Path,
) -> None:
    project = build_direct_sld_project(tmp_path / "source")
    batch_id = project.datasets[0].automation.import_batch_id
    checkpoint_path = tmp_path / "automatic-checkpoint.xrrproj.json"

    job = api.start_automatic_fit_job(project, batch_id, checkpoint_path)
    events = collect_events(job)

    _assert_terminal_then_stopped(events, "fit_result")
    partial = _checkpoints_before_terminal(events, "fit_result")
    assert partial
    assert _contains_dataset_checkpoint(partial)
    result = _event(events, "fit_result").fit_result
    assert result.mode == "automatic"
    assert result.updated_project.datasets[0].last_valid_result is not None
    assert checkpoint_path.is_file()
    assert job.is_running is False
    job.close()


@pytest.mark.spawn
def test_real_spawn_automatic_joint_worker_serializes_checkpoints_and_projection(
    tmp_path: Path,
) -> None:
    project = build_two_point_joint_project(tmp_path / "source")
    batch_id = project.datasets[0].automation.import_batch_id
    checkpoint_path = tmp_path / "automatic-joint-checkpoint.xrrproj.json"

    job = api.start_automatic_fit_job(project, batch_id, checkpoint_path)
    events = collect_events(job)

    _assert_terminal_then_stopped(events, "fit_result")
    partial = _checkpoints_before_terminal(events, "fit_result")
    assert _contains_joint_checkpoint(partial)
    result = _event(events, "fit_result").fit_result
    datasets = result.updated_project.datasets
    assert result.mode == "automatic"
    assert all(dataset.last_valid_result is not None for dataset in datasets)
    assert {dataset.automation.role for dataset in datasets} == {api.AutomaticRole.JOINT}
    assert len({dataset.automation.fit_group_id for dataset in datasets}) == 1
    assert checkpoint_path.is_file()
    assert job.is_running is False
    job.close()


@pytest.mark.spawn
def test_real_spawn_cooperative_cancel_and_failure_propagation() -> None:
    project = _fast_project()
    cancelled_job = api.start_fit_job(project)
    cancelled_job.cancel()
    cancelled_events = collect_events(cancelled_job)

    _assert_terminal_then_stopped(cancelled_events, "cancelled")
    assert _event(cancelled_events, "cancelled").cancellation == "requested"
    cancelled_job.close()

    failed_job = api.start_mcmc_job(
        project,
        project.datasets[0].dataset_id,
        "missing-candidate",
        api.McmcConfig(walkers=4, burn_in=0, production_steps=4),
    )
    failed_events = collect_events(failed_job)

    _assert_terminal_then_stopped(failed_events, "error")
    error = _event(failed_events, "error").error
    assert (
        error.exception_type,
        "valid uncertainty result" in error.message,
        "Traceback" in error.traceback,
    ) == ("ValueError", True, True)
    failed_job.close()
