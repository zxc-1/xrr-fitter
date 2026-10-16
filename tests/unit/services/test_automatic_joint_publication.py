"""Exercise real project validation with inexpensive solver boundary fixtures."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.support.model_cases import final_fit_result, fit_candidate, project
from tests.unit.services.test_automatic_batch import RecordingAutomaticFits, _automatic_dataset, _preset

from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.model.automation import AutomaticRole, AutomaticStatus
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.services import batch
from xrr_fitter.services.fitting import AutomaticPreparedResult


def _checkpoint(result):
    return FitCheckpoint(
        data_sha256="a" * 64,
        structure_fingerprint="b" * 64,
        config_fingerprint="c" * 64,
        stage="E",
        candidates=result.candidates,
        child_seeds=result.child_seeds,
    )


class JointPublicationFits(RecordingAutomaticFits):
    def __init__(self):
        super().__init__()
        self.preparation_modes = []
        self.fitted_ids = []

    def prepare(self, value, dataset_id, seed):
        self.preparation_modes.append(value.batch_mode)
        return super().prepare(value, dataset_id, seed)

    def fit_dataset(self, prepared, *, checkpoint, **_kwargs):
        self.fitted_ids.append(prepared.dataset_id)
        candidate = fit_candidate(f"prefit-{prepared.dataset_id}")
        result = replace(final_fit_result(candidate), child_seeds=(prepared.dataset_index + 1,))
        checkpoint(_checkpoint(result))
        return AutomaticPreparedResult(prepared, result, True, None)

    def fit_joint(self, prepared, prefits, fit_group_id, *, checkpoint, **_kwargs):
        super().fit_joint(prepared, prefits, fit_group_id)
        results = tuple(
            final_fit_result(replace(fit_candidate("joint-winner", cost), ranking_objective=1.5)) for cost in (1.0, 2.0)
        )
        checkpoint(tuple(_checkpoint(result) for result in results))
        return tuple(
            AutomaticPreparedResult(item, result, True, None) for item, result in zip(prepared, results, strict=True)
        )


@pytest.fixture
def joint_case(monkeypatch):
    value = replace(
        project(_automatic_dataset("left", ("Zr",)), _automatic_dataset("right", ("Zr",))),
        batch_mode="joint",
        measurement_preset=_preset(),
    )
    records = tuple(
        SimpleNamespace(dataset_id=dataset.dataset_id, status=SimpleNamespace(value="ok")) for dataset in value.datasets
    )
    monkeypatch.setattr(batch, "inspect_sources", lambda _value: SimpleNamespace(datasets=records))
    return value, JointPublicationFits()


def _run(value, calls, *, import_batch_id=None, cancelled=None, **overrides):
    callbacks = dict(
        seed_branches=calls.seeds,
        prepare_dataset=calls.prepare,
        fit_dataset=calls.fit_dataset,
        fit_joint=calls.fit_joint,
    )
    callbacks.update(overrides)
    return batch.fit_automatic_transaction(value, import_batch_id, None, calls.checkpoint, cancelled, **callbacks)


def _result_ids(value):
    return tuple(
        None if dataset.last_valid_result is None else dataset.last_valid_result.best_candidate.candidate_id
        for dataset in value.datasets
    )


def test_explicit_joint_automatic_publishes_the_actual_joint_results(joint_case):
    value, calls = joint_case

    result = _run(value, calls)

    assert result.cancelled is False
    assert result.updated_project.batch_mode == "joint"
    assert result.updated_project.fit_config is value.fit_config
    assert calls.preparation_modes == ["joint", "joint"]
    assert calls.joint_groups == [("left", "right")]
    assert _result_ids(result.updated_project) == ("joint-winner", "joint-winner")
    assert tuple(item.dataset_id for item in result.datasets) == ("left", "right")
    assert tuple(item.fit_result.best_candidate.objective for item in result.datasets) == (1.0, 2.0)
    assert all(dataset.automation.status is AutomaticStatus.PASSED for dataset in result.updated_project.datasets)


def test_explicit_joint_checkpoints_never_publish_prefit_or_partial_results(joint_case):
    value, calls = joint_case

    _run(value, calls)

    assert calls.checkpoints
    assert all(snapshot.batch_mode == "joint" for snapshot in calls.checkpoints)
    assert all(
        _result_ids(snapshot) in {(None, None), ("joint-winner", "joint-winner")} for snapshot in calls.checkpoints
    )
    for snapshot in calls.checkpoints:
        checkpoints = tuple(dataset.checkpoint for dataset in snapshot.datasets)
        assert all(item is None for item in checkpoints) or all(item is not None for item in checkpoints)
        for item in checkpoints:
            if item is not None:
                assert item.candidates[0].candidate_id == "joint-winner"


def _assert_group_failure(result, reason):
    assert result.cancelled is False
    assert result.updated_project.batch_mode == "joint"
    assert _result_ids(result.updated_project) == (None, None)
    assert tuple(item.dataset_id for item in result.datasets) == ("left", "right")
    for dataset, item in zip(result.updated_project.datasets, result.datasets, strict=True):
        assert dataset.automation.status is AutomaticStatus.FAILED
        assert reason in dataset.automation.reason
        assert dataset.checkpoint is None
        assert item.fit_result.best_candidate is None


@pytest.mark.parametrize("phase", ["prepare_dataset", "fit_dataset", "fit_joint"])
@pytest.mark.parametrize("has_previous_results", [False, True])
def test_explicit_joint_member_failure_is_atomic(joint_case, phase, has_previous_results):
    value, calls = joint_case
    if has_previous_results:
        value = _with_old_results(value)

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{phase} failed")

    result = _run(value, calls, **{phase: fail})

    _assert_group_failure(result, f"{phase} failed")
    assert all(_result_ids(snapshot) == (None, None) for snapshot in calls.checkpoints)


def test_explicit_joint_source_failure_does_not_run_prefits(joint_case, monkeypatch):
    value, calls = joint_case
    record = SimpleNamespace(dataset_id="left", status=SimpleNamespace(value="missing"), message="source missing")
    monkeypatch.setattr(batch, "inspect_sources", lambda _value: SimpleNamespace(datasets=(record,)))

    result = _run(value, calls)

    _assert_group_failure(result, "source missing")
    assert calls.fitted_ids == []
    assert calls.joint_groups == []


def _with_old_results(value):
    old = final_fit_result(replace(fit_candidate("old-joint"), ranking_objective=1.0))
    return replace(value, datasets=tuple(replace(dataset, last_valid_result=old) for dataset in value.datasets))


@pytest.mark.parametrize("phase", ["fit_dataset", "fit_joint"])
@pytest.mark.parametrize("error_type", [InterruptedError, SearchCancelled])
def test_explicit_joint_cancel_restores_original_project(joint_case, phase, error_type):
    value, calls = joint_case
    value = _with_old_results(value)

    def cancel(*_args, **_kwargs):
        raise error_type("cancelled")

    result = _run(value, calls, **{phase: cancel})

    assert result.cancelled is True
    assert result.updated_project is value
    assert result.datasets == ()
    assert result.warnings == ()
    assert calls.checkpoints[-1] is value
    assert _result_ids(value) == ("old-joint", "old-joint")


@pytest.mark.parametrize("damage", ["short", "reversed", "lineage", "failed-member"])
def test_explicit_joint_rejects_incomplete_or_unaligned_final_batch(joint_case, damage):
    value, calls = joint_case

    def broken(*args, **kwargs):
        results = calls.fit_joint(*args, **kwargs)
        if damage == "short":
            return results[:1]
        if damage == "reversed":
            return tuple(reversed(results))
        if damage == "failed-member":
            replacement = results[1].fit_result.failed("member invalid", 4)
        else:
            replacement = final_fit_result(fit_candidate("not-joint"))
        return (results[0], replace(results[1], fit_result=replacement, passed=False, reason="invalid joint fixture"))

    result = _run(value, calls, fit_joint=broken)

    _assert_group_failure(result, "joint")
    assert all(_result_ids(snapshot) == (None, None) for snapshot in calls.checkpoints)


@pytest.mark.parametrize("scope", ["partial-batch", "different-structure"])
def test_explicit_joint_never_silently_splits_into_independent_routes(joint_case, scope):
    value, calls = joint_case
    left, right = value.datasets
    if scope == "partial-batch":
        right = replace(right, automation=replace(right.automation, import_batch_id="other-batch"))
        batch_id = "batch-1"
    else:
        right = _automatic_dataset("right", ("TaN",))
        batch_id = None
    value = replace(value, datasets=(left, right))

    result = _run(value, calls, import_batch_id=batch_id)

    _assert_group_failure(result, "joint automatic fit requires")
    assert calls.prefit_dataset_ids == set()
    assert calls.joint_groups == []


def test_explicit_joint_rejects_isolated_members_even_with_coherent_candidate_ids(joint_case):
    value, calls = joint_case

    def isolated(*args, **kwargs):
        results = calls.fit_joint(*args, **kwargs)
        prepared = results[1].prepared
        automation = replace(
            prepared.updated_dataset.automation,
            role=AutomaticRole.ISOLATED_RETRY,
            reason="prefit parameter boundary hit",
        )
        changed = SimpleNamespace(**vars(prepared))
        changed.updated_dataset = replace(prepared.updated_dataset, automation=automation)
        return (results[0], replace(results[1], prepared=changed))

    result = _run(value, calls, fit_joint=isolated)

    _assert_group_failure(result, "prefit parameter boundary hit")


def test_explicit_joint_rejects_partial_checkpoint_batch(joint_case):
    value, calls = joint_case

    def incomplete(prepared, prefits, group, *, checkpoint, **kwargs):
        checkpoint((_checkpoint(prefits[0].fit_result), None))
        return calls.fit_joint(prepared, prefits, group, checkpoint=checkpoint, **kwargs)

    result = _run(value, calls, fit_joint=incomplete)

    _assert_group_failure(result, "joint checkpoint")
    assert all(all(dataset.checkpoint is None for dataset in snapshot.datasets) for snapshot in calls.checkpoints)


def test_explicit_joint_cancel_between_prefits_and_refinement_keeps_original(joint_case):
    value, calls = joint_case
    requested = []

    def prefit(*args, **kwargs):
        result = calls.fit_dataset(*args, **kwargs)
        requested.append(True)
        return result

    result = _run(value, calls, fit_dataset=prefit, cancelled=lambda: bool(requested))

    assert result.cancelled is True
    assert result.updated_project is value
    assert result.datasets == ()
    assert calls.joint_groups == []
    assert calls.checkpoints[-1] is value
