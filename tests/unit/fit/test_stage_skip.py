"""Real search regressions for explicit stage skips and atomic resume."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.unit.fit.test_joint_pipeline import _assert_equivalent_results as _assert_joint_equal
from tests.unit.fit.test_joint_pipeline import _joint_problem
from tests.unit.fit.test_resume import _assert_equivalent_results as _assert_single_equal
from tests.unit.fit.test_resume import _problem

from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.local_search import StageSkipped
from xrr_fitter.fit.pipeline import FitSearchRequest, run_fit_search
from xrr_fitter.fit.resume import validate_resume_checkpoint
from xrr_fitter.io.codec_results import _checkpoint_from_dict, _checkpoint_to_dict


class _SkipAfterCheckpoint:
    def __init__(self, stages: tuple[str, ...], *, joint: bool = False):
        self.stages = stages
        self.joint = joint
        self.checkpoints = []
        self.skipped = []

    def __call__(self) -> bool:
        if not self.checkpoints or len(self.skipped) == len(self.stages):
            return False
        value = self.checkpoints[-1]
        stage = value[0].stage if self.joint else value.stage
        next_stage = "E" if stage == "E" else "ABCDE"["ABCDE".index(stage) + 1]
        if next_stage == self.stages[len(self.skipped)]:
            self.skipped.append(next_stage)
            raise StageSkipped("skip stage")
        return False


def test_single_skip_keeps_every_published_checkpoint_resumable() -> None:
    problem = _problem(seed=811)
    probe = _SkipAfterCheckpoint(("C",))
    result = run_fit_search(FitSearchRequest("curve", problem), cancelled=probe, checkpoint=probe.checkpoints.append)

    assert probe.skipped == ["C"]
    assert result.best_candidate is not None
    for checkpoint in probe.checkpoints:
        validate_resume_checkpoint(problem, checkpoint, reserved_child_seeds=result.child_seeds)


def test_joint_skip_is_a_stage_transition_not_a_worker_error() -> None:
    probe = _SkipAfterCheckpoint(("C",), joint=True)
    results = run_joint_fit(JointFitRequest(_joint_problem()), cancelled=probe, checkpoint=probe.checkpoints.append)

    assert probe.skipped == ["C"]
    assert len(results) == 2
    assert all(result.best_candidate is not None for result in results)
    assert all("C" not in [summary.stage for summary in result.stage_summaries] for result in results)


@pytest.mark.parametrize("stages", [("C",), ("D",), ("E",), ("C", "D"), ("C", "E"), ("D", "E"), ("C", "D", "E")])
def test_single_skip_decision_survives_codec_and_exact_resume(stages) -> None:
    problem = _problem(seed=811)
    probe = _SkipAfterCheckpoint(stages)
    fresh = run_fit_search(FitSearchRequest("curve", problem), cancelled=probe, checkpoint=probe.checkpoints.append)

    assert tuple(probe.skipped) == stages
    assert fresh.skipped_stages == stages
    assert probe.checkpoints[-1].stage == "E"
    assert tuple(summary.stage for summary in fresh.stage_summaries) == tuple(
        stage for stage in "ABCDE" if stage not in stages
    )
    for checkpoint in probe.checkpoints:
        # A resume can retain only decisions already made at its checkpoint;
        # later user requests are not part of the persisted state.
        if checkpoint.skipped_stages != stages:
            continue
        decoded = _checkpoint_from_dict(_checkpoint_to_dict(checkpoint))
        replay = []
        resumed = run_fit_search(FitSearchRequest("curve", problem, decoded), checkpoint=replay.append)
        _assert_single_equal(fresh, resumed)
        assert resumed.skipped_stages == stages
        assert not set(checkpoint.skipped_stages).intersection(value.stage for value in replay)


def _assert_joint_resume(fresh, stages, resumed, replay):
    _assert_joint_equal(fresh, resumed)
    assert all(result.skipped_stages == stages for result in resumed)
    assert not set(stages).intersection(value[0].stage for value in replay)


@pytest.mark.parametrize("stages", [("C",), ("D",), ("E",), ("C", "D"), ("C", "E"), ("D", "E"), ("C", "D", "E")])
def test_joint_skip_decision_is_atomic_and_survives_resume(stages) -> None:
    problem = _joint_problem()
    probe = _SkipAfterCheckpoint(stages, joint=True)
    fresh = run_joint_fit(JointFitRequest(problem), cancelled=probe, checkpoint=probe.checkpoints.append)

    assert tuple(probe.skipped) == stages
    assert all(result.skipped_stages == stages for result in fresh)
    assert probe.checkpoints[-1][0].stage == "E"
    for batch in probe.checkpoints:
        assert len(batch) == 2
        assert batch[0].skipped_stages == batch[1].skipped_stages
        if batch[0].skipped_stages != stages:
            continue
        decoded = tuple(_checkpoint_from_dict(_checkpoint_to_dict(value)) for value in batch)
        replay = []
        resumed = run_joint_fit(JointFitRequest(problem, decoded), checkpoint=replay.append)
        _assert_joint_resume(fresh, stages, resumed, replay)


def test_joint_skip_preserves_committed_final_seed_prefix_without_replaying_it() -> None:
    problem = _joint_problem()
    checkpoints = []
    skipped = []

    def probe() -> bool:
        if checkpoints and checkpoints[-1][0].stage == "E" and not skipped:
            skipped.append("E")
            raise StageSkipped("skip remaining final seeds")
        return False

    fresh = run_joint_fit(JointFitRequest(problem), cancelled=probe, checkpoint=checkpoints.append)

    assert skipped == ["E"]
    assert len(checkpoints[-1][0].stage_summaries[-1].candidate_ids) == 1
    assert checkpoints[-1][0].stage_summaries == checkpoints[-2][0].stage_summaries
    assert len(checkpoints[-1][0].child_seeds) == 2
    assert all(result.skipped_stages == ("E",) for result in fresh)
    replay = []
    resumed = run_joint_fit(JointFitRequest(problem, checkpoints[-1]), checkpoint=replay.append)
    assert replay == []
    _assert_joint_equal(fresh, resumed)


@pytest.mark.parametrize("replacement", [(), ("D",), ("A", "C")])
def test_resume_rejects_missing_or_falsely_declared_skip_records(replacement) -> None:
    problem = _problem(seed=811)
    probe = _SkipAfterCheckpoint(("C",))
    result = run_fit_search(FitSearchRequest("curve", problem), cancelled=probe, checkpoint=probe.checkpoints.append)
    corrupted = replace(probe.checkpoints[-1], skipped_stages=replacement)

    with pytest.raises(ValueError, match="skip|history|stage"):
        validate_resume_checkpoint(problem, corrupted, reserved_child_seeds=result.child_seeds)


def test_joint_resume_rejects_fabricated_empty_final_summary() -> None:
    from xrr_fitter.model.fitting import FitStageSummary

    problem = _joint_problem()
    probe = _SkipAfterCheckpoint(("E",), joint=True)
    results = run_joint_fit(JointFitRequest(problem), cancelled=probe, checkpoint=probe.checkpoints.append)
    checkpoint = probe.checkpoints[-1][0]
    empty_final = FitStageSummary("E", (), float("inf"), 0, ())
    corrupted = replace(checkpoint, stage_summaries=checkpoint.stage_summaries + (empty_final,))
    with pytest.raises(ValueError, match="empty|prefix|Stage-E"):
        validate_resume_checkpoint(
            problem.problems[0],
            corrupted,
            reserved_child_seeds=results[0].child_seeds,
            expected_joint_layout_fingerprint=problem.layout_fingerprint,
        )


def test_joint_resume_rejects_individually_valid_but_inconsistent_skip_decisions() -> None:
    problem = _joint_problem()
    checkpoints = []
    run_joint_fit(JointFitRequest(problem), checkpoint=checkpoints.append)
    partial = next(batch for batch in checkpoints if batch[0].stage == "E")
    inconsistent = (partial[0], replace(partial[1], skipped_stages=("E",)))
    replay = []
    with pytest.raises(ValueError, match="mismatch|skip"):
        run_joint_fit(JointFitRequest(problem, inconsistent), checkpoint=replay.append)
    assert replay == []
