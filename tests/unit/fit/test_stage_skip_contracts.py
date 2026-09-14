"""Skip evidence across early termination, analysis, and persisted value boundaries."""

from __future__ import annotations

import pickle
from dataclasses import fields, replace

import pytest
from tests.support.model_cases import dataset_project, project
from tests.unit.fit.test_joint_pipeline import _fully_locked_joint_problem, _joint_problem
from tests.unit.fit.test_resume import _problem
from tests.unit.fit.test_stage_skip import _SkipAfterCheckpoint

from xrr_fitter.analysis.report import AnalysisRequest, run_analysis
from xrr_fitter.fit import pipeline
from xrr_fitter.fit.joint_pipeline import JointFitRequest, run_joint_fit
from xrr_fitter.fit.local_search import StageSkipped
from xrr_fitter.fit.pipeline import FitSearchRequest, run_fit_search
from xrr_fitter.fit.resume import validate_resume_checkpoint
from xrr_fitter.io.codec_results import _checkpoint_to_dict, fit_result_to_dict
from xrr_fitter.io.project_codec import project_from_dict, project_to_dict
from xrr_fitter.model.analysis import ConfidenceClass, FitResult
from xrr_fitter.model.provenance import _context_identity, _identity_sha256, fit_search_provenance_sha256
from xrr_fitter.services import fitting
from xrr_fitter.services.fitting_phases.common import PreparedDatasetFit


@pytest.fixture(scope="module")
def skipped_c():
    problem = _problem(seed=811)
    probe = _SkipAfterCheckpoint(("C",))
    search = run_fit_search(FitSearchRequest("curve", problem), cancelled=probe, checkpoint=probe.checkpoints.append)
    return problem, search, probe.checkpoints[-1]


@pytest.fixture(scope="module")
def skipped_e():
    problem = _problem(seed=811)
    probe = _SkipAfterCheckpoint(("E",))
    search = run_fit_search(FitSearchRequest("curve", problem), cancelled=probe, checkpoint=probe.checkpoints.append)
    return problem, search, probe.checkpoints[-1]


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("stage", ["A", "B"])
def test_early_skip_returns_only_committed_evidence(joint, stage) -> None:
    pending = stage == "A"
    checkpoints = []

    def progress(value):
        nonlocal pending
        if stage == "B" and value.stage == "A" and value.completed == value.total:
            pending = True

    def probe():
        nonlocal pending
        if pending:
            pending = False
            raise StageSkipped("early skip")
        return False

    if joint:
        results = run_joint_fit(
            JointFitRequest(_joint_problem()), cancelled=probe, progress=progress, checkpoint=checkpoints.append
        )
    else:
        results = (
            run_fit_search(
                FitSearchRequest("curve", _problem()), cancelled=probe, progress=progress, checkpoint=checkpoints.append
            ),
        )
    assert checkpoints == []
    for result in results:
        assert result.skipped_stages == (stage,)
        assert tuple(value.stage for value in result.stage_summaries) == (() if stage == "A" else ("A",))
        assert bool(result.candidates) == (joint and stage == "B")


def test_joint_skip_before_stage_a_preserves_preparation_warnings() -> None:
    problem = _joint_problem()
    local_problems = tuple(
        replace(local, warnings=("shared input warning", f"{dataset_id} input warning"))
        for dataset_id, local in zip(problem.dataset_ids, problem.problems, strict=True)
    )
    problem = replace(problem, problems=local_problems)

    def probe() -> bool:
        raise StageSkipped("skip before evaluation")

    results = run_joint_fit(JointFitRequest(problem), cancelled=probe)

    assert len(results) == 2
    for result in results:
        assert result.candidates == ()
        assert result.warnings == (
            "shared input warning",
            "left input warning",
            "right input warning",
            "Stage A skipped by user.",
        )


def test_locked_joint_still_consumes_skips_at_stage_boundaries() -> None:
    probe = _SkipAfterCheckpoint(("C", "D", "E"), joint=True)
    results = run_joint_fit(
        JointFitRequest(_fully_locked_joint_problem()), cancelled=probe, checkpoint=probe.checkpoints.append
    )
    assert tuple(probe.skipped) == ("C", "D", "E")
    assert all(result.skipped_stages == ("C", "D", "E") for result in results)


@pytest.mark.parametrize("joint", [False, True])
def test_checkpoint_callback_skip_exception_is_not_a_solver_skip(joint) -> None:
    def checkpoint(_value):
        raise StageSkipped("checkpoint callback sentinel")

    with pytest.raises(StageSkipped, match="checkpoint callback sentinel"):
        if joint:
            run_joint_fit(JointFitRequest(_joint_problem()), checkpoint=checkpoint)
        else:
            run_fit_search(FitSearchRequest("curve", _problem()), checkpoint=checkpoint)


def test_analysis_preserves_skip_evidence_when_final_stage_completed(skipped_c) -> None:
    problem, search, _checkpoint = skipped_c
    result = run_analysis(AnalysisRequest("curve", problem, search, profile_names=(), bootstrap_enabled=False))
    assert result.skipped_stages == ("C",)
    assert result.uncertainty is not None


def test_analysis_reports_skipped_final_stage_without_uncertainty(skipped_e) -> None:
    problem, search, _checkpoint = skipped_e
    progress = []
    result = run_analysis(AnalysisRequest("curve", problem, search), progress=progress.append)
    assert result.skipped_stages == ("E",)
    assert result.confidence is ConfidenceClass.UNTRUSTED
    assert result.uncertainty is None
    assert result.classification_evidence
    assert not any(value.stage in {"bootstrap", "profile"} for value in progress)


@pytest.mark.parametrize("partial", [False, True])
def test_joint_analysis_does_not_claim_complete_skipped_ensemble(partial) -> None:
    problem = _joint_problem()
    checkpoints = []
    skipped = []

    def probe():
        trigger = "E" if partial else "D"
        if checkpoints and checkpoints[-1][0].stage == trigger and not skipped:
            skipped.append("E")
            raise StageSkipped("skip E")
        return False

    searches = run_joint_fit(JointFitRequest(problem), cancelled=probe, checkpoint=checkpoints.append)
    results = fitting._analyze_joint_searches(problem, searches, ((), ()))
    for result in results:
        assert result.skipped_stages == ("E",)
        assert result.confidence is ConfidenceClass.UNTRUSTED
        assert result.uncertainty is None
        assert result.classification_evidence


@pytest.mark.parametrize("automatic", [False, True])
def test_service_stops_analysis_and_recovery_after_final_skip(monkeypatch, automatic) -> None:
    problem = _problem(seed=811)
    prepared = PreparedDatasetFit("curve", 0, dataset_project(), problem)
    probe = _SkipAfterCheckpoint(("E",))

    def unexpected(*_args, **_kwargs):
        pytest.fail("an incomplete search must not enter analysis or profile recovery")

    monkeypatch.setattr(fitting, "run_analysis", unexpected)
    monkeypatch.setattr(fitting, "recover_profile_basin", unexpected)
    run = fitting.fit_automatic_prepared_dataset if automatic else fitting.fit_prepared_dataset
    value = run(prepared, cancelled=probe, checkpoint=probe.checkpoints.append)
    result = value.fit_result if automatic else value
    assert result.skipped_stages == ("E",)
    assert result.confidence is ConfidenceClass.UNTRUSTED
    assert result.uncertainty is None
    assert result.classification_evidence
    if automatic:
        assert not value.passed
        assert "skip" in value.reason.lower()


def test_profile_replacement_and_checkpoint_preserve_prior_skip(skipped_c) -> None:
    problem, search, _checkpoint = skipped_c
    finals = pipeline._profile_stage_candidates(search)
    replaced = pipeline._replace_profile_stage(problem, search, finals, finals)
    assert replaced is not None
    assert replaced.skipped_stages == ("C",)
    checkpoint = pipeline._profile_checkpoint(problem, replaced)
    assert checkpoint.skipped_stages == ("C",)
    validate_resume_checkpoint(problem, checkpoint, reserved_child_seeds=search.child_seeds)


def test_public_result_project_and_pickle_keep_skip_ledger(skipped_e) -> None:
    _problem, search, checkpoint = skipped_e
    result = FitResult.from_search(search, confidence=ConfidenceClass.UNTRUSTED, uncertainty=None)
    value = project(replace(dataset_project(result=result), checkpoint=checkpoint))
    restored = project_from_dict(project_to_dict(value))
    assert restored.datasets[0].checkpoint.skipped_stages == ("E",)
    assert restored.datasets[0].last_valid_result.skipped_stages == ("E",)
    for original in (search, checkpoint, result):
        decoded = pickle.loads(pickle.dumps(original))
        assert decoded.skipped_stages == ("E",)
        assert not decoded.candidates[0].unit_vector.flags.writeable


@pytest.mark.parametrize("invalid", ["C", ("C", "C"), ("E", "C"), ("uncertainty",), (1,)])
def test_malformed_skip_ledgers_are_rejected(skipped_c, invalid) -> None:
    _problem, search, checkpoint = skipped_c
    for original in (search, checkpoint):
        with pytest.raises((ValueError, TypeError), match="skipped_stages"):
            replace(original, skipped_stages=invalid)


def test_empty_ledger_keeps_legacy_encoding_and_provenance(skipped_c) -> None:
    problem, search, checkpoint = skipped_c
    empty = replace(search, skipped_stages=())
    payload = {
        field.name: getattr(empty, field.name)
        for field in fields(empty)
        if field.name not in {"skipped_stages", "provenance_sha256"}
    }
    expected = _identity_sha256({"context": _context_identity(problem), "result": payload})
    assert fit_search_provenance_sha256(problem, empty) == expected
    assert fit_search_provenance_sha256(problem, search) != expected
    result = FitResult.from_search(empty, confidence=ConfidenceClass.UNTRUSTED, uncertainty=None)
    assert "skipped_stages" not in fit_result_to_dict(result)
    assert "skipped_stages" not in _checkpoint_to_dict(replace(checkpoint, skipped_stages=()))


def test_profile_continuation_does_not_restart_a_skipped_final_stage(monkeypatch, skipped_e) -> None:
    problem, search, _checkpoint = skipped_e

    def unexpected(*_args, **_kwargs):
        pytest.fail("profile continuation must not restart a terminal skip")

    monkeypatch.setattr(pipeline, "evaluate_model", unexpected)
    continued = pipeline.continue_profile_basin(
        problem, search, search.best_candidate.unit_vector, parameter_name="component.0.thickness_a"
    )
    assert continued is search


def test_incomplete_result_cannot_be_relabelled_trusted(skipped_e) -> None:
    _problem, search, _checkpoint = skipped_e
    result = FitResult.from_search(search, confidence=ConfidenceClass.UNTRUSTED, uncertainty=None)
    with pytest.raises(ValueError, match="incomplete|skipped"):
        replace(result, confidence=ConfidenceClass.TRUSTED)
